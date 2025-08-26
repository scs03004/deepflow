#!/usr/bin/env python3
"""
Deepflow MCP Server
==================

Model Context Protocol server implementation for Deepflow tools.
Exposes deepflow functionality through MCP protocol for AI assistants.

This server provides MCP tools for:
- Dependency visualization and analysis
- Code quality analysis  
- Architecture validation
- Technical debt assessment
- AI context optimization

Usage:
    python -m deepflow.mcp.server
    deepflow-mcp-server
    
Requirements:
    pip install deepflow[mcp]
"""

import asyncio
import json
import sys
import hashlib
import time
import os
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Import our enhanced error handling
try:
    from .error_handler import setup_mcp_error_handling, with_error_handling, ErrorContext, PerformanceMetrics
    ERROR_HANDLING_AVAILABLE = True
except ImportError:
    ERROR_HANDLING_AVAILABLE = False
    print("WARNING: Enhanced error handling not available")

# Import real-time intelligence system
try:
    from .realtime_intelligence import (
        RealTimeIntelligenceEngine, 
        RealTimeNotificationService,
        get_intelligence_engine,
        get_notification_service
    )
    REALTIME_AVAILABLE = True
except ImportError as e:
    REALTIME_AVAILABLE = False
    print(f"WARNING: Real-time intelligence not available: {e}")

# Graceful MCP imports
try:
    from mcp.server import Server
    from mcp.types import (
        Tool,
        TextContent,
        ImageContent,
        CallToolResult,
        ListToolsRequest,
        CallToolRequest,
    )
    # from mcp.server.models import ListToolsResult  # Not available in this MCP version
    import mcp.server.stdio
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    # Note: Don't exit here to allow module import for testing
    # Exit will happen in main() or async_main() when actually trying to run server
    
    # Provide fallback definitions for testing when MCP is not available
    class Tool:
        def __init__(self, name, description, inputSchema):
            self.name = name
            self.description = description
            self.inputSchema = inputSchema
    
    class TextContent:
        def __init__(self, type, text):
            self.type = type
            self.text = text
    
    class CallToolResult:
        def __init__(self, content):
            self.content = content
    
    class ListToolsResult:
        def __init__(self, tools):
            self.tools = tools
    
    class Server:
        def __init__(self, name):
            self.name = name
        def call_tool(self):
            return lambda func: func
        def run(self, *args, **kwargs):
            pass
        def create_initialization_options(self):
            return {}
    
    # Create a dummy stdio module structure
    class stdio:
        @staticmethod
        async def stdio_server():
            # Return a context manager that yields dummy streams
            class DummyContext:
                async def __aenter__(self):
                    return None, None
                async def __aexit__(self, *args):
                    pass
            return DummyContext()
    
    # Create a dummy mcp module structure for reference
    class mcp:
        class server:
            stdio = stdio()

# Import deepflow tools - graceful fallbacks
try:
    import sys
    
    # Add tools directory to path for imports
    tools_dir = Path(__file__).parent.parent.parent / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    
    from dependency_visualizer import DependencyVisualizer, DependencyAnalyzer
    from code_analyzer import CodeAnalyzer
    from pre_commit_validator import DependencyValidator
    from doc_generator import DocumentationGenerator
    
    TOOLS_AVAILABLE = True
except ImportError as e:
    TOOLS_AVAILABLE = False
    print(f"WARNING: Could not import deepflow tools: {e}")

# Configure logging with enhanced error handling if available
if ERROR_HANDLING_AVAILABLE:
    error_handler = setup_mcp_error_handling("deepflow.mcp")
    logger = error_handler.logger
else:
    # Fallback logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    error_handler = None

class DeepflowMCPServer:
    """MCP Server for Deepflow tools with performance optimizations."""
    
    def __init__(self):
        """Initialize the Deepflow MCP server."""
        self.server = Server("deepflow")
        
        # Performance optimization: Add caching system
        self._cache: Dict[str, Any] = {}
        self._cache_ttl: Dict[str, float] = {}
        self._cache_timeout = 300  # 5 minutes cache timeout
        
        # Lazy loading: Initialize tool instances only when needed
        self._tool_instances: Dict[str, Any] = {}
        
        # Performance monitoring
        self._stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_response_time': 0.0,
            'errors': 0
        }
        
        # Real-time intelligence integration
        self._realtime_engine: Optional[RealTimeIntelligenceEngine] = None
        self._notification_service: Optional[RealTimeNotificationService] = None
        self._realtime_monitoring = False
        
        self._setup_tools()
        
    def _setup_tools(self):
        """Set up MCP tool handlers."""
        
        # Register the list_tools handler
        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """Handle list tools request."""
            return self.get_tools()
        
        # Single call_tool handler that routes to the appropriate function
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict):
            """Route tool calls to appropriate handlers."""
            
            if name == "analyze_dependencies":
                return await self._handle_analyze_dependencies(arguments)
            elif name == "analyze_code_quality":
                return await self._handle_analyze_code_quality(arguments)
            elif name == "validate_commit":
                return await self._handle_validate_commit(arguments)
            elif name == "generate_documentation":
                return await self._handle_generate_documentation(arguments)
            elif name == "start_realtime_monitoring":
                return await self._handle_start_realtime_monitoring(arguments)
            elif name == "stop_realtime_monitoring":
                return await self._handle_stop_realtime_monitoring(arguments)
            elif name == "get_realtime_activity":
                return await self._handle_get_realtime_activity(arguments)
            elif name == "get_realtime_stats":
                return await self._handle_get_realtime_stats(arguments)
            else:
                return [TextContent(
                    type="text",
                    text=f"Unknown tool: {name}"
                )]

    def _get_cache_key(self, tool_name: str, arguments: dict) -> str:
        """Generate cache key from tool name and arguments."""
        # Create a deterministic hash of the arguments
        cache_data = f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"
        return hashlib.md5(cache_data.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is still valid."""
        if cache_key not in self._cache_ttl:
            return False
        return time.time() - self._cache_ttl[cache_key] < self._cache_timeout
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get value from cache if valid."""
        if self._is_cache_valid(cache_key):
            self._stats['cache_hits'] += 1
            logger.debug(f"Cache hit for key: {cache_key[:8]}...")
            return self._cache.get(cache_key)
        self._stats['cache_misses'] += 1
        logger.debug(f"Cache miss for key: {cache_key[:8]}...")
        return None
    
    def _set_cache(self, cache_key: str, value: Any) -> None:
        """Set value in cache."""
        self._cache[cache_key] = value
        self._cache_ttl[cache_key] = time.time()
        logger.debug(f"Cached result for key: {cache_key[:8]}...")
    
    def _get_tool_instance(self, tool_class, *args, **kwargs):
        """Lazy loading of tool instances with caching."""
        instance_key = f"{tool_class.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
        
        if instance_key not in self._tool_instances:
            logger.debug(f"Creating new instance of {tool_class.__name__}")
            self._tool_instances[instance_key] = tool_class(*args, **kwargs)
        else:
            logger.debug(f"Reusing cached instance of {tool_class.__name__}")
        
        return self._tool_instances[instance_key]
    
    def _update_stats(self, start_time: float, error: bool = False) -> None:
        """Update performance statistics."""
        self._stats['total_requests'] += 1
        
        if error:
            self._stats['errors'] += 1
        
        response_time = time.time() - start_time
        
        # Update average response time (moving average)
        if self._stats['avg_response_time'] == 0:
            self._stats['avg_response_time'] = response_time
        else:
            self._stats['avg_response_time'] = (
                (self._stats['avg_response_time'] * (self._stats['total_requests'] - 1) + response_time) / 
                self._stats['total_requests']
            )
        
        logger.info(f"Request completed in {response_time:.3f}s (avg: {self._stats['avg_response_time']:.3f}s)")
    
    def _should_invalidate_cache(self, project_path: str) -> bool:
        """Check if project files have changed and cache should be invalidated."""
        try:
            # Simple file modification time check for basic cache invalidation
            project_path_obj = Path(project_path)
            if not project_path_obj.exists():
                return True
            
            # Get the most recent modification time of Python files
            latest_mod_time = 0
            for py_file in project_path_obj.rglob('*.py'):
                try:
                    mod_time = py_file.stat().st_mtime
                    latest_mod_time = max(latest_mod_time, mod_time)
                except (OSError, IOError):
                    continue
            
            # Check if any file is newer than our cache timeout
            cache_creation_time = time.time() - self._cache_timeout
            return latest_mod_time > cache_creation_time
            
        except Exception as e:
            logger.warning(f"Cache invalidation check failed: {e}")
            return True  # Err on side of caution
    
    async def _handle_analyze_dependencies(self, arguments: dict, request_id: str = None):
        """Analyze project dependencies and create visualization with caching."""
        start_time = time.time()
        
        try:
            # Performance optimization: Check cache first
            cache_key = self._get_cache_key('analyze_dependencies', arguments)
            cached_result = self._get_from_cache(cache_key)
            
            if cached_result and not self._should_invalidate_cache(arguments.get("project_path", ".")):
                logger.info("Returning cached dependency analysis result")
                self._update_stats(start_time)
                return cached_result
            project_path = arguments.get("project_path", ".")
            output_format = arguments.get("format", "text")
            ai_awareness = arguments.get("ai_awareness", True)
            
            if not TOOLS_AVAILABLE:
                return [TextContent(
                    type="text",
                    text="Error: Deepflow tools not available. Please check installation."
                )]
            
            # Use lazy loading for tool instances
            analyzer = self._get_tool_instance(DependencyAnalyzer, project_path, ai_awareness=ai_awareness)
            dependency_graph = analyzer.analyze_project()
            visualizer = self._get_tool_instance(DependencyVisualizer, dependency_graph)
            
            if output_format == "html":
                # Generate HTML output to a temporary file and return the content
                import tempfile
                import os
                with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp:
                    output_path = tmp.name
                visualizer.generate_mermaid_html(output_path, ai_awareness)
                
                # Read the generated file content
                with open(output_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                
                # Clean up temp file
                os.unlink(output_path)
                
                result = [TextContent(
                    type="text", 
                    text=f"HTML visualization generated with {len(html_content)} characters"
                )]
                
                # Cache the result
                self._set_cache(cache_key, result)
                self._update_stats(start_time)
                return result
            elif output_format == "json":
                # For JSON, we need to convert the graph to a JSON-serializable format
                json_data = {
                    "nodes": [
                        {
                            "name": name,
                            "file_path": str(node.file_path),
                            "imports": list(node.imports),
                            "token_count": getattr(node, 'token_count', 0)
                        }
                        for name, node in dependency_graph.nodes.items()
                    ]
                }
                result = [TextContent(
                        type="text",
                        text=json.dumps(json_data, indent=2)
                    )]
                
                # Cache the result
                self._set_cache(cache_key, result)
                self._update_stats(start_time)
                return result
            else:
                text_output = visualizer.generate_text_tree()
                result = [TextContent(
                        type="text",
                        text=text_output
                    )]
                
                # Cache the result
                self._set_cache(cache_key, result)
                self._update_stats(start_time)
                return result
                
        except Exception as e:
            logger.error(f"Error in analyze_dependencies: {e}", exc_info=True)
            self._update_stats(start_time, error=True)
            return [TextContent(
                    type="text",
                    text=f"Error analyzing dependencies: {str(e)}"
                )]

    async def _handle_analyze_code_quality(self, arguments: dict, request_id: str = None):
        """Analyze code quality and detect issues with performance optimizations."""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._get_cache_key('analyze_code_quality', arguments)
            cached_result = self._get_from_cache(cache_key)
            
            if cached_result and not self._should_invalidate_cache(arguments.get("project_path", ".")):
                logger.info("Returning cached code quality analysis result")
                self._update_stats(start_time)
                return cached_result
            project_path = arguments.get("project_path", ".")
            analysis_type = arguments.get("analysis_type", "all")
            fix_imports = arguments.get("fix_imports", False)
            
            if not TOOLS_AVAILABLE:
                return [TextContent(
                    type="text",
                    text="Error: Deepflow tools not available. Please check installation."
                )]
            
            # Use lazy loading for analyzer
            analyzer = self._get_tool_instance(CodeAnalyzer, project_path)
            results = {}
            
            if analysis_type in ["all", "imports"]:
                import_analysis = analyzer.analyze_unused_imports(fix_mode=fix_imports)
                results["unused_imports"] = [
                    {
                        "file": analysis.file_path,
                        "import": analysis.import_name,
                        "used": analysis.is_used,
                        "suggestions": analysis.suggestions
                    }
                    for analysis in import_analysis
                ]
            
            if analysis_type in ["all", "coupling"]:
                coupling_analysis = analyzer.analyze_coupling()
                results["coupling_metrics"] = [
                    {
                        "module_a": metric.module_a,
                        "module_b": metric.module_b,
                        "strength": metric.coupling_strength,
                        "type": metric.coupling_type,
                        "refactoring_opportunity": metric.refactoring_opportunity
                    }
                    for metric in coupling_analysis
                ]
            
            if analysis_type in ["all", "architecture"]:
                arch_violations = analyzer.detect_architecture_violations()
                results["architecture_violations"] = [
                    {
                        "file": violation.file_path,
                        "type": violation.violation_type,
                        "severity": violation.severity,
                        "description": violation.description,
                        "suggestion": violation.suggestion
                    }
                    for violation in arch_violations
                ]
            
            if analysis_type in ["all", "debt"]:
                tech_debt = analyzer.calculate_technical_debt()
                results["technical_debt"] = [
                    {
                        "file": debt.file_path,
                        "score": debt.debt_score,
                        "priority": debt.refactoring_priority,
                        "effort": debt.estimated_effort,
                        "indicators": debt.debt_indicators
                    }
                    for debt in tech_debt
                ]
            
            if analysis_type in ["all", "ai_context"]:
                ai_analysis = analyzer.analyze_ai_context_windows()
                results["ai_context_analysis"] = [
                    {
                        "file": analysis.file_path,
                        "token_count": analysis.token_count,
                        "health": analysis.context_health,
                        "split_points": analysis.estimated_split_points,
                        "ai_score": analysis.ai_friendliness_score
                    }
                    for analysis in ai_analysis
                ]
            
            result = [TextContent(
                    type="text",
                    text=json.dumps(results, indent=2)
                )]
            
            # Cache the result
            self._set_cache(cache_key, result)
            self._update_stats(start_time)
            return result
            
        except Exception as e:
            logger.error(f"Error in analyze_code_quality: {e}", exc_info=True)
            self._update_stats(start_time, error=True)
            return [TextContent(
                    type="text",
                    text=f"Error analyzing code quality: {str(e)}"
                )]

    async def _handle_validate_commit(self, arguments: dict):
        """Validate code changes before commit with caching."""
        start_time = time.time()
        
        try:
            project_path = arguments.get("project_path", ".")
            check_dependencies = arguments.get("check_dependencies", True)
            check_patterns = arguments.get("check_patterns", True)
            
            if not TOOLS_AVAILABLE:
                return [TextContent(
                    type="text",
                    text="Error: Deepflow tools not available. Please check installation."
                )]
            
            # Get changed files from git status
            import subprocess
            try:
                result = subprocess.run(['git', 'status', '--porcelain'], 
                                      capture_output=True, text=True, cwd=project_path)
                changed_files = []
                for line in result.stdout.strip().split('\n'):
                    if line and line.endswith('.py'):
                        # Extract filename from git status format (skip first 3 chars for status)
                        filename = line[3:].strip()  # Skip status flags like " M " or "?? "
                        changed_files.append(filename)
                
                if not changed_files:
                    return [TextContent(
                        type="text",
                        text=json.dumps({
                            "valid": True,
                            "message": "No Python files changed",
                            "changed_files": []
                        }, indent=2)
                    )]
                
                # Use lazy loading for validator
                validator = self._get_tool_instance(DependencyValidator, project_path)
                
                results = {}
                
                if check_dependencies:
                    # Validate imports in changed files
                    import_results = validator.validate_imports(changed_files)
                    results["import_validation"] = [
                        {
                            "file": r.file_path,
                            "issues": r.issues,
                            "warnings": r.warnings,
                            "risk_level": r.risk_level,
                            "requires_testing": r.requires_testing,
                            "affected_components": r.affected_components,
                            "valid": len(r.issues) == 0  # Derive validity from issues
                        }
                        for r in import_results
                    ]
                
                if check_patterns:
                    # Analyze change impact
                    impact = validator.analyze_change_impact(changed_files)
                    results["change_impact"] = {
                        "risk_assessment": impact.risk_assessment,
                        "affected_modules": list(impact.affected_modules),
                        "required_tests": impact.required_tests,
                        "documentation_updates": impact.documentation_updates,
                        "deployment_impact": impact.deployment_impact
                    }
                
                # Determine overall validation status
                all_valid = True
                if "import_validation" in results:
                    all_valid = all(len(r["issues"]) == 0 for r in results["import_validation"])
                
                results["valid"] = all_valid
                results["changed_files"] = changed_files
                
                result = [TextContent(
                        type="text",
                        text=json.dumps(results, indent=2)
                    )]
                
                self._update_stats(start_time)
                return result
                
            except subprocess.CalledProcessError:
                result = [TextContent(
                    type="text",
                    text=json.dumps({
                        "valid": False,
                        "error": "Not a git repository or git not available"
                    }, indent=2)
                )]
                
                self._update_stats(start_time, error=True)
                return result
            
        except Exception as e:
            logger.error(f"Error in validate_commit: {e}", exc_info=True)
            self._update_stats(start_time, error=True)
            return [TextContent(
                    type="text",
                    text=f"Error validating commit: {str(e)}"
                )]

    async def _handle_generate_documentation(self, arguments: dict):
        """Generate project documentation with caching."""
        start_time = time.time()
        
        try:
            # Check cache for documentation generation
            cache_key = self._get_cache_key('generate_documentation', arguments)
            cached_result = self._get_from_cache(cache_key)
            
            if cached_result and not self._should_invalidate_cache(arguments.get("project_path", ".")):
                logger.info("Returning cached documentation generation result")
                self._update_stats(start_time)
                return cached_result
            project_path = arguments.get("project_path", ".")
            doc_type = arguments.get("doc_type", "dependency_map")
            output_path = arguments.get("output_path", None)
            
            if not TOOLS_AVAILABLE:
                return [TextContent(
                    type="text",
                    text="Error: Deepflow tools not available. Please check installation."
                )]
            
            # Use lazy loading for documentation generator
            doc_generator = self._get_tool_instance(DocumentationGenerator, project_path)
            
            if doc_type == "dependency_map":
                output_file = doc_generator.generate_dependency_map(output_path)
            elif doc_type == "architecture_overview":
                output_file = doc_generator.generate_architecture_overview(output_path)
            elif doc_type == "api_docs":
                output_file = doc_generator.generate_api_docs(output_path)
            else:
                return [TextContent(
                        type="text",
                        text=f"Unknown documentation type: {doc_type}"
                    )]
            
            result = [TextContent(
                    type="text",
                    text=f"Documentation generated: {output_file}"
                )]
            
            # Cache the result
            self._set_cache(cache_key, result)
            self._update_stats(start_time)
            return result
            
        except Exception as e:
            logger.error(f"Error in generate_documentation: {e}", exc_info=True)
            self._update_stats(start_time, error=True)
            return [TextContent(
                    type="text",
                    text=f"Error generating documentation: {str(e)}"
                )]

    async def _handle_start_realtime_monitoring(self, arguments: dict):
        """Start real-time file monitoring and analysis."""
        if not REALTIME_AVAILABLE:
            return [TextContent(
                type="text",
                text="Error: Real-time intelligence not available. Install with: pip install deepflow[mcp]"
            )]
        
        try:
            project_path = arguments.get("project_path", ".")
            ai_awareness = arguments.get("ai_awareness", True)
            
            # Initialize real-time engine
            self._realtime_engine = get_intelligence_engine(project_path, ai_awareness)
            self._notification_service = get_notification_service()
            
            # Set up MCP notification callback
            async def mcp_notification_callback(notification_data):
                """Handle real-time notifications for MCP clients."""
                logger.info(f"Real-time notification: {notification_data['type']}")
                # Store notification for retrieval by get_realtime_activity
                if not hasattr(self, '_realtime_notifications'):
                    self._realtime_notifications = []
                self._realtime_notifications.append(notification_data)
                # Keep only last 100 notifications
                if len(self._realtime_notifications) > 100:
                    self._realtime_notifications = self._realtime_notifications[-100:]
            
            # Add callback to engine
            self._realtime_engine.add_notification_callback(mcp_notification_callback)
            
            # Start monitoring
            success = await self._realtime_engine.start_monitoring()
            
            if success:
                self._realtime_monitoring = True
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "started",
                        "message": f"Real-time monitoring started for {project_path}",
                        "ai_awareness": ai_awareness,
                        "features": [
                            "Live file watching",
                            "Incremental dependency analysis",
                            "Architectural violation detection", 
                            "AI context window monitoring"
                        ]
                    }, indent=2)
                )]
            else:
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "status": "failed",
                        "error": "Could not start real-time monitoring"
                    }, indent=2)
                )]
                
        except Exception as e:
            logger.error(f"Error starting real-time monitoring: {e}", exc_info=True)
            return [TextContent(
                type="text",
                text=f"Error starting real-time monitoring: {str(e)}"
            )]

    async def _handle_stop_realtime_monitoring(self, arguments: dict):
        """Stop real-time file monitoring."""
        if not REALTIME_AVAILABLE or not self._realtime_engine:
            return [TextContent(
                type="text",
                text="Real-time monitoring not running"
            )]
        
        try:
            await self._realtime_engine.stop_monitoring()
            self._realtime_monitoring = False
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "status": "stopped",
                    "message": "Real-time monitoring stopped"
                }, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Error stopping real-time monitoring: {e}", exc_info=True)
            return [TextContent(
                type="text",
                text=f"Error stopping real-time monitoring: {str(e)}"
            )]

    async def _handle_get_realtime_activity(self, arguments: dict):
        """Get recent real-time monitoring activity."""
        if not REALTIME_AVAILABLE or not self._realtime_engine:
            return [TextContent(
                type="text",
                text="Real-time monitoring not available or not started"
            )]
        
        try:
            limit = arguments.get("limit", 20)
            activity = self._realtime_engine.get_recent_activity(limit)
            
            return [TextContent(
                type="text",
                text=json.dumps(activity, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Error getting real-time activity: {e}", exc_info=True)
            return [TextContent(
                type="text",
                text=f"Error getting real-time activity: {str(e)}"
            )]

    async def _handle_get_realtime_stats(self, arguments: dict):
        """Get real-time monitoring statistics."""
        if not REALTIME_AVAILABLE or not self._realtime_engine:
            return [TextContent(
                type="text",
                text="Real-time monitoring not available or not started"
            )]
        
        try:
            stats = self._realtime_engine.get_real_time_stats()
            
            # Add recent notifications if available
            if hasattr(self, '_realtime_notifications'):
                stats['recent_notifications'] = len(self._realtime_notifications)
            
            return [TextContent(
                type="text",
                text=json.dumps(stats, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Error getting real-time stats: {e}", exc_info=True)
            return [TextContent(
                type="text", 
                text=f"Error getting real-time stats: {str(e)}"
            )]

    def get_tools(self) -> List[Tool]:
        """Get available MCP tools."""
        return [
            Tool(
                name="analyze_dependencies",
                description="Analyze project dependencies and create visualizations",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Path to the project to analyze",
                            "default": "."
                        },
                        "format": {
                            "type": "string",
                            "enum": ["text", "html", "json"],
                            "description": "Output format for the analysis",
                            "default": "text"
                        },
                        "ai_awareness": {
                            "type": "boolean",
                            "description": "Enable AI-aware dependency analysis",
                            "default": True
                        }
                    }
                }
            ),
            Tool(
                name="analyze_code_quality",
                description="Analyze code quality, detect unused imports, coupling issues, and technical debt",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string", 
                            "description": "Path to the project to analyze",
                            "default": "."
                        },
                        "analysis_type": {
                            "type": "string",
                            "enum": ["all", "imports", "coupling", "architecture", "debt", "ai_context"],
                            "description": "Type of analysis to perform",
                            "default": "all"
                        },
                        "fix_imports": {
                            "type": "boolean",
                            "description": "Automatically fix unused imports",
                            "default": False
                        }
                    }
                }
            ),
            Tool(
                name="validate_commit",
                description="Validate code changes before commit using pre-commit hooks",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Path to the project to validate", 
                            "default": "."
                        },
                        "check_dependencies": {
                            "type": "boolean",
                            "description": "Check for dependency issues",
                            "default": True
                        },
                        "check_patterns": {
                            "type": "boolean", 
                            "description": "Check for pattern consistency",
                            "default": True
                        }
                    }
                }
            ),
            Tool(
                name="generate_documentation",
                description="Generate project documentation including dependency maps and architecture overviews",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Path to the project",
                            "default": "."
                        },
                        "doc_type": {
                            "type": "string",
                            "enum": ["dependency_map", "architecture_overview", "api_docs"],
                            "description": "Type of documentation to generate",
                            "default": "dependency_map"
                        },
                        "output_path": {
                            "type": "string",
                            "description": "Output path for generated documentation",
                            "default": None
                        }
                    }
                }
            ),
            # Real-time intelligence tools
            Tool(
                name="start_realtime_monitoring",
                description="Start real-time file monitoring and incremental dependency analysis",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Path to the project to monitor",
                            "default": "."
                        },
                        "ai_awareness": {
                            "type": "boolean", 
                            "description": "Enable AI-aware analysis features",
                            "default": True
                        }
                    }
                }
            ),
            Tool(
                name="stop_realtime_monitoring",
                description="Stop real-time file monitoring",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            Tool(
                name="get_realtime_activity",
                description="Get recent real-time monitoring activity and events",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "number",
                            "description": "Maximum number of events to return",
                            "default": 20
                        }
                    }
                }
            ),
            Tool(
                name="get_realtime_stats",
                description="Get real-time monitoring statistics and status",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            )
        ]

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        cache_hit_rate = (
            self._stats['cache_hits'] / (self._stats['cache_hits'] + self._stats['cache_misses'])
            if (self._stats['cache_hits'] + self._stats['cache_misses']) > 0
            else 0
        )
        
        return {
            **self._stats,
            'cache_size': len(self._cache),
            'cache_hit_rate': f"{cache_hit_rate:.2%}",
            'active_tool_instances': len(self._tool_instances)
        }
    
    def cleanup_cache(self, max_age_seconds: Optional[float] = None) -> int:
        """Clean up expired cache entries."""
        if max_age_seconds is None:
            max_age_seconds = self._cache_timeout
        
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self._cache_ttl.items()
            if current_time - timestamp > max_age_seconds
        ]
        
        for key in expired_keys:
            self._cache.pop(key, None)
            self._cache_ttl.pop(key, None)
        
        logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        return len(expired_keys)
    
    async def run(self):
        """Run the MCP server with performance monitoring."""
        logger.info("Starting Deepflow MCP server with performance optimizations")
        logger.info(f"Cache timeout: {self._cache_timeout}s")
        
        try:
            async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
                # Log startup stats
                stats = self.get_performance_stats()
                logger.info(f"Server initialized - {stats}")
                
                await self.server.run(
                    read_stream,
                    write_stream,
                    self.server.create_initialization_options()
                )
        except Exception as e:
            logger.error(f"MCP server error during run: {e}", exc_info=True)
            raise
        finally:
            # Log final statistics
            stats = self.get_performance_stats()
            logger.info(f"Server shutting down - Final stats: {stats}")
            
            # Cleanup on shutdown
            self.cleanup_cache()


async def async_main():
    """Async main entry point for the MCP server."""
    if not MCP_AVAILABLE:
        print("ERROR: MCP dependencies not found. Install with: pip install mcp")
        sys.exit(1)
        return  # This line won't be reached in normal execution, but helps in tests
    
    try:
        server = DeepflowMCPServer()
        await server.run()
    except KeyboardInterrupt:
        logger.info("Shutting down MCP server gracefully...")
    except Exception as e:
        logger.error(f"MCP server error: {e}")
        raise


def main():
    """Sync entry point for the MCP server (called by setuptools)."""
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        # Handle gracefully without propagating the exception
        print("\nMCP server stopped.")
    except Exception as e:
        print(f"Error running MCP server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()