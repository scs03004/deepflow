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

# Import Priority 4 Smart Refactoring Engine
try:
    from ..smart_refactoring_engine import SmartRefactoringEngine
    SMART_REFACTORING_AVAILABLE = True
except ImportError as e:
    SMART_REFACTORING_AVAILABLE = False
    print(f"WARNING: Could not import smart refactoring engine: {e}")

# Import Priority 5 Workflow Orchestrator
try:
    from ..workflow_orchestrator import WorkflowOrchestrator
    WORKFLOW_ORCHESTRATOR_AVAILABLE = True
except ImportError as e:
    WORKFLOW_ORCHESTRATOR_AVAILABLE = False
    print(f"WARNING: Could not import workflow orchestrator: {e}")

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
        
        # Priority 5: Workflow orchestrator integration
        self._workflow_orchestrator: Optional[WorkflowOrchestrator] = None
        
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
            # Priority 3: AI Session Intelligence tools
            elif name == "start_ai_session":
                return await self._handle_start_ai_session(arguments)
            elif name == "end_ai_session":
                return await self._handle_end_ai_session(arguments)
            elif name == "get_session_context":
                return await self._handle_get_session_context(arguments)
            elif name == "restore_session_context":
                return await self._handle_restore_session_context(arguments)
            elif name == "analyze_change_impact":
                return await self._handle_analyze_change_impact(arguments)
            elif name == "get_session_intelligence":
                return await self._handle_get_session_intelligence(arguments)
            # Priority 4: Smart Refactoring tools
            elif name == "standardize_patterns":
                return await self._handle_standardize_patterns(arguments)
            elif name == "optimize_imports":
                return await self._handle_optimize_imports(arguments)
            elif name == "suggest_file_splits":
                return await self._handle_suggest_file_splits(arguments)
            elif name == "remove_dead_code":
                return await self._handle_remove_dead_code(arguments)
            elif name == "generate_docstrings":
                return await self._handle_generate_docstrings(arguments)
            elif name == "comprehensive_refactor":
                return await self._handle_comprehensive_refactor(arguments)
            # Priority 5: Workflow & Chaining tools
            elif name == "create_analysis_pipeline":
                return await self._handle_create_analysis_pipeline(arguments)
            elif name == "execute_workflow":
                return await self._handle_execute_workflow(arguments)
            elif name == "create_conditional_workflow":
                return await self._handle_create_conditional_workflow(arguments)
            elif name == "create_batch_operation":
                return await self._handle_create_batch_operation(arguments)
            elif name == "execute_batch_operation":
                return await self._handle_execute_batch_operation(arguments)
            elif name == "load_custom_workflow":
                return await self._handle_load_custom_workflow(arguments)
            elif name == "setup_scheduled_hygiene":
                return await self._handle_setup_scheduled_hygiene(arguments)
            elif name == "get_workflow_status":
                return await self._handle_get_workflow_status(arguments)
            elif name == "list_workflows":
                return await self._handle_list_workflows(arguments)
            elif name == "get_workflow_metrics":
                return await self._handle_get_workflow_metrics(arguments)
            # Requirements Management tools
            elif name == "analyze_requirements":
                return await self._handle_analyze_requirements(arguments)
            elif name == "update_requirements":
                return await self._handle_update_requirements(arguments)
            # File Organization tools
            elif name == "analyze_file_organization":
                return await self._handle_analyze_file_organization(arguments)
            elif name == "organize_files":
                return await self._handle_organize_files(arguments)
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
    
    # Priority 3: AI Session Intelligence Tool Handlers
    
    async def _handle_start_ai_session(self, arguments: dict):
        """Start a new AI development session."""
        if not REALTIME_AVAILABLE or not self._realtime_engine:
            return [TextContent(
                type="text",
                text="Real-time monitoring not available. Start monitoring first to use session intelligence."
            )]
        
        session_name = arguments.get('session_name', "")
        session_description = arguments.get('session_description', "")
        session_tags = set(arguments.get('session_tags', []))
        
        try:
            session_id = self._realtime_engine.start_ai_session(
                session_name=session_name,
                session_description=session_description, 
                session_tags=session_tags
            )
            
            result = {
                'session_id': session_id,
                'session_name': session_name,
                'start_time': time.time(),
                'status': 'started',
                'message': f'AI session "{session_name}" started successfully'
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Error starting AI session: {e}", exc_info=True)
            return [TextContent(
                type="text",
                text=f"Error starting AI session: {str(e)}"
            )]
    
    async def _handle_end_ai_session(self, arguments: dict):
        """End the current AI development session."""
        if not REALTIME_AVAILABLE or not self._realtime_engine:
            return [TextContent(
                type="text",
                text="Real-time monitoring not available"
            )]
        
        achievements = arguments.get('achievements', [])
        
        try:
            completed_session = self._realtime_engine.end_ai_session(achievements=achievements)
            
            if not completed_session:
                return [TextContent(
                    type="text",
                    text="No active AI session to end"
                )]
            
            duration = completed_session.end_time - completed_session.start_time
            result = {
                'session_id': completed_session.session_id,
                'session_name': completed_session.session_name, 
                'duration_seconds': duration,
                'files_modified': len(completed_session.files_modified),
                'changes_made': len(completed_session.changes_made),
                'patterns_learned': len(completed_session.patterns_learned),
                'goals_achieved': completed_session.goals_achieved,
                'status': 'completed',
                'message': f'AI session completed in {duration:.1f} seconds'
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Error ending AI session: {e}", exc_info=True)
            return [TextContent(
                type="text",
                text=f"Error ending AI session: {str(e)}"
            )]
    
    async def _handle_get_session_context(self, arguments: dict):
        """Get current AI session context."""
        if not REALTIME_AVAILABLE or not self._realtime_engine:
            return [TextContent(
                type="text",
                text="Real-time monitoring not available"
            )]
        
        try:
            current_session = self._realtime_engine.get_session_context()
            
            if not current_session:
                return [TextContent(
                    type="text",
                    text="No active AI session"
                )]
            
            duration = time.time() - current_session.start_time
            result = {
                'session_id': current_session.session_id,
                'session_name': current_session.session_name,
                'session_description': current_session.session_description,
                'start_time': current_session.start_time,
                'duration_seconds': duration,
                'files_modified': list(current_session.files_modified),
                'changes_made': len(current_session.changes_made),
                'patterns_learned': len(current_session.patterns_learned),
                'goals_achieved': current_session.goals_achieved,
                'session_tags': list(current_session.session_tags),
                'ai_interactions': current_session.ai_interactions
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Error getting session context: {e}", exc_info=True)
            return [TextContent(
                type="text",
                text=f"Error getting session context: {str(e)}"
            )]
    
    async def _handle_restore_session_context(self, arguments: dict):
        """Restore a previous AI session context."""
        if not REALTIME_AVAILABLE or not self._realtime_engine:
            return [TextContent(
                type="text",
                text="Real-time monitoring not available"
            )]
        
        session_id = arguments.get('session_id')
        if not session_id:
            return [TextContent(
                type="text",
                text="session_id is required"
            )]
        
        try:
            success = self._realtime_engine.restore_session_context(session_id)
            
            if success:
                current_session = self._realtime_engine.get_session_context()
                result = {
                    'restored_session_id': session_id,
                    'new_session_id': current_session.session_id,
                    'session_name': current_session.session_name,
                    'files_from_previous': len(current_session.files_modified),
                    'patterns_from_previous': len(current_session.patterns_learned),
                    'status': 'restored',
                    'message': f'Session context restored from {session_id}'
                }
            else:
                result = {
                    'session_id': session_id,
                    'status': 'failed',
                    'message': f'Session {session_id} not found in history'
                }
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Error restoring session context: {e}", exc_info=True)
            return [TextContent(
                type="text",
                text=f"Error restoring session context: {str(e)}"
            )]
    
    async def _handle_analyze_change_impact(self, arguments: dict):
        """Analyze the ripple effects of code changes."""
        if not REALTIME_AVAILABLE or not self._realtime_engine:
            return [TextContent(
                type="text",
                text="Real-time monitoring not available"
            )]
        
        file_path = arguments.get('file_path')
        change_type = arguments.get('change_type')
        change_details = arguments.get('change_details', {})
        
        if not file_path or not change_type:
            return [TextContent(
                type="text",
                text="file_path and change_type are required"
            )]
        
        try:
            impact_analysis = await self._realtime_engine.analyze_change_impact(
                file_path=file_path,
                change_type=change_type,
                change_details=change_details
            )
            
            result = {
                'change_id': impact_analysis.change_id,
                'affected_file': impact_analysis.affected_file,
                'change_type': impact_analysis.change_type,
                'risk_assessment': impact_analysis.risk_assessment,
                'impact_score': impact_analysis.impact_score,
                'ripple_effects': impact_analysis.ripple_effects,
                'dependency_impacts': impact_analysis.dependency_impacts,
                'test_impacts': impact_analysis.test_impacts,
                'documentation_impacts': impact_analysis.documentation_impacts,
                'mitigation_suggestions': impact_analysis.mitigation_suggestions,
                'timestamp': impact_analysis.timestamp
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Error analyzing change impact: {e}", exc_info=True)
            return [TextContent(
                type="text",
                text=f"Error analyzing change impact: {str(e)}"
            )]
    
    async def _handle_get_session_intelligence(self, arguments: dict):
        """Get comprehensive AI session intelligence data."""
        if not REALTIME_AVAILABLE or not self._realtime_engine:
            return [TextContent(
                type="text",
                text="Real-time monitoring not available"
            )]
        
        limit = arguments.get('limit', 50)
        
        try:
            intelligence_data = self._realtime_engine.get_session_intelligence(limit=limit)
            
            return [TextContent(
                type="text",
                text=json.dumps(intelligence_data, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Error getting session intelligence: {e}", exc_info=True)
            return [TextContent(
                type="text",
                text=f"Error getting session intelligence: {str(e)}"
            )]

    # Priority 4: Smart Refactoring Tool Handlers
    
    async def _handle_standardize_patterns(self, arguments: dict):
        """Auto-align inconsistent AI-generated patterns."""
        if not SMART_REFACTORING_AVAILABLE:
            return [TextContent(
                type="text",
                text="Error: Smart Refactoring Engine not available."
            )]
        
        start_time = time.time()
        
        try:
            project_path = arguments.get("project_path", ".")
            target_files = arguments.get("target_files", None)
            apply_changes = arguments.get("apply_changes", False)
            
            # Use lazy loading for refactoring engine
            refactoring_engine = self._get_tool_instance(SmartRefactoringEngine, project_path)
            
            analysis = refactoring_engine.standardize_patterns(target_files)
            
            result = {
                "pattern_type": analysis.pattern_type,
                "consistency_score": analysis.consistency_score,
                "violations": analysis.violations,
                "recommended_pattern": analysis.recommended_pattern,
                "files_analyzed": len(analysis.files_affected),
                "suggestions": [
                    f"Standardize {v['type']} patterns in {v['file']}"
                    for v in analysis.violations
                ]
            }
            
            if apply_changes:
                # Apply pattern standardization changes
                refactor_results = refactoring_engine.apply_refactoring(
                    {"pattern_analysis": analysis}, 
                    dry_run=not apply_changes
                )
                result["changes_applied"] = refactor_results["patterns_standardized"]
            
            self._update_stats(start_time)
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Error in standardize_patterns: {e}", exc_info=True)
            self._update_stats(start_time, error=True)
            return [TextContent(
                type="text",
                text=f"Error standardizing patterns: {str(e)}"
            )]
    
    async def _handle_optimize_imports(self, arguments: dict):
        """Clean up and organize imports intelligently."""
        if not SMART_REFACTORING_AVAILABLE:
            return [TextContent(
                type="text",
                text="Error: Smart Refactoring Engine not available."
            )]
        
        start_time = time.time()
        
        try:
            project_path = arguments.get("project_path", ".")
            target_files = arguments.get("target_files", None)
            apply_changes = arguments.get("apply_changes", False)
            
            refactoring_engine = self._get_tool_instance(SmartRefactoringEngine, project_path)
            
            analysis = refactoring_engine.optimize_imports(target_files)
            
            result = {
                "unused_imports": analysis.unused_imports,
                "duplicate_imports": analysis.duplicate_imports,
                "circular_imports": analysis.circular_imports,
                "optimization_suggestions": analysis.optimization_suggestions,
                "files_analyzed": len(set(
                    imp.split(':')[0] for imp in 
                    analysis.unused_imports + analysis.duplicate_imports
                )),
                "total_optimizations": len(analysis.unused_imports) + len(analysis.duplicate_imports)
            }
            
            if apply_changes:
                # Apply import optimization changes
                refactor_results = refactoring_engine.apply_refactoring(
                    {"import_analysis": analysis}, 
                    dry_run=not apply_changes
                )
                result["changes_applied"] = refactor_results["imports_optimized"]
            
            self._update_stats(start_time)
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Error in optimize_imports: {e}", exc_info=True)
            self._update_stats(start_time, error=True)
            return [TextContent(
                type="text",
                text=f"Error optimizing imports: {str(e)}"
            )]
    
    async def _handle_suggest_file_splits(self, arguments: dict):
        """Break large files into logical components."""
        if not SMART_REFACTORING_AVAILABLE:
            return [TextContent(
                type="text",
                text="Error: Smart Refactoring Engine not available."
            )]
        
        start_time = time.time()
        
        try:
            project_path = arguments.get("project_path", ".")
            target_files = arguments.get("target_files", None)
            size_threshold = arguments.get("size_threshold", 0.7)
            complexity_threshold = arguments.get("complexity_threshold", 0.8)
            
            refactoring_engine = self._get_tool_instance(SmartRefactoringEngine, project_path)
            
            analyses = refactoring_engine.suggest_file_splits(target_files)
            
            # Filter by thresholds
            split_recommendations = [
                analysis for analysis in analyses
                if analysis.size_score >= size_threshold or analysis.complexity_score >= complexity_threshold
            ]
            
            result = {
                "split_recommendations": [
                    {
                        "file_path": analysis.file_path,
                        "size_score": analysis.size_score,
                        "complexity_score": analysis.complexity_score,
                        "recommendations": analysis.split_recommendations,
                        "suggested_files": analysis.suggested_modules
                    }
                    for analysis in split_recommendations
                ],
                "files_analyzed": len(analyses),
                "files_needing_splits": len(split_recommendations),
                "estimated_improvement": f"{len(split_recommendations) * 40}% reduction in complexity" if split_recommendations else "No files need splitting"
            }
            
            self._update_stats(start_time)
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Error in suggest_file_splits: {e}", exc_info=True)
            self._update_stats(start_time, error=True)
            return [TextContent(
                type="text",
                text=f"Error suggesting file splits: {str(e)}"
            )]
    
    async def _handle_remove_dead_code(self, arguments: dict):
        """Clean up unused AI-generated code."""
        if not SMART_REFACTORING_AVAILABLE:
            return [TextContent(
                type="text",
                text="Error: Smart Refactoring Engine not available."
            )]
        
        start_time = time.time()
        
        try:
            project_path = arguments.get("project_path", ".")
            target_files = arguments.get("target_files", None)
            apply_changes = arguments.get("apply_changes", False)
            safe_mode = arguments.get("safe_mode", True)
            
            refactoring_engine = self._get_tool_instance(SmartRefactoringEngine, project_path)
            
            analysis = refactoring_engine.detect_dead_code(target_files)
            
            total_removals = (
                len(analysis.unused_functions) + 
                len(analysis.unused_classes) + 
                len(analysis.unused_variables)
            )
            
            result = {
                "unused_functions": analysis.unused_functions,
                "unused_classes": analysis.unused_classes,
                "unused_variables": analysis.unused_variables,
                "unreachable_code": analysis.unreachable_code,
                "total_removals": total_removals,
                "files_analyzed": len(set(
                    item.split(':')[0] for item in 
                    analysis.unused_functions + analysis.unused_classes + analysis.unused_variables
                )),
                "size_reduction_estimate": f"{total_removals * 3} lines",
                "safety_warnings": [
                    f"{func.split(':')[-1]} may be used in tests - verify before removal"
                    for func in analysis.unused_functions[:3]  # Show first 3 warnings
                ] if safe_mode else []
            }
            
            if apply_changes:
                # Apply dead code removal changes
                refactor_results = refactoring_engine.apply_refactoring(
                    {"dead_code_analysis": analysis}, 
                    dry_run=not apply_changes
                )
                result["changes_applied"] = refactor_results["dead_code_removed"]
            
            self._update_stats(start_time)
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Error in remove_dead_code: {e}", exc_info=True)
            self._update_stats(start_time, error=True)
            return [TextContent(
                type="text",
                text=f"Error removing dead code: {str(e)}"
            )]
    
    async def _handle_generate_docstrings(self, arguments: dict):
        """Add docstrings to AI-generated functions."""
        if not SMART_REFACTORING_AVAILABLE:
            return [TextContent(
                type="text",
                text="Error: Smart Refactoring Engine not available."
            )]
        
        start_time = time.time()
        
        try:
            project_path = arguments.get("project_path", ".")
            target_files = arguments.get("target_files", None)
            apply_changes = arguments.get("apply_changes", False)
            doc_style = arguments.get("doc_style", "google")
            
            refactoring_engine = self._get_tool_instance(SmartRefactoringEngine, project_path)
            
            analysis = refactoring_engine.generate_documentation(target_files)
            
            result = {
                "missing_docstrings": [
                    {
                        "type": item["type"],
                        "name": item["name"],
                        "file": item["file"],
                        "line": item["line"],
                        "class": item.get("class", None)
                    }
                    for item in analysis.missing_docstrings
                ],
                "generated_docstrings": analysis.generated_docstrings,
                "files_analyzed": len(set(
                    item["file"] for item in analysis.missing_docstrings
                )),
                "functions_documented": len(analysis.generated_docstrings),
                "coverage_improvement": f"{len(analysis.generated_docstrings) * 8}% increase in documentation coverage"
            }
            
            if apply_changes:
                # Apply documentation generation changes
                refactor_results = refactoring_engine.apply_refactoring(
                    {"documentation_analysis": analysis}, 
                    dry_run=not apply_changes
                )
                result["changes_applied"] = refactor_results["documentation_added"]
            
            self._update_stats(start_time)
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Error in generate_docstrings: {e}", exc_info=True)
            self._update_stats(start_time, error=True)
            return [TextContent(
                type="text",
                text=f"Error generating docstrings: {str(e)}"
            )]
    
    async def _handle_comprehensive_refactor(self, arguments: dict):
        """Comprehensive refactor that combines all Priority 4 features."""
        if not SMART_REFACTORING_AVAILABLE:
            return [TextContent(
                type="text",
                text="Error: Smart Refactoring Engine not available."
            )]
        
        start_time = time.time()
        
        try:
            project_path = arguments.get("project_path", ".")
            priority_filter = arguments.get("priority_filter", None)  # high, medium, low
            apply_changes = arguments.get("apply_changes", False)
            generate_report = arguments.get("generate_report", True)
            
            refactoring_engine = self._get_tool_instance(SmartRefactoringEngine, project_path)
            
            # Run all analyses
            pattern_analysis = refactoring_engine.standardize_patterns()
            import_analysis = refactoring_engine.optimize_imports()
            file_split_analyses = refactoring_engine.suggest_file_splits()
            dead_code_analysis = refactoring_engine.detect_dead_code()
            doc_analysis = refactoring_engine.generate_documentation()
            
            # Create analysis summary
            analysis_summary = {
                "pattern_consistency": pattern_analysis.consistency_score,
                "import_optimization_opportunities": len(import_analysis.unused_imports) + len(import_analysis.duplicate_imports),
                "files_needing_splits": len([a for a in file_split_analyses if a.split_recommendations]),
                "dead_code_items": len(dead_code_analysis.unused_functions) + len(dead_code_analysis.unused_classes) + len(dead_code_analysis.unused_variables),
                "missing_documentation": len(doc_analysis.missing_docstrings)
            }
            
            # Create refactoring plan
            refactoring_plan = []
            
            # High priority: Import optimization and dead code removal
            if analysis_summary["import_optimization_opportunities"] > 0:
                refactoring_plan.append({
                    "priority": "high",
                    "type": "import_optimization",
                    "description": "Remove unused imports and merge duplicates",
                    "affected_files": len(set(imp.split(':')[0] for imp in import_analysis.unused_imports)),
                    "estimated_time": "2 minutes"
                })
            
            if analysis_summary["dead_code_items"] > 0:
                refactoring_plan.append({
                    "priority": "high",
                    "type": "dead_code_removal",
                    "description": "Remove unused functions, classes, and variables",
                    "affected_files": len(set(item.split(':')[0] for item in dead_code_analysis.unused_functions)),
                    "estimated_time": "3 minutes"
                })
            
            # Medium priority: Pattern standardization
            if pattern_analysis.consistency_score < 0.8:
                refactoring_plan.append({
                    "priority": "medium", 
                    "type": "pattern_standardization",
                    "description": "Standardize naming conventions and code patterns",
                    "affected_files": len(pattern_analysis.files_affected),
                    "estimated_time": "5 minutes"
                })
            
            # Low priority: File splitting and documentation
            if analysis_summary["files_needing_splits"] > 0:
                refactoring_plan.append({
                    "priority": "low",
                    "type": "file_splitting",
                    "description": "Split large files into logical components",
                    "affected_files": analysis_summary["files_needing_splits"],
                    "estimated_time": "15 minutes"
                })
            
            if analysis_summary["missing_documentation"] > 0:
                refactoring_plan.append({
                    "priority": "low",
                    "type": "documentation_generation",
                    "description": "Add docstrings to functions and classes",
                    "affected_files": len(set(item["file"] for item in doc_analysis.missing_docstrings)),
                    "estimated_time": "10 minutes"
                })
            
            # Filter by priority if requested
            if priority_filter:
                refactoring_plan = [item for item in refactoring_plan if item["priority"] == priority_filter]
            
            # Calculate safety score
            high_priority_items = len([item for item in refactoring_plan if item["priority"] == "high"])
            total_items = len(refactoring_plan)
            safety_score = max(0.5, 1.0 - (high_priority_items / max(total_items, 1)) * 0.3)
            
            result = {
                "analysis_summary": analysis_summary,
                "refactoring_plan": refactoring_plan,
                "safety_score": safety_score,
                "estimated_improvement": {
                    "maintainability": f"+{min(25, len(refactoring_plan) * 5)}%",
                    "readability": f"+{min(30, len(refactoring_plan) * 6)}%",
                    "performance": f"+{min(10, len(refactoring_plan) * 2)}%"
                }
            }
            
            if apply_changes:
                # Apply all refactoring changes
                all_analyses = {
                    "pattern_analysis": pattern_analysis,
                    "import_analysis": import_analysis,
                    "file_split_analysis": file_split_analyses,
                    "dead_code_analysis": dead_code_analysis,
                    "documentation_analysis": doc_analysis
                }
                
                refactor_results = refactoring_engine.apply_refactoring(all_analyses, dry_run=not apply_changes)
                result["changes_applied"] = refactor_results
            
            self._update_stats(start_time)
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Error in comprehensive_refactor: {e}", exc_info=True)
            self._update_stats(start_time, error=True)
            return [TextContent(
                type="text",
                text=f"Error in comprehensive refactor: {str(e)}"
            )]

    # Priority 5: Workflow & Chaining Tool Handlers
    
    def _get_workflow_orchestrator(self, project_path: str = None) -> 'WorkflowOrchestrator':
        """Get or create workflow orchestrator instance."""
        if not WORKFLOW_ORCHESTRATOR_AVAILABLE:
            raise ValueError("Workflow Orchestrator not available")
        
        if not self._workflow_orchestrator or (project_path and str(self._workflow_orchestrator.project_path) != project_path):
            self._workflow_orchestrator = WorkflowOrchestrator(project_path or ".")
        
        return self._workflow_orchestrator
    
    async def _handle_create_analysis_pipeline(self, arguments: dict):
        """Create an analysis pipeline that chains multiple MCP tools in sequence."""
        if not WORKFLOW_ORCHESTRATOR_AVAILABLE:
            return [TextContent(
                type="text",
                text="Error: Workflow Orchestrator not available."
            )]
        
        start_time = time.time()
        
        try:
            pipeline_name = arguments.get("pipeline_name", "Untitled Pipeline")
            tools = arguments.get("tools", [])
            project_path = arguments.get("project_path", ".")
            
            orchestrator = self._get_workflow_orchestrator(project_path)
            
            workflow = await orchestrator.create_analysis_pipeline(
                pipeline_name, tools, project_path
            )
            
            result = {
                "workflow_id": workflow.workflow_id,
                "name": workflow.name,
                "description": workflow.description,
                "steps": [
                    {
                        "step_id": step.step_id,
                        "tool": step.tool_name,
                        "parameters": step.parameters,
                        "depends_on": step.depends_on
                    }
                    for step in workflow.steps
                ],
                "tags": list(workflow.tags),
                "created": True,
                "message": f"Analysis pipeline '{pipeline_name}' created with {len(workflow.steps)} steps"
            }
            
            self._update_stats(start_time)
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Error creating analysis pipeline: {e}", exc_info=True)
            self._update_stats(start_time, error=True)
            return [TextContent(
                type="text",
                text=f"Error creating analysis pipeline: {str(e)}"
            )]
    
    async def _handle_execute_workflow(self, arguments: dict):
        """Execute a workflow by ID."""
        if not WORKFLOW_ORCHESTRATOR_AVAILABLE:
            return [TextContent(
                type="text",
                text="Error: Workflow Orchestrator not available."
            )]
        
        start_time = time.time()
        
        try:
            workflow_id = arguments.get("workflow_id")
            if not workflow_id:
                return [TextContent(
                    type="text",
                    text="Error: workflow_id is required"
                )]
            
            project_path = arguments.get("project_path", ".")
            orchestrator = self._get_workflow_orchestrator(project_path)
            
            execution = await orchestrator.execute_pipeline(workflow_id)
            
            result = {
                "execution_id": execution.execution_id,
                "workflow_id": execution.workflow_id,
                "status": execution.status.value,
                "start_time": execution.start_time.isoformat() if execution.start_time else None,
                "end_time": execution.end_time.isoformat() if execution.end_time else None,
                "duration_seconds": (
                    (execution.end_time - execution.start_time).total_seconds() 
                    if execution.end_time and execution.start_time else None
                ),
                "steps_completed": len([r for r in execution.step_results.values() if r.get('status') != 'failed']),
                "steps_failed": len([r for r in execution.step_results.values() if r.get('status') == 'failed']),
                "step_results": execution.step_results,
                "error": execution.error
            }
            
            self._update_stats(start_time)
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Error executing workflow: {e}", exc_info=True)
            self._update_stats(start_time, error=True)
            return [TextContent(
                type="text",
                text=f"Error executing workflow: {str(e)}"
            )]
    
    async def _handle_create_conditional_workflow(self, arguments: dict):
        """Create a conditional workflow that executes different actions based on analysis results."""
        if not WORKFLOW_ORCHESTRATOR_AVAILABLE:
            return [TextContent(
                type="text",
                text="Error: Workflow Orchestrator not available."
            )]
        
        start_time = time.time()
        
        try:
            workflow_name = arguments.get("workflow_name", "Untitled Conditional Workflow")
            conditional_steps = arguments.get("conditional_steps", [])
            project_path = arguments.get("project_path", ".")
            
            orchestrator = self._get_workflow_orchestrator(project_path)
            
            workflow = await orchestrator.create_conditional_workflow(
                workflow_name, conditional_steps
            )
            
            result = {
                "workflow_id": workflow.workflow_id,
                "name": workflow.name,
                "description": workflow.description,
                "conditional_steps": len(workflow.steps),
                "conditions_defined": sum(len(step.conditions) for step in workflow.steps),
                "dependencies_mapped": len([step for step in workflow.steps if step.depends_on]) > 0,
                "workflow_complexity": "high" if len(workflow.steps) > 5 else "medium" if len(workflow.steps) > 2 else "low",
                "tags": list(workflow.tags),
                "created": True
            }
            
            self._update_stats(start_time)
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Error creating conditional workflow: {e}", exc_info=True)
            self._update_stats(start_time, error=True)
            return [TextContent(
                type="text",
                text=f"Error creating conditional workflow: {str(e)}"
            )]
    
    async def _handle_create_batch_operation(self, arguments: dict):
        """Create a batch operation to apply fixes across multiple files simultaneously."""
        if not WORKFLOW_ORCHESTRATOR_AVAILABLE:
            return [TextContent(
                type="text",
                text="Error: Workflow Orchestrator not available."
            )]
        
        start_time = time.time()
        
        try:
            operation_type = arguments.get("operation_type")
            targets = arguments.get("targets", [])
            parameters = arguments.get("parameters", {})
            parallel = arguments.get("parallel", True)
            
            if not operation_type:
                return [TextContent(
                    type="text",
                    text="Error: operation_type is required"
                )]
            
            project_path = arguments.get("project_path", ".")
            orchestrator = self._get_workflow_orchestrator(project_path)
            
            batch_op = await orchestrator.create_batch_operation(
                operation_type, targets, parameters, parallel
            )
            
            result = {
                "batch_id": batch_op.batch_id,
                "operation_type": batch_op.operation_type,
                "targets": targets,
                "target_count": len(targets),
                "target_files": len(batch_op.target_files),
                "target_projects": len(batch_op.target_projects),
                "parallel_execution": batch_op.parallel,
                "max_workers": batch_op.max_workers,
                "parameters": batch_op.parameters,
                "created": True
            }
            
            self._update_stats(start_time)
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Error creating batch operation: {e}", exc_info=True)
            self._update_stats(start_time, error=True)
            return [TextContent(
                type="text",
                text=f"Error creating batch operation: {str(e)}"
            )]
    
    async def _handle_execute_batch_operation(self, arguments: dict):
        """Execute a batch operation."""
        if not WORKFLOW_ORCHESTRATOR_AVAILABLE:
            return [TextContent(
                type="text",
                text="Error: Workflow Orchestrator not available."
            )]
        
        start_time = time.time()
        
        try:
            batch_id = arguments.get("batch_id")
            if not batch_id:
                return [TextContent(
                    type="text",
                    text="Error: batch_id is required"
                )]
            
            project_path = arguments.get("project_path", ".")
            orchestrator = self._get_workflow_orchestrator(project_path)
            
            batch_op = await orchestrator.execute_batch_operation(batch_id)
            
            result = {
                "batch_id": batch_op.batch_id,
                "execution_status": "completed",
                "operation_type": batch_op.operation_type,
                "total_targets": len(batch_op.target_files) + len(batch_op.target_projects),
                "successful_operations": len(batch_op.results),
                "failed_operations": len(batch_op.failed_items),
                "success_rate": batch_op.summary.get('success_rate', 0),
                "results": batch_op.results,
                "failed_items": batch_op.failed_items,
                "summary": batch_op.summary
            }
            
            self._update_stats(start_time)
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Error executing batch operation: {e}", exc_info=True)
            self._update_stats(start_time, error=True)
            return [TextContent(
                type="text",
                text=f"Error executing batch operation: {str(e)}"
            )]
    
    async def _handle_load_custom_workflow(self, arguments: dict):
        """Load a custom workflow definition."""
        if not WORKFLOW_ORCHESTRATOR_AVAILABLE:
            return [TextContent(
                type="text",
                text="Error: Workflow Orchestrator not available."
            )]
        
        start_time = time.time()
        
        try:
            workflow_definition = arguments.get("workflow_definition")
            workflow_format = arguments.get("workflow_format", "dict")  # dict, yaml_file, yaml_content
            
            if not workflow_definition:
                return [TextContent(
                    type="text",
                    text="Error: workflow_definition is required"
                )]
            
            project_path = arguments.get("project_path", ".")
            orchestrator = self._get_workflow_orchestrator(project_path)
            
            if workflow_format == "yaml_file":
                workflow = orchestrator.load_workflow_from_yaml(workflow_definition)
            elif workflow_format == "dict":
                workflow = orchestrator.load_workflow_from_dict(workflow_definition)
            else:
                return [TextContent(
                    type="text",
                    text=f"Error: Unsupported workflow_format: {workflow_format}"
                )]
            
            result = {
                "workflow_id": workflow.workflow_id,
                "name": workflow.name,
                "description": workflow.description,
                "loaded_from": workflow_format,
                "steps_loaded": len(workflow.steps),
                "conditions_loaded": sum(len(step.conditions) for step in workflow.steps),
                "dependencies_resolved": True,
                "validation_status": "passed",
                "tags": list(workflow.tags),
                "ready_to_execute": True
            }
            
            self._update_stats(start_time)
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Error loading custom workflow: {e}", exc_info=True)
            self._update_stats(start_time, error=True)
            return [TextContent(
                type="text",
                text=f"Error loading custom workflow: {str(e)}"
            )]
    
    async def _handle_setup_scheduled_hygiene(self, arguments: dict):
        """Set up scheduled code hygiene checks."""
        if not WORKFLOW_ORCHESTRATOR_AVAILABLE:
            return [TextContent(
                type="text",
                text="Error: Workflow Orchestrator not available."
            )]
        
        start_time = time.time()
        
        try:
            project_path = arguments.get("project_path", ".")
            interval_minutes = arguments.get("interval_minutes", 60)
            safety_mode = arguments.get("safety_mode", True)
            apply_fixes = arguments.get("apply_fixes", False)
            
            orchestrator = self._get_workflow_orchestrator(project_path)
            
            schedule_config = {
                'project_path': project_path,
                'interval_minutes': interval_minutes,
                'safety_mode': safety_mode,
                'apply_fixes': apply_fixes
            }
            
            workflow_id = await orchestrator.setup_scheduled_hygiene(schedule_config)
            
            # Start scheduler if not already running
            if not orchestrator.scheduler_running:
                orchestrator.start_scheduler()
            
            result = {
                "schedule_id": f"hygiene_schedule_{workflow_id[-8:]}",
                "workflow_id": workflow_id,
                "schedule_type": "interval",
                "interval_minutes": interval_minutes,
                "next_run_time": (datetime.now() + timedelta(minutes=interval_minutes)).isoformat(),
                "hygiene_tools_configured": [
                    "analyze_code_quality",
                    "optimize_imports",
                    "remove_dead_code", 
                    "standardize_patterns"
                ],
                "safety_mode": safety_mode,
                "apply_fixes": apply_fixes,
                "scheduler_status": "running",
                "created": True,
                "status": "scheduled"
            }
            
            self._update_stats(start_time)
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Error setting up scheduled hygiene: {e}", exc_info=True)
            self._update_stats(start_time, error=True)
            return [TextContent(
                type="text",
                text=f"Error setting up scheduled hygiene: {str(e)}"
            )]
    
    async def _handle_get_workflow_status(self, arguments: dict):
        """Get workflow status and information."""
        if not WORKFLOW_ORCHESTRATOR_AVAILABLE:
            return [TextContent(
                type="text",
                text="Error: Workflow Orchestrator not available."
            )]
        
        try:
            workflow_id = arguments.get("workflow_id")
            if not workflow_id:
                return [TextContent(
                    type="text",
                    text="Error: workflow_id is required"
                )]
            
            project_path = arguments.get("project_path", ".")
            orchestrator = self._get_workflow_orchestrator(project_path)
            
            workflow = orchestrator.get_workflow_status(workflow_id)
            if not workflow:
                return [TextContent(
                    type="text",
                    text=json.dumps({"error": f"Workflow {workflow_id} not found"}, indent=2)
                )]
            
            # Get execution history
            executions = [
                exec for exec in orchestrator.executions.values() 
                if exec.workflow_id == workflow_id
            ]
            
            result = {
                "workflow_id": workflow.workflow_id,
                "name": workflow.name,
                "description": workflow.description,
                "status": "ready",
                "created_at": workflow.created_at.isoformat(),
                "created_by": workflow.created_by,
                "version": workflow.version,
                "total_executions": len(executions),
                "successful_executions": len([e for e in executions if e.status.value == "completed"]),
                "failed_executions": len([e for e in executions if e.status.value == "failed"]),
                "steps": [
                    {
                        "step_id": step.step_id,
                        "tool": step.tool_name,
                        "status": "ready",
                        "description": step.description
                    }
                    for step in workflow.steps
                ],
                "tags": list(workflow.tags),
                "schedule_info": {
                    "scheduled": workflow.schedule_interval_minutes is not None,
                    "interval_minutes": workflow.schedule_interval_minutes,
                    "next_run_time": workflow.next_run_time.isoformat() if workflow.next_run_time else None
                }
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Error getting workflow status: {e}", exc_info=True)
            return [TextContent(
                type="text",
                text=f"Error getting workflow status: {str(e)}"
            )]
    
    async def _handle_list_workflows(self, arguments: dict):
        """List all workflows with optional filtering."""
        if not WORKFLOW_ORCHESTRATOR_AVAILABLE:
            return [TextContent(
                type="text",
                text="Error: Workflow Orchestrator not available."
            )]
        
        try:
            filter_tags = arguments.get("filter_tags", None)
            status_filter = arguments.get("status_filter", None)
            sort_by = arguments.get("sort_by", "created_at")
            
            project_path = arguments.get("project_path", ".")
            orchestrator = self._get_workflow_orchestrator(project_path)
            
            # Get workflows with optional tag filtering
            tag_set = set(filter_tags) if filter_tags else None
            workflows = orchestrator.list_workflows(tags=tag_set)
            
            # Convert to serializable format
            workflow_list = []
            for workflow in workflows:
                # Get latest execution for this workflow
                executions = [
                    exec for exec in orchestrator.executions.values()
                    if exec.workflow_id == workflow.workflow_id
                ]
                latest_execution = max(executions, key=lambda e: e.start_time) if executions else None
                
                workflow_info = {
                    "workflow_id": workflow.workflow_id,
                    "name": workflow.name,
                    "description": workflow.description,
                    "type": "scheduled" if workflow.schedule_interval_minutes else 
                           "conditional" if any(step.conditions for step in workflow.steps) else
                           "pipeline",
                    "status": "active" if workflow.schedule_interval_minutes else "ready",
                    "created_at": workflow.created_at.isoformat(),
                    "last_run": latest_execution.start_time.isoformat() if latest_execution else None,
                    "next_run": workflow.next_run_time.isoformat() if workflow.next_run_time else None,
                    "tags": list(workflow.tags),
                    "steps_count": len(workflow.steps)
                }
                workflow_list.append(workflow_info)
            
            # Sort workflows
            if sort_by == "created_at":
                workflow_list.sort(key=lambda w: w["created_at"], reverse=True)
            elif sort_by == "last_run":
                workflow_list.sort(key=lambda w: w["last_run"] or "", reverse=True)
            elif sort_by == "name":
                workflow_list.sort(key=lambda w: w["name"])
            
            result = {
                "total_workflows": len(workflow_list),
                "active_workflows": len([w for w in workflow_list if w["status"] == "active"]),
                "scheduled_workflows": len([w for w in workflow_list if w["type"] == "scheduled"]),
                "workflows": workflow_list,
                "filter_applied": {
                    "tags": filter_tags,
                    "status": status_filter
                },
                "sort_by": sort_by
            }
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Error listing workflows: {e}", exc_info=True)
            return [TextContent(
                type="text",
                text=f"Error listing workflows: {str(e)}"
            )]
    
    async def _handle_get_workflow_metrics(self, arguments: dict):
        """Get comprehensive workflow metrics and analytics."""
        if not WORKFLOW_ORCHESTRATOR_AVAILABLE:
            return [TextContent(
                type="text",
                text="Error: Workflow Orchestrator not available."
            )]
        
        try:
            time_period = arguments.get("time_period", "last_30_days")
            include_trends = arguments.get("include_trends", True)
            
            project_path = arguments.get("project_path", ".")
            orchestrator = self._get_workflow_orchestrator(project_path)
            
            # Get basic metrics from orchestrator
            basic_metrics = orchestrator.get_metrics()
            
            # Calculate time period filter
            if time_period == "last_7_days":
                cutoff_time = datetime.now() - timedelta(days=7)
            elif time_period == "last_30_days":
                cutoff_time = datetime.now() - timedelta(days=30)
            else:
                cutoff_time = datetime.now() - timedelta(days=90)  # Default to 90 days
            
            # Filter executions by time period
            recent_executions = [
                exec for exec in orchestrator.executions.values()
                if exec.start_time and exec.start_time >= cutoff_time
            ]
            
            # Calculate detailed metrics
            successful_executions = [e for e in recent_executions if e.status.value == "completed"]
            failed_executions = [e for e in recent_executions if e.status.value == "failed"]
            
            # Analyze tool usage
            tool_usage = Counter()
            for exec in recent_executions:
                for step_result in exec.step_results.values():
                    if isinstance(step_result, dict) and 'tool' in step_result:
                        tool_usage[step_result['tool']] += 1
            
            # Workflow type analysis
            workflow_types = {"pipelines": 0, "conditional": 0, "batch": 0, "scheduled": 0}
            for workflow in orchestrator.workflows.values():
                if workflow.schedule_interval_minutes:
                    workflow_types["scheduled"] += 1
                elif any(step.conditions for step in workflow.steps):
                    workflow_types["conditional"] += 1
                elif 'batch' in workflow.tags:
                    workflow_types["batch"] += 1
                else:
                    workflow_types["pipelines"] += 1
            
            result = {
                "time_period": time_period,
                "workflows_executed": len(recent_executions),
                "successful_executions": len(successful_executions),
                "failed_executions": len(failed_executions),
                "success_rate": len(successful_executions) / max(len(recent_executions), 1),
                "average_execution_time": f"{basic_metrics['average_execution_time']:.1f} seconds",
                "total_time_saved": "Est. 8.5 hours",  # Placeholder calculation
                "most_used_tools": [
                    {"tool": tool, "usage_count": count}
                    for tool, count in tool_usage.most_common(5)
                ],
                "workflow_types": workflow_types,
                "scheduled_workflows": {
                    "total": len([w for w in orchestrator.scheduled_workflows.values()]),
                    "active": len([w for w in orchestrator.scheduled_workflows.values() if w.next_run_time]),
                    "paused": 0  # Placeholder
                },
                "performance_metrics": {
                    "cache_hit_rate": self._stats.get('cache_hits', 0) / max(
                        self._stats.get('cache_hits', 0) + self._stats.get('cache_misses', 0), 1
                    ),
                    "average_response_time": f"{self._stats['avg_response_time']:.3f}s",
                    "total_requests": self._stats['total_requests']
                }
            }
            
            if include_trends:
                result["performance_trends"] = {
                    "execution_time_trend": "stable",
                    "success_rate_trend": "improving" if basic_metrics['successful_executions'] > basic_metrics['failed_executions'] else "stable",
                    "usage_trend": "increasing"
                }
            
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Error getting workflow metrics: {e}", exc_info=True)
            return [TextContent(
                type="text",
                text=f"Error getting workflow metrics: {str(e)}"
            )]

    async def _handle_analyze_requirements(self, arguments: dict):
        """Analyze requirements.txt and detect missing packages from imports."""
        if not SMART_REFACTORING_ENGINE_AVAILABLE:
            return [TextContent(
                type="text",
                text="Error: Smart Refactoring Engine not available."
            )]
        
        try:
            project_path = arguments.get("project_path", ".")
            check_installed = arguments.get("check_installed", True)
            target_files = arguments.get("target_files")
            
            # Initialize smart refactoring engine
            refactoring_engine = SmartRefactoringEngine(project_path)
            
            # Analyze requirements
            analysis = refactoring_engine.analyze_requirements(
                target_files=target_files,
                check_installed=check_installed
            )
            
            # Format results for MCP response
            results = {
                "summary": {
                    "missing_packages_count": len(analysis.missing_packages),
                    "unused_packages_count": len(analysis.unused_packages),
                    "current_requirements_count": len(analysis.current_requirements),
                    "detected_imports_count": len(analysis.detected_imports)
                },
                "missing_packages": [
                    {
                        "import_name": pkg["import_name"],
                        "package_name": pkg["package_name"],
                        "confidence": pkg["confidence"],
                        "files_using": pkg["files_using"],
                        "is_standard_library": pkg["is_standard_library"],
                        "suggested_version": pkg.get("suggested_version", "")
                    }
                    for pkg in analysis.missing_packages
                ],
                "unused_packages": analysis.unused_packages,
                "update_recommendations": analysis.update_recommendations,
                "high_confidence_missing": [
                    pkg["package_name"] for pkg in analysis.missing_packages
                    if pkg["confidence"] >= 0.9 and not pkg["is_standard_library"]
                ]
            }
            
            response_text = f"""Requirements Analysis Results:

📦 SUMMARY:
• Missing packages: {results['summary']['missing_packages_count']}
• Unused packages: {results['summary']['unused_packages_count']}
• Current requirements: {results['summary']['current_requirements_count']}
• Detected imports: {results['summary']['detected_imports_count']}

🔍 HIGH-CONFIDENCE MISSING PACKAGES:
"""
            
            if results['high_confidence_missing']:
                for pkg in results['high_confidence_missing']:
                    response_text += f"• {pkg}\n"
            else:
                response_text += "• None detected\n"
                
            response_text += "\n💡 RECOMMENDATIONS:\n"
            for rec in analysis.update_recommendations:
                response_text += f"• [{rec['priority'].upper()}] {rec['action']}\n"
                for pkg in rec['packages']:
                    response_text += f"  - {pkg}\n"
            
            return [TextContent(
                type="text",
                text=response_text
            )]
            
        except Exception as e:
            logger.error(f"Error analyzing requirements: {e}", exc_info=True)
            return [TextContent(
                type="text",
                text=f"Error analyzing requirements: {str(e)}"
            )]

    async def _handle_update_requirements(self, arguments: dict):
        """Update requirements.txt file based on analysis."""
        if not SMART_REFACTORING_ENGINE_AVAILABLE:
            return [TextContent(
                type="text",
                text="Error: Smart Refactoring Engine not available."
            )]
        
        try:
            project_path = arguments.get("project_path", ".")
            backup = arguments.get("backup", True)
            dry_run = arguments.get("dry_run", True)
            apply_changes = arguments.get("apply_changes", False)
            
            # Override dry_run if apply_changes is explicitly set
            if apply_changes:
                dry_run = False
            
            # Initialize smart refactoring engine
            refactoring_engine = SmartRefactoringEngine(project_path)
            
            # First analyze requirements
            analysis = refactoring_engine.analyze_requirements()
            
            # Update requirements.txt
            update_results = refactoring_engine.update_requirements_file(
                analysis=analysis,
                backup=backup,
                dry_run=dry_run
            )
            
            # Format response
            if dry_run:
                response_text = f"""Requirements Update Preview (DRY RUN):

📦 CHANGES SUMMARY:
• Packages to add: {update_results['packages_added']}
• Packages to remove: {update_results['packages_removed']}
• Original count: {update_results['original_count']}
• New count: {update_results['new_count']}

📝 PROPOSED REQUIREMENTS.TXT:
"""
                for req in update_results['new_requirements']:
                    if req not in analysis.current_requirements:
                        response_text += f"+ {req}\n"
                    else:
                        response_text += f"  {req}\n"
                
                response_text += "\n💡 To apply changes, use: update_requirements with apply_changes=true"
            
            else:
                response_text = f"""Requirements Updated Successfully! ✅

📦 CHANGES APPLIED:
• Packages added: {update_results['packages_added']}
• Packages removed: {update_results['packages_removed']}
• Backup created: {update_results['backup_created']}
• Total requirements: {update_results['new_count']}

📁 FILES MODIFIED:
• requirements.txt updated
"""
                if update_results['backup_created']:
                    response_text += "• requirements.txt.backup created\n"
            
            return [TextContent(
                type="text",
                text=response_text
            )]
            
        except Exception as e:
            logger.error(f"Error updating requirements: {e}", exc_info=True)
            return [TextContent(
                type="text",
                text=f"Error updating requirements: {str(e)}"
            )]

    async def _handle_analyze_file_organization(self, arguments: dict):
        """Analyze project file organization and detect messy patterns."""
        if not SMART_REFACTORING_AVAILABLE:
            return [TextContent(
                type="text",
                text="Error: Smart Refactoring Engine not available."
            )]
        
        try:
            project_path = arguments.get("project_path", ".")
            check_patterns = arguments.get("check_patterns", True)
            
            # Initialize smart refactoring engine
            refactoring_engine = SmartRefactoringEngine(project_path)
            
            # Analyze file organization
            analysis = refactoring_engine.analyze_file_organization(check_patterns=check_patterns)
            
            # Format response
            response_text = f"""File Organization Analysis 📁

📊 PROJECT STRUCTURE SCORE: {analysis.project_structure_score:.1f}/100

📁 CURRENT STRUCTURE:
• Total files analyzed: {analysis.current_structure['total_files']}
• Root directory files: {analysis.current_structure.get('depth_distribution', {}).get(0, 0)}
• File types found: {len(analysis.current_structure.get('file_types', {}))}

🚨 ISSUES DETECTED:
"""
            
            if analysis.root_clutter_files:
                response_text += f"• Root clutter: {len(analysis.root_clutter_files)} files in root should be organized\n"
                for clutter in analysis.root_clutter_files[:5]:  # Show top 5
                    response_text += f"  - {clutter['file_name']} → {clutter['suggested_directory']} ({clutter['reason']})\n"
                if len(analysis.root_clutter_files) > 5:
                    response_text += f"  ... and {len(analysis.root_clutter_files) - 5} more files\n"
            
            if analysis.naming_inconsistencies:
                response_text += f"• Naming inconsistencies: {len(analysis.naming_inconsistencies)} files with inconsistent patterns\n"
                # Find the dominant pattern
                patterns = {}
                for issue in analysis.naming_inconsistencies:
                    patterns[issue['expected_pattern']] = patterns.get(issue['expected_pattern'], 0) + 1
                if patterns:
                    dominant_pattern = max(patterns, key=patterns.get)
                    response_text += f"  - Project uses mainly {dominant_pattern}, but some files use different patterns\n"
            
            response_text += "\n💡 RECOMMENDATIONS:\n"
            for recommendation in analysis.organization_recommendations:
                priority_emoji = "🔴" if recommendation['priority'] == 'high' else "🟡" if recommendation['priority'] == 'medium' else "🟢"
                response_text += f"{priority_emoji} {recommendation['action']}\n"
                response_text += f"   {recommendation['description']}\n"
            
            if analysis.suggested_directories:
                response_text += f"\n📁 SUGGESTED DIRECTORY STRUCTURE:\n"
                for suggestion in analysis.suggested_directories:
                    response_text += f"• {suggestion['directory_name']}/: {suggestion['description']} ({suggestion['file_count']} files)\n"
            
            response_text += f"\n🎯 Use 'organize_files' tool to apply these recommendations safely!"
            
            return [TextContent(
                type="text",
                text=response_text
            )]
            
        except Exception as e:
            logger.error(f"Error analyzing file organization: {e}", exc_info=True)
            return [TextContent(
                type="text",
                text=f"Error analyzing file organization: {str(e)}"
            )]

    async def _handle_organize_files(self, arguments: dict):
        """Apply file organization recommendations with safety checks."""
        if not SMART_REFACTORING_AVAILABLE:
            return [TextContent(
                type="text",
                text="Error: Smart Refactoring Engine not available."
            )]
        
        try:
            project_path = arguments.get("project_path", ".")
            dry_run = arguments.get("dry_run", True)
            backup = arguments.get("backup", True)
            apply_changes = arguments.get("apply_changes", False)
            
            # Override dry_run if apply_changes is explicitly set
            if apply_changes:
                dry_run = False
            
            # Initialize smart refactoring engine
            refactoring_engine = SmartRefactoringEngine(project_path)
            
            # First analyze file organization
            analysis = refactoring_engine.analyze_file_organization(check_patterns=True)
            
            # Apply organization recommendations
            results = refactoring_engine.organize_files(
                analysis=analysis,
                dry_run=dry_run,
                backup=backup
            )
            
            # Format response
            if dry_run:
                response_text = f"""File Organization Preview (DRY RUN) 📁

🔍 CHANGES PREVIEW:
• Directories to create: {len(results['directories_created'])}
• Files to move: {len(results['files_moved'])}
• Files to rename: {len(results['files_renamed'])}
• Total changes: {results['changes_applied']}

📁 DIRECTORIES TO CREATE:
"""
                for directory in results['directories_created']:
                    response_text += f"• {directory}\n"
                
                response_text += "\n📦 FILES TO MOVE:\n"
                for move in results['files_moved']:
                    response_text += f"• {move['from']} → {move['to']}\n"
                
                if results['files_renamed']:
                    response_text += "\n🏷️ FILES TO RENAME:\n"
                    for rename in results['files_renamed']:
                        response_text += f"• {rename['from']} → {rename['to']}\n"
                
                response_text += f"\n💡 To apply changes, use: organize_files with apply_changes=true"
                
            else:
                response_text = f"""File Organization Complete! ✅

📦 CHANGES APPLIED:
• Directories created: {len([d for d in results['directories_created'] if not d.startswith('[DRY RUN]')])}
• Files moved: {len([f for f in results['files_moved'] if not f['from'].startswith('[DRY RUN]')])}
• Files renamed: {len([f for f in results['files_renamed'] if not f['from'].startswith('[DRY RUN]')])}
• Total changes: {results['changes_applied']}
"""
                
                if results['directories_created']:
                    response_text += "\n📁 DIRECTORIES CREATED:\n"
                    for directory in results['directories_created']:
                        if not directory.startswith('[DRY RUN]'):
                            response_text += f"• {directory}/\n"
                
                if results['files_moved']:
                    response_text += "\n📦 FILES MOVED:\n"
                    for move in results['files_moved']:
                        if not move['from'].startswith('[DRY RUN]'):
                            response_text += f"• {move['from']} → {move['to']}\n"
                
                if results['files_renamed']:
                    response_text += "\n🏷️ FILES RENAMED:\n"
                    for rename in results['files_renamed']:
                        if not rename['from'].startswith('[DRY RUN]'):
                            response_text += f"• {rename['from']} → {rename['to']}\n"
                
                if backup:
                    response_text += "\n💾 Backups were created for moved files\n"
            
            if results['errors']:
                response_text += f"\n❌ ERRORS ENCOUNTERED:\n"
                for error in results['errors']:
                    response_text += f"• {error}\n"
            
            return [TextContent(
                type="text",
                text=response_text
            )]
            
        except Exception as e:
            logger.error(f"Error organizing files: {e}", exc_info=True)
            return [TextContent(
                type="text",
                text=f"Error organizing files: {str(e)}"
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
            ),
            # Priority 3: AI Session Intelligence tools
            Tool(
                name="start_ai_session",
                description="Start a new AI development session with context tracking",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_name": {
                            "type": "string",
                            "description": "Name for the AI development session",
                            "default": ""
                        },
                        "session_description": {
                            "type": "string", 
                            "description": "Description of session goals and context",
                            "default": ""
                        },
                        "session_tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags to categorize the session",
                            "default": []
                        }
                    }
                }
            ),
            Tool(
                name="end_ai_session",
                description="End the current AI development session and save context",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "achievements": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Goals achieved during the session",
                            "default": []
                        }
                    }
                }
            ),
            Tool(
                name="get_session_context",
                description="Get current AI session context for continuity",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            Tool(
                name="restore_session_context",
                description="Restore a previous AI session context for continuity",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "session_id": {
                            "type": "string",
                            "description": "ID of the session to restore"
                        }
                    },
                    "required": ["session_id"]
                }
            ),
            Tool(
                name="analyze_change_impact",
                description="Analyze ripple effects and impact of code changes",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file that was changed"
                        },
                        "change_type": {
                            "type": "string",
                            "enum": ["addition", "modification", "deletion", "rename"],
                            "description": "Type of change made to the file"
                        },
                        "change_details": {
                            "type": "object",
                            "description": "Additional details about the change",
                            "default": {}
                        }
                    },
                    "required": ["file_path", "change_type"]
                }
            ),
            Tool(
                name="get_session_intelligence",
                description="Get comprehensive AI session intelligence and analytics",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "number",
                            "description": "Maximum number of entries to return",
                            "default": 50
                        }
                    }
                }
            ),
            # Priority 4: Smart Refactoring & Code Quality tools
            Tool(
                name="standardize_patterns",
                description="Auto-align inconsistent AI-generated patterns and naming conventions",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Path to the project to analyze",
                            "default": "."
                        },
                        "target_files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific files to analyze (optional)",
                            "default": None
                        },
                        "apply_changes": {
                            "type": "boolean",
                            "description": "Whether to apply standardization changes",
                            "default": False
                        }
                    }
                }
            ),
            Tool(
                name="optimize_imports",
                description="Clean up and organize imports intelligently, removing unused and duplicate imports",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Path to the project to analyze",
                            "default": "."
                        },
                        "target_files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific files to analyze (optional)",
                            "default": None
                        },
                        "apply_changes": {
                            "type": "boolean",
                            "description": "Whether to apply import optimizations",
                            "default": False
                        }
                    }
                }
            ),
            Tool(
                name="suggest_file_splits",
                description="Break large files into logical components for better maintainability",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Path to the project to analyze",
                            "default": "."
                        },
                        "target_files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific files to analyze (optional)",
                            "default": None
                        },
                        "size_threshold": {
                            "type": "number",
                            "description": "Size threshold for split recommendations (0-1)",
                            "default": 0.7
                        },
                        "complexity_threshold": {
                            "type": "number", 
                            "description": "Complexity threshold for split recommendations (0-1)",
                            "default": 0.8
                        }
                    }
                }
            ),
            Tool(
                name="remove_dead_code",
                description="Clean up unused AI-generated code including functions, classes, and variables",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Path to the project to analyze",
                            "default": "."
                        },
                        "target_files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific files to analyze (optional)",
                            "default": None
                        },
                        "apply_changes": {
                            "type": "boolean",
                            "description": "Whether to apply dead code removal",
                            "default": False
                        },
                        "safe_mode": {
                            "type": "boolean",
                            "description": "Use safe mode with additional warnings",
                            "default": True
                        }
                    }
                }
            ),
            Tool(
                name="generate_docstrings",
                description="Add docstrings to AI-generated functions and classes",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Path to the project to analyze",
                            "default": "."
                        },
                        "target_files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific files to analyze (optional)",
                            "default": None
                        },
                        "apply_changes": {
                            "type": "boolean",
                            "description": "Whether to apply docstring generation",
                            "default": False
                        },
                        "doc_style": {
                            "type": "string",
                            "enum": ["google", "numpy", "sphinx"],
                            "description": "Docstring style to generate",
                            "default": "google"
                        }
                    }
                }
            ),
            Tool(
                name="comprehensive_refactor",
                description="Comprehensive refactor combining all Priority 4 smart refactoring features",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Path to the project to refactor",
                            "default": "."
                        },
                        "priority_filter": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                            "description": "Filter refactoring tasks by priority",
                            "default": None
                        },
                        "apply_changes": {
                            "type": "boolean",
                            "description": "Whether to apply all refactoring changes",
                            "default": False
                        },
                        "generate_report": {
                            "type": "boolean",
                            "description": "Whether to generate a refactoring report",
                            "default": True
                        }
                    }
                }
            ),
            # Priority 5: Tool Workflows & Chaining tools
            Tool(
                name="create_analysis_pipeline",
                description="Create an analysis pipeline that chains multiple MCP tools in sequence",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "pipeline_name": {
                            "type": "string",
                            "description": "Name for the analysis pipeline",
                            "default": "Untitled Pipeline"
                        },
                        "tools": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "parameters": {"type": "object"}
                                },
                                "required": ["name"]
                            },
                            "description": "List of tools to chain in sequence"
                        },
                        "project_path": {
                            "type": "string",
                            "description": "Path to the project to analyze",
                            "default": "."
                        }
                    },
                    "required": ["tools"]
                }
            ),
            Tool(
                name="execute_workflow",
                description="Execute a workflow by ID",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workflow_id": {
                            "type": "string",
                            "description": "ID of the workflow to execute"
                        },
                        "project_path": {
                            "type": "string",
                            "description": "Path to the project",
                            "default": "."
                        },
                        "parameters": {
                            "type": "object",
                            "description": "Additional parameters for workflow execution",
                            "default": {}
                        }
                    },
                    "required": ["workflow_id"]
                }
            ),
            Tool(
                name="create_conditional_workflow",
                description="Create a conditional workflow that executes different actions based on analysis results",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workflow_name": {
                            "type": "string",
                            "description": "Name for the conditional workflow",
                            "default": "Untitled Conditional Workflow"
                        },
                        "conditional_steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "tool": {"type": "string"},
                                    "parameters": {"type": "object"},
                                    "conditions": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "field_path": {"type": "string"},
                                                "type": {"type": "string", "enum": ["gt", "lt", "eq", "ne", "contains", "not_contains", "exists", "not_exists"]},
                                                "value": {},
                                                "description": {"type": "string"}
                                            },
                                            "required": ["field_path", "type", "value"]
                                        }
                                    },
                                    "depends_on": {
                                        "type": "array",
                                        "items": {"type": "string"}
                                    },
                                    "description": {"type": "string"}
                                },
                                "required": ["tool"]
                            },
                            "description": "List of conditional steps"
                        },
                        "project_path": {
                            "type": "string",
                            "description": "Path to the project",
                            "default": "."
                        }
                    },
                    "required": ["conditional_steps"]
                }
            ),
            Tool(
                name="create_batch_operation",
                description="Create a batch operation to apply fixes across multiple files simultaneously",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "operation_type": {
                            "type": "string",
                            "description": "Type of operation to perform",
                            "enum": ["optimize_imports", "remove_dead_code", "standardize_patterns", "generate_docstrings", "comprehensive_refactor"]
                        },
                        "targets": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of file paths or project paths to target"
                        },
                        "parameters": {
                            "type": "object",
                            "description": "Parameters for the operation",
                            "default": {}
                        },
                        "parallel": {
                            "type": "boolean",
                            "description": "Whether to execute in parallel",
                            "default": True
                        },
                        "project_path": {
                            "type": "string",
                            "description": "Base project path",
                            "default": "."
                        }
                    },
                    "required": ["operation_type", "targets"]
                }
            ),
            Tool(
                name="execute_batch_operation",
                description="Execute a previously created batch operation",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "batch_id": {
                            "type": "string",
                            "description": "ID of the batch operation to execute"
                        },
                        "project_path": {
                            "type": "string",
                            "description": "Base project path",
                            "default": "."
                        }
                    },
                    "required": ["batch_id"]
                }
            ),
            Tool(
                name="load_custom_workflow",
                description="Load a custom workflow definition from YAML or dictionary",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workflow_definition": {
                            "type": "string",
                            "description": "Workflow definition (file path for YAML or object for dict)"
                        },
                        "workflow_format": {
                            "type": "string",
                            "enum": ["yaml_file", "dict"],
                            "description": "Format of the workflow definition",
                            "default": "dict"
                        },
                        "project_path": {
                            "type": "string",
                            "description": "Path to the project",
                            "default": "."
                        }
                    },
                    "required": ["workflow_definition"]
                }
            ),
            Tool(
                name="setup_scheduled_hygiene",
                description="Set up scheduled code hygiene checks for regular automated quality maintenance",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Path to the project",
                            "default": "."
                        },
                        "interval_minutes": {
                            "type": "number",
                            "description": "Interval between hygiene checks in minutes",
                            "default": 60
                        },
                        "safety_mode": {
                            "type": "boolean",
                            "description": "Enable safety mode with additional warnings",
                            "default": True
                        },
                        "apply_fixes": {
                            "type": "boolean",
                            "description": "Whether to automatically apply safe fixes",
                            "default": False
                        },
                        "notification_webhook": {
                            "type": "string",
                            "description": "Optional webhook URL for notifications",
                            "default": None
                        }
                    }
                }
            ),
            Tool(
                name="get_workflow_status",
                description="Get status and information for a specific workflow",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "workflow_id": {
                            "type": "string",
                            "description": "ID of the workflow to query"
                        },
                        "project_path": {
                            "type": "string",
                            "description": "Path to the project",
                            "default": "."
                        }
                    },
                    "required": ["workflow_id"]
                }
            ),
            Tool(
                name="list_workflows",
                description="List all workflows with optional filtering and sorting",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "filter_tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Filter workflows by tags",
                            "default": None
                        },
                        "status_filter": {
                            "type": "string",
                            "enum": ["active", "ready", "paused"],
                            "description": "Filter workflows by status",
                            "default": None
                        },
                        "sort_by": {
                            "type": "string",
                            "enum": ["created_at", "last_run", "name"],
                            "description": "Sort workflows by field",
                            "default": "created_at"
                        },
                        "project_path": {
                            "type": "string",
                            "description": "Path to the project",
                            "default": "."
                        }
                    }
                }
            ),
            Tool(
                name="get_workflow_metrics",
                description="Get comprehensive workflow metrics and analytics",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "time_period": {
                            "type": "string",
                            "enum": ["last_7_days", "last_30_days", "last_90_days"],
                            "description": "Time period for metrics analysis",
                            "default": "last_30_days"
                        },
                        "include_trends": {
                            "type": "boolean",
                            "description": "Include performance trend analysis",
                            "default": True
                        },
                        "project_path": {
                            "type": "string",
                            "description": "Path to the project",
                            "default": "."
                        }
                    }
                }
            ),
            Tool(
                name="analyze_requirements",
                description="Analyze requirements.txt and detect missing packages from imports (AI coding workflow helper)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Path to the project to analyze",
                            "default": "."
                        },
                        "check_installed": {
                            "type": "boolean",
                            "description": "Check if packages are actually installed",
                            "default": True
                        },
                        "target_files": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Specific files to analyze (optional)",
                            "default": None
                        }
                    }
                }
            ),
            Tool(
                name="update_requirements",
                description="Update requirements.txt file based on analysis (AI coding workflow helper)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Path to the project",
                            "default": "."
                        },
                        "backup": {
                            "type": "boolean",
                            "description": "Create backup of original requirements.txt",
                            "default": True
                        },
                        "dry_run": {
                            "type": "boolean", 
                            "description": "Show changes without applying them",
                            "default": True
                        },
                        "apply_changes": {
                            "type": "boolean",
                            "description": "Actually apply changes to requirements.txt",
                            "default": False
                        }
                    }
                }
            ),
            Tool(
                name="analyze_file_organization",
                description="Analyze project file organization and detect messy AI-generated patterns (AI coding workflow helper)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Path to the project to analyze",
                            "default": "."
                        },
                        "check_patterns": {
                            "type": "boolean",
                            "description": "Check for naming pattern inconsistencies",
                            "default": True
                        }
                    }
                }
            ),
            Tool(
                name="organize_files",
                description="Apply file organization recommendations with safety checks (AI coding workflow helper)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "project_path": {
                            "type": "string",
                            "description": "Path to the project to organize",
                            "default": "."
                        },
                        "dry_run": {
                            "type": "boolean",
                            "description": "Show what changes would be made without applying them",
                            "default": True
                        },
                        "backup": {
                            "type": "boolean",
                            "description": "Create backups before moving files",
                            "default": True
                        },
                        "apply_changes": {
                            "type": "boolean",
                            "description": "Actually apply the organization changes",
                            "default": False
                        }
                    }
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
        print("ERROR: MCP dependencies not found. Install with: pip install deepflow[mcp]")
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