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
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import logging

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
    import os
    
    # Add tools directory to path for imports
    tools_dir = Path(__file__).parent.parent.parent / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    
    from dependency_visualizer import DependencyVisualizer
    from code_analyzer import CodeAnalyzer
    from pre_commit_validator import DependencyValidator
    from monitoring_dashboard import DependencyMonitor 
    from doc_generator import DocumentationGenerator
    
    TOOLS_AVAILABLE = True
except ImportError as e:
    TOOLS_AVAILABLE = False
    print(f"WARNING: Could not import deepflow tools: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepflowMCPServer:
    """MCP Server for Deepflow tools."""
    
    def __init__(self):
        """Initialize the Deepflow MCP server."""
        self.server = Server("deepflow")
        self._setup_tools()
        
    def _setup_tools(self):
        """Set up MCP tool handlers."""
        
        # Register the list_tools handler
        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """Handle list tools request."""
            return self.get_tools()
        
        # Dependency visualization tools
        @self.server.call_tool()
        async def handle_analyze_dependencies(tool_name: str, arguments: dict):
            """Analyze project dependencies and create visualization."""
            try:
                project_path = arguments.get("project_path", ".")
                output_format = arguments.get("format", "text")
                ai_awareness = arguments.get("ai_awareness", True)
                
                if not TOOLS_AVAILABLE:
                    return [TextContent(
                        type="text",
                        text="Error: Deepflow tools not available. Please check installation."
                    )]
                
                visualizer = DependencyVisualizer(project_path, ai_awareness=ai_awareness)
                dependency_graph = visualizer.analyze_project()
                
                if output_format == "html":
                    html_output = visualizer.generate_interactive_html(dependency_graph)
                    return [TextContent(
                        type="text", 
                        text=f"HTML visualization generated: {html_output}"
                    )]
                elif output_format == "json":
                    json_data = visualizer.export_to_json(dependency_graph)
                    return [TextContent(
                            type="text",
                            text=json.dumps(json_data, indent=2)
                        )]
                else:
                    text_output = visualizer.generate_text_report(dependency_graph)
                    return [TextContent(
                            type="text",
                            text=text_output
                        )]
                    
            except Exception as e:
                logger.error(f"Error in analyze_dependencies: {e}")
                return [TextContent(
                        type="text",
                        text=f"Error analyzing dependencies: {str(e)}"
                    )]

        @self.server.call_tool()
        async def handle_analyze_code_quality(tool_name: str, arguments: dict):
            """Analyze code quality and detect issues."""
            try:
                project_path = arguments.get("project_path", ".")
                analysis_type = arguments.get("analysis_type", "all")
                fix_imports = arguments.get("fix_imports", False)
                
                if not TOOLS_AVAILABLE:
                    return [TextContent(
                        type="text",
                        text="Error: Deepflow tools not available. Please check installation."
                    )]
                
                analyzer = CodeAnalyzer(project_path)
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
                
                return [TextContent(
                        type="text",
                        text=json.dumps(results, indent=2)
                    )]
                
            except Exception as e:
                logger.error(f"Error in analyze_code_quality: {e}")
                return [TextContent(
                        type="text",
                        text=f"Error analyzing code quality: {str(e)}"
                    )]

        @self.server.call_tool()
        async def handle_validate_commit(tool_name: str, arguments: dict):
            """Validate code changes before commit."""
            try:
                project_path = arguments.get("project_path", ".")
                check_dependencies = arguments.get("check_dependencies", True)
                check_patterns = arguments.get("check_patterns", True)
                
                if not TOOLS_AVAILABLE:
                    return [TextContent(
                        type="text",
                        text="Error: Deepflow tools not available. Please check installation."
                    )]
                
                validator = DependencyValidator(project_path)
                validation_result = validator.validate_changes(
                    check_dependencies=check_dependencies,
                    check_patterns=check_patterns
                )
                
                return [TextContent(
                        type="text",
                        text=json.dumps({
                            "valid": validation_result.is_valid,
                            "errors": validation_result.errors,
                            "warnings": validation_result.warnings,
                            "suggestions": validation_result.suggestions
                        }, indent=2)
                    )]
                
            except Exception as e:
                logger.error(f"Error in validate_commit: {e}")
                return [TextContent(
                        type="text",
                        text=f"Error validating commit: {str(e)}"
                    )]

        @self.server.call_tool()
        async def handle_generate_documentation(tool_name: str, arguments: dict):
            """Generate project documentation."""
            try:
                project_path = arguments.get("project_path", ".")
                doc_type = arguments.get("doc_type", "dependency_map")
                output_path = arguments.get("output_path", None)
                
                if not TOOLS_AVAILABLE:
                    return [TextContent(
                        type="text",
                        text="Error: Deepflow tools not available. Please check installation."
                    )]
                
                doc_generator = DocumentationGenerator(project_path)
                
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
                
                return [TextContent(
                        type="text",
                        text=f"Documentation generated: {output_file}"
                    )]
                
            except Exception as e:
                logger.error(f"Error in generate_documentation: {e}")
                return [TextContent(
                        type="text",
                        text=f"Error generating documentation: {str(e)}"
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
            )
        ]

    async def run(self):
        """Run the MCP server."""
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options()
            )


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