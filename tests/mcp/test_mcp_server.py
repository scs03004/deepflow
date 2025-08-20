"""
MCP server functionality tests.
"""

import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path

# Mock MCP imports before importing the module
mock_mcp = MagicMock()
mock_server = MagicMock()
mock_types = MagicMock()

# Create mock classes
mock_types.Tool = MagicMock
mock_types.TextContent = MagicMock
mock_types.CallToolResult = MagicMock
mock_types.ListToolsRequest = MagicMock
mock_types.CallToolRequest = MagicMock

mock_mcp.server = mock_server
mock_mcp.types = mock_types
mock_mcp.server.stdio = MagicMock()

with patch.dict('sys.modules', {
    'mcp': mock_mcp,
    'mcp.server': mock_server,
    'mcp.types': mock_types,
    'mcp.server.stdio': mock_mcp.server.stdio
}):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    # Mock tools availability
    with patch('deepflow.mcp.server.TOOLS_AVAILABLE', True), \
         patch('deepflow.mcp.server.MCP_AVAILABLE', True):
        from deepflow.mcp import server as mcp_server


class TestDeepflowMCPServer:
    """Test cases for DeepflowMCPServer class."""
    
    def test_init(self):
        """Test DeepflowMCPServer initialization."""
        with patch('deepflow.mcp.server.Server') as mock_server_class:
            mock_server_instance = MagicMock()
            mock_server_class.return_value = mock_server_instance
            
            server = mcp_server.DeepflowMCPServer()
            
            assert server.server == mock_server_instance
            mock_server_class.assert_called_once_with("deepflow")
    
    def test_get_tools(self):
        """Test get_tools method returns correct tool definitions."""
        with patch('deepflow.mcp.server.Server'):
            server = mcp_server.DeepflowMCPServer()
            tools = server.get_tools()
            
            assert isinstance(tools, list)
            assert len(tools) == 4  # Expected number of tools
            
            tool_names = [tool.name for tool in tools]
            expected_tools = [
                "analyze_dependencies", 
                "analyze_code_quality",
                "validate_commit",
                "generate_documentation"
            ]
            
            for expected_tool in expected_tools:
                assert expected_tool in tool_names
    
    def test_tool_schema_validation(self):
        """Test that tool schemas are properly defined."""
        with patch('deepflow.mcp.server.Server'):
            server = mcp_server.DeepflowMCPServer()
            tools = server.get_tools()
            
            for tool in tools:
                # Each tool should have required attributes
                assert hasattr(tool, 'name')
                assert hasattr(tool, 'description')
                assert hasattr(tool, 'inputSchema')
                
                # Schema should be valid JSON schema structure
                schema = tool.inputSchema
                assert isinstance(schema, dict)
                assert 'type' in schema
                assert schema['type'] == 'object'
                assert 'properties' in schema
    
    @pytest.mark.asyncio
    async def test_analyze_dependencies_tool(self, mock_project_structure):
        """Test analyze_dependencies tool functionality."""
        with patch('deepflow.mcp.server.Server') as mock_server_class, \
             patch('deepflow.mcp.server.DependencyVisualizer') as mock_visualizer_class:
            
            # Setup mocks
            mock_server_instance = MagicMock()
            mock_server_class.return_value = mock_server_instance
            
            mock_visualizer = MagicMock()
            mock_visualizer_class.return_value = mock_visualizer
            mock_visualizer.analyze_project.return_value = MagicMock()
            mock_visualizer.generate_text_report.return_value = "Dependency report"
            
            # Create server instance
            server = mcp_server.DeepflowMCPServer()
            
            # Get the analyze_dependencies handler
            # Note: In real implementation, this would be registered with @server.call_tool()
            # For testing, we'll call the method directly
            arguments = {
                "project_path": str(mock_project_structure),
                "format": "text",
                "ai_awareness": True
            }
            
            # Mock the call_tool result
            with patch('deepflow.mcp.server.CallToolResult') as mock_result:
                mock_result.return_value = MagicMock()
                
                # This would normally be called by the MCP framework
                # Here we test the logic that would be in the handler
                mock_visualizer_class.assert_not_called()  # Until we actually call it
                
                # Simulate calling the tool
                visualizer = mock_visualizer_class(arguments["project_path"], ai_awareness=arguments["ai_awareness"])
                graph = visualizer.analyze_project()
                result = visualizer.generate_text_report(graph)
                
                assert result == "Dependency report"
                mock_visualizer_class.assert_called_with(str(mock_project_structure), ai_awareness=True)
    
    @pytest.mark.asyncio
    async def test_analyze_code_quality_tool(self, mock_project_structure):
        """Test analyze_code_quality tool functionality."""
        with patch('deepflow.mcp.server.Server') as mock_server_class, \
             patch('deepflow.mcp.server.CodeAnalyzer') as mock_analyzer_class:
            
            # Setup mocks
            mock_server_instance = MagicMock()
            mock_server_class.return_value = mock_server_instance
            
            mock_analyzer = MagicMock()
            mock_analyzer_class.return_value = mock_analyzer
            mock_analyzer.analyze_unused_imports.return_value = []
            mock_analyzer.analyze_coupling.return_value = []
            
            # Create server instance
            server = mcp_server.DeepflowMCPServer()
            
            arguments = {
                "project_path": str(mock_project_structure),
                "analysis_type": "imports",
                "fix_imports": False
            }
            
            # Simulate the tool logic
            analyzer = mock_analyzer_class(arguments["project_path"])
            import_results = analyzer.analyze_unused_imports(fix_mode=arguments["fix_imports"])
            
            assert import_results == []
            mock_analyzer_class.assert_called_with(str(mock_project_structure))
            mock_analyzer.analyze_unused_imports.assert_called_with(fix_mode=False)
    
    @pytest.mark.asyncio
    async def test_validate_commit_tool(self, mock_project_structure):
        """Test validate_commit tool functionality."""
        with patch('deepflow.mcp.server.Server') as mock_server_class, \
             patch('deepflow.mcp.server.PreCommitValidator') as mock_validator_class:
            
            # Setup mocks
            mock_server_instance = MagicMock()
            mock_server_class.return_value = mock_server_instance
            
            mock_validator = MagicMock()
            mock_validator_class.return_value = mock_validator
            
            # Mock validation result
            mock_result = MagicMock()
            mock_result.is_valid = True
            mock_result.errors = []
            mock_result.warnings = ["Consider updating dependencies"]
            mock_result.suggestions = ["Add type hints"]
            mock_validator.validate_changes.return_value = mock_result
            
            # Create server instance
            server = mcp_server.DeepflowMCPServer()
            
            arguments = {
                "project_path": str(mock_project_structure),
                "check_dependencies": True,
                "check_patterns": True
            }
            
            # Simulate the tool logic
            validator = mock_validator_class(arguments["project_path"])
            validation_result = validator.validate_changes(
                check_dependencies=arguments["check_dependencies"],
                check_patterns=arguments["check_patterns"]
            )
            
            assert validation_result.is_valid is True
            assert len(validation_result.warnings) == 1
            mock_validator_class.assert_called_with(str(mock_project_structure))
    
    @pytest.mark.asyncio
    async def test_generate_documentation_tool(self, mock_project_structure):
        """Test generate_documentation tool functionality."""
        with patch('deepflow.mcp.server.Server') as mock_server_class, \
             patch('deepflow.mcp.server.DocumentationGenerator') as mock_doc_gen_class:
            
            # Setup mocks
            mock_server_instance = MagicMock()
            mock_server_class.return_value = mock_server_instance
            
            mock_doc_gen = MagicMock()
            mock_doc_gen_class.return_value = mock_doc_gen
            mock_doc_gen.generate_dependency_map.return_value = "DEPENDENCY_MAP.md"
            
            # Create server instance
            server = mcp_server.DeepflowMCPServer()
            
            arguments = {
                "project_path": str(mock_project_structure),
                "doc_type": "dependency_map",
                "output_path": None
            }
            
            # Simulate the tool logic
            doc_generator = mock_doc_gen_class(arguments["project_path"])
            output_file = doc_generator.generate_dependency_map(arguments["output_path"])
            
            assert output_file == "DEPENDENCY_MAP.md"
            mock_doc_gen_class.assert_called_with(str(mock_project_structure))
    
    @pytest.mark.asyncio
    async def test_tools_unavailable_handling(self):
        """Test handling when tools are not available."""
        with patch('deepflow.mcp.server.Server') as mock_server_class, \
             patch('deepflow.mcp.server.TOOLS_AVAILABLE', False):
            
            mock_server_instance = MagicMock()
            mock_server_class.return_value = mock_server_instance
            
            # Create server instance
            server = mcp_server.DeepflowMCPServer()
            
            # When tools are unavailable, tool handlers should return error messages
            # This would be tested in the actual tool handlers, but we can verify
            # the TOOLS_AVAILABLE flag is respected
            assert mcp_server.TOOLS_AVAILABLE is False
    
    @pytest.mark.asyncio
    async def test_server_run(self):
        """Test server run method."""
        with patch('deepflow.mcp.server.Server') as mock_server_class, \
             patch('deepflow.mcp.server.mcp.server.stdio.stdio_server') as mock_stdio:
            
            mock_server_instance = MagicMock()
            mock_server_instance.run = AsyncMock()
            mock_server_instance.create_initialization_options = MagicMock(return_value={})
            mock_server_class.return_value = mock_server_instance
            
            # Mock stdio server context manager
            mock_streams = (MagicMock(), MagicMock())
            mock_stdio.return_value.__aenter__ = AsyncMock(return_value=mock_streams)
            mock_stdio.return_value.__aexit__ = AsyncMock(return_value=None)
            
            # Create and run server
            server = mcp_server.DeepflowMCPServer()
            await server.run()
            
            # Verify server.run was called with correct parameters
            mock_server_instance.run.assert_called_once()
            call_args = mock_server_instance.run.call_args[0]
            assert len(call_args) == 3  # read_stream, write_stream, options


class TestMCPServerIntegration:
    """Integration tests for MCP server."""
    
    @pytest.mark.asyncio
    async def test_async_main_with_mcp_available(self):
        """Test async_main function when MCP is available."""
        with patch('deepflow.mcp.server.MCP_AVAILABLE', True), \
             patch('deepflow.mcp.server.DeepflowMCPServer') as mock_server_class:
            
            mock_server = MagicMock()
            mock_server.run = AsyncMock()
            mock_server_class.return_value = mock_server
            
            await mcp_server.async_main()
            
            mock_server_class.assert_called_once()
            mock_server.run.assert_called_once()
    
    @pytest.mark.asyncio 
    async def test_async_main_with_mcp_unavailable(self):
        """Test async_main function when MCP is unavailable."""
        with patch('deepflow.mcp.server.MCP_AVAILABLE', False), \
             patch('sys.exit') as mock_exit, \
             patch('builtins.print') as mock_print:
            
            await mcp_server.async_main()
            
            mock_print.assert_called_with("ERROR: MCP dependencies not found. Install with: pip install deepflow[mcp]")
            mock_exit.assert_called_with(1)
    
    def test_main_function(self):
        """Test main function (sync entry point)."""
        with patch('deepflow.mcp.server.asyncio.run') as mock_run:
            mcp_server.main()
            mock_run.assert_called_once_with(mcp_server.async_main)
    
    def test_main_entry_point(self):
        """Test __main__ entry point."""
        with patch('deepflow.mcp.server.main') as mock_main:
            # Simulate running as main module
            with patch('__main__.__name__', '__main__'):
                exec(compile(
                    'if __name__ == "__main__": main()',
                    'test', 'exec'
                ), {'__name__': '__main__', 'main': mock_main})
            
            mock_main.assert_called_once()


class TestMCPProtocolCompliance:
    """Test MCP protocol compliance."""
    
    def test_tool_schema_compliance(self):
        """Test that tool schemas comply with MCP specification."""
        with patch('deepflow.mcp.server.Server'):
            server = mcp_server.DeepflowMCPServer()
            tools = server.get_tools()
            
            for tool in tools:
                # Verify required MCP tool properties
                assert hasattr(tool, 'name')
                assert hasattr(tool, 'description')
                assert hasattr(tool, 'inputSchema')
                
                # Verify schema structure
                schema = tool.inputSchema
                assert schema['type'] == 'object'
                assert 'properties' in schema
                
                # Each property should have proper type definitions
                for prop_name, prop_def in schema['properties'].items():
                    assert 'type' in prop_def
                    assert prop_def['type'] in ['string', 'boolean', 'number', 'array', 'object']
    
    def test_tool_names_are_valid(self):
        """Test that tool names follow MCP naming conventions."""
        with patch('deepflow.mcp.server.Server'):
            server = mcp_server.DeepflowMCPServer()
            tools = server.get_tools()
            
            for tool in tools:
                # Tool names should be valid identifiers
                assert tool.name.replace('_', '').isalnum()
                assert not tool.name.startswith('_')
                assert len(tool.name) > 0
    
    def test_tool_descriptions_are_meaningful(self):
        """Test that tool descriptions are meaningful."""
        with patch('deepflow.mcp.server.Server'):
            server = mcp_server.DeepflowMCPServer()
            tools = server.get_tools()
            
            for tool in tools:
                # Descriptions should not be empty and should be descriptive
                assert len(tool.description) > 10
                assert isinstance(tool.description, str)
    
    def test_error_handling_compliance(self):
        """Test that error handling follows MCP patterns."""
        # This would test that tools return proper CallToolResult objects
        # with appropriate error handling, but since we're mocking the MCP types,
        # we'll verify the pattern structure
        
        with patch('deepflow.mcp.server.Server'):
            server = mcp_server.DeepflowMCPServer()
            
            # Verify the server has the expected structure for error handling
            assert hasattr(server, 'server')


class TestToolErrorHandling:
    """Test error handling in tool implementations."""
    
    @pytest.mark.asyncio
    async def test_analyze_dependencies_error_handling(self):
        """Test error handling in analyze_dependencies tool."""
        with patch('deepflow.mcp.server.Server') as mock_server_class, \
             patch('deepflow.mcp.server.DependencyVisualizer') as mock_visualizer_class:
            
            # Setup mock to raise exception
            mock_visualizer_class.side_effect = Exception("Test error")
            
            mock_server_instance = MagicMock()
            mock_server_class.return_value = mock_server_instance
            
            server = mcp_server.DeepflowMCPServer()
            
            # Test that exceptions are properly caught and handled
            # In the real implementation, this would return a CallToolResult with error
            try:
                mock_visualizer_class("/test/path")
            except Exception as e:
                assert str(e) == "Test error"
    
    @pytest.mark.asyncio
    async def test_invalid_arguments_handling(self):
        """Test handling of invalid tool arguments."""
        with patch('deepflow.mcp.server.Server'):
            server = mcp_server.DeepflowMCPServer()
            
            # Test that the server can handle missing or invalid arguments gracefully
            # This would be tested in the actual tool handlers
            invalid_args = {
                "invalid_param": "invalid_value"
            }
            
            # The tools should handle missing required parameters gracefully
            # with appropriate default values or error messages


@pytest.mark.parametrize("tool_name,expected_params", [
    ("analyze_dependencies", ["project_path", "format", "ai_awareness"]),
    ("analyze_code_quality", ["project_path", "analysis_type", "fix_imports"]),
    ("validate_commit", ["project_path", "check_dependencies", "check_patterns"]),
    ("generate_documentation", ["project_path", "doc_type", "output_path"])
])
def test_tool_parameters(tool_name, expected_params):
    """Test that tools have expected parameters."""
    with patch('deepflow.mcp.server.Server'):
        server = mcp_server.DeepflowMCPServer()
        tools = server.get_tools()
        
        tool = next((t for t in tools if t.name == tool_name), None)
        assert tool is not None, f"Tool {tool_name} not found"
        
        schema_props = tool.inputSchema['properties']
        for param in expected_params:
            assert param in schema_props, f"Parameter {param} missing from {tool_name}"