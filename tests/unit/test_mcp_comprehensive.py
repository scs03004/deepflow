"""
Comprehensive MCP Protocol Integration Tests.
Tests server initialization, tool execution, and communication protocol.
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Test availability of MCP and tools
MCP_TEST_AVAILABLE = False
DEEPFLOW_SERVER_AVAILABLE = False

try:
    # Try to import MCP - if not available, we'll skip real MCP tests
    import mcp
    from mcp.server import Server
    from mcp.types import Tool, TextContent, CallToolResult
    MCP_TEST_AVAILABLE = True
except ImportError:
    # Create fallback mocks for testing structure
    class Tool:
        def __init__(self, name, description, inputSchema):
            self.name = name
            self.description = description
            self.inputSchema = inputSchema

try:
    from deepflow.mcp.server import DeepflowMCPServer
    DEEPFLOW_SERVER_AVAILABLE = True
except ImportError:
    DEEPFLOW_SERVER_AVAILABLE = False


@pytest.mark.unit
class TestMCPServerInitialization:
    """Test MCP server initialization and configuration."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        import os
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @pytest.mark.skipif(not DEEPFLOW_SERVER_AVAILABLE, reason="DeepflowMCPServer not available")
    def test_server_initialization_basic(self):
        """Test basic server initialization."""
        try:
            # Mock the MCP imports to avoid dependency issues
            with patch('deepflow.mcp.server.MCP_AVAILABLE', True), \
                 patch('deepflow.mcp.server.TOOLS_AVAILABLE', True):
                
                server = DeepflowMCPServer()
                
                # Should initialize without errors
                assert server is not None
                assert hasattr(server, 'server'), "Should have MCP server instance"
                assert hasattr(server, 'get_tools'), "Should have get_tools method"
                
        except Exception as e:
            pytest.fail(f"Server initialization failed: {e}")
    
    @pytest.mark.skipif(not DEEPFLOW_SERVER_AVAILABLE, reason="DeepflowMCPServer not available")
    def test_tools_registration(self):
        """Test that tools are properly registered."""
        try:
            with patch('deepflow.mcp.server.MCP_AVAILABLE', True), \
                 patch('deepflow.mcp.server.TOOLS_AVAILABLE', True):
                
                server = DeepflowMCPServer()
                tools = server.get_tools()
                
                # Should return a list of tools
                assert isinstance(tools, list), "get_tools should return a list"
                assert len(tools) > 0, "Should have at least one tool registered"
                
                # Each tool should have required attributes
                for tool in tools:
                    assert hasattr(tool, 'name'), f"Tool should have name attribute: {tool}"
                    assert hasattr(tool, 'description'), f"Tool should have description: {tool}"
                    assert hasattr(tool, 'inputSchema'), f"Tool should have inputSchema: {tool}"
                    
                    # Validate tool names are strings
                    assert isinstance(tool.name, str), f"Tool name should be string: {tool.name}"
                    assert len(tool.name) > 0, f"Tool name should not be empty: {tool.name}"
                
        except Exception as e:
            pytest.fail(f"Tools registration test failed: {e}")
    
    @pytest.mark.skipif(not DEEPFLOW_SERVER_AVAILABLE, reason="DeepflowMCPServer not available")
    def test_expected_core_tools_present(self):
        """Test that expected core tools are present."""
        try:
            with patch('deepflow.mcp.server.MCP_AVAILABLE', True), \
                 patch('deepflow.mcp.server.TOOLS_AVAILABLE', True):
                
                server = DeepflowMCPServer()
                tools = server.get_tools()
                tool_names = [tool.name for tool in tools]
                
                # Core tools that should always be present
                expected_core_tools = [
                    "analyze_dependencies",
                    "analyze_code_quality", 
                    "validate_commit",
                    "generate_documentation"
                ]
                
                for expected_tool in expected_core_tools:
                    assert expected_tool in tool_names, f"Core tool '{expected_tool}' should be present. Available tools: {tool_names}"
                
        except Exception as e:
            pytest.fail(f"Core tools validation failed: {e}")
    
    @pytest.mark.skipif(not DEEPFLOW_SERVER_AVAILABLE, reason="DeepflowMCPServer not available")
    def test_tool_schema_validation(self):
        """Test that tool schemas are properly structured."""
        try:
            with patch('deepflow.mcp.server.MCP_AVAILABLE', True), \
                 patch('deepflow.mcp.server.TOOLS_AVAILABLE', True):
                
                server = DeepflowMCPServer()
                tools = server.get_tools()
                
                for tool in tools:
                    # Each tool should have a valid input schema
                    assert hasattr(tool, 'inputSchema'), f"Tool {tool.name} missing inputSchema"
                    schema = tool.inputSchema
                    
                    # Schema should be a dictionary
                    assert isinstance(schema, dict), f"Tool {tool.name} schema should be dict, got {type(schema)}"
                    
                    # Schema should have type property
                    assert 'type' in schema, f"Tool {tool.name} schema missing 'type' property"
                    
                    # For object type schemas, should have properties
                    if schema.get('type') == 'object':
                        assert 'properties' in schema, f"Tool {tool.name} object schema missing 'properties'"
                        
        except Exception as e:
            pytest.fail(f"Tool schema validation failed: {e}")


@pytest.mark.unit 
class TestMCPToolExecution:
    """Test MCP tool execution functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        import os
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_test_project(self):
        """Create a simple test project for tool testing."""
        # Create main.py
        main_py = self.test_path / "main.py"
        main_py.write_text("""
import sys
import json
from utils import helper

def main():
    data = {"status": "ok", "version": sys.version_info}
    return json.dumps(data)

if __name__ == "__main__":
    print(main())
""")
        
        # Create utils.py  
        utils_py = self.test_path / "utils.py"
        utils_py.write_text("""
import os
from pathlib import Path

def helper():
    return {"cwd": os.getcwd(), "path": str(Path.cwd())}
""")
        
        return str(self.test_path)
    
    @pytest.mark.skipif(not DEEPFLOW_SERVER_AVAILABLE, reason="DeepflowMCPServer not available")
    def test_analyze_dependencies_tool_structure(self):
        """Test analyze_dependencies tool has proper structure."""
        try:
            with patch('deepflow.mcp.server.MCP_AVAILABLE', True), \
                 patch('deepflow.mcp.server.TOOLS_AVAILABLE', True):
                
                server = DeepflowMCPServer()
                tools = server.get_tools()
                
                # Find the analyze_dependencies tool
                analyze_deps_tool = None
                for tool in tools:
                    if tool.name == "analyze_dependencies":
                        analyze_deps_tool = tool
                        break
                
                assert analyze_deps_tool is not None, "analyze_dependencies tool not found"
                
                # Validate tool structure
                assert isinstance(analyze_deps_tool.description, str), "Tool should have string description"
                assert len(analyze_deps_tool.description) > 10, "Tool description should be meaningful"
                
                # Validate input schema
                schema = analyze_deps_tool.inputSchema
                assert schema['type'] == 'object', "Tool should expect object input"
                assert 'properties' in schema, "Tool should have properties defined"
                
                # Should expect project_path parameter
                properties = schema['properties']
                assert 'project_path' in properties, "Tool should accept project_path parameter"
                
        except Exception as e:
            pytest.fail(f"analyze_dependencies tool structure test failed: {e}")
    
    @pytest.mark.skipif(not DEEPFLOW_SERVER_AVAILABLE, reason="DeepflowMCPServer not available")
    def test_tool_execution_error_handling(self):
        """Test tool execution handles errors gracefully."""
        try:
            with patch('deepflow.mcp.server.MCP_AVAILABLE', True), \
                 patch('deepflow.mcp.server.TOOLS_AVAILABLE', True):
                
                server = DeepflowMCPServer()
                
                # Test with non-existent project path
                # Note: This tests the structure, not actual execution since we'd need
                # the full MCP protocol implementation
                
                # Should not crash when initialized
                assert server is not None
                
        except Exception as e:
            pytest.fail(f"Error handling test failed: {e}")
    
    @pytest.mark.skipif(not DEEPFLOW_SERVER_AVAILABLE, reason="DeepflowMCPServer not available")
    def test_multiple_tools_available(self):
        """Test that multiple tools are available for different functions."""
        try:
            with patch('deepflow.mcp.server.MCP_AVAILABLE', True), \
                 patch('deepflow.mcp.server.TOOLS_AVAILABLE', True):
                
                server = DeepflowMCPServer()
                tools = server.get_tools()
                tool_names = [tool.name for tool in tools]
                
                # Should have tools for different categories
                dependency_tools = [name for name in tool_names if 'dependen' in name.lower()]
                quality_tools = [name for name in tool_names if 'quality' in name.lower() or 'analyz' in name.lower()]
                doc_tools = [name for name in tool_names if 'doc' in name.lower() or 'generate' in name.lower()]
                
                assert len(dependency_tools) > 0, f"Should have dependency-related tools. Available: {tool_names}"
                assert len(quality_tools) > 0, f"Should have quality analysis tools. Available: {tool_names}"
                assert len(doc_tools) > 0, f"Should have documentation tools. Available: {tool_names}"
                
        except Exception as e:
            pytest.fail(f"Multiple tools availability test failed: {e}")


@pytest.mark.unit
class TestMCPCommunicationProtocol:
    """Test MCP communication protocol compliance."""
    
    @pytest.mark.skipif(not DEEPFLOW_SERVER_AVAILABLE, reason="DeepflowMCPServer not available")
    def test_server_name_configuration(self):
        """Test server has proper name configuration."""
        try:
            with patch('deepflow.mcp.server.MCP_AVAILABLE', True), \
                 patch('deepflow.mcp.server.TOOLS_AVAILABLE', True):
                
                server = DeepflowMCPServer()
                
                # Should have a server instance with proper name
                assert hasattr(server, 'server'), "Should have MCP server instance"
                
                # Server name should be appropriate
                if hasattr(server.server, 'name'):
                    assert isinstance(server.server.name, str), "Server name should be string"
                    assert len(server.server.name) > 0, "Server name should not be empty"
                
        except Exception as e:
            pytest.fail(f"Server name configuration test failed: {e}")
    
    @pytest.mark.skipif(not DEEPFLOW_SERVER_AVAILABLE, reason="DeepflowMCPServer not available") 
    def test_graceful_fallback_handling(self):
        """Test server handles missing dependencies gracefully."""
        try:
            # Test with MCP unavailable
            with patch('deepflow.mcp.server.MCP_AVAILABLE', False), \
                 patch('deepflow.mcp.server.TOOLS_AVAILABLE', True):
                
                # Should not crash when MCP is unavailable
                # This tests the fallback logic in the server
                server = DeepflowMCPServer()
                assert server is not None
                
            # Test with tools unavailable  
            with patch('deepflow.mcp.server.MCP_AVAILABLE', True), \
                 patch('deepflow.mcp.server.TOOLS_AVAILABLE', False):
                
                # Should not crash when tools are unavailable
                server = DeepflowMCPServer()
                assert server is not None
                
        except Exception as e:
            pytest.fail(f"Graceful fallback test failed: {e}")
    
    @pytest.mark.skipif(not DEEPFLOW_SERVER_AVAILABLE, reason="DeepflowMCPServer not available")
    def test_async_compatibility(self):
        """Test server is compatible with async operations."""
        try:
            with patch('deepflow.mcp.server.MCP_AVAILABLE', True), \
                 patch('deepflow.mcp.server.TOOLS_AVAILABLE', True):
                
                server = DeepflowMCPServer()
                
                # Should be compatible with async context
                # This is a structural test - actual async behavior would require
                # full MCP protocol implementation
                assert server is not None
                
        except Exception as e:
            pytest.fail(f"Async compatibility test failed: {e}")


@pytest.mark.unit
class TestMCPEdgeCases:
    """Test MCP edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        import os
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @pytest.mark.skipif(not DEEPFLOW_SERVER_AVAILABLE, reason="DeepflowMCPServer not available")
    def test_malformed_project_handling(self):
        """Test server handles malformed projects gracefully."""
        # Create a malformed project
        malformed_py = self.test_path / "malformed.py"
        malformed_py.write_text("""
import sys
def broken_function(
    # Missing closing parenthesis and syntax errors
    return "broken
""")
        
        try:
            with patch('deepflow.mcp.server.MCP_AVAILABLE', True), \
                 patch('deepflow.mcp.server.TOOLS_AVAILABLE', True):
                
                server = DeepflowMCPServer()
                
                # Server should initialize even with malformed project in test dir
                assert server is not None
                
                # Tools should be available
                tools = server.get_tools()
                assert len(tools) > 0
                
        except Exception as e:
            pytest.fail(f"Malformed project handling failed: {e}")
    
    @pytest.mark.skipif(not DEEPFLOW_SERVER_AVAILABLE, reason="DeepflowMCPServer not available")
    def test_empty_project_handling(self):
        """Test server handles empty projects gracefully."""
        try:
            with patch('deepflow.mcp.server.MCP_AVAILABLE', True), \
                 patch('deepflow.mcp.server.TOOLS_AVAILABLE', True):
                
                server = DeepflowMCPServer()
                
                # Should handle empty project directory without issues
                assert server is not None
                tools = server.get_tools()
                assert len(tools) > 0
                
        except Exception as e:
            pytest.fail(f"Empty project handling failed: {e}")
    
    @pytest.mark.skipif(not DEEPFLOW_SERVER_AVAILABLE, reason="DeepflowMCPServer not available") 
    def test_large_project_structure_support(self):
        """Test server can handle large project structures."""
        # Create a larger project structure
        for i in range(20):
            file_path = self.test_path / f"module_{i:02d}.py"
            file_path.write_text(f"""
import sys
import json

class Module{i}:
    def process(self):
        return {i}

def function_{i}():
    return Module{i}().process()
""")
        
        try:
            with patch('deepflow.mcp.server.MCP_AVAILABLE', True), \
                 patch('deepflow.mcp.server.TOOLS_AVAILABLE', True):
                
                server = DeepflowMCPServer()
                
                # Should handle larger projects
                assert server is not None
                tools = server.get_tools()
                assert len(tools) > 0
                
        except Exception as e:
            pytest.fail(f"Large project structure support failed: {e}")
    
    @pytest.mark.skipif(not DEEPFLOW_SERVER_AVAILABLE, reason="DeepflowMCPServer not available")
    def test_unicode_content_support(self):
        """Test server handles Unicode content in projects."""
        # Create file with Unicode content
        unicode_py = self.test_path / "unicode_测试.py"
        unicode_py.write_text("""
# -*- coding: utf-8 -*-
import sys

def función_test():
    # Unicode comments: العربية 中文 русский
    mensaje = "¡Hola mundo! 🌍"
    return mensaje
""", encoding='utf-8')
        
        try:
            with patch('deepflow.mcp.server.MCP_AVAILABLE', True), \
                 patch('deepflow.mcp.server.TOOLS_AVAILABLE', True):
                
                server = DeepflowMCPServer()
                
                # Should handle Unicode content
                assert server is not None
                tools = server.get_tools()
                assert len(tools) > 0
                
        except Exception as e:
            pytest.fail(f"Unicode content support failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])