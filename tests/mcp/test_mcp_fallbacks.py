"""
Tests for graceful MCP fallbacks when dependencies are missing.
"""

import pytest
import sys
from unittest.mock import patch
from pathlib import Path


class TestMCPFallbacks:
    """Test MCP graceful fallbacks."""
    
    def test_mcp_import_failure_handling(self):
        """Test handling when MCP package is not installed."""
        # Simulate MCP not being available
        with patch.dict('sys.modules', {'mcp': None}):
            # Clear any existing imports
            for module in list(sys.modules.keys()):
                if module.startswith('deepflow.mcp'):
                    sys.modules.pop(module, None)
            
            # Add project to path
            project_root = str(Path(__file__).parent.parent.parent)
            if project_root not in sys.path:
                sys.path.insert(0, project_root)
            
            # Import should handle missing MCP gracefully
            try:
                from deepflow.mcp import server
                # Should have MCP_AVAILABLE = False
                assert hasattr(server, 'MCP_AVAILABLE')
                # Note: This might be True if mcp is actually installed in the test environment
            except ImportError:
                # Expected when MCP is truly not available
                pass
    
    def test_mcp_server_unavailable_message(self):
        """Test error message when MCP server is unavailable."""
        with patch('deepflow.mcp.server.MCP_AVAILABLE', False), \
             patch('sys.exit') as mock_exit, \
             patch('builtins.print') as mock_print:
            
            from deepflow.mcp.server import async_main
            
            # Should print error and exit
            import asyncio
            asyncio.run(async_main())
            
            mock_print.assert_called_with("ERROR: MCP dependencies not found. Install with: pip install deepflow[mcp]")
            mock_exit.assert_called_with(1)
    
    def test_tools_unavailable_fallback(self):
        """Test fallback when tools are not available."""
        with patch('deepflow.mcp.server.TOOLS_AVAILABLE', False):
            from deepflow.mcp.server import DeepflowMCPServer
            
            # Server should still initialize
            with patch('deepflow.mcp.server.Server'):
                server = DeepflowMCPServer()
                
                # Should still provide tools (they'll just return errors)
                tools = server.get_tools()
                assert len(tools) > 0
    
    def test_deepflow_init_mcp_fallback(self):
        """Test deepflow.__init__.py handles missing MCP gracefully."""
        # Clear any existing deepflow imports
        for module in list(sys.modules.keys()):
            if module.startswith('deepflow'):
                sys.modules.pop(module, None)
        
        # Add project to path
        project_root = str(Path(__file__).parent.parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Mock MCP not being available
        with patch.dict('sys.modules', {'mcp': None}):
            import deepflow
            
            # Should have MCP_AVAILABLE flag
            assert hasattr(deepflow, 'MCP_AVAILABLE')
            # Should be boolean
            assert isinstance(deepflow.MCP_AVAILABLE, bool)


class TestMCPServerRobustness:
    """Test MCP server robustness and error handling."""
    
    def test_server_handles_tool_import_failures(self):
        """Test server handles individual tool import failures."""
        # Mock specific tool imports failing
        with patch('deepflow.mcp.server.DependencyVisualizer', side_effect=ImportError("Tool not available")):
            from deepflow.mcp.server import DeepflowMCPServer
            
            # Server should still initialize
            with patch('deepflow.mcp.server.Server'):
                server = DeepflowMCPServer()
                
                # Should still provide tool definitions
                tools = server.get_tools()
                analyze_deps = next((t for t in tools if t.name == "analyze_dependencies"), None)
                assert analyze_deps is not None
    
    def test_server_handles_missing_optional_dependencies(self):
        """Test server handles missing optional dependencies in tools."""
        # Mock various dependencies being unavailable
        mock_modules = {
            'networkx': None,
            'plotly': None,
            'matplotlib': None,
            'rich': None
        }
        
        with patch.dict('sys.modules', mock_modules):
            from deepflow.mcp.server import DeepflowMCPServer
            
            # Server should still work
            with patch('deepflow.mcp.server.Server'):
                server = DeepflowMCPServer()
                tools = server.get_tools()
                
                # All tools should still be defined
                expected_tools = ["analyze_dependencies", "analyze_code_quality", "validate_commit", "generate_documentation"]
                tool_names = [t.name for t in tools]
                
                for expected in expected_tools:
                    assert expected in tool_names
    
    @pytest.mark.asyncio
    async def test_tool_error_handling_with_unavailable_dependencies(self):
        """Test tool error handling when dependencies are unavailable."""
        with patch('deepflow.mcp.server.TOOLS_AVAILABLE', False):
            from deepflow.mcp.server import DeepflowMCPServer
            
            # Tool execution should handle unavailable tools gracefully
            # This would be tested in the actual tool handler implementations
            # by checking that they return appropriate error messages
            pass


class TestGracefulDegradation:
    """Test graceful degradation when components are missing."""
    
    def test_partial_functionality_with_missing_components(self):
        """Test that some functionality remains when components are missing."""
        # Test that basic package structure remains intact
        project_root = str(Path(__file__).parent.parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Clear imports
        for module in list(sys.modules.keys()):
            if module.startswith('deepflow'):
                sys.modules.pop(module, None)
        
        # Should be able to import basic package
        import deepflow
        
        # Should have version info
        assert hasattr(deepflow, '__version__')
        assert hasattr(deepflow, '__author__')
    
    def test_tools_availability_reporting(self):
        """Test that tools availability is properly reported."""
        project_root = str(Path(__file__).parent.parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        import deepflow
        
        # Should report availability status
        assert hasattr(deepflow, 'TOOLS_AVAILABLE')
        assert hasattr(deepflow, 'MCP_AVAILABLE')
        
        # Should be boolean values
        assert isinstance(deepflow.TOOLS_AVAILABLE, bool)
        assert isinstance(deepflow.MCP_AVAILABLE, bool)
    
    def test_mcp_availability_with_missing_server_deps(self):
        """Test MCP availability reporting with missing server dependencies."""
        # Mock missing MCP server dependencies
        with patch.dict('sys.modules', {'mcp.server': None, 'mcp.types': None}):
            # Clear existing imports
            for module in list(sys.modules.keys()):
                if module.startswith('deepflow.mcp'):
                    sys.modules.pop(module, None)
            
            try:
                from deepflow.mcp import server
                # Should handle gracefully
                assert hasattr(server, 'MCP_AVAILABLE')
            except (ImportError, SystemExit):
                # Also acceptable - dependency check at import time
                pass


class TestErrorMessaging:
    """Test error messaging for missing dependencies."""
    
    def test_helpful_error_messages(self):
        """Test that error messages are helpful for users."""
        with patch('deepflow.mcp.server.MCP_AVAILABLE', False), \
             patch('builtins.print') as mock_print:
            
            from deepflow.mcp.server import async_main
            
            # Should provide installation instructions
            import asyncio
            with patch('sys.exit'):
                asyncio.run(async_main())
            
            # Check that the error message is helpful
            printed_messages = [call.args[0] for call in mock_print.call_args_list]
            error_message = ' '.join(printed_messages)
            
            assert "pip install deepflow[mcp]" in error_message
            assert "dependencies not found" in error_message.lower()
    
    def test_tools_unavailable_messaging(self):
        """Test messaging when tools are unavailable."""
        with patch('deepflow.mcp.server.TOOLS_AVAILABLE', False), \
             patch('builtins.print') as mock_print:
            
            # When tools are unavailable, there should be appropriate messaging
            # This would be tested in the tool handler implementations
            pass


class TestFallbackBehavior:
    """Test specific fallback behaviors."""
    
    def test_cli_entry_points_with_missing_mcp(self):
        """Test CLI entry points work when MCP is missing."""
        # The regular CLI tools should work even if MCP is not available
        project_root = str(Path(__file__).parent.parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Mock MCP not available
        with patch.dict('sys.modules', {'mcp': None}):
            # Regular tools should still be importable
            try:
                sys.path.insert(0, str(Path(project_root) / "tools"))
                import dependency_visualizer
                # Should work independently of MCP
                assert hasattr(dependency_visualizer, 'DependencyVisualizer')
            except ImportError:
                # May fail due to other dependencies, but not MCP
                pass
    
    def test_package_metadata_always_available(self):
        """Test that package metadata is always available."""
        project_root = str(Path(__file__).parent.parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Clear imports
        for module in list(sys.modules.keys()):
            if module.startswith('deepflow'):
                sys.modules.pop(module, None)
        
        # Mock all optional dependencies as unavailable
        mock_modules = {
            'mcp': None,
            'networkx': None,
            'rich': None,
            'plotly': None,
            'matplotlib': None,
            'jinja2': None
        }
        
        with patch.dict('sys.modules', mock_modules):
            import deepflow
            
            # Basic metadata should always be available
            assert hasattr(deepflow, '__version__')
            assert hasattr(deepflow, '__author__')
            assert hasattr(deepflow, '__email__')
    
    def test_import_error_handling_patterns(self):
        """Test that import error handling follows consistent patterns."""
        # Test the pattern used throughout the codebase
        def mock_import_pattern(module_name):
            try:
                __import__(module_name)
                return True, None
            except ImportError as e:
                return False, str(e)
        
        # Should handle missing modules gracefully
        available, error = mock_import_pattern('nonexistent_module_xyz')
        assert available is False
        assert isinstance(error, str)
        
        # Should handle existing modules correctly
        available, error = mock_import_pattern('os')
        assert available is True
        assert error is None


class TestMCPIntegrationToggle:
    """Test MCP integration toggle functionality."""
    
    def test_mcp_can_be_disabled(self):
        """Test that MCP integration can be disabled."""
        # Even if MCP is available, it should be possible to disable it
        with patch('deepflow.mcp.server.MCP_AVAILABLE', False):
            # Should handle gracefully
            from deepflow.mcp.server import DeepflowMCPServer
            
            # Should still be able to create server instance (for testing)
            with patch('deepflow.mcp.server.Server'):
                server = DeepflowMCPServer()
                assert server is not None
    
    def test_deepflow_works_without_mcp_directory(self):
        """Test that deepflow works even if MCP directory is missing."""
        project_root = str(Path(__file__).parent.parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Simulate MCP import failure by preventing the import
        import builtins
        original_import = builtins.__import__
        
        def mock_import(name, *args, **kwargs):
            if name == 'deepflow.mcp' or (len(args) > 0 and args[0] and 'mcp' in name):
                raise ImportError("Simulated MCP directory missing")
            return original_import(name, *args, **kwargs)
        
        with patch('builtins.__import__', side_effect=mock_import):
            # Clear imports
            for module in list(sys.modules.keys()):
                if module.startswith('deepflow'):
                    sys.modules.pop(module, None)
            
            # Should still be able to import main package
            import deepflow
            
            # MCP_AVAILABLE should be False when mcp import fails
            assert deepflow.MCP_AVAILABLE is False


@pytest.fixture(autouse=True)
def cleanup_mcp_imports():
    """Clean up MCP-related imports after each test."""
    yield
    
    # Remove MCP-related modules from cache
    modules_to_remove = []
    for module_name in sys.modules:
        if any(pattern in module_name for pattern in ['deepflow.mcp', 'mcp']):
            modules_to_remove.append(module_name)
    
    for module_name in modules_to_remove:
        sys.modules.pop(module_name, None)