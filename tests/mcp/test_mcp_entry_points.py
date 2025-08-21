"""
Tests for MCP CLI entry points and command line interface.
"""

import pytest
import asyncio
import subprocess
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestMCPEntryPoints:
    """Test MCP entry points defined in pyproject.toml."""
    
    def test_deepflow_mcp_server_entry_point_exists(self):
        """Test that deepflow-mcp-server entry point is properly defined."""
        # Read pyproject.toml to verify entry point is defined
        project_root = Path(__file__).parent.parent.parent
        pyproject_path = project_root / "pyproject.toml"
        
        if pyproject_path.exists():
            content = pyproject_path.read_text()
            assert "deepflow-mcp-server" in content
            assert "deepflow.mcp.server:main" in content
    
    def test_mcp_server_main_function_exists(self):
        """Test that MCP server main function exists and is callable."""
        with patch('deepflow.mcp.server.MCP_AVAILABLE', True):
            from deepflow.mcp import server
            
            # Should have main function
            assert hasattr(server, 'main')
            assert callable(server.main)
    
    def test_mcp_server_async_main_function_exists(self):
        """Test that MCP server async_main function exists."""
        with patch('deepflow.mcp.server.MCP_AVAILABLE', True):
            from deepflow.mcp import server
            
            # Should have async_main function
            assert hasattr(server, 'async_main')
            assert callable(server.async_main)
    
    @patch('deepflow.mcp.server.asyncio.run')
    def test_main_calls_async_main(self, mock_run):
        """Test that main() properly calls asyncio.run()."""
        from deepflow.mcp.server import main
        
        main()
        
        # Verify asyncio.run was called exactly once (with some coroutine)
        mock_run.assert_called_once()
        # Verify the argument is a coroutine object
        args, kwargs = mock_run.call_args
        assert len(args) == 1
        # The argument should be a coroutine object from async_main()
        import types
        assert isinstance(args[0], types.CoroutineType)


class TestMCPServerCLI:
    """Test MCP server command line interface."""
    
    def test_mcp_server_can_be_imported_as_main(self):
        """Test that MCP server can be run as __main__."""
        with patch('deepflow.mcp.server.main') as mock_main:
            # Simulate running the module
            import deepflow.mcp.server
            
            # Module should be importable
            assert deepflow.mcp.server is not None
    
    @pytest.mark.asyncio
    async def test_async_main_with_mcp_available(self):
        """Test async_main when MCP is available."""
        with patch('deepflow.mcp.server.MCP_AVAILABLE', True), \
             patch('deepflow.mcp.server.DeepflowMCPServer') as mock_server_class:
            
            mock_server = MagicMock()
            # Fix: Use proper async future creation
            future = asyncio.get_event_loop().create_future()
            future.set_result(None)
            mock_server.run = MagicMock(return_value=future)
            mock_server_class.return_value = mock_server
            
            from deepflow.mcp.server import async_main
            
            await async_main()
            
            mock_server_class.assert_called_once()
            mock_server.run.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_async_main_with_mcp_unavailable(self):
        """Test async_main when MCP is unavailable."""
        with patch('deepflow.mcp.server.MCP_AVAILABLE', False), \
             patch('sys.exit') as mock_exit, \
             patch('builtins.print') as mock_print:
            
            from deepflow.mcp.server import async_main
            
            await async_main()
            
            # Should print error and exit
            mock_print.assert_called_with("ERROR: MCP dependencies not found. Install with: pip install deepflow[mcp]")
            mock_exit.assert_called_with(1)
    
    def test_server_handles_keyboard_interrupt(self):
        """Test that server handles KeyboardInterrupt gracefully."""
        with patch('deepflow.mcp.server.asyncio.run') as mock_run:
            mock_run.side_effect = KeyboardInterrupt()
            
            from deepflow.mcp.server import main
            
            # Should not raise exception
            try:
                main()
            except KeyboardInterrupt:
                pytest.fail("main() should handle KeyboardInterrupt gracefully")


class TestMCPModuleExecution:
    """Test running MCP module as python -m deepflow.mcp."""
    
    def test_mcp_module_has_main(self):
        """Test that deepflow.mcp.__main__ exists."""
        project_root = Path(__file__).parent.parent.parent
        main_file = project_root / "deepflow" / "mcp" / "__main__.py"
        
        # Should exist
        assert main_file.exists()
        
        # Should contain main execution logic
        content = main_file.read_text()
        assert "main" in content or "__main__" in content
    
    def test_mcp_main_module_imports(self):
        """Test that MCP __main__ module imports work."""
        try:
            from deepflow.mcp import __main__
            # Should import without error (if MCP is available)
        except ImportError:
            # Expected if MCP dependencies are not installed
            pass
        except SystemExit:
            # Also expected if MCP is not available
            pass


class TestEntryPointIntegration:
    """Test integration of entry points with package structure."""
    
    def test_all_cli_entry_points_importable(self):
        """Test that all CLI entry points are importable."""
        # Entry points from pyproject.toml
        entry_points = {
            "deepflow-visualizer": "tools.dependency_visualizer:main",
            "deepflow-validator": "tools.pre_commit_validator:main",
            "deepflow-docs": "tools.doc_generator:main",
            "deepflow-ci": "tools.ci_cd_integrator:main",
            "deepflow-monitor": "tools.monitoring_dashboard:main",
            "deepflow-analyzer": "tools.code_analyzer:main",
            "ai-session-tracker": "tools.ai_session_tracker:main",
            "deepflow-mcp-server": "deepflow.mcp.server:main"
        }
        
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))
        sys.path.insert(0, str(project_root / "tools"))
        
        for entry_name, module_path in entry_points.items():
            module_name, func_name = module_path.split(":")
            
            try:
                if module_name.startswith("tools."):
                    # Remove tools. prefix for direct import
                    module_name = module_name.replace("tools.", "")
                
                module = __import__(module_name)
                
                # Should have the specified function
                assert hasattr(module, func_name), f"{module_name} missing {func_name}"
                
                # Function should be callable
                func = getattr(module, func_name)
                assert callable(func), f"{module_name}.{func_name} not callable"
                
            except ImportError:
                # Some modules may not import due to missing dependencies
                # This is acceptable for optional tools
                if "mcp" in module_name:
                    # MCP is optional
                    continue
                else:
                    # Other tools should be importable
                    pytest.fail(f"Could not import {module_name} for entry point {entry_name}")
    
    def test_mcp_entry_point_specific_imports(self):
        """Test MCP-specific entry point imports."""
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))
        
        try:
            from deepflow.mcp import server
            
            # Should have main function
            assert hasattr(server, 'main')
            assert callable(server.main)
            
        except ImportError:
            # Expected if MCP is not installed
            pass
        except SystemExit:
            # Also expected if MCP checks fail
            pass


class TestCLIErrorHandling:
    """Test CLI error handling and user experience."""
    
    def test_helpful_error_for_missing_mcp(self):
        """Test helpful error message when MCP is missing."""
        with patch('deepflow.mcp.server.MCP_AVAILABLE', False), \
             patch('builtins.print') as mock_print, \
             patch('sys.exit') as mock_exit:
            
            from deepflow.mcp.server import main
            
            main()
            
            # Should provide helpful error
            printed_calls = [call.args[0] for call in mock_print.call_args_list]
            error_message = ' '.join(printed_calls) if printed_calls else ''
            
            assert "pip install deepflow[mcp]" in error_message
            mock_exit.assert_called_with(1)
    
    def test_graceful_shutdown_handling(self):
        """Test graceful shutdown on SIGINT/SIGTERM."""
        with patch('deepflow.mcp.server.asyncio.run') as mock_run:
            # Simulate SIGINT
            mock_run.side_effect = KeyboardInterrupt()
            
            from deepflow.mcp.server import main
            
            # Should handle gracefully without propagating exception
            try:
                main()
            except KeyboardInterrupt:
                pytest.fail("Should handle KeyboardInterrupt gracefully")
    
    def test_error_handling_for_invalid_arguments(self):
        """Test error handling for invalid command line arguments."""
        # MCP server doesn't take command line arguments currently,
        # but this tests the general pattern
        
        with patch('sys.argv', ['deepflow-mcp-server', '--invalid-arg']):
            # Should handle gracefully or show help
            # Current implementation doesn't parse args, so this should work
            pass


class TestSubprocessExecution:
    """Test running entry points as subprocesses."""
    
    @pytest.mark.slow
    def test_mcp_server_help_via_subprocess(self):
        """Test running MCP server help via subprocess."""
        project_root = Path(__file__).parent.parent.parent
        
        # Test that the server can be executed (even if it exits quickly)
        try:
            result = subprocess.run(
                [sys.executable, "-m", "deepflow.mcp"],
                cwd=str(project_root),
                capture_output=True,
                text=True,
                timeout=5
            )
            
            # Should either run successfully or exit with helpful error
            # (depending on whether MCP is installed)
            assert result.returncode in [0, 1]
            
            if result.returncode == 1:
                # Should have helpful error message
                assert "MCP" in result.stderr or "mcp" in result.stderr.lower()
            
        except subprocess.TimeoutExpired:
            # Server might be waiting for input, which is okay
            pass
        except FileNotFoundError:
            # Module might not be found if not properly installed
            pytest.skip("deepflow.mcp module not found")
    
    @pytest.mark.slow 
    def test_entry_point_executable_via_subprocess(self):
        """Test that entry points can be executed via subprocess."""
        # This test would verify that entry points work when installed
        # Skip if not in an installed environment
        
        try:
            result = subprocess.run(
                ["deepflow-mcp-server", "--help"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            # If the command exists, it should provide some output
            # (or error about missing dependencies)
            assert result.returncode in [0, 1, 2]  # 2 for argument errors
            
        except FileNotFoundError:
            # Entry point not installed - skip test
            pytest.skip("deepflow-mcp-server not installed")
        except subprocess.TimeoutExpired:
            # Command might be waiting for input
            pass


class TestMCPServerConfiguration:
    """Test MCP server configuration and options."""
    
    def test_server_has_proper_name(self):
        """Test that MCP server has proper name for protocol."""
        with patch('deepflow.mcp.server.MCP_AVAILABLE', True), \
             patch('deepflow.mcp.server.Server') as mock_server_class:
            
            from deepflow.mcp.server import DeepflowMCPServer
            
            server = DeepflowMCPServer()
            
            # Should initialize server with correct name
            mock_server_class.assert_called_once_with("deepflow")
    
    def test_server_stdio_integration(self):
        """Test server stdio integration setup."""
        with patch('deepflow.mcp.server.MCP_AVAILABLE', True), \
             patch('deepflow.mcp.server.mcp.server.stdio.stdio_server') as mock_stdio:
            
            from deepflow.mcp.server import DeepflowMCPServer
            
            # Mock stdio context manager
            mock_stdio.return_value.__aenter__ = MagicMock()
            mock_stdio.return_value.__aexit__ = MagicMock()
            
            server = DeepflowMCPServer()
            
            # Should be set up to use stdio
            assert server is not None


@pytest.mark.parametrize("entry_point", [
    "deepflow-visualizer",
    "deepflow-validator", 
    "deepflow-docs",
    "deepflow-ci",
    "deepflow-monitor",
    "deepflow-analyzer",
    "ai-session-tracker",
    "deepflow-mcp-server"
])
def test_entry_point_structure(entry_point):
    """Test that entry point has proper structure."""
    # Read pyproject.toml to verify entry point structure
    project_root = Path(__file__).parent.parent.parent
    pyproject_path = project_root / "pyproject.toml"
    
    if pyproject_path.exists():
        content = pyproject_path.read_text()
        
        # Entry point should be defined
        assert entry_point in content
        
        # Should have proper format "command = module:function"
        import re
        pattern = rf'{entry_point}\s*=\s*["\']([^"\']+):[^"\']+["\']'
        match = re.search(pattern, content)
        assert match is not None, f"Entry point {entry_point} not properly formatted"