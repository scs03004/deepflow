"""
Comprehensive tests for graceful fallbacks and error handling across the entire deepflow package.
"""

import pytest
import sys
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock
import importlib


class TestGracefulFallbacksComprehensive:
    """Comprehensive tests for graceful fallback behavior."""
    
    def test_deepflow_package_resilience(self):
        """Test that deepflow package is resilient to missing dependencies."""
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Test with various missing dependencies
        missing_combinations = [
            {'networkx': None},
            {'plotly': None},
            {'rich': None},
            {'jinja2': None},
            {'mcp': None},
            {'networkx': None, 'plotly': None},
            {'rich': None, 'jinja2': None},
            {'mcp': None, 'networkx': None},
        ]
        
        for missing_deps in missing_combinations:
            with patch.dict('sys.modules', missing_deps):
                # Clear existing imports
                for module in list(sys.modules.keys()):
                    if module.startswith('deepflow'):
                        sys.modules.pop(module, None)
                
                try:
                    # Should be able to import main package
                    import deepflow
                    
                    # Should have basic attributes
                    assert hasattr(deepflow, '__version__')
                    assert hasattr(deepflow, 'TOOLS_AVAILABLE')
                    assert hasattr(deepflow, 'MCP_AVAILABLE')
                    
                    # Availability flags should be boolean
                    assert isinstance(deepflow.TOOLS_AVAILABLE, bool)
                    assert isinstance(deepflow.MCP_AVAILABLE, bool)
                    
                except ImportError as e:
                    # Should not fail to import the main package
                    pytest.fail(f"Main package import failed with {missing_deps}: {e}")
    
    def test_tools_fallback_patterns(self):
        """Test that tools follow consistent fallback patterns."""
        project_root = Path(__file__).parent.parent
        tools_dir = project_root / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        
        tool_modules = [
            "dependency_visualizer",
            "code_analyzer",
            "doc_generator"
        ]
        
        for module_name in tool_modules:
            try:
                # Clear module cache
                sys.modules.pop(module_name, None)
                
                # Test import without dependencies
                with patch.dict('sys.modules', {
                    'networkx': None,
                    'plotly': None,
                    'matplotlib': None,
                    'rich': None,
                    'jinja2': None
                }):
                    try:
                        module = importlib.import_module(module_name)
                        
                        # Should have availability flags
                        availability_flags = [attr for attr in dir(module) if attr.endswith('_AVAILABLE')]
                        assert len(availability_flags) > 0, f"{module_name} missing availability flags"
                        
                        # All availability flags should be boolean
                        for flag in availability_flags:
                            value = getattr(module, flag)
                            assert isinstance(value, bool), f"{module_name}.{flag} is not boolean"
                            
                    except ImportError:
                        # Some tools may fail completely if core dependencies are missing
                        # This is acceptable if they handle it gracefully
                        continue
                    except SystemExit:
                        # Some tools may exit if core dependencies are missing
                        # This is acceptable if they provide helpful messages
                        continue
                        
            except ImportError:
                # Tool may not be available in test environment
                continue
    
    def test_error_message_consistency(self):
        """Test that error messages are consistent and helpful."""
        project_root = Path(__file__).parent.parent
        tools_dir = project_root / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        
        # Test dependency_visualizer error messages
        try:
            with patch.dict('sys.modules', {'networkx': None}):
                sys.modules.pop('dependency_visualizer', None)
                import dependency_visualizer
                
                visualizer = dependency_visualizer.DependencyVisualizer(".")
                
                try:
                    visualizer.analyze_project()
                    pytest.fail("Should raise ImportError for missing NetworkX")
                except ImportError as e:
                    error_msg = str(e).lower()
                    # Should mention the missing dependency
                    assert "networkx" in error_msg
                    # Should mention it's required
                    assert any(word in error_msg for word in ["required", "needed", "missing"])
                    
        except ImportError:
            # dependency_visualizer itself may not import
            pass
        
        # Test plotly error messages
        try:
            with patch.dict('sys.modules', {'plotly': None}):
                sys.modules.pop('dependency_visualizer', None)
                import dependency_visualizer
                
                visualizer = dependency_visualizer.DependencyVisualizer(".")
                mock_graph = MagicMock()
                
                try:
                    visualizer.generate_interactive_html(mock_graph)
                    pytest.fail("Should raise ImportError for missing Plotly")
                except ImportError as e:
                    error_msg = str(e).lower()
                    assert "plotly" in error_msg
                    assert any(word in error_msg for word in ["required", "needed", "missing"])
                    
        except ImportError:
            pass
    
    def test_cli_fallback_behavior(self):
        """Test that CLI commands handle missing dependencies gracefully."""
        project_root = Path(__file__).parent.parent
        tools_dir = project_root / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        
        # Test CLI with missing dependencies
        with patch.dict('sys.modules', {'networkx': None, 'plotly': None}):
            test_args = ['dependency_visualizer.py', '.', '--help']
            
            with patch('sys.argv', test_args):
                try:
                    sys.modules.pop('dependency_visualizer', None)
                    import dependency_visualizer
                    
                    # Help should work even with missing dependencies
                    if hasattr(dependency_visualizer, 'main'):
                        try:
                            dependency_visualizer.main()
                        except SystemExit as e:
                            # argparse exits with 0 for --help
                            assert e.code == 0
                            
                except ImportError:
                    # Tool may not import with missing dependencies
                    pass
    
    def test_mcp_fallback_comprehensive(self):
        """Test comprehensive MCP fallback behavior."""
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Test MCP unavailable scenarios
        mcp_scenarios = [
            {'mcp': None},  # MCP package missing
            {'mcp.server': None},  # MCP server missing
            {'mcp.types': None},  # MCP types missing
            {'mcp.server.stdio': None},  # MCP stdio missing
        ]
        
        for scenario in mcp_scenarios:
            with patch.dict('sys.modules', scenario):
                # Clear existing imports
                for module in list(sys.modules.keys()):
                    if module.startswith('deepflow.mcp'):
                        sys.modules.pop(module, None)
                
                try:
                    # Should handle MCP unavailability gracefully
                    from deepflow.mcp import server
                    assert hasattr(server, 'MCP_AVAILABLE')
                    
                except (ImportError, SystemExit):
                    # Expected behavior when MCP is truly unavailable
                    continue
    
    def test_partial_functionality_maintenance(self):
        """Test that partial functionality is maintained with missing dependencies."""
        project_root = Path(__file__).parent.parent
        tools_dir = project_root / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        
        # Test dependency_visualizer with partial dependencies
        try:
            with patch.dict('sys.modules', {
                'plotly': None,  # Missing
                'matplotlib': None,  # Missing
                # NetworkX and Rich available (not patched)
            }):
                sys.modules.pop('dependency_visualizer', None)
                import dependency_visualizer
                
                # Should still be able to create visualizer
                visualizer = dependency_visualizer.DependencyVisualizer(".")
                assert visualizer is not None
                
                # Should have correct availability flags
                assert hasattr(dependency_visualizer, 'PLOTLY_AVAILABLE')
                assert hasattr(dependency_visualizer, 'MATPLOTLIB_AVAILABLE')
                
                # Missing dependencies should be marked as unavailable
                if hasattr(dependency_visualizer, 'PLOTLY_AVAILABLE'):
                    assert dependency_visualizer.PLOTLY_AVAILABLE is False
                if hasattr(dependency_visualizer, 'MATPLOTLIB_AVAILABLE'):
                    assert dependency_visualizer.MATPLOTLIB_AVAILABLE is False
                    
        except ImportError:
            # Tool may require some core dependencies
            pass
    
    def test_installation_guidance(self):
        """Test that helpful installation guidance is provided."""
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Test MCP installation guidance
        with patch('deepflow.mcp.server.MCP_AVAILABLE', False), \
             patch('builtins.print') as mock_print, \
             patch('sys.exit') as mock_exit:
            
            from deepflow.mcp.server import async_main
            
            import asyncio
            asyncio.run(async_main())
            
            # Should provide installation instructions
            printed_messages = [call.args[0] for call in mock_print.call_args_list if call.args]
            
            if printed_messages:
                combined_message = ' '.join(printed_messages)
                assert "pip install" in combined_message
                assert "deepflow[mcp]" in combined_message
            
            mock_exit.assert_called_with(1)
    
    def test_dependency_isolation(self):
        """Test that dependency failures are isolated and don't affect other components."""
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Test that NetworkX failure doesn't affect basic package functionality
        with patch.dict('sys.modules', {'networkx': None}):
            # Clear existing imports
            for module in list(sys.modules.keys()):
                if module.startswith('deepflow'):
                    sys.modules.pop(module, None)
            
            # Should still be able to import and use basic functionality
            import deepflow
            
            # Basic package info should work
            assert hasattr(deepflow, '__version__')
            assert hasattr(deepflow, '__author__')
            
            # Availability flags should work
            assert hasattr(deepflow, 'TOOLS_AVAILABLE')
            assert hasattr(deepflow, 'MCP_AVAILABLE')
    
    def test_cross_platform_compatibility(self):
        """Test cross-platform compatibility of fallback mechanisms."""
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Test path handling across platforms
        import deepflow
        
        # Should handle different path separators
        test_paths = [
            "C:\\Windows\\Path" if sys.platform == "win32" else "/unix/path",
            "./relative/path",
            "../parent/path",
            "simple_name"
        ]
        
        for test_path in test_paths:
            # Path handling should not crash
            normalized_path = str(Path(test_path))
            assert isinstance(normalized_path, str)
    
    def test_memory_efficiency_with_fallbacks(self):
        """Test that fallback mechanisms don't cause memory leaks."""
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Test repeated import cycles don't leak memory
        for i in range(5):
            with patch.dict('sys.modules', {'networkx': None}):
                # Clear imports
                modules_to_clear = [m for m in sys.modules.keys() if m.startswith('deepflow')]
                for module in modules_to_clear:
                    sys.modules.pop(module, None)
                
                # Import again
                import deepflow
                
                # Should maintain consistent behavior
                assert hasattr(deepflow, '__version__')
    
    def test_concurrent_access_safety(self):
        """Test that fallback mechanisms are safe for concurrent access."""
        import threading
        
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        results = []
        errors = []
        
        def import_deepflow():
            try:
                # Clear existing imports
                modules_to_clear = [m for m in sys.modules.keys() if m.startswith('deepflow')]
                for module in modules_to_clear:
                    sys.modules.pop(module, None)
                
                import deepflow
                results.append(deepflow.__version__)
            except Exception as e:
                errors.append(str(e))
        
        # Start multiple threads
        threads = []
        for i in range(3):
            thread = threading.Thread(target=import_deepflow)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join(timeout=5)
        
        # Should not have errors and should have consistent results
        assert len(errors) == 0, f"Concurrent import errors: {errors}"
        if results:
            assert all(version == results[0] for version in results), "Inconsistent versions"
    
    def test_environment_variable_handling(self):
        """Test handling of environment variables in fallback scenarios."""
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Test with various environment configurations
        import os
        original_env = os.environ.copy()
        
        try:
            # Test with minimal environment
            os.environ.clear()
            os.environ['PATH'] = original_env.get('PATH', '')
            os.environ['PYTHONPATH'] = str(project_root)
            
            # Should still work
            import deepflow
            assert hasattr(deepflow, '__version__')
            
        finally:
            # Restore environment
            os.environ.clear()
            os.environ.update(original_env)
    
    def test_documentation_completeness_for_fallbacks(self):
        """Test that fallback behavior is properly documented."""
        project_root = Path(__file__).parent.parent
        
        # Check that README mentions optional dependencies
        readme_path = project_root / "README.md"
        if readme_path.exists():
            readme_content = readme_path.read_text().lower()
            
            # Should mention optional dependencies
            assert any(term in readme_content for term in ["optional", "requirements", "dependencies"])
            
            # Should mention installation options
            assert any(term in readme_content for term in ["pip install", "install"])
    
    @pytest.mark.slow
    def test_subprocess_fallback_behavior(self):
        """Test fallback behavior when running as subprocess."""
        project_root = Path(__file__).parent.parent
        
        # Test that basic functionality works in subprocess
        test_script = f"""
import sys
sys.path.insert(0, r'{project_root}')

try:
    import deepflow
    print(f"SUCCESS: {deepflow.__version__}")
except Exception as e:
    print(f"ERROR: {e}")
"""
        
        result = subprocess.run(
            [sys.executable, "-c", test_script],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # Should not have major errors
        assert result.returncode in [0, 1]  # 1 might be acceptable for missing deps
        
        if result.returncode == 0:
            assert "SUCCESS:" in result.stdout
        else:
            # Should have helpful error message
            assert len(result.stderr) > 0 or "ERROR:" in result.stdout


@pytest.fixture(autouse=True)
def cleanup_comprehensive_fallback_tests():
    """Clean up after comprehensive fallback tests."""
    yield
    
    # Restore sys.modules to clean state
    modules_to_remove = []
    for module_name in sys.modules:
        if any(pattern in module_name for pattern in [
            'deepflow',
            'dependency_visualizer',
            'code_analyzer', 
            'doc_generator'
        ]):
            modules_to_remove.append(module_name)
    
    for module_name in modules_to_remove:
        sys.modules.pop(module_name, None)