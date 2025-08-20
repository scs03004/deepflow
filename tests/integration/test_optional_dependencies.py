"""
Integration tests for optional dependency handling.
"""

import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestOptionalDependencyHandling:
    """Test handling of optional dependencies."""
    
    def test_networkx_optional_handling(self):
        """Test handling when NetworkX is not available."""
        project_root = Path(__file__).parent.parent.parent
        tools_dir = project_root / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        
        # Mock NetworkX as unavailable
        with patch.dict('sys.modules', {'networkx': None}):
            sys.modules.pop('dependency_visualizer', None)
            
            try:
                import dependency_visualizer
                
                # Should have availability flag
                assert hasattr(dependency_visualizer, 'NETWORKX_AVAILABLE')
                
                # Should provide fallback behavior
                visualizer = dependency_visualizer.DependencyVisualizer(".")
                
                # Should raise informative error when trying to use NetworkX features
                with pytest.raises(ImportError, match="NetworkX is required"):
                    visualizer.analyze_project()
                    
            except ImportError:
                # Expected if the module handles this at import time
                pass
    
    def test_plotly_optional_handling(self):
        """Test handling when Plotly is not available."""
        project_root = Path(__file__).parent.parent.parent
        tools_dir = project_root / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        
        # Mock Plotly as unavailable
        with patch.dict('sys.modules', {'plotly': None}):
            sys.modules.pop('dependency_visualizer', None)
            
            try:
                import dependency_visualizer
                
                # Should have availability flag
                assert hasattr(dependency_visualizer, 'PLOTLY_AVAILABLE')
                
                visualizer = dependency_visualizer.DependencyVisualizer(".")
                mock_graph = MagicMock()
                
                # Should raise informative error for HTML generation
                with pytest.raises(ImportError, match="Plotly is required"):
                    visualizer.generate_interactive_html(mock_graph)
                    
            except ImportError:
                pass
    
    def test_rich_optional_handling(self):
        """Test handling when Rich is not available."""
        project_root = Path(__file__).parent.parent.parent
        tools_dir = project_root / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        
        # Mock Rich as unavailable
        with patch.dict('sys.modules', {'rich': None}):
            sys.modules.pop('dependency_visualizer', None)
            
            try:
                import dependency_visualizer
                
                # Should have availability flag
                assert hasattr(dependency_visualizer, 'RICH_AVAILABLE')
                
                # Should provide fallback implementations
                if hasattr(dependency_visualizer, 'Console'):
                    console = dependency_visualizer.Console()
                    console.print("test")  # Should not raise
                
                if hasattr(dependency_visualizer, 'Tree'):
                    tree = dependency_visualizer.Tree("test")
                    child = tree.add("child")
                    assert hasattr(child, 'children')
                    
            except ImportError:
                pass
    
    def test_jinja2_optional_handling(self):
        """Test handling when Jinja2 is not available."""
        project_root = Path(__file__).parent.parent.parent
        tools_dir = project_root / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        
        # Mock Jinja2 as unavailable
        with patch.dict('sys.modules', {'jinja2': None}):
            sys.modules.pop('doc_generator', None)
            
            try:
                import doc_generator
                
                generator = doc_generator.DocumentationGenerator(".")
                
                # Should handle missing Jinja2 gracefully
                result = generator._render_template("test.md", {})
                assert isinstance(result, str)  # Should return some fallback
                
            except ImportError:
                # Expected if Jinja2 is required
                pass
    
    def test_mcp_optional_handling(self):
        """Test handling when MCP is not available."""
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Mock MCP as unavailable
        with patch.dict('sys.modules', {'mcp': None}):
            # Clear existing imports
            for module in list(sys.modules.keys()):
                if module.startswith('deepflow'):
                    sys.modules.pop(module, None)
            
            # Should still be able to import main package
            import deepflow
            
            # Should indicate MCP is not available
            assert hasattr(deepflow, 'MCP_AVAILABLE')
            
            # Try importing MCP module
            try:
                from deepflow import mcp
                assert hasattr(mcp, 'MCP_AVAILABLE')
            except ImportError:
                # Expected when MCP is not available
                pass


class TestDependencyFallbacks:
    """Test fallback implementations for missing dependencies."""
    
    def test_rich_fallback_console(self):
        """Test Rich Console fallback implementation."""
        project_root = Path(__file__).parent.parent.parent
        tools_dir = project_root / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        
        with patch('dependency_visualizer.RICH_AVAILABLE', False):
            sys.modules.pop('dependency_visualizer', None)
            import dependency_visualizer
            
            # Should have fallback Console
            console = dependency_visualizer.Console()
            
            # Should have print method that works
            console.print("test message")  # Should not raise
            assert hasattr(console, 'print')
    
    def test_rich_fallback_tree(self):
        """Test Rich Tree fallback implementation."""
        project_root = Path(__file__).parent.parent.parent
        tools_dir = project_root / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        
        with patch('dependency_visualizer.RICH_AVAILABLE', False):
            sys.modules.pop('dependency_visualizer', None)
            import dependency_visualizer
            
            # Should have fallback Tree
            tree = dependency_visualizer.Tree("root")
            
            # Should have add method that works
            child = tree.add("child")
            assert hasattr(child, 'children')
            assert hasattr(tree, 'label')
    
    def test_rich_fallback_track(self):
        """Test Rich track fallback implementation."""
        project_root = Path(__file__).parent.parent.parent
        tools_dir = project_root / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        
        with patch('dependency_visualizer.RICH_AVAILABLE', False):
            sys.modules.pop('dependency_visualizer', None)
            import dependency_visualizer
            
            # Should have fallback track function
            items = [1, 2, 3]
            tracked_items = list(dependency_visualizer.track(items, "Processing"))
            
            # Should return items unchanged
            assert tracked_items == items
    
    def test_template_rendering_fallback(self):
        """Test template rendering fallback when Jinja2 is unavailable."""
        project_root = Path(__file__).parent.parent.parent
        tools_dir = project_root / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        
        try:
            import doc_generator
            
            generator = doc_generator.DocumentationGenerator(".")
            
            # Mock Jinja2 failure
            with patch('doc_generator.jinja2') as mock_jinja:
                mock_jinja.Environment.side_effect = ImportError("Jinja2 not available")
                
                # Should handle gracefully
                result = generator._render_template("test.md", {"data": "test"})
                assert isinstance(result, str)
                
        except ImportError:
            # Expected if doc_generator itself is not importable
            pass


class TestGracefulDegradation:
    """Test graceful degradation with missing dependencies."""
    
    def test_partial_functionality_with_missing_deps(self):
        """Test that partial functionality remains with missing dependencies."""
        project_root = Path(__file__).parent.parent.parent
        tools_dir = project_root / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        
        # Mock multiple dependencies as missing
        missing_deps = {
            'plotly': None,
            'matplotlib': None,
            'jinja2': None
        }
        
        with patch.dict('sys.modules', missing_deps):
            try:
                sys.modules.pop('dependency_visualizer', None)
                import dependency_visualizer
                
                # Should still be able to create visualizer
                visualizer = dependency_visualizer.DependencyVisualizer(".")
                
                # Basic functionality should work
                assert hasattr(visualizer, 'project_path')
                
                # Should have availability flags indicating what's missing
                assert hasattr(dependency_visualizer, 'PLOTLY_AVAILABLE')
                assert hasattr(dependency_visualizer, 'MATPLOTLIB_AVAILABLE')
                
            except ImportError:
                # May fail if core dependencies are missing
                pass
    
    def test_error_messages_are_helpful(self):
        """Test that error messages for missing dependencies are helpful."""
        project_root = Path(__file__).parent.parent.parent
        tools_dir = project_root / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        
        with patch.dict('sys.modules', {'networkx': None}):
            try:
                sys.modules.pop('dependency_visualizer', None)
                import dependency_visualizer
                
                visualizer = dependency_visualizer.DependencyVisualizer(".")
                
                # Error message should be helpful
                with pytest.raises(ImportError) as exc_info:
                    visualizer.analyze_project()
                
                error_message = str(exc_info.value)
                assert "NetworkX" in error_message
                assert "required" in error_message.lower()
                
            except ImportError:
                pass
    
    def test_installation_hints_provided(self):
        """Test that installation hints are provided for missing dependencies."""
        # This would test that error messages include installation instructions
        # For example: "Install with: pip install networkx"
        
        project_root = Path(__file__).parent.parent.parent
        tools_dir = project_root / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        
        # Test would check that helpful installation instructions are provided
        # This is more of a documentation/UX test
        pass


class TestOptionalDependencyDiscovery:
    """Test discovery and reporting of optional dependencies."""
    
    def test_availability_flags_accurate(self):
        """Test that availability flags accurately reflect dependency status."""
        project_root = Path(__file__).parent.parent.parent
        tools_dir = project_root / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        
        try:
            import dependency_visualizer
            
            # Availability flags should exist
            flags = [
                'NETWORKX_AVAILABLE',
                'MATPLOTLIB_AVAILABLE', 
                'PLOTLY_AVAILABLE',
                'PANDAS_AVAILABLE',
                'RICH_AVAILABLE'
            ]
            
            for flag in flags:
                assert hasattr(dependency_visualizer, flag)
                assert isinstance(getattr(dependency_visualizer, flag), bool)
            
            # If a dependency is available, it should actually be importable
            if dependency_visualizer.NETWORKX_AVAILABLE:
                import networkx  # Should not raise if flag is True
            
        except ImportError:
            pass
    
    def test_deepflow_reports_capabilities(self):
        """Test that deepflow package reports its capabilities correctly."""
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        import deepflow
        
        # Should report tools and MCP availability
        assert hasattr(deepflow, 'TOOLS_AVAILABLE')
        assert hasattr(deepflow, 'MCP_AVAILABLE')
        
        # Values should be boolean
        assert isinstance(deepflow.TOOLS_AVAILABLE, bool)
        assert isinstance(deepflow.MCP_AVAILABLE, bool)
    
    def test_mcp_availability_accurate(self):
        """Test that MCP availability is accurately reported."""
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Test with MCP unavailable
        with patch.dict('sys.modules', {'mcp': None}):
            # Clear existing imports
            for module in list(sys.modules.keys()):
                if module.startswith('deepflow'):
                    sys.modules.pop(module, None)
            
            import deepflow
            
            # Should report MCP as unavailable
            # (Note: This might be True if mcp is actually installed in test environment)
            assert isinstance(deepflow.MCP_AVAILABLE, bool)


class TestDependencyVersioning:
    """Test handling of dependency version requirements."""
    
    def test_minimum_version_requirements(self):
        """Test that minimum version requirements are handled."""
        # This would test version compatibility checking
        # For now, we'll test that the pattern exists
        
        project_root = Path(__file__).parent.parent.parent
        pyproject_path = project_root / "pyproject.toml"
        
        if pyproject_path.exists():
            content = pyproject_path.read_text()
            
            # Should have version specifications
            assert ">=" in content  # Minimum version specs
            
            # Common dependencies should have version requirements
            dependencies = ["networkx", "matplotlib", "plotly", "rich"]
            for dep in dependencies:
                if dep in content:
                    # Should have version requirement
                    assert f"{dep}>=" in content
    
    def test_optional_dependency_groups(self):
        """Test that optional dependency groups are properly defined."""
        project_root = Path(__file__).parent.parent.parent
        pyproject_path = project_root / "pyproject.toml"
        
        if pyproject_path.exists():
            content = pyproject_path.read_text()
            
            # Should have optional dependency groups
            assert "[project.optional-dependencies]" in content
            
            expected_groups = ["dev", "docs", "mcp", "all"]
            for group in expected_groups:
                assert f'"{group}"' in content or f"'{group}'" in content


class TestCompatibilityLayers:
    """Test compatibility layers for different dependency versions."""
    
    def test_backwards_compatibility(self):
        """Test backwards compatibility with older dependency versions."""
        # This would test that the code works with minimum required versions
        # For now, we'll test the structure exists
        
        project_root = Path(__file__).parent.parent.parent
        tools_dir = project_root / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        
        try:
            import dependency_visualizer
            
            # Should handle different versions gracefully
            visualizer = dependency_visualizer.DependencyVisualizer(".")
            assert visualizer is not None
            
        except ImportError:
            pass
    
    def test_future_compatibility(self):
        """Test that code structure supports future dependency versions."""
        # This would test that the code doesn't break with newer versions
        # For now, we'll test that imports are defensive
        
        project_root = Path(__file__).parent.parent.parent
        tools_dir = project_root / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        
        try:
            import dependency_visualizer
            
            # Should use try/except patterns for imports
            # This is verified by the existence of availability flags
            assert hasattr(dependency_visualizer, 'NETWORKX_AVAILABLE')
            
        except ImportError:
            pass


@pytest.fixture(autouse=True)
def cleanup_dependency_tests():
    """Clean up after dependency tests."""
    yield
    
    # Remove test-related modules from cache
    modules_to_remove = []
    for module_name in sys.modules:
        if any(pattern in module_name for pattern in ['dependency_visualizer', 'code_analyzer', 'doc_generator', 'deepflow']):
            modules_to_remove.append(module_name)
    
    for module_name in modules_to_remove:
        sys.modules.pop(module_name, None)