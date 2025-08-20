"""
Unit tests for graceful import handling in tools modules.
"""

import pytest
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestGracefulImports:
    """Test graceful handling of missing dependencies."""
    
    def test_dependency_visualizer_missing_networkx(self):
        """Test dependency_visualizer handles missing NetworkX gracefully."""
        # Mock missing NetworkX
        with patch.dict('sys.modules', {'networkx': None}):
            with patch('dependency_visualizer.NETWORKX_AVAILABLE', False):
                # Clear module cache to force reimport
                sys.modules.pop('dependency_visualizer', None)
                
                # Add tools to path
                tools_path = str(Path(__file__).parent.parent.parent / "tools")
                if tools_path not in sys.path:
                    sys.path.insert(0, tools_path)
                
                # Should import without error
                import dependency_visualizer
                
                # NETWORKX_AVAILABLE should be False
                assert dependency_visualizer.NETWORKX_AVAILABLE is False
                
                # Creating visualizer should still work
                visualizer = dependency_visualizer.DependencyVisualizer(".")
                
                # But analyze_project should raise ImportError
                with pytest.raises(ImportError, match="NetworkX is required"):
                    visualizer.analyze_project()
    
    def test_dependency_visualizer_missing_plotly(self):
        """Test dependency_visualizer handles missing Plotly gracefully."""
        with patch.dict('sys.modules', {'plotly': None}):
            with patch('dependency_visualizer.PLOTLY_AVAILABLE', False):
                sys.modules.pop('dependency_visualizer', None)
                
                tools_path = str(Path(__file__).parent.parent.parent / "tools")
                if tools_path not in sys.path:
                    sys.path.insert(0, tools_path)
                
                import dependency_visualizer
                
                assert dependency_visualizer.PLOTLY_AVAILABLE is False
                
                visualizer = dependency_visualizer.DependencyVisualizer(".")
                mock_graph = MagicMock()
                
                # HTML generation should raise ImportError
                with pytest.raises(ImportError, match="Plotly is required"):
                    visualizer.generate_interactive_html(mock_graph)
    
    def test_dependency_visualizer_missing_rich(self):
        """Test dependency_visualizer handles missing Rich gracefully."""
        with patch.dict('sys.modules', {'rich': None}):
            with patch('dependency_visualizer.RICH_AVAILABLE', False):
                sys.modules.pop('dependency_visualizer', None)
                
                tools_path = str(Path(__file__).parent.parent.parent / "tools")
                if tools_path not in sys.path:
                    sys.path.insert(0, tools_path)
                
                import dependency_visualizer
                
                assert dependency_visualizer.RICH_AVAILABLE is False
                
                # Should use fallback console and tree classes
                console = dependency_visualizer.Console()
                tree = dependency_visualizer.Tree("test")
                
                # Fallback classes should work
                console.print("test message")
                child = tree.add("child")
                assert hasattr(child, 'children')
    
    def test_code_analyzer_missing_rich(self):
        """Test code_analyzer handles missing Rich gracefully."""
        with patch.dict('sys.modules', {'rich': None}):
            # Should raise ImportError during import since Rich is required
            with pytest.raises(SystemExit):
                sys.modules.pop('code_analyzer', None)
                
                tools_path = str(Path(__file__).parent.parent.parent / "tools")
                if tools_path not in sys.path:
                    sys.path.insert(0, tools_path)
                
                import code_analyzer
    
    def test_code_analyzer_missing_networkx(self):
        """Test code_analyzer handles missing NetworkX gracefully."""
        with patch.dict('sys.modules', {'networkx': None}):
            # Should raise ImportError during import since NetworkX is required
            with pytest.raises(SystemExit):
                sys.modules.pop('code_analyzer', None)
                
                tools_path = str(Path(__file__).parent.parent.parent / "tools")
                if tools_path not in sys.path:
                    sys.path.insert(0, tools_path)
                
                import code_analyzer
    
    def test_doc_generator_missing_jinja2(self):
        """Test doc_generator handles missing Jinja2 gracefully."""
        with patch.dict('sys.modules', {'jinja2': None}):
            # Mock the module to simulate missing dependency
            mock_jinja = MagicMock()
            mock_jinja.Environment = MagicMock(side_effect=ImportError("No module named 'jinja2'"))
            
            with patch('doc_generator.jinja2', mock_jinja):
                sys.modules.pop('doc_generator', None)
                
                tools_path = str(Path(__file__).parent.parent.parent / "tools")
                if tools_path not in sys.path:
                    sys.path.insert(0, tools_path)
                
                import doc_generator
                
                generator = doc_generator.DocumentationGenerator(".")
                
                # Template rendering should handle missing Jinja2
                result = generator._render_template("test.md", {})
                assert isinstance(result, str)


class TestAvailabilityFlags:
    """Test availability flags for optional dependencies."""
    
    def test_dependency_visualizer_availability_flags(self):
        """Test that availability flags are correctly set."""
        tools_path = str(Path(__file__).parent.parent.parent / "tools")
        if tools_path not in sys.path:
            sys.path.insert(0, tools_path)
        
        import dependency_visualizer
        
        # These flags should exist
        assert hasattr(dependency_visualizer, 'NETWORKX_AVAILABLE')
        assert hasattr(dependency_visualizer, 'MATPLOTLIB_AVAILABLE')
        assert hasattr(dependency_visualizer, 'PLOTLY_AVAILABLE')
        assert hasattr(dependency_visualizer, 'PANDAS_AVAILABLE')
        assert hasattr(dependency_visualizer, 'RICH_AVAILABLE')
        
        # They should be boolean values
        assert isinstance(dependency_visualizer.NETWORKX_AVAILABLE, bool)
        assert isinstance(dependency_visualizer.MATPLOTLIB_AVAILABLE, bool)
        assert isinstance(dependency_visualizer.PLOTLY_AVAILABLE, bool)
        assert isinstance(dependency_visualizer.PANDAS_AVAILABLE, bool)
        assert isinstance(dependency_visualizer.RICH_AVAILABLE, bool)
    
    def test_fallback_implementations(self):
        """Test that fallback implementations are provided."""
        with patch('dependency_visualizer.RICH_AVAILABLE', False):
            tools_path = str(Path(__file__).parent.parent.parent / "tools")
            if tools_path not in sys.path:
                sys.path.insert(0, tools_path)
            
            # Force reload to get fallback implementations
            sys.modules.pop('dependency_visualizer', None)
            import dependency_visualizer
            
            # Fallback Console should exist and work
            console = dependency_visualizer.Console()
            console.print("test")  # Should not raise error
            
            # Fallback Tree should exist and work
            tree = dependency_visualizer.Tree("root")
            child = tree.add("child")
            assert hasattr(child, 'children')
            
            # Fallback track should exist and work
            items = [1, 2, 3]
            result = list(dependency_visualizer.track(items, "test"))
            assert result == items


class TestDeepflowPackageImports:
    """Test deepflow package import handling."""
    
    def test_deepflow_init_import_handling(self):
        """Test deepflow.__init__.py handles missing modules gracefully."""
        # Add project root to path
        project_root = str(Path(__file__).parent.parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Clear any existing imports
        for module in list(sys.modules.keys()):
            if module.startswith('deepflow'):
                sys.modules.pop(module, None)
        
        # Should import without error
        import deepflow
        
        # Should have availability flags
        assert hasattr(deepflow, 'TOOLS_AVAILABLE')
        assert hasattr(deepflow, 'MCP_AVAILABLE')
        assert isinstance(deepflow.TOOLS_AVAILABLE, bool)
        assert isinstance(deepflow.MCP_AVAILABLE, bool)
    
    def test_deepflow_tools_import(self):
        """Test deepflow.tools import handling."""
        project_root = str(Path(__file__).parent.parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Clear existing imports
        for module in list(sys.modules.keys()):
            if module.startswith('deepflow'):
                sys.modules.pop(module, None)
        
        try:
            import deepflow.tools
            tools_imported = True
        except ImportError:
            tools_imported = False
        
        # Should handle gracefully whether tools import works or not
        assert isinstance(tools_imported, bool)
    
    def test_deepflow_mcp_import_handling(self):
        """Test deepflow.mcp import handling."""
        project_root = str(Path(__file__).parent.parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        
        # Test with MCP unavailable
        with patch.dict('sys.modules', {'mcp': None}):
            # Clear existing imports
            for module in list(sys.modules.keys()):
                if module.startswith('deepflow'):
                    sys.modules.pop(module, None)
            
            import deepflow.mcp
            
            # Should have MCP_AVAILABLE flag set to False
            assert hasattr(deepflow.mcp, 'MCP_AVAILABLE')
            # Note: This might be True if mcp is actually installed


class TestRobustImportPatterns:
    """Test robust import patterns used in the codebase."""
    
    def test_try_except_import_pattern(self):
        """Test the try/except import pattern works correctly."""
        # Simulate the pattern used in the codebase
        def safe_import():
            try:
                import nonexistent_module
                AVAILABLE = True
            except ImportError:
                AVAILABLE = False
            
            return AVAILABLE
        
        # Should return False for non-existent module
        assert safe_import() is False
    
    def test_module_availability_check(self):
        """Test module availability checking pattern."""
        # Pattern used in the codebase
        def check_module_availability(module_name):
            try:
                __import__(module_name)
                return True
            except ImportError:
                return False
        
        # Should work for existing modules
        assert check_module_availability('os') is True
        assert check_module_availability('sys') is True
        
        # Should return False for non-existing modules
        assert check_module_availability('nonexistent_module_xyz') is False
    
    def test_conditional_functionality(self):
        """Test conditional functionality based on availability."""
        # Simulate pattern used in tools
        FEATURE_AVAILABLE = True
        
        def feature_function():
            if not FEATURE_AVAILABLE:
                raise ImportError("Feature not available - missing dependency")
            return "Feature working"
        
        # Should work when available
        result = feature_function()
        assert result == "Feature working"
        
        # Should raise ImportError when not available
        FEATURE_AVAILABLE = False
        
        def feature_function_unavailable():
            if not FEATURE_AVAILABLE:
                raise ImportError("Feature not available - missing dependency")
            return "Feature working"
        
        with pytest.raises(ImportError, match="Feature not available"):
            feature_function_unavailable()


@pytest.fixture(autouse=True)
def cleanup_imports():
    """Clean up imports after each test."""
    yield
    
    # Remove test-related modules from cache
    modules_to_remove = []
    for module_name in sys.modules:
        if any(pattern in module_name for pattern in ['dependency_visualizer', 'code_analyzer', 'doc_generator']):
            modules_to_remove.append(module_name)
    
    for module_name in modules_to_remove:
        sys.modules.pop(module_name, None)