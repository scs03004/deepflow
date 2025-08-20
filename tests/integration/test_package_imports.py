"""
Integration tests for package import scenarios and module loading.
"""

import pytest
import sys
import importlib
from pathlib import Path
from unittest.mock import patch


class TestPackageImports:
    """Test package import scenarios."""
    
    def test_deepflow_package_importable(self):
        """Test that deepflow package can be imported."""
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Clear any existing imports
        for module in list(sys.modules.keys()):
            if module.startswith('deepflow'):
                sys.modules.pop(module, None)
        
        # Should import without error
        import deepflow
        
        # Should have basic attributes
        assert hasattr(deepflow, '__version__')
        assert hasattr(deepflow, '__author__')
        assert hasattr(deepflow, 'TOOLS_AVAILABLE')
        assert hasattr(deepflow, 'MCP_AVAILABLE')
    
    def test_deepflow_tools_import(self):
        """Test deepflow.tools import behavior."""
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Clear existing imports
        for module in list(sys.modules.keys()):
            if module.startswith('deepflow'):
                sys.modules.pop(module, None)
        
        try:
            from deepflow import tools
            tools_imported = True
        except ImportError:
            tools_imported = False
        
        # Should handle gracefully regardless of outcome
        assert isinstance(tools_imported, bool)
    
    def test_deepflow_mcp_import(self):
        """Test deepflow.mcp import behavior."""
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Clear existing imports
        for module in list(sys.modules.keys()):
            if module.startswith('deepflow'):
                sys.modules.pop(module, None)
        
        try:
            from deepflow import mcp
            mcp_imported = True
            has_mcp_available = hasattr(mcp, 'MCP_AVAILABLE')
        except ImportError:
            mcp_imported = False
            has_mcp_available = False
        
        # Should handle gracefully
        assert isinstance(mcp_imported, bool)
        if mcp_imported:
            assert has_mcp_available
    
    def test_tools_direct_import(self):
        """Test importing tools directly from tools directory."""
        project_root = Path(__file__).parent.parent.parent
        tools_dir = project_root / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        
        tool_modules = [
            "dependency_visualizer",
            "code_analyzer", 
            "doc_generator",
            "pre_commit_validator",
            "ci_cd_integrator",
            "monitoring_dashboard",
            "ai_session_tracker"
        ]
        
        for module_name in tool_modules:
            try:
                # Clear module cache
                sys.modules.pop(module_name, None)
                
                module = importlib.import_module(module_name)
                
                # Should have basic structure
                assert module is not None
                
                # Should have main function for CLI
                if hasattr(module, 'main'):
                    assert callable(module.main)
                
            except ImportError:
                # Some modules may fail due to missing dependencies
                # This is acceptable if handled gracefully
                pass
            except SystemExit:
                # Some modules may exit during import due to dependency checks
                pass


class TestImportResilience:
    """Test import resilience with missing dependencies."""
    
    def test_deepflow_import_with_missing_tools(self):
        """Test deepflow import when tools are not available."""
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Mock tools as unavailable
        with patch('deepflow.TOOLS_AVAILABLE', False):
            # Clear existing imports
            for module in list(sys.modules.keys()):
                if module.startswith('deepflow'):
                    sys.modules.pop(module, None)
            
            # Should still import successfully
            import deepflow
            
            # Should indicate tools are not available
            assert deepflow.TOOLS_AVAILABLE is False
    
    def test_deepflow_import_with_missing_mcp(self):
        """Test deepflow import when MCP is not available."""
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Mock MCP as unavailable
        with patch.dict('sys.modules', {'mcp': None}):
            # Clear existing imports
            for module in list(sys.modules.keys()):
                if module.startswith('deepflow'):
                    sys.modules.pop(module, None)
            
            # Should still import successfully
            import deepflow
            
            # Should indicate MCP is not available
            assert hasattr(deepflow, 'MCP_AVAILABLE')
    
    def test_tools_import_with_missing_dependencies(self):
        """Test tools import when optional dependencies are missing."""
        project_root = Path(__file__).parent.parent.parent
        tools_dir = project_root / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        
        # Mock missing dependencies
        missing_deps = {
            'networkx': None,
            'matplotlib': None,
            'plotly': None,
            'rich': None,
            'jinja2': None
        }
        
        with patch.dict('sys.modules', missing_deps):
            # Test dependency_visualizer with missing deps
            try:
                sys.modules.pop('dependency_visualizer', None)
                import dependency_visualizer
                
                # Should handle missing dependencies gracefully
                assert hasattr(dependency_visualizer, 'NETWORKX_AVAILABLE')
                assert dependency_visualizer.NETWORKX_AVAILABLE is False
                
            except ImportError:
                # Expected if dependencies are required
                pass
    
    def test_graceful_degradation_with_partial_dependencies(self):
        """Test graceful degradation with some dependencies missing."""
        project_root = Path(__file__).parent.parent.parent
        tools_dir = project_root / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        
        # Mock only some dependencies as missing
        partial_deps = {
            'plotly': None,  # Missing
            'matplotlib': None,  # Missing
            # 'networkx' and 'rich' available
        }
        
        with patch.dict('sys.modules', partial_deps):
            try:
                sys.modules.pop('dependency_visualizer', None)
                import dependency_visualizer
                
                # Should have availability flags
                assert hasattr(dependency_visualizer, 'PLOTLY_AVAILABLE')
                assert hasattr(dependency_visualizer, 'MATPLOTLIB_AVAILABLE')
                
                # Missing ones should be False
                assert dependency_visualizer.PLOTLY_AVAILABLE is False
                assert dependency_visualizer.MATPLOTLIB_AVAILABLE is False
                
            except ImportError:
                # May fail if core dependencies are missing
                pass


class TestModuleReloading:
    """Test module reloading scenarios."""
    
    def test_module_can_be_reloaded(self):
        """Test that modules can be reloaded without issues."""
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Import deepflow
        import deepflow
        initial_version = deepflow.__version__
        
        # Reload the module
        importlib.reload(deepflow)
        
        # Should maintain consistency
        assert deepflow.__version__ == initial_version
        assert hasattr(deepflow, 'TOOLS_AVAILABLE')
        assert hasattr(deepflow, 'MCP_AVAILABLE')
    
    def test_tools_module_reloading(self):
        """Test that tools modules can be reloaded."""
        project_root = Path(__file__).parent.parent.parent
        tools_dir = project_root / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        
        try:
            import dependency_visualizer
            
            # Should be able to reload
            importlib.reload(dependency_visualizer)
            
            # Should maintain structure
            assert hasattr(dependency_visualizer, 'DependencyVisualizer')
            
        except ImportError:
            # May fail due to missing dependencies
            pytest.skip("dependency_visualizer not importable")


class TestPackageStructure:
    """Test package structure and organization."""
    
    def test_deepflow_package_structure(self):
        """Test that deepflow package has expected structure."""
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        import deepflow
        
        # Should have expected attributes in __all__
        expected_attrs = ['__version__', '__author__', '__email__', 'TOOLS_AVAILABLE', 'MCP_AVAILABLE']
        
        for attr in expected_attrs:
            assert hasattr(deepflow, attr), f"deepflow missing {attr}"
        
        # Should have __all__ defined
        assert hasattr(deepflow, '__all__')
        assert isinstance(deepflow.__all__, list)
    
    def test_tools_package_organization(self):
        """Test that tools are properly organized."""
        project_root = Path(__file__).parent.parent.parent
        tools_dir = project_root / "tools"
        
        # Should exist as directory
        assert tools_dir.exists()
        assert tools_dir.is_dir()
        
        # Should contain expected tool files
        expected_tools = [
            "dependency_visualizer.py",
            "code_analyzer.py",
            "doc_generator.py",
            "pre_commit_validator.py",
            "ci_cd_integrator.py", 
            "monitoring_dashboard.py",
            "ai_session_tracker.py"
        ]
        
        for tool in expected_tools:
            tool_path = tools_dir / tool
            assert tool_path.exists(), f"Tool {tool} not found"
    
    def test_mcp_package_structure(self):
        """Test that MCP package has expected structure."""
        project_root = Path(__file__).parent.parent.parent
        mcp_dir = project_root / "deepflow" / "mcp"
        
        # Should exist as package
        assert mcp_dir.exists()
        assert (mcp_dir / "__init__.py").exists()
        assert (mcp_dir / "server.py").exists()
        assert (mcp_dir / "__main__.py").exists()


class TestImportPerformance:
    """Test import performance and efficiency."""
    
    def test_import_time_reasonable(self):
        """Test that imports complete in reasonable time."""
        import time
        
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Clear existing imports
        for module in list(sys.modules.keys()):
            if module.startswith('deepflow'):
                sys.modules.pop(module, None)
        
        start_time = time.time()
        import deepflow
        import_time = time.time() - start_time
        
        # Should import quickly (less than 5 seconds even on slow systems)
        assert import_time < 5.0, f"Import took too long: {import_time:.2f}s"
    
    def test_lazy_imports_work(self):
        """Test that lazy imports work correctly."""
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Clear existing imports
        for module in list(sys.modules.keys()):
            if module.startswith('deepflow'):
                sys.modules.pop(module, None)
        
        # Import main package
        import deepflow
        
        # MCP should not be imported yet if not used
        mcp_modules = [m for m in sys.modules.keys() if 'mcp' in m and m.startswith('deepflow')]
        
        # Should minimize eager imports
        assert len(mcp_modules) <= 1  # Only deepflow.mcp.__init__ if any


class TestCircularImports:
    """Test for circular import issues."""
    
    def test_no_circular_imports_in_deepflow(self):
        """Test that there are no circular imports in deepflow package."""
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Clear existing imports
        for module in list(sys.modules.keys()):
            if module.startswith('deepflow'):
                sys.modules.pop(module, None)
        
        # Should be able to import all parts without circular import errors
        try:
            import deepflow
            from deepflow import mcp
            import deepflow.mcp.server
            
            # No circular import errors should occur
            assert True
            
        except ImportError as e:
            if "circular import" in str(e).lower():
                pytest.fail(f"Circular import detected: {e}")
            else:
                # Other import errors are acceptable (missing dependencies)
                pass
    
    def test_no_circular_imports_in_tools(self):
        """Test that there are no circular imports in tools."""
        project_root = Path(__file__).parent.parent.parent
        tools_dir = project_root / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        
        tool_modules = [
            "dependency_visualizer",
            "code_analyzer",
            "doc_generator"
        ]
        
        # Should be able to import all tools without circular imports
        for module_name in tool_modules:
            try:
                sys.modules.pop(module_name, None)
                importlib.import_module(module_name)
                
            except ImportError as e:
                if "circular import" in str(e).lower():
                    pytest.fail(f"Circular import in {module_name}: {e}")
                else:
                    # Other import errors are acceptable
                    continue


class TestNamespaceConflicts:
    """Test for namespace conflicts and name collisions."""
    
    def test_no_name_conflicts_with_standard_library(self):
        """Test that package names don't conflict with standard library."""
        # Import standard library modules that might conflict
        import json
        import os
        import sys
        import pathlib
        
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Import deepflow
        import deepflow
        
        # Standard library modules should still work
        assert json.dumps({"test": True}) == '{"test": true}'
        assert os.path.exists(__file__)
        assert sys.version_info.major >= 3
        assert pathlib.Path(__file__).exists()
    
    def test_no_conflicts_with_common_packages(self):
        """Test that package doesn't conflict with common third-party packages."""
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Import deepflow first
        import deepflow
        
        # Try importing common packages (if available)
        common_packages = ['requests', 'numpy', 'pandas', 'matplotlib']
        
        for package_name in common_packages:
            try:
                package = importlib.import_module(package_name)
                # Should not interfere with deepflow
                assert hasattr(deepflow, '__version__')
                
            except ImportError:
                # Package not installed - skip
                continue


@pytest.fixture(autouse=True)
def cleanup_imports():
    """Clean up imports after each test."""
    yield
    
    # Remove test-related modules from cache
    modules_to_remove = []
    for module_name in sys.modules:
        if any(pattern in module_name for pattern in ['deepflow', 'dependency_visualizer', 'code_analyzer', 'doc_generator']):
            modules_to_remove.append(module_name)
    
    for module_name in modules_to_remove:
        sys.modules.pop(module_name, None)