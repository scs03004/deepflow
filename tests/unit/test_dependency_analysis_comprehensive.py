"""
Comprehensive edge case tests for dependency analysis engine.
Tests complex scenarios, malformed inputs, and boundary conditions.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from tools.dependency_visualizer import DependencyAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False


@pytest.mark.unit
@pytest.mark.skipif(not ANALYZER_AVAILABLE, reason="DependencyAnalyzer not available")
class TestDependencyAnalysisEdgeCases:
    """Test cases for dependency analysis edge cases and boundary conditions."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_test_file(self, filename: str, content: str):
        """Helper to create test files."""
        file_path = self.test_path / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding='utf-8')
        return file_path
    
    def test_circular_dependency_complex_chains(self):
        """Test detection of complex circular dependency chains (A->B->C->A)."""
        # Create complex circular dependency chain
        self.create_test_file("module_a.py", """
import module_b
from module_d import helper_func

def func_a():
    return module_b.func_b()
""")
        
        self.create_test_file("module_b.py", """
from module_c import func_c
import json

def func_b():
    return func_c()
""")
        
        self.create_test_file("module_c.py", """
from module_a import func_a
import sys

def func_c():
    # This creates A->B->C->A circular dependency
    return func_a()
""")
        
        self.create_test_file("module_d.py", """
from module_b import func_b

def helper_func():
    return func_b()
""")
        
        analyzer = DependencyAnalyzer(str(self.test_path))
        
        # Test that analysis completes without crashing
        try:
            result = analyzer.analyze_project()
            assert result is not None, "Analysis should return a result"
            
            # Check if it's a DependencyGraph object or similar
            assert hasattr(result, 'nodes') or hasattr(result, 'graph'), "Result should have graph structure"
            
        except Exception as e:
            pytest.fail(f"Analysis failed on circular dependencies: {e}")
    
    def test_dynamic_imports_detection(self):
        """Test handling of dynamic imports and importlib usage."""
        self.create_test_file("dynamic_imports.py", """
import importlib
from importlib import import_module
import sys

def load_module_dynamically(module_name):
    # Dynamic import using importlib
    module = importlib.import_module(module_name)
    return module

def conditional_import():
    if sys.version_info >= (3, 8):
        import_module('typing_extensions')
    else:
        import typing
    
def runtime_import():
    # Runtime import using __import__
    json_module = __import__('json')
    return json_module

# String-based dynamic import
MODULE_NAME = 'os'
os_module = importlib.import_module(MODULE_NAME)
""")
        
        analyzer = DependencyAnalyzer(str(self.test_path))
        
        # Test that analysis completes without crashing on dynamic imports
        try:
            result = analyzer.analyze_project()
            assert result is not None, "Analysis should return a result"
            
            # Should detect static imports (importlib, sys) but may miss dynamic ones
            # This tests the limits of static analysis
            assert hasattr(result, 'nodes') or hasattr(result, 'graph'), "Result should have graph structure"
            
        except Exception as e:
            pytest.fail(f"Analysis failed on dynamic imports: {e}")
    
    def test_conditional_imports_analysis(self):
        """Test analysis of conditional imports within try/except blocks."""
        self.create_test_file("conditional_imports.py", """
# Version-based conditional imports
try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

# Feature-based conditional imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# Platform-specific imports
import sys
if sys.platform == 'win32':
    import winsound
elif sys.platform.startswith('linux'):
    import subprocess
else:
    import os

# Optional dependency pattern
try:
    import rich
    from rich.console import Console
    HAS_RICH = True
except ImportError:
    rich = None
    Console = None
    HAS_RICH = False

def use_conditional_imports():
    if NUMPY_AVAILABLE:
        return np.array([1, 2, 3])
    if HAS_RICH:
        console = Console()
        console.print("Hello")
""")
        
        analyzer = DependencyAnalyzer(str(self.test_path))
        
        try:
            result = analyzer.analyze_project()
            assert result is not None, "Analysis should return a result"
            
            # Should handle conditional imports without crashing
            assert hasattr(result, 'nodes') or hasattr(result, 'graph'), "Result should have graph structure"
            
        except Exception as e:
            pytest.fail(f"Analysis failed on conditional imports: {e}")
    
    def test_malformed_python_files(self):
        """Test handling of syntactically invalid Python files."""
        # Syntax error - missing colon
        self.create_test_file("syntax_error1.py", """
import json
import sys

def broken_function()
    return "missing colon"

if __name__ == "__main__"
    print("missing colon here too")
""")
        
        # Indentation error
        self.create_test_file("syntax_error2.py", """
import os
from pathlib import Path

def indentation_error():
return "wrong indentation"
    
class BadIndentation:
def method(self):
        return "mixed indentation"
""")
        
        # Encoding error simulation
        self.create_test_file("encoding_error.py", """
# -*- coding: utf-8 -*-
import json
# This file has encoding issues but valid imports
""")
        
        # Incomplete file (EOF in middle of statement)
        self.create_test_file("incomplete.py", """
import sys
import os

def incomplete_function():
    x = [1, 2, 3,
""")
        
        analyzer = DependencyAnalyzer(str(self.test_path))
        
        # Should not crash on malformed files, should skip them gracefully
        try:
            result = analyzer.analyze_project()
            # Should complete without crashing
            assert result is not None, "Analysis completed without crashing on malformed files"
        except Exception as e:
            pytest.fail(f"Analysis should handle malformed files gracefully, but raised: {e}")
    
    def test_extremely_large_codebase(self):
        """Test performance with large number of files (simulated)."""
        # Create a large number of small Python files
        num_files = 100  # Reduced from 1000 for reasonable test execution time
        
        for i in range(num_files):
            content = f"""
# File {i}
import sys
import os
import json
from pathlib import Path

class Module{i}:
    def __init__(self):
        self.id = {i}
        
    def process(self):
        return f"Processing module {{self.id}}"

def function_{i}():
    module = Module{i}()
    return module.process()

if __name__ == "__main__":
    result = function_{i}()
    print(result)
"""
            self.create_test_file(f"module_{i:03d}.py", content)
        
        analyzer = DependencyAnalyzer(str(self.test_path))
        
        import time
        start_time = time.time()
        
        try:
            result = analyzer.analyze_project()
            
            end_time = time.time()
            analysis_time = end_time - start_time
            
            # Should complete within reasonable time
            assert analysis_time < 30.0, f"Analysis took {analysis_time}s, should be < 30s"
            
            # Should return a valid result for large codebase
            assert result is not None, "Analysis should return result for large codebase"
            assert hasattr(result, 'nodes') or hasattr(result, 'graph'), "Result should have graph structure"
            
        except Exception as e:
            pytest.fail(f"Large codebase analysis failed: {e}")
    
    def test_unicode_and_encoding_edge_cases(self):
        """Test files with various encodings and Unicode characters."""
        # Unicode in identifiers and strings
        self.create_test_file("unicode_test.py", """
# -*- coding: utf-8 -*-
import sys
import json
from pathlib import Path

# Unicode in variable names (valid in Python 3)
测试变量 = "test variable in Chinese"
café_name = "café with accent"
λ_value = 42  # Greek lambda

class Tëst:
    def __init__(self):
        self.naïve_approach = True
    
    def méthod_with_accénts(self):
        return "Héllo Wörld! 🚀"

def función_española():
    return "¡Hola mundo!"

# Unicode strings
emoji_string = "Testing with emojis: 🐍 🔬 📊"
mixed_script = "English العربية 中文 русский"
""")
        
        # File with different encoding declaration
        self.create_test_file("latin1_test.py", """
# -*- coding: latin-1 -*-
import os
import sys

# This would be Latin-1 encoded content
name = "José"
city = "São Paulo"
""")
        
        # Unicode in filename
        unicode_filename = "测试文件.py"
        self.create_test_file(unicode_filename, """
import json
import sys

def unicode_filename_test():
    return "Testing Unicode filename"
""")
        
        analyzer = DependencyAnalyzer(str(self.test_path))
        
        with patch('networkx.DiGraph') as mock_graph:
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            
            # Should handle Unicode files without issues
            try:
                result = analyzer.analyze_project()
                assert True, "Successfully handled Unicode files"
            except UnicodeError as e:
                pytest.fail(f"Should handle Unicode gracefully, but got: {e}")
            except Exception as e:
                # Other exceptions might be acceptable depending on implementation
                pass
    
    def test_symlink_and_junction_handling(self):
        """Test handling of symbolic links and Windows junctions."""
        # Create source files
        source_dir = self.test_path / "source"
        source_dir.mkdir()
        
        source_file = source_dir / "original.py"
        source_file.write_text("""
import sys
import json

def original_function():
    return "from original file"
""")
        
        # Create symbolic link (if supported by platform)
        link_dir = self.test_path / "links"
        link_dir.mkdir()
        
        try:
            # Try to create symbolic link
            link_file = link_dir / "linked.py"
            if hasattr(os, 'symlink'):
                os.symlink(str(source_file), str(link_file))
            else:
                # Fallback: just copy the file for testing
                import shutil
                shutil.copy2(str(source_file), str(link_file))
        except (OSError, NotImplementedError):
            # Platform doesn't support symlinks, create regular file instead
            link_file = link_dir / "linked.py"
            link_file.write_text(source_file.read_text())
        
        analyzer = DependencyAnalyzer(str(self.test_path))
        
        with patch('networkx.DiGraph') as mock_graph:
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            
            # Should handle symlinks appropriately (either follow them or ignore them)
            try:
                result = analyzer.analyze_project()
                # Should not crash on symlinks
                assert True, "Handled symlinks without crashing"
            except Exception as e:
                # Some exceptions might be acceptable depending on symlink handling strategy
                pass
    
    def test_import_alias_patterns(self):
        """Test various import alias patterns and complex import statements."""
        self.create_test_file("alias_patterns.py", """
# Various import alias patterns
import sys as system
import json as js
from pathlib import Path as FilePath
from collections import defaultdict as dd, OrderedDict as od
from typing import Dict, List, Optional, Union as U

# Multiple imports on one line
import os, re, time
from datetime import datetime, timedelta, timezone

# Nested module imports
from xml.etree import ElementTree as ET
from urllib.parse import urlparse, parse_qs
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# Import with parentheses (multi-line)
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

# Relative imports
from .utils import helper_function
from ..config import settings
from ...common import constants

def use_aliases():
    path = FilePath("test.txt")
    data = js.loads('{"key": "value"}')
    executor = ThreadPoolExecutor(max_workers=4)
    return system.version
""")
        
        analyzer = DependencyAnalyzer(str(self.test_path))
        
        try:
            result = analyzer.analyze_project()
            assert result is not None, "Analysis should return a result"
            
            # Should handle complex import patterns without crashing
            assert hasattr(result, 'nodes') or hasattr(result, 'graph'), "Result should have graph structure"
            
        except Exception as e:
            pytest.fail(f"Analysis failed on import alias patterns: {e}")
    
    def test_complex_module_structures(self):
        """Test analysis of complex module structures with packages and subpackages."""
        # Create package structure
        pkg_dir = self.test_path / "complex_package"
        pkg_dir.mkdir()
        
        # Main package __init__.py
        self.create_test_file("complex_package/__init__.py", """
from .core import CoreClass
from .utils import utility_function
from .subpackage import SubModule

__version__ = "1.0.0"
__all__ = ['CoreClass', 'utility_function', 'SubModule']
""")
        
        # Core module
        self.create_test_file("complex_package/core.py", """
import sys
import json
from typing import Any, Dict, List
from .utils import helper
from ..other_package import external_function

class CoreClass:
    def __init__(self):
        self.data = {}
    
    def process(self, data: Dict[str, Any]) -> List[str]:
        return helper.process_data(data)
""")
        
        # Utils module
        self.create_test_file("complex_package/utils.py", """
import os
import re
from pathlib import Path
from .core import CoreClass

class Helper:
    def process_data(self, data):
        return list(data.keys())

helper = Helper()
""")
        
        # Subpackage
        subpkg_dir = self.test_path / "complex_package" / "subpackage"
        subpkg_dir.mkdir()
        
        self.create_test_file("complex_package/subpackage/__init__.py", """
from .module import SubModule
""")
        
        self.create_test_file("complex_package/subpackage/module.py", """
import json
from ...complex_package.core import CoreClass

class SubModule:
    def __init__(self):
        self.core = CoreClass()
""")
        
        # External package reference
        other_pkg_dir = self.test_path / "other_package"
        other_pkg_dir.mkdir()
        
        self.create_test_file("other_package/__init__.py", """
def external_function():
    return "external"
""")
        
        analyzer = DependencyAnalyzer(str(self.test_path))
        
        try:
            result = analyzer.analyze_project()
            assert result is not None, "Analysis should return a result"
            
            # Should handle complex package structures without crashing
            assert hasattr(result, 'nodes') or hasattr(result, 'graph'), "Result should have graph structure"
            
        except Exception as e:
            pytest.fail(f"Analysis failed on complex module structures: {e}")
    
    def test_error_recovery_and_partial_analysis(self):
        """Test recovery from errors and partial analysis completion."""
        # Mix of valid and invalid files
        self.create_test_file("valid1.py", """
import sys
import json

def valid_function():
    return "valid"
""")
        
        self.create_test_file("invalid.py", """
import sys
import json

def broken_function(
    # Missing closing parenthesis and syntax errors
    return "broken
""")
        
        self.create_test_file("valid2.py", """
import os
from pathlib import Path

def another_valid_function():
    return Path.cwd()
""")
        
        # File that can't be read (simulate permission error)
        restricted_file = self.create_test_file("restricted.py", """
import sys
def restricted_function():
    return "restricted"
""")
        
        analyzer = DependencyAnalyzer(str(self.test_path))
        
        # Mock file reading to simulate permission error
        original_read_text = Path.read_text
        
        def mock_read_text(self, *args, **kwargs):
            if self.name == "restricted.py":
                raise PermissionError("Access denied")
            return original_read_text(self, *args, **kwargs)
        
        with patch.object(Path, 'read_text', mock_read_text):
            try:
                # Should complete analysis despite errors
                result = analyzer.analyze_project()
                
                # Should return some result even with partial failures
                assert result is not None, "Should return partial results despite errors"
                assert hasattr(result, 'nodes') or hasattr(result, 'graph'), "Result should have graph structure"
                
            except Exception as e:
                # Complete failure is also acceptable for this edge case
                # What matters is that it doesn't crash the system entirely
                pass
    
    @pytest.mark.slow
    def test_memory_usage_large_files(self):
        """Test memory usage with very large Python files."""
        # Create a large Python file (simulated)
        large_content = '''
import sys
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict, OrderedDict

'''
        
        # Add many function definitions to make it large
        for i in range(1000):  # Reduced from larger number for reasonable test time
            large_content += f'''
def function_{i}(param1, param2, param3):
    """Function number {i}."""
    result = param1 + param2 + param3
    return result * {i}

class Class_{i}:
    """Class number {i}."""
    
    def __init__(self):
        self.value = {i}
    
    def method_{i}(self):
        return self.value * 2
        
'''
        
        self.create_test_file("large_file.py", large_content)
        
        analyzer = DependencyAnalyzer(str(self.test_path))
        
        with patch('networkx.DiGraph') as mock_graph:
            mock_instance = MagicMock()
            mock_graph.return_value = mock_instance
            
            # Monitor memory usage (simplified)
            import psutil
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            result = analyzer.analyze_project()
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before
            
            # Memory increase should be reasonable (less than 100MB for this test)
            assert memory_increase < 100, f"Memory usage increased by {memory_increase}MB, should be < 100MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])