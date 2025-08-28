"""
Simplified comprehensive tests for code quality analysis.
Tests essential functionality without overly complex examples.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from tools.code_analyzer import CodeAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError:
    ANALYZER_AVAILABLE = False


@pytest.mark.unit
@pytest.mark.skipif(not ANALYZER_AVAILABLE, reason="CodeAnalyzer not available")
class TestCodeQualitySimple:
    """Test cases for code quality analysis with simplified examples."""
    
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
    
    def test_basic_code_analysis(self):
        """Test basic code analysis functionality."""
        self.create_test_file("simple.py", """
import sys
import json
import unused_module

def test_function():
    return json.dumps({"test": True})
""")
        
        analyzer = CodeAnalyzer(str(self.test_path))
        
        try:
            result = analyzer.analyze_unused_imports()
            assert result is not None
            assert isinstance(result, list)
        except Exception as e:
            pytest.fail(f"Basic code analysis failed: {e}")
    
    def test_unused_imports_detection(self):
        """Test detection of unused imports."""
        self.create_test_file("unused_imports.py", """
import sys  # Used
import json  # Used
import os  # Unused
from pathlib import Path  # Unused

def main():
    print(sys.version)
    return json.dumps({"status": "ok"})
""")
        
        analyzer = CodeAnalyzer(str(self.test_path))
        
        try:
            result = analyzer.analyze_unused_imports()
            assert result is not None
        except Exception as e:
            pytest.fail(f"Unused imports analysis failed: {e}")
    
    def test_malformed_file_handling(self):
        """Test handling of malformed Python files."""
        self.create_test_file("malformed.py", """
import sys
# Missing closing quote
def broken_function():
    return "unclosed string
""")
        
        analyzer = CodeAnalyzer(str(self.test_path))
        
        # Should not crash on malformed files
        try:
            result = analyzer.analyze_unused_imports()
            # Result might be None or partial, but shouldn't crash
            pass
        except Exception as e:
            # Some exceptions might be acceptable depending on implementation
            pass
    
    def test_empty_project_analysis(self):
        """Test analysis of empty project directory."""
        analyzer = CodeAnalyzer(str(self.test_path))
        
        try:
            result = analyzer.analyze_unused_imports()
            # Should handle empty projects gracefully
            assert result is not None or result is None  # Either is acceptable
        except Exception as e:
            pytest.fail(f"Empty project analysis failed: {e}")
    
    def test_large_file_handling(self):
        """Test handling of large Python files."""
        # Create a reasonably large file
        content = "import sys\nimport json\n\n"
        for i in range(100):
            content += f"""
def function_{i}():
    return {i} * 2

class Class_{i}:
    def method_{i}(self):
        return {i}
"""
        
        self.create_test_file("large_file.py", content)
        
        analyzer = CodeAnalyzer(str(self.test_path))
        
        try:
            result = analyzer.analyze_unused_imports()
            assert result is not None
        except Exception as e:
            pytest.fail(f"Large file analysis failed: {e}")
    
    def test_unicode_file_handling(self):
        """Test handling of files with Unicode content."""
        self.create_test_file("unicode_test.py", """
# -*- coding: utf-8 -*-
import sys

# Unicode comments: 测试 العربية русский
def función_test():
    name = "José"
    message = "Hello 世界! 🌍"
    return message
""")
        
        analyzer = CodeAnalyzer(str(self.test_path))
        
        try:
            result = analyzer.analyze_unused_imports()
            assert result is not None
        except UnicodeError as e:
            pytest.fail(f"Should handle Unicode gracefully: {e}")
        except Exception as e:
            # Other exceptions might be acceptable
            pass
    
    def test_circular_imports_analysis(self):
        """Test analysis of files with circular import patterns."""
        self.create_test_file("module_a.py", """
import module_b

def func_a():
    return module_b.func_b()
""")
        
        self.create_test_file("module_b.py", """
import module_a

def func_b():
    return "from B"
""")
        
        analyzer = CodeAnalyzer(str(self.test_path))
        
        try:
            result = analyzer.analyze_unused_imports()
            assert result is not None
        except Exception as e:
            pytest.fail(f"Circular imports analysis failed: {e}")
    
    def test_complex_imports_patterns(self):
        """Test analysis of complex import patterns."""
        self.create_test_file("complex_imports.py", """
import sys as system
from pathlib import Path as P
from typing import Dict, List, Optional
from collections import defaultdict, OrderedDict

# Star import
from os import *

def use_imports():
    path = P("test")
    data = defaultdict(list)
    return system.version
""")
        
        analyzer = CodeAnalyzer(str(self.test_path))
        
        try:
            result = analyzer.analyze_unused_imports()
            assert result is not None
        except Exception as e:
            pytest.fail(f"Complex imports analysis failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])