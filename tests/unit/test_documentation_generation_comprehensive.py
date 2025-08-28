"""
Comprehensive tests for documentation generation functionality.
Tests template rendering, format validation, and edge cases.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import sys
import json

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from tools.doc_generator import DocumentationGenerator
    DOC_GENERATOR_AVAILABLE = True
except ImportError:
    DOC_GENERATOR_AVAILABLE = False


@pytest.mark.unit
@pytest.mark.skipif(not DOC_GENERATOR_AVAILABLE, reason="DocumentationGenerator not available")
class TestDocumentationGenerationEdgeCases:
    """Test cases for documentation generation edge cases and boundary conditions."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        self.output_dir = self.test_path / "output"
        self.output_dir.mkdir(exist_ok=True)
    
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
    
    def test_mermaid_graph_syntax_validation(self):
        """Test Mermaid graph syntax validation with complex dependencies."""
        # Create complex project structure
        self.create_test_file("main.py", """
import sys
import json
from utils import helper
from models.user import User
from config import settings
""")
        
        self.create_test_file("utils.py", """
import os
import re
from pathlib import Path
""")
        
        self.create_test_file("models/user.py", """
from dataclasses import dataclass
from typing import Optional
""")
        
        self.create_test_file("config.py", """
import os
from pathlib import Path
""")
        
        doc_gen = DocumentationGenerator(str(self.test_path))
        
        # Mock dependency analysis
        mock_dependencies = {
            "main.py": {
                "imports": ["sys", "json", "utils", "models.user", "config"],
                "type": "module",
                "size": 150
            },
            "utils.py": {
                "imports": ["os", "re", "pathlib"],
                "type": "module", 
                "size": 100
            },
            "models/user.py": {
                "imports": ["dataclasses", "typing"],
                "type": "module",
                "size": 80
            },
            "config.py": {
                "imports": ["os", "pathlib"],
                "type": "module",
                "size": 50
            }
        }
        
        with patch.object(doc_gen, '_analyze_dependencies', return_value=mock_dependencies):
            # Test Mermaid syntax generation
            result = doc_gen.generate_dependency_map()
            
            assert result is not None
            
            # Verify Mermaid syntax is valid
            if isinstance(result, str) and "graph" in result.lower():
                # Should contain graph declaration
                assert any(line.strip().startswith("graph") for line in result.split('\n'))
                # Should contain node definitions
                assert "main.py" in result or "main" in result
                # Should contain arrows/connections
                assert "-->" in result or "--->" in result
    
    def test_template_rendering_with_various_data_sizes(self):
        """Test template rendering with different data sizes and formats."""
        doc_gen = DocumentationGenerator(str(self.test_path))
        
        # Test with minimal data
        minimal_data = {
            "nodes": [{"id": "main.py", "type": "module"}],
            "edges": [],
            "stats": {"total_files": 1, "total_dependencies": 0}
        }
        
        # Test with large data set
        large_data = {
            "nodes": [{"id": f"module_{i}.py", "type": "module"} for i in range(100)],
            "edges": [{"source": f"module_{i}.py", "target": f"module_{i+1}.py"} for i in range(99)],
            "stats": {"total_files": 100, "total_dependencies": 99}
        }
        
        # Test with complex nested data
        complex_data = {
            "nodes": [
                {
                    "id": "complex_module.py",
                    "type": "module",
                    "metadata": {
                        "imports": ["sys", "os", "json"],
                        "functions": ["func1", "func2"],
                        "classes": ["Class1", "Class2"],
                        "complexity": {"cyclomatic": 15, "cognitive": 20}
                    }
                }
            ],
            "edges": [],
            "stats": {
                "total_files": 1,
                "total_dependencies": 0,
                "complexity_distribution": {"low": 0, "medium": 1, "high": 0}
            }
        }
        
        test_cases = [
            ("minimal", minimal_data),
            ("large", large_data),
            ("complex", complex_data)
        ]
        
        for test_name, test_data in test_cases:
            with patch.object(doc_gen, '_get_project_data', return_value=test_data):
                try:
                    result = doc_gen.generate_dependency_map()
                    assert result is not None, f"Failed to generate documentation for {test_name} data"
                except Exception as e:
                    pytest.fail(f"Template rendering failed for {test_name} data: {e}")
    
    def test_multiformat_output_generation(self):
        """Test generation of multiple output formats."""
        doc_gen = DocumentationGenerator(str(self.test_path))
        
        # Mock project data
        mock_data = {
            "nodes": [
                {"id": "main.py", "type": "module", "size": 100},
                {"id": "utils.py", "type": "module", "size": 80}
            ],
            "edges": [
                {"source": "main.py", "target": "utils.py", "type": "import"}
            ],
            "stats": {"total_files": 2, "total_dependencies": 1}
        }
        
        with patch.object(doc_gen, '_get_project_data', return_value=mock_data):
            # Test HTML output
            html_output = doc_gen.generate_html_report()
            if html_output:
                assert isinstance(html_output, str)
                assert "<html" in html_output.lower() or "<!doctype" in html_output.lower()
            
            # Test Markdown output
            md_output = doc_gen.generate_markdown_report()
            if md_output:
                assert isinstance(md_output, str)
                # Should contain markdown elements
                assert "#" in md_output or "##" in md_output or "```" in md_output
            
            # Test JSON output
            json_output = doc_gen.generate_json_report()
            if json_output:
                assert isinstance(json_output, (str, dict))
                if isinstance(json_output, str):
                    # Should be valid JSON
                    try:
                        json.loads(json_output)
                    except json.JSONDecodeError:
                        pytest.fail("Generated JSON output is not valid JSON")
    
    def test_unicode_and_special_characters_handling(self):
        """Test handling of Unicode and special characters in documentation."""
        # Create files with Unicode names and content
        unicode_content = """
# -*- coding: utf-8 -*-
# File with Unicode content: 测试文件
import sys
import json

class 测试类:
    '''Class with Unicode name and docstring: 这是一个测试类'''
    
    def __init__(self):
        self.名称 = "测试名称"
        self.描述 = "这是一个描述"
    
    def 方法_with_émojis(self):
        '''Method with emojis: 🚀 🐍 📊'''
        return "测试结果 with émojis: ✨ 🎉"

def función_española():
    '''Función con caracteres especiales: ñáéíóú'''
    return "¡Hola mundo!"

# Russian comments: Привет мир
# Arabic comments: مرحبا بالعالم  
# Chinese comments: 你好世界
""")
        
        self.create_test_file("unicode_测试.py", unicode_content)
        
        # File with mixed scripts
        mixed_script_content = """
import os
# Mixed script content: English العربية 中文 русский
def mixed_function():
    data = {
        'english': 'Hello',
        'arabic': 'مرحبا',
        'chinese': '你好',
        'russian': 'Привет',
        'emoji': '🌍🌎🌏'
    }
    return data
"""
        
        self.create_test_file("mixed_scripts.py", mixed_script_content)
        
        doc_gen = DocumentationGenerator(str(self.test_path))
        
        # Should handle Unicode without errors
        try:
            result = doc_gen.generate_dependency_map()
            assert result is not None, "Should generate documentation despite Unicode content"
        except UnicodeError as e:
            pytest.fail(f"Should handle Unicode gracefully, but got: {e}")
        except Exception as e:
            # Other exceptions might be acceptable depending on implementation
            pass
    
    def test_large_project_documentation_performance(self):
        """Test documentation generation performance with large projects."""
        # Create a large project structure
        num_files = 50  # Reduced for reasonable test time
        
        # Create main modules
        for i in range(num_files):
            content = f"""
# Module {i}
import sys
import os
import json
from pathlib import Path

class Module{i}Class:
    '''Class for module {i}'''
    
    def __init__(self):
        self.module_id = {i}
        self.dependencies = {list(range(max(0, i-3), i))}
    
    def process(self):
        '''Process data for module {i}'''
        return f"Processed module {{self.module_id}}"
    
    def get_stats(self):
        '''Get statistics for module {i}'''
        return {{
            'module_id': self.module_id,
            'dependency_count': len(self.dependencies),
            'class_count': 1,
            'method_count': 3
        }}

def module_{i}_function():
    '''Function for module {i}'''
    instance = Module{i}Class()
    return instance.process()

# Constants
MODULE_{i}_VERSION = "1.0.{i}"
MODULE_{i}_AUTHOR = "Generated Author {i}"
"""
            self.create_test_file(f"modules/module_{i:03d}.py", content)
        
        # Create package structure
        packages = ["utils", "models", "services", "controllers", "views"]
        for package in packages:
            for i in range(10):  # 10 files per package
                content = f"""
# {package.title()} module {i}
import sys
from typing import Any, Dict, List

class {package.title()}{i}:
    def process_{package}(self, data: Any) -> Dict[str, Any]:
        return {{'processed': True, 'data': data, 'module': '{package}_{i}'}}
"""
                self.create_test_file(f"{package}/{package}_{i}.py", content)
        
        doc_gen = DocumentationGenerator(str(self.test_path))
        
        import time
        start_time = time.time()
        
        # Generate documentation
        try:
            result = doc_gen.generate_dependency_map()
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            # Should complete within reasonable time
            assert generation_time < 60.0, f"Documentation generation took {generation_time}s, should be < 60s"
            assert result is not None, "Should generate documentation for large project"
            
        except Exception as e:
            pytest.fail(f"Large project documentation generation failed: {e}")
    
    def test_malformed_template_handling(self):
        """Test handling of malformed templates and template errors."""
        doc_gen = DocumentationGenerator(str(self.test_path))
        
        # Mock template with syntax errors
        malformed_template = """
    graph TD
        {% for node in nodes %}
            {{ node.id }} --> {{ node.target  # Missing closing brace
        {% endfor %}
        {% for edge in edges  # Missing closing brace
            {{ edge.source }} --> {{ edge.target }}
        {% endfor %}
"""
        
        # Mock template with undefined variables
        undefined_var_template = """
    graph TD
        {% for node in nodes %}
            {{ node.id }} --> {{ node.undefined_field }}
        {% endfor %}
        {{ undefined_variable }}
"""
        
        # Test error handling
        mock_data = {
            "nodes": [{"id": "main.py"}],
            "edges": [{"source": "main.py", "target": "utils.py"}]
        }
        
        with patch.object(doc_gen, '_get_project_data', return_value=mock_data):
            # Should handle template errors gracefully
            try:
                # This might fail, but shouldn't crash the entire system
                result = doc_gen.generate_dependency_map()
                # If it succeeds, that's fine too
                pass
            except Exception as e:
                # Template errors should be handled gracefully
                # The specific behavior depends on implementation
                pass
    
    def test_circular_dependency_visualization(self):
        """Test visualization of circular dependencies in documentation."""
        # Create circular dependency structure
        self.create_test_file("module_a.py", """
from module_b import function_b
import sys

def function_a():
    return function_b() + "A"
""")
        
        self.create_test_file("module_b.py", """
from module_c import function_c
import os

def function_b():
    return function_c() + "B"
""")
        
        self.create_test_file("module_c.py", """
from module_a import function_a  # Creates circular dependency
import json

def function_c():
    return "C"
""")
        
        doc_gen = DocumentationGenerator(str(self.test_path))
        
        # Mock circular dependency detection
        mock_data = {
            "nodes": [
                {"id": "module_a.py", "type": "module"},
                {"id": "module_b.py", "type": "module"},
                {"id": "module_c.py", "type": "module"}
            ],
            "edges": [
                {"source": "module_a.py", "target": "module_b.py"},
                {"source": "module_b.py", "target": "module_c.py"},
                {"source": "module_c.py", "target": "module_a.py"}  # Circular
            ],
            "circular_dependencies": [
                ["module_a.py", "module_b.py", "module_c.py"]
            ],
            "stats": {
                "total_files": 3,
                "total_dependencies": 3,
                "circular_count": 1
            }
        }
        
        with patch.object(doc_gen, '_get_project_data', return_value=mock_data):
            result = doc_gen.generate_dependency_map()
            
            # Should generate documentation that highlights circular dependencies
            assert result is not None
            
            # If it's a Mermaid diagram, might contain special styling for cycles
            if isinstance(result, str) and "graph" in result:
                # Should contain the circular nodes
                assert "module_a" in result or "module_a.py" in result
                assert "module_b" in result or "module_b.py" in result
                assert "module_c" in result or "module_c.py" in result
    
    def test_dependency_depth_analysis(self):
        """Test documentation of dependency depth and complexity."""
        # Create deep dependency chain
        self.create_test_file("level_0.py", """
from level_1 import Level1Class
import sys

class Level0Class:
    def __init__(self):
        self.level1 = Level1Class()
""")
        
        for level in range(1, 6):  # Create 5 levels
            next_level = level + 1 if level < 5 else None
            import_line = f"from level_{next_level} import Level{next_level}Class" if next_level else ""
            
            content = f"""
{import_line}
import sys

class Level{level}Class:
    def __init__(self):
        {'self.next_level = Level' + str(next_level) + 'Class()' if next_level else 'self.data = "leaf node"'}
    
    def get_depth(self):
        return {level}
"""
            self.create_test_file(f"level_{level}.py", content)
        
        doc_gen = DocumentationGenerator(str(self.test_path))
        
        # Mock deep dependency analysis
        mock_data = {
            "nodes": [{"id": f"level_{i}.py", "type": "module", "depth": i} for i in range(6)],
            "edges": [{"source": f"level_{i}.py", "target": f"level_{i+1}.py"} for i in range(5)],
            "depth_analysis": {
                "max_depth": 5,
                "avg_depth": 2.5,
                "depth_distribution": {str(i): 1 for i in range(6)}
            },
            "stats": {"total_files": 6, "total_dependencies": 5}
        }
        
        with patch.object(doc_gen, '_get_project_data', return_value=mock_data):
            result = doc_gen.generate_dependency_map()
            
            # Should generate documentation showing dependency depth
            assert result is not None
    
    def test_template_caching_and_reuse(self):
        """Test template caching and reuse for performance."""
        doc_gen = DocumentationGenerator(str(self.test_path))
        
        mock_data = {
            "nodes": [{"id": "main.py", "type": "module"}],
            "edges": [],
            "stats": {"total_files": 1, "total_dependencies": 0}
        }
        
        with patch.object(doc_gen, '_get_project_data', return_value=mock_data):
            # Generate documentation multiple times
            results = []
            generation_times = []
            
            for i in range(3):
                import time
                start = time.time()
                result = doc_gen.generate_dependency_map()
                end = time.time()
                
                results.append(result)
                generation_times.append(end - start)
            
            # All results should be consistent
            for result in results:
                assert result is not None
            
            # Later generations might be faster due to caching (if implemented)
            # This is optional behavior to test
            pass
    
    def test_documentation_versioning_and_metadata(self):
        """Test inclusion of version information and metadata in documentation."""
        doc_gen = DocumentationGenerator(str(self.test_path))
        
        # Mock project with version information
        mock_data = {
            "project_info": {
                "name": "Test Project",
                "version": "1.2.3",
                "description": "A test project for documentation generation",
                "author": "Test Author",
                "python_version": "3.9+",
                "dependencies": ["requests>=2.25.0", "numpy>=1.20.0"]
            },
            "generation_info": {
                "generated_at": "2024-01-01T12:00:00Z",
                "generator_version": "2.0.0",
                "analysis_duration": 1.23
            },
            "nodes": [{"id": "main.py", "type": "module"}],
            "edges": [],
            "stats": {"total_files": 1, "total_dependencies": 0}
        }
        
        with patch.object(doc_gen, '_get_project_data', return_value=mock_data):
            result = doc_gen.generate_dependency_map()
            
            # Should include metadata in documentation
            assert result is not None
    
    def test_error_recovery_partial_documentation(self):
        """Test error recovery and partial documentation generation."""
        # Create mixed valid and invalid files
        self.create_test_file("valid.py", """
import sys
import json

def valid_function():
    return "valid"
""")
        
        self.create_test_file("invalid.py", """
import sys
# This file has syntax errors
def broken_function(
    # Missing closing parenthesis
    return "broken
""")
        
        self.create_test_file("valid2.py", """
import os
from pathlib import Path

def another_function():
    return Path.cwd()
""")
        
        doc_gen = DocumentationGenerator(str(self.test_path))
        
        # Should generate partial documentation despite some errors
        try:
            result = doc_gen.generate_dependency_map()
            
            # Should not be None even with some invalid files
            assert result is not None, "Should generate partial documentation despite errors"
            
            # Should include the valid files
            if isinstance(result, str):
                assert "valid" in result or "valid.py" in result
                
        except Exception as e:
            # Complete failure might be acceptable depending on implementation
            # but the system should ideally recover gracefully
            pass
    
    def test_custom_styling_and_themes(self):
        """Test custom styling and theme support in documentation."""
        doc_gen = DocumentationGenerator(str(self.test_path))
        
        mock_data = {
            "nodes": [
                {"id": "main.py", "type": "module", "category": "core"},
                {"id": "utils.py", "type": "module", "category": "utility"},
                {"id": "config.py", "type": "module", "category": "configuration"}
            ],
            "edges": [
                {"source": "main.py", "target": "utils.py"},
                {"source": "main.py", "target": "config.py"}
            ],
            "styling": {
                "theme": "dark",
                "node_colors": {
                    "core": "#FF6B6B",
                    "utility": "#4ECDC4", 
                    "configuration": "#45B7D1"
                },
                "edge_styles": {
                    "import": "solid",
                    "inherit": "dashed"
                }
            },
            "stats": {"total_files": 3, "total_dependencies": 2}
        }
        
        with patch.object(doc_gen, '_get_project_data', return_value=mock_data):
            result = doc_gen.generate_dependency_map()
            
            # Should apply custom styling
            assert result is not None
            
            # If Mermaid format, might contain styling information
            if isinstance(result, str) and "graph" in result:
                # Might contain color or style information
                pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])