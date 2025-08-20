"""
Unit tests for doc_generator.py
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

# Mock imports before importing the module
with patch.dict('sys.modules', {
    'jinja2': MagicMock(),
    'markdown': MagicMock(),
    'rich': MagicMock()
}):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))
    import doc_generator


class TestDocumentationGenerator:
    """Test cases for DocumentationGenerator class."""
    
    def test_init(self, mock_project_structure):
        """Test DocumentationGenerator initialization."""
        generator = doc_generator.DocumentationGenerator(str(mock_project_structure))
        
        assert generator.project_path == str(mock_project_structure)
        assert hasattr(generator, 'console')
    
    def test_init_with_nonexistent_path(self):
        """Test DocumentationGenerator initialization with nonexistent path."""
        with pytest.raises(FileNotFoundError):
            doc_generator.DocumentationGenerator("/nonexistent/path")
    
    def test_generate_dependency_map(self, mock_project_structure):
        """Test dependency map generation."""
        generator = doc_generator.DocumentationGenerator(str(mock_project_structure))
        
        with patch('builtins.open', mock_open()) as mock_file, \
             patch.object(generator, '_analyze_project_structure') as mock_analyze, \
             patch.object(generator, '_render_template') as mock_render:
            
            mock_analyze.return_value = {
                "modules": ["main.py", "utils.py"],
                "dependencies": [{"source": "main.py", "target": "utils.py"}]
            }
            mock_render.return_value = "# Dependency Map\n\nGenerated documentation..."
            
            output_path = generator.generate_dependency_map()
            
            assert output_path.endswith("DEPENDENCY_MAP.md")
            mock_analyze.assert_called_once()
            mock_render.assert_called_once()
            mock_file.assert_called()
    
    def test_generate_dependency_map_with_custom_output(self, mock_project_structure):
        """Test dependency map generation with custom output path."""
        generator = doc_generator.DocumentationGenerator(str(mock_project_structure))
        custom_path = "custom_deps.md"
        
        with patch('builtins.open', mock_open()) as mock_file, \
             patch.object(generator, '_analyze_project_structure') as mock_analyze, \
             patch.object(generator, '_render_template') as mock_render:
            
            mock_analyze.return_value = {"modules": [], "dependencies": []}
            mock_render.return_value = "Generated content"
            
            output_path = generator.generate_dependency_map(custom_path)
            
            assert output_path == custom_path
    
    def test_generate_architecture_overview(self, mock_project_structure):
        """Test architecture overview generation."""
        generator = doc_generator.DocumentationGenerator(str(mock_project_structure))
        
        with patch('builtins.open', mock_open()) as mock_file, \
             patch.object(generator, '_analyze_architecture') as mock_analyze, \
             patch.object(generator, '_render_template') as mock_render:
            
            mock_analyze.return_value = {
                "layers": ["presentation", "business", "data"],
                "components": ["web", "api", "database"],
                "patterns": ["MVC", "Repository"]
            }
            mock_render.return_value = "# Architecture Overview\n\nSystem design..."
            
            output_path = generator.generate_architecture_overview()
            
            assert output_path.endswith("ARCHITECTURE.md")
            mock_analyze.assert_called_once()
            mock_render.assert_called_once()
    
    def test_generate_api_docs(self, mock_project_structure):
        """Test API documentation generation."""
        generator = doc_generator.DocumentationGenerator(str(mock_project_structure))
        
        with patch('builtins.open', mock_open()) as mock_file, \
             patch.object(generator, '_extract_api_documentation') as mock_extract, \
             patch.object(generator, '_render_template') as mock_render:
            
            mock_extract.return_value = {
                "functions": [
                    {
                        "name": "calculate_total",
                        "args": ["items", "tax_rate"],
                        "docstring": "Calculate total with tax",
                        "file": "utils.py"
                    }
                ],
                "classes": []
            }
            mock_render.return_value = "# API Documentation\n\nFunction reference..."
            
            output_path = generator.generate_api_docs()
            
            assert output_path.endswith("API.md")
            mock_extract.assert_called_once()
            mock_render.assert_called_once()
    
    def test_analyze_project_structure(self, mock_project_structure):
        """Test project structure analysis."""
        generator = doc_generator.DocumentationGenerator(str(mock_project_structure))
        
        # Create additional test files
        (mock_project_structure / "api" / "__init__.py").parent.mkdir(exist_ok=True)
        (mock_project_structure / "api" / "__init__.py").write_text("")
        (mock_project_structure / "api" / "routes.py").write_text("from utils import process_data")
        
        structure = generator._analyze_project_structure()
        
        assert "modules" in structure
        assert "dependencies" in structure
        assert len(structure["modules"]) > 0
    
    def test_analyze_architecture(self, mock_project_structure):
        """Test architecture analysis."""
        generator = doc_generator.DocumentationGenerator(str(mock_project_structure))
        
        # Create architecture-like structure
        for layer in ["presentation", "business", "data"]:
            layer_dir = mock_project_structure / layer
            layer_dir.mkdir(exist_ok=True)
            (layer_dir / "__init__.py").write_text("")
            (layer_dir / f"{layer}_module.py").write_text(f"# {layer} layer")
        
        architecture = generator._analyze_architecture()
        
        assert "layers" in architecture
        assert "components" in architecture
        assert "patterns" in architecture
        assert len(architecture["layers"]) > 0
    
    def test_extract_api_documentation(self, mock_project_structure):
        """Test API documentation extraction."""
        # Create a file with documented functions
        api_file = mock_project_structure / "api.py"
        api_file.write_text('''
def calculate_total(items, tax_rate=0.1):
    """
    Calculate the total price including tax.
    
    Args:
        items (list): List of item prices
        tax_rate (float): Tax rate as decimal (default: 0.1)
    
    Returns:
        float: Total price including tax
    """
    subtotal = sum(items)
    return subtotal * (1 + tax_rate)

class Calculator:
    """A simple calculator class."""
    
    def add(self, a, b):
        """Add two numbers."""
        return a + b
''')
        
        generator = doc_generator.DocumentationGenerator(str(mock_project_structure))
        api_docs = generator._extract_api_documentation()
        
        assert "functions" in api_docs
        assert "classes" in api_docs
        assert len(api_docs["functions"]) > 0
        assert len(api_docs["classes"]) > 0
        
        # Check function extraction
        calc_func = next((f for f in api_docs["functions"] if f["name"] == "calculate_total"), None)
        assert calc_func is not None
        assert "items" in calc_func["args"]
        assert "tax_rate" in calc_func["args"]
    
    def test_extract_docstring(self):
        """Test docstring extraction from AST node."""
        import ast
        
        code = '''
def example_function():
    """This is a docstring."""
    pass
'''
        
        tree = ast.parse(code)
        func_node = tree.body[0]
        
        generator = doc_generator.DocumentationGenerator(".")
        docstring = generator._extract_docstring(func_node)
        
        assert docstring == "This is a docstring."
    
    def test_extract_docstring_no_docstring(self):
        """Test docstring extraction when no docstring exists."""
        import ast
        
        code = '''
def example_function():
    pass
'''
        
        tree = ast.parse(code)
        func_node = tree.body[0]
        
        generator = doc_generator.DocumentationGenerator(".")
        docstring = generator._extract_docstring(func_node)
        
        assert docstring == ""
    
    def test_extract_function_args(self):
        """Test function argument extraction."""
        import ast
        
        code = '''
def example_function(arg1, arg2="default", *args, **kwargs):
    pass
'''
        
        tree = ast.parse(code)
        func_node = tree.body[0]
        
        generator = doc_generator.DocumentationGenerator(".")
        args = generator._extract_function_args(func_node)
        
        assert "arg1" in args
        assert "arg2" in args
        assert "args" in args
        assert "kwargs" in args
    
    @patch('doc_generator.jinja2.Environment')
    def test_render_template(self, mock_jinja_env):
        """Test template rendering."""
        mock_template = MagicMock()
        mock_template.render.return_value = "Rendered content"
        mock_env = MagicMock()
        mock_env.get_template.return_value = mock_template
        mock_jinja_env.return_value = mock_env
        
        generator = doc_generator.DocumentationGenerator(".")
        result = generator._render_template("dependency_map.md", {"data": "test"})
        
        assert result == "Rendered content"
        mock_env.get_template.assert_called_once_with("dependency_map.md")
        mock_template.render.assert_called_once_with({"data": "test"})
    
    def test_get_template_path(self, mock_project_structure):
        """Test template path resolution."""
        generator = doc_generator.DocumentationGenerator(str(mock_project_structure))
        
        # Mock templates directory
        with patch('pathlib.Path.exists', return_value=True):
            template_path = generator._get_template_path()
            assert "templates" in str(template_path)
    
    def test_detect_patterns(self, mock_project_structure):
        """Test architectural pattern detection."""
        # Create MVC-like structure
        for directory in ["models", "views", "controllers"]:
            dir_path = mock_project_structure / directory
            dir_path.mkdir(exist_ok=True)
            (dir_path / "__init__.py").write_text("")
            (dir_path / f"{directory[:-1]}.py").write_text(f"# {directory} code")
        
        generator = doc_generator.DocumentationGenerator(str(mock_project_structure))
        patterns = generator._detect_patterns()
        
        assert isinstance(patterns, list)
        # Should detect MVC pattern
        assert any("MVC" in pattern.upper() for pattern in patterns)
    
    def test_calculate_complexity_metrics(self, mock_project_structure):
        """Test complexity metrics calculation."""
        # Create a complex file
        complex_file = mock_project_structure / "complex.py"
        complex_file.write_text('''
def complex_function(x, y, z):
    if x > 0:
        if y > 0:
            if z > 0:
                return x + y + z
            else:
                return x + y
        else:
            if z > 0:
                return x + z
            else:
                return x
    else:
        return 0

class ComplexClass:
    def __init__(self):
        self.data = []
    
    def method1(self):
        pass
    
    def method2(self):
        pass
''')
        
        generator = doc_generator.DocumentationGenerator(str(mock_project_structure))
        metrics = generator._calculate_complexity_metrics()
        
        assert "files" in metrics
        assert "total_lines" in metrics
        assert "average_complexity" in metrics
        assert metrics["total_lines"] > 0


class TestDocumentationGeneratorIntegration:
    """Integration tests for DocumentationGenerator."""
    
    def test_full_documentation_generation_workflow(self, mock_project_structure):
        """Test complete documentation generation workflow."""
        generator = doc_generator.DocumentationGenerator(str(mock_project_structure))
        
        with patch('builtins.open', mock_open()) as mock_file, \
             patch.object(generator, '_render_template', return_value="Generated content"):
            
            # Generate all types of documentation
            dep_map = generator.generate_dependency_map()
            arch_overview = generator.generate_architecture_overview()
            api_docs = generator.generate_api_docs()
            
            assert dep_map.endswith(".md")
            assert arch_overview.endswith(".md")
            assert api_docs.endswith(".md")
            
            # Should have created multiple files
            assert mock_file.call_count >= 3


class TestTemplateHandling:
    """Test template handling functionality."""
    
    def test_template_not_found_handling(self, mock_project_structure):
        """Test handling when template is not found."""
        generator = doc_generator.DocumentationGenerator(str(mock_project_structure))
        
        with patch('doc_generator.jinja2.Environment') as mock_env:
            mock_env.return_value.get_template.side_effect = FileNotFoundError("Template not found")
            
            # Should handle template not found gracefully
            result = generator._render_template("nonexistent.md", {})
            
            # Should return a default template or error message
            assert isinstance(result, str)
    
    def test_template_rendering_error(self, mock_project_structure):
        """Test handling of template rendering errors."""
        generator = doc_generator.DocumentationGenerator(str(mock_project_structure))
        
        with patch('doc_generator.jinja2.Environment') as mock_env:
            mock_template = MagicMock()
            mock_template.render.side_effect = Exception("Rendering error")
            mock_env.return_value.get_template.return_value = mock_template
            
            # Should handle rendering errors gracefully
            result = generator._render_template("test.md", {})
            
            # Should return an error message or empty string
            assert isinstance(result, str)


class TestCommandLineInterface:
    """Test the command line interface."""
    
    @patch('doc_generator.DocumentationGenerator')
    def test_main_function(self, mock_generator_class):
        """Test main function execution."""
        mock_generator = MagicMock()
        mock_generator_class.return_value = mock_generator
        mock_generator.generate_dependency_map.return_value = "output.md"
        
        with patch('sys.argv', ['doc_generator.py', '/test/path']), \
             patch('doc_generator.argparse.ArgumentParser.parse_args') as mock_parse:
            
            mock_args = MagicMock()
            mock_args.project_path = '/test/path'
            mock_args.doc_type = 'dependency_map'
            mock_args.output = None
            mock_parse.return_value = mock_args
            
            # Import and call main function if it exists
            if hasattr(doc_generator, 'main'):
                doc_generator.main()
                
                mock_generator_class.assert_called_once_with('/test/path')
                mock_generator.generate_dependency_map.assert_called_once()


@pytest.mark.parametrize("doc_type", ["dependency_map", "architecture_overview", "api_docs"])
def test_doc_type_parameter(mock_project_structure, doc_type):
    """Test different documentation types."""
    generator = doc_generator.DocumentationGenerator(str(mock_project_structure))
    
    with patch('builtins.open', mock_open()) as mock_file, \
         patch.object(generator, '_render_template', return_value="Generated content"):
        
        if doc_type == "dependency_map":
            output = generator.generate_dependency_map()
        elif doc_type == "architecture_overview":
            output = generator.generate_architecture_overview()
        elif doc_type == "api_docs":
            output = generator.generate_api_docs()
        
        assert output.endswith(".md")
        assert mock_file.called