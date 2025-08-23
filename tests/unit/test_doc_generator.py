"""
Unit tests for doc_generator.py
"""

import pytest
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
    
    def test_generate_all_documentation(self, mock_project_structure):
        """Test documentation generation."""
        generator = doc_generator.DocumentationGenerator(str(mock_project_structure))
        
        with patch('builtins.open', mock_open()) as mock_file:
            output_files = generator.generate_all_documentation()
            
            assert isinstance(output_files, dict)
            # Should write to files
            mock_file.assert_called()
    
    def test_generate_all_documentation_with_custom_output(self, mock_project_structure):
        """Test documentation generation with custom output path."""
        generator = doc_generator.DocumentationGenerator(str(mock_project_structure))
        custom_path = "custom_docs"
        
        with patch('builtins.open', mock_open()):
            output_files = generator.generate_all_documentation(custom_path)
            
            assert isinstance(output_files, dict)
    
    def test_extract_project_metadata(self, mock_project_structure):
        """Test project metadata extraction."""
        generator = doc_generator.DocumentationGenerator(str(mock_project_structure))
        
        metadata = generator._extract_project_metadata()
        
        assert isinstance(metadata, doc_generator.ProjectMetadata)
        assert metadata.name is not None
        assert metadata.path is not None
    
    def test_extract_api_endpoints(self, mock_project_structure):
        """Test API endpoint extraction."""
        generator = doc_generator.DocumentationGenerator(str(mock_project_structure))
        
        endpoints = generator._extract_api_endpoints()
        
        assert isinstance(endpoints, list)
    
    def test_analyze_project(self, mock_project_structure):
        """Test project analysis."""
        generator = doc_generator.DocumentationGenerator(str(mock_project_structure))
        
        # Create additional test files
        (mock_project_structure / "api" / "__init__.py").parent.mkdir(exist_ok=True)
        (mock_project_structure / "api" / "__init__.py").write_text("")
        (mock_project_structure / "api" / "routes.py").write_text("from utils import process_data")
        
        generator._analyze_project()
        
        # Should have some metadata and dependency graph
        assert generator.project_metadata is not None or generator.dependency_graph is not None
    
    def test_analyze_components(self, mock_project_structure):
        """Test component analysis."""
        generator = doc_generator.DocumentationGenerator(str(mock_project_structure))
        
        # Create architecture-like structure
        for layer in ["presentation", "business", "data"]:
            layer_dir = mock_project_structure / layer
            layer_dir.mkdir(exist_ok=True)
            (layer_dir / "__init__.py").write_text("")
            (layer_dir / f"{layer}_module.py").write_text(f"# {layer} layer")
        
        components = generator._analyze_components()
        
        assert isinstance(components, dict)
    
    def test_extract_api_endpoints_with_functions(self, mock_project_structure):
        """Test API endpoint extraction with functions."""
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
        endpoints = generator._extract_api_endpoints()
        
        assert isinstance(endpoints, list)
    
    def test_get_function_parameters(self):
        """Test function parameter extraction."""
        import ast
        
        code = '''
def example_function(arg1, arg2="default"):
    """This is a docstring."""
    pass
'''
        
        tree = ast.parse(code)
        func_node = tree.body[0]
        
        generator = doc_generator.DocumentationGenerator(".")
        params = generator._get_function_parameters(func_node)
        
        assert isinstance(params, list)
    
    def test_get_function_dependencies(self):
        """Test function dependency extraction."""
        import ast
        
        code = '''
def example_function():
    import os
    return os.path.join("a", "b")
'''
        
        tree = ast.parse(code)
        func_node = tree.body[0]
        
        generator = doc_generator.DocumentationGenerator(".")
        deps = generator._get_function_dependencies(func_node)
        
        assert isinstance(deps, list)
    
    def test_get_return_type(self):
        """Test return type extraction."""
        import ast
        
        code = '''
def example_function(arg1, arg2="default", *args, **kwargs):
    return "test"
'''
        
        tree = ast.parse(code)
        func_node = tree.body[0]
        
        generator = doc_generator.DocumentationGenerator(".")
        return_type = generator._get_return_type(func_node)
        
        # May be None or a string
        assert return_type is None or isinstance(return_type, str)
    
    def test_detect_framework(self):
        """Test framework detection."""
        generator = doc_generator.DocumentationGenerator(".")
        framework = generator._detect_framework()
        
        # May be None or a string
        assert framework is None or isinstance(framework, str)
    
    def test_analyze_project_initializes_properly(self, mock_project_structure):
        """Test that project analysis initializes properly."""
        generator = doc_generator.DocumentationGenerator(str(mock_project_structure))
        
        # Test that analyze project works
        generator._analyze_project()
        assert True  # Just test it doesn't crash
    
    def test_detect_backend_tech(self, mock_project_structure):
        """Test backend technology detection."""
        # Create backend-like structure
        for directory in ["models", "views", "controllers"]:
            dir_path = mock_project_structure / directory
            dir_path.mkdir(exist_ok=True)
            (dir_path / "__init__.py").write_text("")
            (dir_path / f"{directory[:-1]}.py").write_text(f"# {directory} code")
        
        generator = doc_generator.DocumentationGenerator(str(mock_project_structure))
        backend_tech = generator._detect_backend_tech()
        
        assert isinstance(backend_tech, list)
    
    def test_detect_database_tech(self, mock_project_structure):
        """Test database technology detection."""
        # Create a complex file
        db_file = mock_project_structure / "models.py"
        db_file.write_text('''
from django.db import models

class User(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
''')
        
        generator = doc_generator.DocumentationGenerator(str(mock_project_structure))
        db_tech = generator._detect_database_tech()
        
        assert isinstance(db_tech, list)


class TestDocumentationGeneratorIntegration:
    """Integration tests for DocumentationGenerator."""
    
    def test_full_documentation_generation_workflow(self, mock_project_structure):
        """Test complete documentation generation workflow."""
        generator = doc_generator.DocumentationGenerator(str(mock_project_structure))
        
        with patch('builtins.open', mock_open()) as mock_file:
            
            # Generate all documentation
            output_files = generator.generate_all_documentation()
            
            assert isinstance(output_files, dict)
            # Should have created files
            mock_file.assert_called()


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
        mock_generator.generate_all_documentation.return_value = {"docs": "output.md"}
        
        with patch('sys.argv', ['doc_generator.py', '/test/path']), \
             patch('doc_generator.argparse.ArgumentParser.parse_args') as mock_parse:
            
            mock_args = MagicMock()
            mock_args.project_path = '/test/path'
            mock_args.type = 'all'
            mock_args.output = None
            mock_parse.return_value = mock_args
            
            # Import and call main function if it exists
            if hasattr(doc_generator, 'main'):
                doc_generator.main()
                
                mock_generator_class.assert_called_once_with('/test/path')


@pytest.mark.parametrize("output_dir", [None, "custom_docs"])
def test_output_directory_parameter(mock_project_structure, output_dir):
    """Test different output directory options."""
    generator = doc_generator.DocumentationGenerator(str(mock_project_structure))
    
    with patch('builtins.open', mock_open()) as mock_file:
        output = generator.generate_all_documentation(output_dir)
        
        assert isinstance(output, dict)
        mock_file.assert_called()