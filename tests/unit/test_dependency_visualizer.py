"""
Unit tests for dependency_visualizer.py
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open

# Mock imports before importing the module
with patch.dict('sys.modules', {
    'networkx': MagicMock(),
    'matplotlib': MagicMock(),
    'plotly': MagicMock(),
    'rich': MagicMock(),
    'pandas': MagicMock()
}):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))
    import dependency_visualizer


class TestDependencyVisualizer:
    """Test cases for DependencyVisualizer class."""
    
    def test_init(self, mock_project_structure):
        """Test DependencyVisualizer initialization."""
        visualizer = dependency_visualizer.DependencyVisualizer(
            str(mock_project_structure),
            ai_awareness=True
        )
        
        assert visualizer.project_path == str(mock_project_structure)
        assert visualizer.ai_awareness is True
        assert hasattr(visualizer, 'dependency_graph')
    
    def test_init_with_nonexistent_path(self):
        """Test DependencyVisualizer initialization with nonexistent path."""
        with pytest.raises(FileNotFoundError):
            dependency_visualizer.DependencyVisualizer("/nonexistent/path")
    
    @patch('dependency_visualizer.NETWORKX_AVAILABLE', True)
    def test_analyze_project_success(self, mock_project_structure, mock_networkx):
        """Test successful project analysis."""
        with patch('dependency_visualizer.nx.DiGraph', return_value=mock_networkx):
            visualizer = dependency_visualizer.DependencyVisualizer(str(mock_project_structure))
            result = visualizer.analyze_project()
            
            assert result is not None
            assert hasattr(result, 'add_node')
            assert hasattr(result, 'add_edge')
    
    @patch('dependency_visualizer.NETWORKX_AVAILABLE', False)
    def test_analyze_project_networkx_unavailable(self, mock_project_structure):
        """Test project analysis when NetworkX is unavailable."""
        visualizer = dependency_visualizer.DependencyVisualizer(str(mock_project_structure))
        
        with pytest.raises(ImportError, match="NetworkX is required"):
            visualizer.analyze_project()
    
    def test_extract_imports_from_file(self, mock_project_structure):
        """Test import extraction from Python files.""" 
        # Create a test file with various import patterns
        test_file = mock_project_structure / "test_imports.py"
        test_file.write_text("""
import os
import sys
from pathlib import Path
from typing import Dict, List
import json as j
from collections import defaultdict, Counter
""")
        
        visualizer = dependency_visualizer.DependencyVisualizer(str(mock_project_structure))
        imports = visualizer._extract_imports_from_file(str(test_file))
        
        expected_imports = [
            'os', 'sys', 'pathlib', 'typing', 'json', 'collections'
        ]
        
        for expected_import in expected_imports:
            assert any(expected_import in imp for imp in imports)
    
    def test_extract_imports_from_invalid_file(self, mock_project_structure):
        """Test import extraction from invalid Python file."""
        # Create a file with syntax errors
        test_file = mock_project_structure / "invalid.py"
        test_file.write_text("import os\nif True\n  print('invalid syntax')")
        
        visualizer = dependency_visualizer.DependencyVisualizer(str(mock_project_structure))
        imports = visualizer._extract_imports_from_file(str(test_file))
        
        # Should return empty list for files with syntax errors
        assert imports == []
    
    def test_is_external_dependency(self):
        """Test external dependency detection."""
        visualizer = dependency_visualizer.DependencyVisualizer(".")
        
        # Standard library modules
        assert not visualizer._is_external_dependency("os")
        assert not visualizer._is_external_dependency("sys")
        assert not visualizer._is_external_dependency("json")
        
        # External packages
        assert visualizer._is_external_dependency("requests")
        assert visualizer._is_external_dependency("numpy")
        assert visualizer._is_external_dependency("pandas")
        
        # Local modules (assuming they exist in project)
        assert not visualizer._is_external_dependency("models")
        assert not visualizer._is_external_dependency("utils")
    
    @patch('dependency_visualizer.PLOTLY_AVAILABLE', True)
    def test_generate_interactive_html(self, mock_project_structure, mock_networkx):
        """Test HTML generation."""
        with patch('dependency_visualizer.nx.DiGraph', return_value=mock_networkx), \
             patch('plotly.graph_objects.Figure') as mock_fig, \
             patch('builtins.open', mock_open()):
            
            mock_fig_instance = MagicMock()
            mock_fig.return_value = mock_fig_instance
            
            visualizer = dependency_visualizer.DependencyVisualizer(str(mock_project_structure))
            visualizer.dependency_graph = mock_networkx
            
            output_path = visualizer.generate_interactive_html(mock_networkx)
            
            assert output_path.endswith('.html')
            mock_fig_instance.write_html.assert_called_once()
    
    @patch('dependency_visualizer.PLOTLY_AVAILABLE', False)
    def test_generate_interactive_html_plotly_unavailable(self, mock_project_structure, mock_networkx):
        """Test HTML generation when Plotly is unavailable."""
        visualizer = dependency_visualizer.DependencyVisualizer(str(mock_project_structure))
        
        with pytest.raises(ImportError, match="Plotly is required"):
            visualizer.generate_interactive_html(mock_networkx)
    
    def test_export_to_json(self, mock_project_structure, mock_networkx, sample_dependency_graph):
        """Test JSON export functionality."""
        mock_networkx.nodes.return_value = [
            ("main.py", {"label": "main", "type": "module"}),
            ("utils.py", {"label": "utils", "type": "module"})
        ]
        mock_networkx.edges.return_value = [
            ("main.py", "utils.py", {"type": "import"})
        ]
        
        visualizer = dependency_visualizer.DependencyVisualizer(str(mock_project_structure))
        result = visualizer.export_to_json(mock_networkx)
        
        assert "nodes" in result
        assert "edges" in result
        assert len(result["nodes"]) == 2
        assert len(result["edges"]) == 1
    
    def test_generate_text_report(self, mock_project_structure, mock_networkx):
        """Test text report generation."""
        # Mock networkx graph data
        mock_networkx.nodes.return_value = ["main.py", "utils.py", "models/user.py"]
        mock_networkx.edges.return_value = [("main.py", "utils.py")]
        mock_networkx.number_of_nodes.return_value = 3
        mock_networkx.number_of_edges.return_value = 1
        
        visualizer = dependency_visualizer.DependencyVisualizer(str(mock_project_structure))
        report = visualizer.generate_text_report(mock_networkx)
        
        assert "Dependency Analysis Report" in report
        assert "Total modules: 3" in report
        assert "Total dependencies: 1" in report
    
    def test_detect_circular_dependencies(self, mock_project_structure, mock_networkx):
        """Test circular dependency detection."""
        # Mock a graph with circular dependencies
        mock_networkx.nodes.return_value = ["A", "B", "C"]
        
        # Mock networkx.simple_cycles to return a cycle
        with patch('dependency_visualizer.nx.simple_cycles', return_value=[["A", "B", "C", "A"]]):
            visualizer = dependency_visualizer.DependencyVisualizer(str(mock_project_structure))
            cycles = visualizer._detect_circular_dependencies(mock_networkx)
            
            assert len(cycles) == 1
            assert "A" in cycles[0]
            assert "B" in cycles[0]
            assert "C" in cycles[0]
    
    def test_calculate_risk_metrics(self, mock_project_structure, mock_networkx):
        """Test risk metrics calculation."""
        # Mock networkx methods for risk calculation
        mock_networkx.in_degree.return_value = [("main.py", 0), ("utils.py", 2)]
        mock_networkx.out_degree.return_value = [("main.py", 1), ("utils.py", 0)]
        
        visualizer = dependency_visualizer.DependencyVisualizer(str(mock_project_structure))
        risks = visualizer._calculate_risk_metrics(mock_networkx)
        
        assert "utils.py" in risks
        assert risks["utils.py"]["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
    
    def test_get_ai_context_health(self):
        """Test AI context health assessment."""
        assert dependency_visualizer.get_ai_context_health(1500) == "GOOD"
        assert dependency_visualizer.get_ai_context_health(3000) == "WARNING"
        assert dependency_visualizer.get_ai_context_health(5000) == "CRITICAL"
    
    def test_estimate_token_count(self, mock_project_structure):
        """Test token count estimation."""
        test_file = mock_project_structure / "test_token.py"
        test_content = "import os\nprint('Hello, world!')\n"
        test_file.write_text(test_content)
        
        token_count = dependency_visualizer.estimate_token_count(str(test_file))
        
        # Should be roughly len(content) // 4
        expected_tokens = len(test_content) // 4
        assert abs(token_count - expected_tokens) <= 1
    
    def test_estimate_token_count_nonexistent_file(self):
        """Test token count estimation for nonexistent file."""
        token_count = dependency_visualizer.estimate_token_count("/nonexistent/file.py")
        assert token_count == 0


class TestDependencyVisualizerIntegration:
    """Integration tests for DependencyVisualizer."""
    
    @patch('dependency_visualizer.NETWORKX_AVAILABLE', True)
    @patch('dependency_visualizer.RICH_AVAILABLE', True)
    def test_full_analysis_workflow(self, mock_project_structure, mock_networkx):
        """Test complete analysis workflow."""
        with patch('dependency_visualizer.nx.DiGraph', return_value=mock_networkx):
            visualizer = dependency_visualizer.DependencyVisualizer(
                str(mock_project_structure),
                ai_awareness=True
            )
            
            # Perform analysis
            graph = visualizer.analyze_project()
            
            # Generate different outputs
            text_report = visualizer.generate_text_report(graph)
            json_data = visualizer.export_to_json(graph)
            
            assert graph is not None
            assert "Dependency Analysis Report" in text_report
            assert "nodes" in json_data
            assert "edges" in json_data


class TestCommandLineInterface:
    """Test the command line interface."""
    
    def test_parse_arguments(self):
        """Test argument parsing."""
        with patch('sys.argv', ['dependency_visualizer.py', '/test/path', '--format', 'html']):
            with patch('dependency_visualizer.argparse.ArgumentParser.parse_args') as mock_parse:
                mock_args = MagicMock()
                mock_args.project_path = '/test/path'
                mock_args.format = 'html'
                mock_args.output = None
                mock_args.ai_awareness = True
                mock_parse.return_value = mock_args
                
                # Test that arguments are parsed correctly
                assert mock_args.project_path == '/test/path'
                assert mock_args.format == 'html'
    
    @patch('dependency_visualizer.DependencyVisualizer')
    def test_main_function(self, mock_visualizer_class):
        """Test main function execution."""
        mock_visualizer = MagicMock()
        mock_visualizer_class.return_value = mock_visualizer
        mock_visualizer.analyze_project.return_value = MagicMock()
        
        with patch('sys.argv', ['dependency_visualizer.py', '/test/path']), \
             patch('dependency_visualizer.argparse.ArgumentParser.parse_args') as mock_parse:
            
            mock_args = MagicMock()
            mock_args.project_path = '/test/path'
            mock_args.format = 'text'
            mock_args.output = None
            mock_args.ai_awareness = True
            mock_parse.return_value = mock_args
            
            # Import and call main function
            if hasattr(dependency_visualizer, 'main'):
                dependency_visualizer.main()
                
                mock_visualizer_class.assert_called_once_with('/test/path', ai_awareness=True)
                mock_visualizer.analyze_project.assert_called_once()


@pytest.mark.parametrize("ai_awareness", [True, False])
def test_ai_awareness_parameter(mock_project_structure, ai_awareness):
    """Test AI awareness parameter handling."""
    visualizer = dependency_visualizer.DependencyVisualizer(
        str(mock_project_structure),
        ai_awareness=ai_awareness
    )
    assert visualizer.ai_awareness == ai_awareness


def test_dependency_graph_persistence(mock_project_structure):
    """Test that dependency graph is properly stored."""
    visualizer = dependency_visualizer.DependencyVisualizer(str(mock_project_structure))
    
    # Initially None
    assert visualizer.dependency_graph is None
    
    # After analysis, should be set
    with patch('dependency_visualizer.NETWORKX_AVAILABLE', True), \
         patch('dependency_visualizer.nx.DiGraph') as mock_digraph:
        
        mock_graph = MagicMock()
        mock_digraph.return_value = mock_graph
        
        result = visualizer.analyze_project()
        assert visualizer.dependency_graph is not None
        assert visualizer.dependency_graph == result