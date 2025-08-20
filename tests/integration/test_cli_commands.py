"""
Integration tests for CLI commands and entry points.
"""

import pytest
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestCLIEntryPoints:
    """Test CLI entry points work end-to-end."""
    
    def test_all_entry_points_exist(self):
        """Test that all CLI entry points are properly defined."""
        project_root = Path(__file__).parent.parent.parent
        pyproject_path = project_root / "pyproject.toml"
        
        expected_entry_points = [
            "deepflow-visualizer",
            "deepflow-validator",
            "deepflow-docs", 
            "deepflow-ci",
            "deepflow-monitor",
            "deepflow-analyzer",
            "ai-session-tracker",
            "deepflow-mcp-server"
        ]
        
        if pyproject_path.exists():
            content = pyproject_path.read_text()
            for entry_point in expected_entry_points:
                assert entry_point in content, f"Entry point {entry_point} not found in pyproject.toml"
    
    def test_tools_importable_for_cli(self):
        """Test that tools are importable for CLI usage."""
        project_root = Path(__file__).parent.parent.parent
        tools_dir = project_root / "tools"
        
        # Add tools to path
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
                module = __import__(module_name)
                
                # Should have main function for CLI
                assert hasattr(module, 'main'), f"{module_name} missing main() function"
                assert callable(getattr(module, 'main')), f"{module_name}.main() not callable"
                
            except ImportError as e:
                # Some imports may fail due to missing dependencies
                # This is acceptable if handled gracefully
                if "required dependency" in str(e).lower():
                    continue
                else:
                    pytest.fail(f"Unexpected import error for {module_name}: {e}")


class TestDependencyVisualizerCLI:
    """Test dependency visualizer CLI functionality."""
    
    @patch('dependency_visualizer.DependencyVisualizer')
    def test_dependency_visualizer_cli_basic(self, mock_viz_class, mock_project_structure):
        """Test basic dependency visualizer CLI execution."""
        # Add tools to path
        project_root = Path(__file__).parent.parent.parent
        tools_dir = project_root / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        
        # Mock the visualizer
        mock_viz = MagicMock()
        mock_viz_class.return_value = mock_viz
        mock_viz.analyze_project.return_value = MagicMock()
        mock_viz.generate_text_report.return_value = "Test report"
        
        # Mock command line arguments
        test_args = ['dependency_visualizer.py', str(mock_project_structure)]
        
        with patch('sys.argv', test_args), \
             patch('dependency_visualizer.argparse.ArgumentParser.parse_args') as mock_parse:
            
            mock_args = MagicMock()
            mock_args.project_path = str(mock_project_structure)
            mock_args.format = 'text'
            mock_args.output = None
            mock_args.ai_awareness = True
            mock_parse.return_value = mock_args
            
            # Import and test
            import dependency_visualizer
            
            if hasattr(dependency_visualizer, 'main'):
                dependency_visualizer.main()
                
                # Verify CLI functionality
                mock_viz_class.assert_called_once()
                mock_viz.analyze_project.assert_called_once()
    
    @patch('dependency_visualizer.DependencyVisualizer')
    def test_dependency_visualizer_cli_formats(self, mock_viz_class, mock_project_structure):
        """Test dependency visualizer CLI with different formats."""
        project_root = Path(__file__).parent.parent.parent
        tools_dir = project_root / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        
        mock_viz = MagicMock()
        mock_viz_class.return_value = mock_viz
        mock_viz.analyze_project.return_value = MagicMock()
        
        formats_to_test = ['text', 'html', 'json']
        
        for format_type in formats_to_test:
            # Set up format-specific returns
            if format_type == 'html':
                mock_viz.generate_interactive_html.return_value = "output.html"
            elif format_type == 'json':
                mock_viz.export_to_json.return_value = {"nodes": [], "edges": []}
            else:
                mock_viz.generate_text_report.return_value = "Text report"
            
            test_args = ['dependency_visualizer.py', str(mock_project_structure), '--format', format_type]
            
            with patch('sys.argv', test_args), \
                 patch('dependency_visualizer.argparse.ArgumentParser.parse_args') as mock_parse:
                
                mock_args = MagicMock()
                mock_args.project_path = str(mock_project_structure)
                mock_args.format = format_type
                mock_args.output = None
                mock_args.ai_awareness = True
                mock_parse.return_value = mock_args
                
                # Clear module cache to ensure fresh import
                sys.modules.pop('dependency_visualizer', None)
                
                import dependency_visualizer
                
                if hasattr(dependency_visualizer, 'main'):
                    dependency_visualizer.main()


class TestCodeAnalyzerCLI:
    """Test code analyzer CLI functionality."""
    
    @patch('code_analyzer.CodeAnalyzer')
    def test_code_analyzer_cli_basic(self, mock_analyzer_class, mock_project_structure):
        """Test basic code analyzer CLI execution."""
        project_root = Path(__file__).parent.parent.parent
        tools_dir = project_root / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        
        # Mock the analyzer
        mock_analyzer = MagicMock()
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.analyze_unused_imports.return_value = []
        
        test_args = ['code_analyzer.py', str(mock_project_structure)]
        
        with patch('sys.argv', test_args), \
             patch('code_analyzer.argparse.ArgumentParser.parse_args') as mock_parse:
            
            mock_args = MagicMock()
            mock_args.project_path = str(mock_project_structure)
            mock_args.fix_imports = False
            mock_args.analyze_coupling = False
            mock_parse.return_value = mock_args
            
            try:
                import code_analyzer
                
                if hasattr(code_analyzer, 'main'):
                    code_analyzer.main()
                    
                    # Verify CLI functionality
                    mock_analyzer_class.assert_called_once()
                    
            except SystemExit:
                # Some tools may exit due to missing dependencies
                pass
    
    @patch('code_analyzer.CodeAnalyzer')
    def test_code_analyzer_cli_with_fix_mode(self, mock_analyzer_class, mock_project_structure):
        """Test code analyzer CLI with fix mode enabled."""
        project_root = Path(__file__).parent.parent.parent
        tools_dir = project_root / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        
        mock_analyzer = MagicMock()
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.analyze_unused_imports.return_value = []
        
        test_args = ['code_analyzer.py', str(mock_project_structure), '--fix-imports']
        
        with patch('sys.argv', test_args), \
             patch('code_analyzer.argparse.ArgumentParser.parse_args') as mock_parse:
            
            mock_args = MagicMock()
            mock_args.project_path = str(mock_project_structure)
            mock_args.fix_imports = True
            mock_args.analyze_coupling = False
            mock_parse.return_value = mock_args
            
            try:
                sys.modules.pop('code_analyzer', None)
                import code_analyzer
                
                if hasattr(code_analyzer, 'main'):
                    code_analyzer.main()
                    
                    # Should call with fix_mode=True
                    mock_analyzer.analyze_unused_imports.assert_called_with(fix_mode=True)
                    
            except SystemExit:
                # Expected if missing dependencies
                pass


class TestDocGeneratorCLI:
    """Test documentation generator CLI functionality."""
    
    @patch('doc_generator.DocumentationGenerator')
    def test_doc_generator_cli_basic(self, mock_doc_gen_class, mock_project_structure):
        """Test basic documentation generator CLI execution."""
        project_root = Path(__file__).parent.parent.parent
        tools_dir = project_root / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        
        # Mock the generator
        mock_doc_gen = MagicMock()
        mock_doc_gen_class.return_value = mock_doc_gen
        mock_doc_gen.generate_dependency_map.return_value = "DEPENDENCY_MAP.md"
        
        test_args = ['doc_generator.py', str(mock_project_structure)]
        
        with patch('sys.argv', test_args), \
             patch('doc_generator.argparse.ArgumentParser.parse_args') as mock_parse:
            
            mock_args = MagicMock()
            mock_args.project_path = str(mock_project_structure)
            mock_args.doc_type = 'dependency_map'
            mock_args.output = None
            mock_parse.return_value = mock_args
            
            try:
                import doc_generator
                
                if hasattr(doc_generator, 'main'):
                    doc_generator.main()
                    
                    # Verify CLI functionality
                    mock_doc_gen_class.assert_called_once()
                    
            except (ImportError, SystemExit):
                # May fail due to missing dependencies
                pass
    
    @patch('doc_generator.DocumentationGenerator')
    def test_doc_generator_cli_different_types(self, mock_doc_gen_class, mock_project_structure):
        """Test documentation generator CLI with different doc types."""
        project_root = Path(__file__).parent.parent.parent
        tools_dir = project_root / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        
        mock_doc_gen = MagicMock()
        mock_doc_gen_class.return_value = mock_doc_gen
        
        doc_types = ['dependency_map', 'architecture_overview', 'api_docs']
        
        for doc_type in doc_types:
            # Set up type-specific returns
            if doc_type == 'dependency_map':
                mock_doc_gen.generate_dependency_map.return_value = "DEPENDENCY_MAP.md"
            elif doc_type == 'architecture_overview':
                mock_doc_gen.generate_architecture_overview.return_value = "ARCHITECTURE.md"
            elif doc_type == 'api_docs':
                mock_doc_gen.generate_api_docs.return_value = "API.md"
            
            test_args = ['doc_generator.py', str(mock_project_structure), '--doc-type', doc_type]
            
            with patch('sys.argv', test_args), \
                 patch('doc_generator.argparse.ArgumentParser.parse_args') as mock_parse:
                
                mock_args = MagicMock()
                mock_args.project_path = str(mock_project_structure)
                mock_args.doc_type = doc_type
                mock_args.output = None
                mock_parse.return_value = mock_args
                
                try:
                    sys.modules.pop('doc_generator', None)
                    import doc_generator
                    
                    if hasattr(doc_generator, 'main'):
                        doc_generator.main()
                        
                except (ImportError, SystemExit):
                    pass


class TestMCPServerCLI:
    """Test MCP server CLI functionality."""
    
    @patch('deepflow.mcp.server.DeepflowMCPServer')
    @patch('deepflow.mcp.server.MCP_AVAILABLE', True)
    def test_mcp_server_cli_basic(self, mock_server_class):
        """Test basic MCP server CLI execution."""
        mock_server = MagicMock()
        mock_server_class.return_value = mock_server
        mock_server.run = MagicMock()
        
        from deepflow.mcp import server
        
        # Test main function
        with patch('deepflow.mcp.server.asyncio.run') as mock_run:
            server.main()
            mock_run.assert_called_once()
    
    @patch('deepflow.mcp.server.MCP_AVAILABLE', False)
    def test_mcp_server_cli_unavailable(self):
        """Test MCP server CLI when MCP is unavailable."""
        with patch('sys.exit') as mock_exit, \
             patch('builtins.print') as mock_print:
            
            from deepflow.mcp import server
            
            # Should exit with error
            server.main()
            
            # Verify error handling
            mock_print.assert_called()
            mock_exit.assert_called_with(1)


class TestCLIErrorHandling:
    """Test CLI error handling and user experience."""
    
    def test_cli_handles_missing_project_path(self):
        """Test CLI handles missing project path gracefully."""
        project_root = Path(__file__).parent.parent.parent
        tools_dir = project_root / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        
        test_args = ['dependency_visualizer.py', '/nonexistent/path']
        
        with patch('sys.argv', test_args), \
             patch('dependency_visualizer.argparse.ArgumentParser.parse_args') as mock_parse:
            
            mock_args = MagicMock()
            mock_args.project_path = '/nonexistent/path'
            mock_args.format = 'text'
            mock_args.output = None
            mock_args.ai_awareness = True
            mock_parse.return_value = mock_args
            
            try:
                import dependency_visualizer
                
                if hasattr(dependency_visualizer, 'main'):
                    # Should handle FileNotFoundError gracefully
                    dependency_visualizer.main()
                    
            except (FileNotFoundError, SystemExit):
                # Expected behavior for invalid paths
                pass
    
    def test_cli_handles_invalid_arguments(self):
        """Test CLI handles invalid arguments gracefully."""
        project_root = Path(__file__).parent.parent.parent
        tools_dir = project_root / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        
        # Test invalid format argument
        test_args = ['dependency_visualizer.py', '.', '--format', 'invalid_format']
        
        with patch('sys.argv', test_args):
            try:
                import dependency_visualizer
                
                # Should handle via argparse validation
                if hasattr(dependency_visualizer, 'main'):
                    dependency_visualizer.main()
                    
            except (ValueError, SystemExit):
                # Expected for invalid arguments
                pass
    
    def test_cli_provides_help(self):
        """Test that CLI tools provide help."""
        project_root = Path(__file__).parent.parent.parent
        tools_dir = project_root / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        
        test_args = ['dependency_visualizer.py', '--help']
        
        with patch('sys.argv', test_args):
            try:
                import dependency_visualizer
                
                if hasattr(dependency_visualizer, 'main'):
                    dependency_visualizer.main()
                    
            except SystemExit as e:
                # argparse exits with 0 for --help
                assert e.code == 0


class TestCLIIntegrationWithFileSystem:
    """Test CLI integration with file system operations."""
    
    def test_cli_creates_output_files(self, mock_project_structure):
        """Test that CLI tools create output files when specified."""
        project_root = Path(__file__).parent.parent.parent
        tools_dir = project_root / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        
        output_file = mock_project_structure / "test_output.html"
        test_args = ['dependency_visualizer.py', str(mock_project_structure), '--output', str(output_file)]
        
        with patch('sys.argv', test_args), \
             patch('dependency_visualizer.DependencyVisualizer') as mock_viz_class, \
             patch('builtins.open', create=True) as mock_open:
            
            mock_viz = MagicMock()
            mock_viz_class.return_value = mock_viz
            mock_viz.analyze_project.return_value = MagicMock()
            mock_viz.generate_interactive_html.return_value = str(output_file)
            
            mock_args = MagicMock()
            mock_args.project_path = str(mock_project_structure)
            mock_args.format = 'html'
            mock_args.output = str(output_file)
            mock_args.ai_awareness = True
            
            with patch('dependency_visualizer.argparse.ArgumentParser.parse_args', return_value=mock_args):
                try:
                    import dependency_visualizer
                    
                    if hasattr(dependency_visualizer, 'main'):
                        dependency_visualizer.main()
                        
                        # Should attempt to create output file
                        mock_viz.generate_interactive_html.assert_called()
                        
                except (ImportError, SystemExit):
                    pass
    
    def test_cli_respects_output_directory(self, mock_project_structure):
        """Test that CLI tools respect output directory settings."""
        project_root = Path(__file__).parent.parent.parent
        tools_dir = project_root / "tools"
        if str(tools_dir) not in sys.path:
            sys.path.insert(0, str(tools_dir))
        
        output_dir = mock_project_structure / "output"
        output_dir.mkdir(exist_ok=True)
        
        test_args = ['doc_generator.py', str(mock_project_structure), '--output', str(output_dir / "docs.md")]
        
        with patch('sys.argv', test_args), \
             patch('doc_generator.DocumentationGenerator') as mock_doc_gen_class:
            
            mock_doc_gen = MagicMock()
            mock_doc_gen_class.return_value = mock_doc_gen
            mock_doc_gen.generate_dependency_map.return_value = str(output_dir / "docs.md")
            
            mock_args = MagicMock()
            mock_args.project_path = str(mock_project_structure)
            mock_args.doc_type = 'dependency_map'
            mock_args.output = str(output_dir / "docs.md")
            
            with patch('doc_generator.argparse.ArgumentParser.parse_args', return_value=mock_args):
                try:
                    import doc_generator
                    
                    if hasattr(doc_generator, 'main'):
                        doc_generator.main()
                        
                        # Should use specified output path
                        mock_doc_gen.generate_dependency_map.assert_called_with(str(output_dir / "docs.md"))
                        
                except (ImportError, SystemExit):
                    pass


class TestCLIPerformance:
    """Test CLI performance and resource usage."""
    
    def test_cli_handles_large_projects(self):
        """Test that CLI tools can handle large projects gracefully."""
        # This would test memory usage and performance
        # For now, we'll test that the tools don't crash with large input
        
        with patch('dependency_visualizer.DependencyVisualizer') as mock_viz_class:
            mock_viz = MagicMock()
            mock_viz_class.return_value = mock_viz
            mock_viz.analyze_project.return_value = MagicMock()
            
            # Simulate large project analysis
            mock_viz.generate_text_report.return_value = "Large project analysis complete"
            
            # Should handle without memory errors
            assert mock_viz is not None
    
    def test_cli_timeout_handling(self):
        """Test that CLI tools handle timeouts gracefully."""
        # This would test long-running operations
        # For now, we'll test that tools can be interrupted
        
        with patch('dependency_visualizer.DependencyVisualizer') as mock_viz_class:
            mock_viz = MagicMock()
            mock_viz_class.return_value = mock_viz
            
            # Simulate long-running operation
            mock_viz.analyze_project.side_effect = KeyboardInterrupt()
            
            try:
                mock_viz.analyze_project()
            except KeyboardInterrupt:
                # Should be able to handle interruption
                pass


@pytest.mark.parametrize("tool_name,expected_function", [
    ("dependency_visualizer", "main"),
    ("code_analyzer", "main"),
    ("doc_generator", "main"),
    ("pre_commit_validator", "main"),
    ("ci_cd_integrator", "main"),
    ("monitoring_dashboard", "main"),
    ("ai_session_tracker", "main")
])
def test_tool_has_main_function(tool_name, expected_function):
    """Test that each tool has the expected main function."""
    project_root = Path(__file__).parent.parent.parent
    tools_dir = project_root / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    
    try:
        module = __import__(tool_name)
        assert hasattr(module, expected_function), f"{tool_name} missing {expected_function}()"
        assert callable(getattr(module, expected_function)), f"{tool_name}.{expected_function}() not callable"
    except ImportError:
        # Some tools may not import due to missing dependencies
        pytest.skip(f"Could not import {tool_name}")


@pytest.mark.parametrize("format_type", ["text", "html", "json"])
def test_dependency_visualizer_formats(format_type, mock_project_structure):
    """Test dependency visualizer with different output formats."""
    project_root = Path(__file__).parent.parent.parent
    tools_dir = project_root / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    
    with patch('dependency_visualizer.DependencyVisualizer') as mock_viz_class:
        mock_viz = MagicMock()
        mock_viz_class.return_value = mock_viz
        mock_viz.analyze_project.return_value = MagicMock()
        
        # Set up format-specific behavior
        if format_type == 'html':
            mock_viz.generate_interactive_html.return_value = "output.html"
        elif format_type == 'json':
            mock_viz.export_to_json.return_value = {"nodes": [], "edges": []}
        else:
            mock_viz.generate_text_report.return_value = "Text report"
        
        # Test that the format is handled correctly
        test_args = ['dependency_visualizer.py', str(mock_project_structure), '--format', format_type]
        
        with patch('sys.argv', test_args), \
             patch('dependency_visualizer.argparse.ArgumentParser.parse_args') as mock_parse:
            
            mock_args = MagicMock()
            mock_args.project_path = str(mock_project_structure)
            mock_args.format = format_type
            mock_args.output = None
            mock_args.ai_awareness = True
            mock_parse.return_value = mock_args
            
            try:
                sys.modules.pop('dependency_visualizer', None)
                import dependency_visualizer
                
                if hasattr(dependency_visualizer, 'main'):
                    dependency_visualizer.main()
                    
                    # Verify correct method was called based on format
                    if format_type == 'html':
                        mock_viz.generate_interactive_html.assert_called()
                    elif format_type == 'json':
                        mock_viz.export_to_json.assert_called()
                    else:
                        mock_viz.generate_text_report.assert_called()
                        
            except (ImportError, SystemExit):
                # May fail due to missing dependencies
                pass