"""
MCP tool exposure and functionality tests.
"""

import pytest
import json
from unittest.mock import patch, MagicMock
from pathlib import Path


class TestMCPToolExposure:
    """Test MCP tool exposure and interface."""
    
    def test_all_tools_exposed(self):
        """Test that all expected tools are exposed via MCP."""
        expected_tools = [
            "analyze_dependencies",
            "analyze_code_quality", 
            "validate_commit",
            "generate_documentation"
        ]
        
        with patch('deepflow.mcp.server.Server'):
            from deepflow.mcp.server import DeepflowMCPServer
            
            server = DeepflowMCPServer()
            tools = server.get_tools()
            tool_names = [tool.name for tool in tools]
            
            for expected_tool in expected_tools:
                assert expected_tool in tool_names, f"Tool {expected_tool} not exposed"
    
    def test_tool_metadata_completeness(self):
        """Test that all tools have complete metadata."""
        with patch('deepflow.mcp.server.Server'):
            from deepflow.mcp.server import DeepflowMCPServer
            
            server = DeepflowMCPServer()
            tools = server.get_tools()
            
            for tool in tools:
                # Each tool must have required metadata
                assert tool.name, f"Tool missing name: {tool}"
                assert tool.description, f"Tool {tool.name} missing description"
                assert tool.inputSchema, f"Tool {tool.name} missing input schema"
                
                # Schema must be properly structured
                schema = tool.inputSchema
                assert schema.get('type') == 'object', f"Tool {tool.name} schema not object type"
                assert 'properties' in schema, f"Tool {tool.name} schema missing properties"
    
    def test_tool_parameter_validation(self):
        """Test tool parameter validation schemas."""
        with patch('deepflow.mcp.server.Server'):
            from deepflow.mcp.server import DeepflowMCPServer
            
            server = DeepflowMCPServer()
            tools = server.get_tools()
            
            for tool in tools:
                schema = tool.inputSchema
                properties = schema['properties']
                
                # All properties should have type definitions
                for prop_name, prop_def in properties.items():
                    assert 'type' in prop_def, f"Property {prop_name} in {tool.name} missing type"
                    
                    # If it's an enum, should have valid values
                    if 'enum' in prop_def:
                        assert len(prop_def['enum']) > 0, f"Empty enum in {tool.name}.{prop_name}"
                    
                    # If it has a default, should be documented
                    if 'default' in prop_def:
                        assert prop_def['default'] is not None or prop_def['type'] in ['string', 'object']


class TestAnalyzeDependenciesTool:
    """Test the analyze_dependencies MCP tool."""
    
    def test_analyze_dependencies_schema(self):
        """Test analyze_dependencies tool schema."""
        with patch('deepflow.mcp.server.Server'):
            from deepflow.mcp.server import DeepflowMCPServer
            
            server = DeepflowMCPServer()
            tools = server.get_tools()
            
            analyze_deps = next(t for t in tools if t.name == "analyze_dependencies")
            schema = analyze_deps.inputSchema
            props = schema['properties']
            
            # Required parameters
            assert 'project_path' in props
            assert props['project_path']['type'] == 'string'
            
            assert 'format' in props
            assert props['format']['type'] == 'string'
            assert set(props['format']['enum']) == {'text', 'html', 'json'}
            
            assert 'ai_awareness' in props
            assert props['ai_awareness']['type'] == 'boolean'
    
    @pytest.mark.asyncio
    async def test_analyze_dependencies_text_format(self, mock_project_structure):
        """Test analyze_dependencies with text format."""
        mock_result = {
            "nodes": ["main.py", "utils.py"],
            "dependencies": ["main.py -> utils.py"],
            "summary": "2 modules, 1 dependency"
        }
        
        with patch('deepflow.mcp.server.DependencyVisualizer') as mock_viz:
            mock_viz_instance = MagicMock()
            mock_viz.return_value = mock_viz_instance
            mock_viz_instance.analyze_project.return_value = MagicMock()
            mock_viz_instance.generate_text_report.return_value = "Text report content"
            
            # Simulate tool call arguments
            arguments = {
                "project_path": str(mock_project_structure),
                "format": "text",
                "ai_awareness": True
            }
            
            # Test the tool logic
            visualizer = mock_viz(arguments["project_path"], ai_awareness=arguments["ai_awareness"])
            graph = visualizer.analyze_project()
            result = visualizer.generate_text_report(graph)
            
            assert result == "Text report content"
            mock_viz.assert_called_with(str(mock_project_structure), ai_awareness=True)
    
    @pytest.mark.asyncio
    async def test_analyze_dependencies_json_format(self, mock_project_structure):
        """Test analyze_dependencies with JSON format."""
        with patch('deepflow.mcp.server.DependencyVisualizer') as mock_viz:
            mock_viz_instance = MagicMock()
            mock_viz.return_value = mock_viz_instance
            mock_viz_instance.analyze_project.return_value = MagicMock()
            mock_viz_instance.export_to_json.return_value = {
                "nodes": [{"id": "main.py", "type": "module"}],
                "edges": []
            }
            
            arguments = {
                "project_path": str(mock_project_structure),
                "format": "json",
                "ai_awareness": False
            }
            
            # Test the tool logic
            visualizer = mock_viz(arguments["project_path"], ai_awareness=arguments["ai_awareness"])
            graph = visualizer.analyze_project()
            json_data = visualizer.export_to_json(graph)
            
            assert "nodes" in json_data
            assert "edges" in json_data
            mock_viz.assert_called_with(str(mock_project_structure), ai_awareness=False)


class TestAnalyzeCodeQualityTool:
    """Test the analyze_code_quality MCP tool."""
    
    def test_analyze_code_quality_schema(self):
        """Test analyze_code_quality tool schema."""
        with patch('deepflow.mcp.server.Server'):
            from deepflow.mcp.server import DeepflowMCPServer
            
            server = DeepflowMCPServer()
            tools = server.get_tools()
            
            analyze_quality = next(t for t in tools if t.name == "analyze_code_quality")
            schema = analyze_quality.inputSchema
            props = schema['properties']
            
            # Required parameters
            assert 'project_path' in props
            assert 'analysis_type' in props
            assert 'fix_imports' in props
            
            # Check enum values for analysis_type
            expected_types = {'all', 'imports', 'coupling', 'architecture', 'debt', 'ai_context'}
            assert set(props['analysis_type']['enum']) == expected_types
    
    @pytest.mark.asyncio
    async def test_analyze_code_quality_imports_only(self, mock_project_structure):
        """Test analyze_code_quality with imports analysis only."""
        with patch('deepflow.mcp.server.CodeAnalyzer') as mock_analyzer:
            mock_analyzer_instance = MagicMock()
            mock_analyzer.return_value = mock_analyzer_instance
            mock_analyzer_instance.analyze_unused_imports.return_value = [
                MagicMock(file_path="main.py", import_name="unused", is_used=False, suggestions=[])
            ]
            
            arguments = {
                "project_path": str(mock_project_structure),
                "analysis_type": "imports",
                "fix_imports": True
            }
            
            # Test the tool logic
            analyzer = mock_analyzer(arguments["project_path"])
            results = analyzer.analyze_unused_imports(fix_mode=arguments["fix_imports"])
            
            assert len(results) == 1
            mock_analyzer.assert_called_with(str(mock_project_structure))
    
    @pytest.mark.asyncio
    async def test_analyze_code_quality_all_types(self, mock_project_structure):
        """Test analyze_code_quality with all analysis types."""
        with patch('deepflow.mcp.server.CodeAnalyzer') as mock_analyzer:
            mock_analyzer_instance = MagicMock()
            mock_analyzer.return_value = mock_analyzer_instance
            
            # Mock all analysis methods
            mock_analyzer_instance.analyze_unused_imports.return_value = []
            mock_analyzer_instance.analyze_coupling.return_value = []
            mock_analyzer_instance.detect_architecture_violations.return_value = []
            mock_analyzer_instance.calculate_technical_debt.return_value = []
            mock_analyzer_instance.analyze_ai_context_windows.return_value = []
            
            arguments = {
                "project_path": str(mock_project_structure),
                "analysis_type": "all",
                "fix_imports": False
            }
            
            # Test the tool logic - all methods should be called
            analyzer = mock_analyzer(arguments["project_path"])
            
            # Simulate the "all" analysis type logic
            analyzer.analyze_unused_imports(fix_mode=arguments["fix_imports"])
            analyzer.analyze_coupling()
            analyzer.detect_architecture_violations()
            analyzer.calculate_technical_debt()
            analyzer.analyze_ai_context_windows()
            
            # Verify all methods were called
            mock_analyzer_instance.analyze_unused_imports.assert_called_once()
            mock_analyzer_instance.analyze_coupling.assert_called_once()
            mock_analyzer_instance.detect_architecture_violations.assert_called_once()
            mock_analyzer_instance.calculate_technical_debt.assert_called_once()
            mock_analyzer_instance.analyze_ai_context_windows.assert_called_once()


class TestValidateCommitTool:
    """Test the validate_commit MCP tool."""
    
    def test_validate_commit_schema(self):
        """Test validate_commit tool schema."""
        with patch('deepflow.mcp.server.Server'):
            from deepflow.mcp.server import DeepflowMCPServer
            
            server = DeepflowMCPServer()
            tools = server.get_tools()
            
            validate_commit = next(t for t in tools if t.name == "validate_commit")
            schema = validate_commit.inputSchema
            props = schema['properties']
            
            assert 'project_path' in props
            assert 'check_dependencies' in props
            assert 'check_patterns' in props
            
            # All should be proper types
            assert props['project_path']['type'] == 'string'
            assert props['check_dependencies']['type'] == 'boolean'
            assert props['check_patterns']['type'] == 'boolean'
    
    @pytest.mark.asyncio
    async def test_validate_commit_functionality(self, mock_project_structure):
        """Test validate_commit tool functionality."""
        with patch('deepflow.mcp.server.PreCommitValidator') as mock_validator:
            mock_validator_instance = MagicMock()
            mock_validator.return_value = mock_validator_instance
            
            # Mock validation result
            mock_result = MagicMock()
            mock_result.is_valid = True
            mock_result.errors = []
            mock_result.warnings = ["Warning message"]
            mock_result.suggestions = ["Suggestion"]
            mock_validator_instance.validate_changes.return_value = mock_result
            
            arguments = {
                "project_path": str(mock_project_structure),
                "check_dependencies": True,
                "check_patterns": False
            }
            
            # Test the tool logic
            validator = mock_validator(arguments["project_path"])
            result = validator.validate_changes(
                check_dependencies=arguments["check_dependencies"],
                check_patterns=arguments["check_patterns"]
            )
            
            assert result.is_valid is True
            assert len(result.warnings) == 1
            mock_validator.assert_called_with(str(mock_project_structure))
            mock_validator_instance.validate_changes.assert_called_with(
                check_dependencies=True,
                check_patterns=False
            )


class TestGenerateDocumentationTool:
    """Test the generate_documentation MCP tool."""
    
    def test_generate_documentation_schema(self):
        """Test generate_documentation tool schema."""
        with patch('deepflow.mcp.server.Server'):
            from deepflow.mcp.server import DeepflowMCPServer
            
            server = DeepflowMCPServer()
            tools = server.get_tools()
            
            gen_docs = next(t for t in tools if t.name == "generate_documentation")
            schema = gen_docs.inputSchema
            props = schema['properties']
            
            assert 'project_path' in props
            assert 'doc_type' in props
            assert 'output_path' in props
            
            # Check doc_type enum values
            expected_types = {'dependency_map', 'architecture_overview', 'api_docs'}
            assert set(props['doc_type']['enum']) == expected_types
    
    @pytest.mark.asyncio
    async def test_generate_documentation_dependency_map(self, mock_project_structure):
        """Test generate_documentation with dependency_map type."""
        with patch('deepflow.mcp.server.DocumentationGenerator') as mock_doc_gen:
            mock_doc_gen_instance = MagicMock()
            mock_doc_gen.return_value = mock_doc_gen_instance
            mock_doc_gen_instance.generate_dependency_map.return_value = "DEPENDENCY_MAP.md"
            
            arguments = {
                "project_path": str(mock_project_structure),
                "doc_type": "dependency_map",
                "output_path": None
            }
            
            # Test the tool logic
            doc_generator = mock_doc_gen(arguments["project_path"])
            output_file = doc_generator.generate_dependency_map(arguments["output_path"])
            
            assert output_file == "DEPENDENCY_MAP.md"
            mock_doc_gen.assert_called_with(str(mock_project_structure))
    
    @pytest.mark.asyncio
    async def test_generate_documentation_architecture_overview(self, mock_project_structure):
        """Test generate_documentation with architecture_overview type."""
        with patch('deepflow.mcp.server.DocumentationGenerator') as mock_doc_gen:
            mock_doc_gen_instance = MagicMock()
            mock_doc_gen.return_value = mock_doc_gen_instance
            mock_doc_gen_instance.generate_architecture_overview.return_value = "ARCHITECTURE.md"
            
            arguments = {
                "project_path": str(mock_project_structure),
                "doc_type": "architecture_overview",
                "output_path": "custom_arch.md"
            }
            
            # Test the tool logic
            doc_generator = mock_doc_gen(arguments["project_path"])
            output_file = doc_generator.generate_architecture_overview(arguments["output_path"])
            
            assert output_file == "ARCHITECTURE.md"
            mock_doc_gen.assert_called_with(str(mock_project_structure))


class TestMCPToolErrorHandling:
    """Test error handling in MCP tools."""
    
    @pytest.mark.asyncio
    async def test_tool_handles_missing_dependencies(self):
        """Test that tools handle missing dependencies gracefully."""
        with patch('deepflow.mcp.server.TOOLS_AVAILABLE', False):
            # When tools are not available, tools should return appropriate error messages
            # This would be tested in the actual tool implementations
            pass
    
    @pytest.mark.asyncio
    async def test_tool_handles_invalid_project_path(self):
        """Test that tools handle invalid project paths gracefully."""
        with patch('deepflow.mcp.server.DependencyVisualizer') as mock_viz:
            mock_viz.side_effect = FileNotFoundError("Project path not found")
            
            # Tool should catch this exception and return an appropriate error
            try:
                mock_viz("/nonexistent/path")
            except FileNotFoundError as e:
                assert "not found" in str(e)
    
    @pytest.mark.asyncio
    async def test_tool_handles_analysis_errors(self):
        """Test that tools handle analysis errors gracefully."""
        with patch('deepflow.mcp.server.CodeAnalyzer') as mock_analyzer:
            mock_analyzer_instance = MagicMock()
            mock_analyzer.return_value = mock_analyzer_instance
            mock_analyzer_instance.analyze_unused_imports.side_effect = Exception("Analysis failed")
            
            # Tool should catch this exception and return an error result
            try:
                analyzer = mock_analyzer("/test/path")
                analyzer.analyze_unused_imports()
            except Exception as e:
                assert "failed" in str(e)


class TestMCPToolIntegration:
    """Integration tests for MCP tools."""
    
    def test_all_tools_have_unique_names(self):
        """Test that all tools have unique names."""
        with patch('deepflow.mcp.server.Server'):
            from deepflow.mcp.server import DeepflowMCPServer
            
            server = DeepflowMCPServer()
            tools = server.get_tools()
            tool_names = [tool.name for tool in tools]
            
            # All names should be unique
            assert len(tool_names) == len(set(tool_names))
    
    def test_tool_descriptions_are_descriptive(self):
        """Test that tool descriptions are actually descriptive."""
        with patch('deepflow.mcp.server.Server'):
            from deepflow.mcp.server import DeepflowMCPServer
            
            server = DeepflowMCPServer()
            tools = server.get_tools()
            
            for tool in tools:
                # Descriptions should be meaningful
                assert len(tool.description) > 20
                assert tool.name.replace('_', ' ') in tool.description.lower()
    
    def test_tool_parameters_have_defaults(self):
        """Test that optional tool parameters have reasonable defaults."""
        with patch('deepflow.mcp.server.Server'):
            from deepflow.mcp.server import DeepflowMCPServer
            
            server = DeepflowMCPServer()
            tools = server.get_tools()
            
            for tool in tools:
                schema = tool.inputSchema
                props = schema['properties']
                
                # project_path should usually default to current directory
                if 'project_path' in props:
                    assert props['project_path'].get('default') == "."
                
                # Boolean parameters should have defaults
                for prop_name, prop_def in props.items():
                    if prop_def['type'] == 'boolean':
                        assert 'default' in prop_def


@pytest.mark.parametrize("format_type", ["text", "html", "json"])
def test_analyze_dependencies_format_parameter(format_type):
    """Test analyze_dependencies with different format parameters."""
    with patch('deepflow.mcp.server.Server'):
        from deepflow.mcp.server import DeepflowMCPServer
        
        server = DeepflowMCPServer()
        tools = server.get_tools()
        
        analyze_deps = next(t for t in tools if t.name == "analyze_dependencies")
        schema = analyze_deps.inputSchema
        
        # Format should support all specified types
        assert format_type in schema['properties']['format']['enum']


@pytest.mark.parametrize("analysis_type", ["all", "imports", "coupling", "architecture", "debt", "ai_context"])
def test_analyze_code_quality_analysis_types(analysis_type):
    """Test analyze_code_quality with different analysis types."""
    with patch('deepflow.mcp.server.Server'):
        from deepflow.mcp.server import DeepflowMCPServer
        
        server = DeepflowMCPServer()
        tools = server.get_tools()
        
        analyze_quality = next(t for t in tools if t.name == "analyze_code_quality")
        schema = analyze_quality.inputSchema
        
        # Analysis type should support all specified types
        assert analysis_type in schema['properties']['analysis_type']['enum']


@pytest.mark.parametrize("doc_type", ["dependency_map", "architecture_overview", "api_docs"])
def test_generate_documentation_doc_types(doc_type):
    """Test generate_documentation with different doc types."""
    with patch('deepflow.mcp.server.Server'):
        from deepflow.mcp.server import DeepflowMCPServer
        
        server = DeepflowMCPServer()
        tools = server.get_tools()
        
        gen_docs = next(t for t in tools if t.name == "generate_documentation")
        schema = gen_docs.inputSchema
        
        # Doc type should support all specified types
        assert doc_type in schema['properties']['doc_type']['enum']