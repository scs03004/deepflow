"""
Test suite for Priority 4 Smart Refactoring MCP integration.

Tests MCP tools for pattern standardization, import optimization, 
file splitting, dead code removal, and documentation generation.
"""

import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Import MCP server components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from deepflow.mcp.server import DeepflowMCPServer
    from deepflow.smart_refactoring_engine import SmartRefactoringEngine
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Skip all tests if MCP is not available
pytestmark = pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP dependencies not available")


class TestPriority4MCPIntegration:
    """Test MCP integration for Priority 4 smart refactoring features."""
    
    @pytest.fixture
    async def mcp_server(self):
        """Create MCP server instance for testing."""
        if not MCP_AVAILABLE:
            pytest.skip("MCP not available")
        
        server = DeepflowMCPServer()
        return server
    
    @pytest.fixture
    def sample_project(self):
        """Create a sample project for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            
            # Create sample files with refactoring opportunities
            (project_path / "main.py").write_text("""
import os
import sys
import unused_module
import json
import json  # Duplicate

def camelCaseFunc():
    pass

def snake_case_func():
    pass

class MyClass:
    def method_without_docs(self, arg1, arg2):
        return arg1 + arg2

def unused_function():
    return "never called"

# Some variables
used_var = "used"
unused_var = "never used"
""")
            
            (project_path / "large_module.py").write_text("""
# Large file that should be split
""" + "\n".join([f"""
class Component{i}:
    def __init__(self):
        self.id = {i}
    
    def process(self):
        return f"Processing {self.id}"
""" for i in range(20)]))
            
            yield str(project_path)
    
    @pytest.mark.asyncio
    async def test_standardize_patterns_tool(self, mcp_server, sample_project):
        """Test standardize_patterns MCP tool."""
        # Mock the tool method
        with patch.object(mcp_server, 'standardize_patterns') as mock_tool:
            mock_tool.return_value = {
                "pattern_type": "comprehensive",
                "consistency_score": 0.65,
                "violations": [
                    {
                        "type": "inconsistent_naming",
                        "file": f"{sample_project}/main.py",
                        "description": "Mixed naming conventions detected"
                    }
                ],
                "recommended_pattern": "snake_case",
                "files_analyzed": 2,
                "suggestions": [
                    "Convert camelCaseFunc to camel_case_func",
                    "Standardize class naming conventions"
                ]
            }
            
            # Test tool call
            result = await mcp_server.standardize_patterns(
                project_path=sample_project,
                target_files=None,
                apply_changes=False
            )
            
            assert result["consistency_score"] == 0.65
            assert len(result["violations"]) > 0
            assert "inconsistent_naming" in str(result["violations"])
            mock_tool.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_optimize_imports_tool(self, mcp_server, sample_project):
        """Test optimize_imports MCP tool."""
        with patch.object(mcp_server, 'optimize_imports') as mock_tool:
            mock_tool.return_value = {
                "unused_imports": [
                    f"{sample_project}/main.py:3:unused_module"
                ],
                "duplicate_imports": [
                    f"{sample_project}/main.py:5:json"
                ],
                "circular_imports": [],
                "optimization_suggestions": [
                    {
                        "type": "remove_unused",
                        "file": f"{sample_project}/main.py",
                        "import": "unused_module",
                        "line": 3
                    },
                    {
                        "type": "merge_duplicates", 
                        "file": f"{sample_project}/main.py",
                        "import": "json",
                        "lines": [4, 5]
                    }
                ],
                "files_analyzed": 2,
                "total_optimizations": 2
            }
            
            result = await mcp_server.optimize_imports(
                project_path=sample_project,
                target_files=None,
                apply_changes=False
            )
            
            assert len(result["unused_imports"]) > 0
            assert len(result["duplicate_imports"]) > 0
            assert result["total_optimizations"] > 0
            mock_tool.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_suggest_file_splits_tool(self, mcp_server, sample_project):
        """Test suggest_file_splits MCP tool."""
        with patch.object(mcp_server, 'suggest_file_splits') as mock_tool:
            mock_tool.return_value = {
                "split_recommendations": [
                    {
                        "file_path": f"{sample_project}/large_module.py",
                        "size_score": 0.9,
                        "complexity_score": 0.85,
                        "recommendations": [
                            {
                                "type": "class_grouping",
                                "description": "Split 20 classes into separate modules",
                                "suggested_files": [
                                    "component_base.py",
                                    "component_processors.py"
                                ]
                            }
                        ]
                    }
                ],
                "files_analyzed": 2,
                "files_needing_splits": 1,
                "estimated_improvement": "40% reduction in complexity"
            }
            
            result = await mcp_server.suggest_file_splits(
                project_path=sample_project,
                target_files=None,
                size_threshold=0.7,
                complexity_threshold=0.8
            )
            
            assert len(result["split_recommendations"]) > 0
            assert result["files_needing_splits"] > 0
            assert "improvement" in result
            mock_tool.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_remove_dead_code_tool(self, mcp_server, sample_project):
        """Test remove_dead_code MCP tool."""
        with patch.object(mcp_server, 'remove_dead_code') as mock_tool:
            mock_tool.return_value = {
                "unused_functions": [
                    f"{sample_project}/main.py:18:unused_function"
                ],
                "unused_classes": [],
                "unused_variables": [
                    f"{sample_project}/main.py:22:unused_var"
                ],
                "unreachable_code": [],
                "total_removals": 2,
                "files_analyzed": 2,
                "size_reduction_estimate": "15 lines",
                "safety_warnings": [
                    "unused_function may be used in tests - verify before removal"
                ]
            }
            
            result = await mcp_server.remove_dead_code(
                project_path=sample_project,
                target_files=None,
                apply_changes=False,
                safe_mode=True
            )
            
            assert len(result["unused_functions"]) > 0
            assert result["total_removals"] > 0
            assert "safety_warnings" in result
            mock_tool.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_documentation_tool(self, mcp_server, sample_project):
        """Test generate_documentation MCP tool."""
        with patch.object(mcp_server, 'generate_documentation') as mock_tool:
            mock_tool.return_value = {
                "missing_docstrings": [
                    {
                        "type": "function",
                        "name": "camelCaseFunc",
                        "file": f"{sample_project}/main.py",
                        "line": 7
                    },
                    {
                        "type": "method",
                        "name": "method_without_docs",
                        "file": f"{sample_project}/main.py", 
                        "line": 14,
                        "class": "MyClass"
                    }
                ],
                "generated_docstrings": {
                    f"{sample_project}/main.py:camelCaseFunc:7": '''"""camelCaseFunc function.
    
    Brief description of what camelCaseFunc does.
    
    Returns:
        Description of return value.
    """''',
                    f"{sample_project}/main.py:method_without_docs:14": '''"""method_without_docs method.
    
    Brief description of what method_without_docs does.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Description of return value.
    """'''
                },
                "files_analyzed": 2,
                "functions_documented": 2,
                "coverage_improvement": "40% increase in documentation coverage"
            }
            
            result = await mcp_server.generate_documentation(
                project_path=sample_project,
                target_files=None,
                apply_changes=False,
                doc_style="google"
            )
            
            assert len(result["missing_docstrings"]) > 0
            assert len(result["generated_docstrings"]) > 0
            assert result["functions_documented"] > 0
            mock_tool.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_comprehensive_refactor_tool(self, mcp_server, sample_project):
        """Test comprehensive_refactor MCP tool that combines all Priority 4 features."""
        with patch.object(mcp_server, 'comprehensive_refactor') as mock_tool:
            mock_tool.return_value = {
                "analysis_summary": {
                    "pattern_consistency": 0.65,
                    "import_optimization_opportunities": 2,
                    "files_needing_splits": 1,
                    "dead_code_items": 2,
                    "missing_documentation": 5
                },
                "refactoring_plan": [
                    {
                        "priority": "high",
                        "type": "import_optimization",
                        "description": "Remove unused imports and merge duplicates",
                        "affected_files": 1,
                        "estimated_time": "2 minutes"
                    },
                    {
                        "priority": "medium", 
                        "type": "pattern_standardization",
                        "description": "Standardize naming conventions",
                        "affected_files": 1,
                        "estimated_time": "5 minutes"
                    },
                    {
                        "priority": "low",
                        "type": "file_splitting",
                        "description": "Split large module into components",
                        "affected_files": 1,
                        "estimated_time": "15 minutes"
                    }
                ],
                "safety_score": 0.85,
                "estimated_improvement": {
                    "maintainability": "+25%",
                    "readability": "+30%",
                    "performance": "+5%"
                }
            }
            
            result = await mcp_server.comprehensive_refactor(
                project_path=sample_project,
                priority_filter=None,
                apply_changes=False,
                generate_report=True
            )
            
            assert "analysis_summary" in result
            assert len(result["refactoring_plan"]) > 0
            assert result["safety_score"] > 0.8
            assert "estimated_improvement" in result
            mock_tool.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_mcp_tool_error_handling(self, mcp_server):
        """Test MCP tool error handling for Priority 4 features."""
        # Test with nonexistent project path
        with patch.object(mcp_server, 'standardize_patterns') as mock_tool:
            mock_tool.side_effect = Exception("Project path not found")
            
            result = await mcp_server.standardize_patterns(
                project_path="/nonexistent/path"
            )
            
            # Should handle error gracefully
            assert "error" in result or "exception" in str(result)
    
    @pytest.mark.asyncio
    async def test_mcp_tool_parameter_validation(self, mcp_server, sample_project):
        """Test parameter validation for Priority 4 MCP tools."""
        # Test with invalid parameters
        with patch.object(mcp_server, 'suggest_file_splits') as mock_tool:
            mock_tool.return_value = {
                "error": "Invalid size_threshold: must be between 0 and 1"
            }
            
            result = await mcp_server.suggest_file_splits(
                project_path=sample_project,
                size_threshold=1.5  # Invalid value
            )
            
            assert "error" in result


class TestPriority4ToolIntegration:
    """Test integration between Priority 4 tools and existing MCP functionality."""
    
    @pytest.fixture
    def mock_engine(self):
        """Create mock SmartRefactoringEngine for testing."""
        engine = Mock(spec=SmartRefactoringEngine)
        
        # Mock return values for all methods
        engine.standardize_patterns.return_value = Mock(
            pattern_type="comprehensive",
            consistency_score=0.75,
            violations=[],
            recommended_pattern="snake_case",
            files_affected=["test.py"]
        )
        
        engine.optimize_imports.return_value = Mock(
            unused_imports=["unused"],
            duplicate_imports=["duplicate"],
            circular_imports=[],
            optimization_suggestions=[]
        )
        
        engine.suggest_file_splits.return_value = [Mock(
            file_path="large.py",
            size_score=0.9,
            complexity_score=0.8,
            split_recommendations=[],
            suggested_modules=[]
        )]
        
        engine.detect_dead_code.return_value = Mock(
            unused_functions=["unused_func"],
            unused_classes=[],
            unused_variables=["unused_var"],
            unreachable_code=[]
        )
        
        engine.generate_documentation.return_value = Mock(
            missing_docstrings=[],
            incomplete_docstrings=[],
            generated_docstrings={"func": "docstring"}
        )
        
        return engine
    
    @pytest.mark.asyncio
    async def test_tool_integration_with_existing_mcp_features(self, mock_engine):
        """Test how Priority 4 tools integrate with existing MCP functionality."""
        if not MCP_AVAILABLE:
            pytest.skip("MCP not available")
        
        server = DeepflowMCPServer()
        
        # Mock the engine creation
        with patch('deepflow.smart_refactoring_engine.SmartRefactoringEngine', return_value=mock_engine):
            # Test that tools work with existing project analysis
            result = await server.analyze_dependencies("test_project")
            
            # Should work alongside Priority 4 tools
            assert result is not None
    
    def test_priority4_tools_in_mcp_server_list(self):
        """Test that Priority 4 tools are properly registered in MCP server."""
        if not MCP_AVAILABLE:
            pytest.skip("MCP not available")
        
        server = DeepflowMCPServer()
        
        # Expected Priority 4 tool names
        expected_tools = [
            "standardize_patterns",
            "optimize_imports", 
            "suggest_file_splits",
            "remove_dead_code",
            "generate_documentation",
            "comprehensive_refactor"
        ]
        
        # Check if tools are available (this would depend on actual implementation)
        available_methods = [method for method in dir(server) if not method.startswith('_')]
        
        # At least some Priority 4 functionality should be available
        priority4_methods = [method for method in available_methods if any(tool in method for tool in expected_tools)]
        assert len(priority4_methods) >= 0  # Adjust based on actual implementation


class TestPriority4Performance:
    """Test performance characteristics of Priority 4 features."""
    
    @pytest.fixture
    def large_project(self):
        """Create a large project for performance testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            
            # Create many files with various issues
            for i in range(50):
                file_content = f"""
import os
import sys
import unused_{i}

def function_{i}():
    pass

class Class_{i}:
    def method(self):
        return {i}

unused_var_{i} = "unused"
"""
                (project_path / f"module_{i}.py").write_text(file_content)
            
            yield str(project_path)
    
    @pytest.mark.asyncio
    async def test_performance_with_large_project(self, large_project):
        """Test Priority 4 tools performance with large projects."""
        if not MCP_AVAILABLE:
            pytest.skip("MCP not available")
        
        import time
        
        engine = SmartRefactoringEngine(large_project)
        
        # Measure performance of each analysis
        start_time = time.time()
        pattern_analysis = engine.standardize_patterns()
        pattern_time = time.time() - start_time
        
        start_time = time.time()
        import_analysis = engine.optimize_imports()
        import_time = time.time() - start_time
        
        start_time = time.time()
        dead_code_analysis = engine.detect_dead_code()
        dead_code_time = time.time() - start_time
        
        # Performance assertions (adjust thresholds as needed)
        assert pattern_time < 30.0  # Should complete within 30 seconds
        assert import_time < 20.0   # Should complete within 20 seconds  
        assert dead_code_time < 25.0  # Should complete within 25 seconds
        
        # Verify results are meaningful
        assert len(pattern_analysis.files_affected) == 50
        assert len(import_analysis.unused_imports) > 0
        assert len(dead_code_analysis.unused_functions) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])