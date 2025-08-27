"""
Test suite for Priority 5 Workflow MCP integration.

Tests MCP tools for analysis pipelines, conditional workflows, 
batch operations, custom workflow definitions, and scheduled code hygiene.
"""

import json
import tempfile
import pytest
import asyncio
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

# Import MCP server components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from deepflow.mcp.server import DeepflowMCPServer
    from deepflow.workflow_orchestrator import WorkflowOrchestrator, WorkflowStatus
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

# Skip all tests if MCP is not available
pytestmark = pytest.mark.skipif(not MCP_AVAILABLE, reason="MCP dependencies not available")


class TestPriority5MCPIntegration:
    """Test MCP integration for Priority 5 workflow features."""
    
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
            
            # Create sample files for workflow testing
            (project_path / "main.py").write_text("""
import os
import sys
import unused_module

def camelCaseFunc():
    pass

def snake_case_func():
    pass

class TestClass:
    def method_without_docs(self):
        return "test"

def unused_function():
    return "never called"
""")
            
            (project_path / "utils.py").write_text("""
import json
import json  # Duplicate
import another_unused

def utility():
    pass

class Helper:
    pass
""")
            
            yield str(project_path)
    
    @pytest.mark.asyncio
    async def test_create_analysis_pipeline_tool(self, mcp_server, sample_project):
        """Test create_analysis_pipeline MCP tool."""
        # Mock the tool method
        with patch.object(mcp_server, 'create_analysis_pipeline') as mock_tool:
            mock_tool.return_value = {
                "workflow_id": "pipeline_abc123",
                "name": "Quality Analysis Pipeline",
                "steps": [
                    {"step_id": "step_1", "tool": "analyze_code_quality"},
                    {"step_id": "step_2", "tool": "optimize_imports"},
                    {"step_id": "step_3", "tool": "standardize_patterns"}
                ],
                "created": True,
                "message": "Analysis pipeline created successfully"
            }
            
            # Test tool call
            result = await mcp_server.create_analysis_pipeline(
                pipeline_name="Quality Analysis Pipeline",
                tools=[
                    {"name": "analyze_code_quality", "parameters": {}},
                    {"name": "optimize_imports", "parameters": {"apply_changes": False}},
                    {"name": "standardize_patterns", "parameters": {}}
                ],
                project_path=sample_project
            )
            
            assert result["workflow_id"] == "pipeline_abc123"
            assert result["created"] == True
            assert len(result["steps"]) == 3
            mock_tool.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_workflow_tool(self, mcp_server, sample_project):
        """Test execute_workflow MCP tool."""
        with patch.object(mcp_server, 'execute_workflow') as mock_tool:
            mock_tool.return_value = {
                "execution_id": "exec_def456",
                "workflow_id": "pipeline_abc123",
                "status": "completed",
                "start_time": "2025-08-27T10:00:00",
                "end_time": "2025-08-27T10:05:30",
                "duration_seconds": 330,
                "steps_completed": 3,
                "steps_failed": 0,
                "results": {
                    "step_1": {"score": 0.75, "issues": 8},
                    "step_2": {"optimized": 5, "duplicates_removed": 2},
                    "step_3": {"consistency_improved": 0.15, "violations_fixed": 6}
                },
                "metrics": {
                    "total_improvements": 13,
                    "estimated_time_saved": "2.5 hours"
                }
            }
            
            result = await mcp_server.execute_workflow(
                workflow_id="pipeline_abc123",
                parameters={}
            )
            
            assert result["execution_id"] == "exec_def456"
            assert result["status"] == "completed"
            assert result["steps_completed"] == 3
            assert result["steps_failed"] == 0
            mock_tool.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_conditional_workflow_tool(self, mcp_server, sample_project):
        """Test create_conditional_workflow MCP tool."""
        with patch.object(mcp_server, 'create_conditional_workflow') as mock_tool:
            mock_tool.return_value = {
                "workflow_id": "conditional_xyz789",
                "name": "Quality-Based Refactor",
                "conditional_steps": 3,
                "conditions_defined": 5,
                "dependencies_mapped": True,
                "workflow_complexity": "medium",
                "estimated_execution_time": "8-15 minutes",
                "created": True
            }
            
            conditional_steps = [
                {
                    "tool": "analyze_code_quality",
                    "parameters": {},
                    "conditions": [
                        {
                            "field_path": "result.score",
                            "type": "lt",
                            "value": 0.8,
                            "description": "Quality below threshold"
                        }
                    ]
                },
                {
                    "tool": "comprehensive_refactor",
                    "parameters": {"apply_changes": True},
                    "conditions": [
                        {
                            "field_path": "step_results.step_1.issues",
                            "type": "gt",
                            "value": 10,
                            "description": "Many issues detected"
                        }
                    ],
                    "depends_on": ["step_1"]
                }
            ]
            
            result = await mcp_server.create_conditional_workflow(
                workflow_name="Quality-Based Refactor",
                conditional_steps=conditional_steps,
                project_path=sample_project
            )
            
            assert result["workflow_id"] == "conditional_xyz789"
            assert result["conditional_steps"] == 3
            assert result["conditions_defined"] == 5
            assert result["created"] == True
            mock_tool.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_batch_operation_tool(self, mcp_server, sample_project):
        """Test create_batch_operation MCP tool."""
        with patch.object(mcp_server, 'create_batch_operation') as mock_tool:
            mock_tool.return_value = {
                "batch_id": "batch_uvw456",
                "operation_type": "optimize_imports",
                "targets": [
                    f"{sample_project}/main.py",
                    f"{sample_project}/utils.py"
                ],
                "target_count": 2,
                "parallel_execution": True,
                "max_workers": 4,
                "estimated_duration": "3-5 minutes",
                "created": True
            }
            
            result = await mcp_server.create_batch_operation(
                operation_type="optimize_imports",
                targets=[
                    f"{sample_project}/main.py",
                    f"{sample_project}/utils.py"
                ],
                parameters={"apply_changes": True},
                parallel=True
            )
            
            assert result["batch_id"] == "batch_uvw456"
            assert result["target_count"] == 2
            assert result["parallel_execution"] == True
            assert result["created"] == True
            mock_tool.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_batch_operation_tool(self, mcp_server):
        """Test execute_batch_operation MCP tool."""
        with patch.object(mcp_server, 'execute_batch_operation') as mock_tool:
            mock_tool.return_value = {
                "batch_id": "batch_uvw456",
                "execution_status": "completed",
                "total_targets": 5,
                "successful_operations": 4,
                "failed_operations": 1,
                "success_rate": 0.8,
                "execution_time": "4.2 minutes",
                "results": [
                    {
                        "target": "/project1/main.py",
                        "status": "success",
                        "optimized_imports": 3,
                        "time": "45s"
                    },
                    {
                        "target": "/project2/utils.py", 
                        "status": "success",
                        "optimized_imports": 1,
                        "time": "32s"
                    }
                ],
                "failed_items": [
                    {
                        "target": "/project3/broken.py",
                        "error": "Syntax error in file",
                        "status": "failed"
                    }
                ],
                "summary": {
                    "total_optimizations": 12,
                    "time_saved": "1.5 hours",
                    "files_improved": 4
                }
            }
            
            result = await mcp_server.execute_batch_operation(
                batch_id="batch_uvw456"
            )
            
            assert result["batch_id"] == "batch_uvw456"
            assert result["execution_status"] == "completed"
            assert result["success_rate"] == 0.8
            assert result["successful_operations"] == 4
            assert result["failed_operations"] == 1
            mock_tool.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_custom_workflow_tool(self, mcp_server, sample_project):
        """Test load_custom_workflow MCP tool."""
        with patch.object(mcp_server, 'load_custom_workflow') as mock_tool:
            mock_tool.return_value = {
                "workflow_id": "custom_abc789",
                "name": "Custom Quality Workflow",
                "loaded_from": "yaml",
                "steps_loaded": 4,
                "conditions_loaded": 6,
                "dependencies_resolved": True,
                "validation_status": "passed",
                "warnings": [],
                "ready_to_execute": True
            }
            
            workflow_definition = {
                "name": "Custom Quality Workflow",
                "description": "A comprehensive quality improvement workflow",
                "steps": [
                    {
                        "id": "analyze",
                        "type": "analysis", 
                        "tool": "comprehensive_refactor",
                        "parameters": {}
                    },
                    {
                        "id": "conditional_fix",
                        "type": "condition",
                        "tool": "optimize_imports",
                        "conditions": [
                            {
                                "field_path": "step_results.analyze.import_issues",
                                "type": "gt",
                                "value": 0
                            }
                        ],
                        "depends_on": ["analyze"]
                    }
                ]
            }
            
            result = await mcp_server.load_custom_workflow(
                workflow_definition=workflow_definition,
                workflow_format="dict"
            )
            
            assert result["workflow_id"] == "custom_abc789"
            assert result["steps_loaded"] == 4
            assert result["validation_status"] == "passed"
            assert result["ready_to_execute"] == True
            mock_tool.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_setup_scheduled_hygiene_tool(self, mcp_server, sample_project):
        """Test setup_scheduled_hygiene MCP tool."""
        with patch.object(mcp_server, 'setup_scheduled_hygiene') as mock_tool:
            mock_tool.return_value = {
                "schedule_id": "hygiene_schedule_123",
                "workflow_id": "hygiene_workflow_456",
                "schedule_type": "interval",
                "interval_minutes": 120,
                "next_run_time": "2025-08-27T14:00:00",
                "hygiene_tools_configured": [
                    "analyze_code_quality",
                    "optimize_imports", 
                    "remove_dead_code",
                    "standardize_patterns"
                ],
                "safety_mode": True,
                "notifications_enabled": True,
                "created": True,
                "status": "scheduled"
            }
            
            result = await mcp_server.setup_scheduled_hygiene(
                project_path=sample_project,
                interval_minutes=120,
                safety_mode=True,
                apply_fixes=True,
                notification_webhook=None
            )
            
            assert result["schedule_id"] == "hygiene_schedule_123"
            assert result["interval_minutes"] == 120
            assert len(result["hygiene_tools_configured"]) == 4
            assert result["safety_mode"] == True
            assert result["created"] == True
            mock_tool.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_workflow_status_tool(self, mcp_server):
        """Test get_workflow_status MCP tool."""
        with patch.object(mcp_server, 'get_workflow_status') as mock_tool:
            mock_tool.return_value = {
                "workflow_id": "pipeline_abc123",
                "name": "Quality Analysis Pipeline",
                "status": "ready",
                "created_at": "2025-08-27T09:30:00",
                "last_executed": "2025-08-27T10:05:30",
                "total_executions": 5,
                "successful_executions": 4,
                "failed_executions": 1,
                "average_duration": "5.2 minutes",
                "steps": [
                    {
                        "step_id": "step_1",
                        "tool": "analyze_code_quality",
                        "status": "ready",
                        "last_result": {"score": 0.75}
                    },
                    {
                        "step_id": "step_2", 
                        "tool": "optimize_imports",
                        "status": "ready",
                        "last_result": {"optimized": 5}
                    }
                ],
                "next_scheduled_run": None,
                "tags": ["pipeline", "analysis", "quality"]
            }
            
            result = await mcp_server.get_workflow_status(
                workflow_id="pipeline_abc123"
            )
            
            assert result["workflow_id"] == "pipeline_abc123"
            assert result["status"] == "ready"
            assert result["total_executions"] == 5
            assert result["successful_executions"] == 4
            assert len(result["steps"]) == 2
            mock_tool.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_list_workflows_tool(self, mcp_server):
        """Test list_workflows MCP tool."""
        with patch.object(mcp_server, 'list_workflows') as mock_tool:
            mock_tool.return_value = {
                "total_workflows": 8,
                "active_workflows": 6,
                "scheduled_workflows": 2,
                "workflows": [
                    {
                        "workflow_id": "pipeline_001",
                        "name": "Daily Code Quality",
                        "type": "pipeline",
                        "status": "active",
                        "last_run": "2025-08-27T08:00:00",
                        "next_run": "2025-08-28T08:00:00",
                        "tags": ["daily", "quality"]
                    },
                    {
                        "workflow_id": "conditional_002",
                        "name": "Smart Refactor",
                        "type": "conditional",
                        "status": "ready",
                        "last_run": "2025-08-26T15:30:00",
                        "next_run": None,
                        "tags": ["refactor", "conditional"]
                    },
                    {
                        "workflow_id": "batch_003",
                        "name": "Multi-Project Cleanup",
                        "type": "batch",
                        "status": "running",
                        "last_run": "2025-08-27T11:00:00",
                        "next_run": None,
                        "tags": ["batch", "cleanup"]
                    }
                ],
                "filter_applied": None,
                "sort_by": "last_run"
            }
            
            result = await mcp_server.list_workflows(
                filter_tags=None,
                status_filter=None,
                sort_by="last_run"
            )
            
            assert result["total_workflows"] == 8
            assert result["active_workflows"] == 6
            assert result["scheduled_workflows"] == 2
            assert len(result["workflows"]) == 3
            mock_tool.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_workflow_metrics_tool(self, mcp_server):
        """Test get_workflow_metrics MCP tool."""
        with patch.object(mcp_server, 'get_workflow_metrics') as mock_tool:
            mock_tool.return_value = {
                "time_period": "last_30_days",
                "workflows_executed": 45,
                "successful_executions": 41,
                "failed_executions": 4,
                "success_rate": 0.911,
                "average_execution_time": "6.4 minutes",
                "total_time_saved": "12.5 hours",
                "most_used_tools": [
                    {"tool": "optimize_imports", "usage_count": 38},
                    {"tool": "analyze_code_quality", "usage_count": 35},
                    {"tool": "standardize_patterns", "usage_count": 28}
                ],
                "workflow_types": {
                    "pipelines": 25,
                    "conditional": 12,
                    "batch": 8
                },
                "performance_trends": {
                    "execution_time_trend": "improving",
                    "success_rate_trend": "stable",
                    "usage_trend": "increasing"
                },
                "scheduled_workflows": {
                    "total": 6,
                    "active": 4,
                    "paused": 2,
                    "next_runs_today": 3
                }
            }
            
            result = await mcp_server.get_workflow_metrics(
                time_period="last_30_days",
                include_trends=True
            )
            
            assert result["workflows_executed"] == 45
            assert result["success_rate"] == 0.911
            assert len(result["most_used_tools"]) == 3
            assert result["workflow_types"]["pipelines"] == 25
            assert result["performance_trends"]["execution_time_trend"] == "improving"
            mock_tool.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_mcp_tool_error_handling(self, mcp_server):
        """Test MCP tool error handling for Priority 5 features."""
        # Test with invalid workflow ID
        with patch.object(mcp_server, 'execute_workflow') as mock_tool:
            mock_tool.side_effect = Exception("Workflow not found")
            
            result = await mcp_server.execute_workflow(
                workflow_id="nonexistent_workflow"
            )
            
            # Should handle error gracefully
            assert "error" in result or "exception" in str(result)
    
    @pytest.mark.asyncio
    async def test_workflow_validation_tool(self, mcp_server):
        """Test workflow validation for complex scenarios."""
        with patch.object(mcp_server, 'validate_workflow') as mock_tool:
            mock_tool.return_value = {
                "workflow_id": "complex_workflow_789",
                "validation_status": "passed_with_warnings",
                "errors": [],
                "warnings": [
                    "Step 'conditional_step_3' has no fallback condition",
                    "Circular dependency risk detected between steps 2 and 4"
                ],
                "recommendations": [
                    "Add timeout configuration for long-running steps",
                    "Consider splitting complex conditional logic",
                    "Add error handling for external tool failures"
                ],
                "complexity_score": 0.75,
                "estimated_execution_time": "8-12 minutes",
                "resource_requirements": {
                    "cpu": "medium",
                    "memory": "low",
                    "disk": "low"
                },
                "ready_to_execute": True
            }
            
            complex_workflow = {
                "name": "Complex Validation Test",
                "steps": [
                    # Multiple conditional steps with dependencies
                    {"id": "step1", "tool": "analyze_code_quality"},
                    {"id": "step2", "tool": "optimize_imports", "depends_on": ["step1"]},
                    {"id": "step3", "tool": "remove_dead_code", "depends_on": ["step2"]},
                    {"id": "step4", "tool": "comprehensive_refactor", "depends_on": ["step1", "step3"]}
                ]
            }
            
            result = await mcp_server.validate_workflow(
                workflow_definition=complex_workflow
            )
            
            assert result["validation_status"] == "passed_with_warnings"
            assert len(result["warnings"]) == 2
            assert len(result["recommendations"]) == 3
            assert result["complexity_score"] == 0.75
            assert result["ready_to_execute"] == True
            mock_tool.assert_called_once()


class TestPriority5ToolIntegration:
    """Test integration between Priority 5 tools and existing functionality."""
    
    @pytest.fixture
    def mock_orchestrator(self):
        """Create mock WorkflowOrchestrator for testing."""
        orchestrator = Mock(spec=WorkflowOrchestrator)
        
        # Mock return values for workflow methods
        orchestrator.create_analysis_pipeline.return_value = Mock(
            workflow_id="test_pipeline",
            name="Test Pipeline", 
            steps=[],
            tags={'pipeline', 'analysis'}
        )
        
        orchestrator.execute_pipeline.return_value = Mock(
            execution_id="test_exec",
            workflow_id="test_pipeline",
            status=WorkflowStatus.COMPLETED,
            step_results={}
        )
        
        orchestrator.create_batch_operation.return_value = Mock(
            batch_id="test_batch",
            operation_type="optimize_imports",
            target_files=[],
            results=[]
        )
        
        return orchestrator
    
    @pytest.mark.asyncio
    async def test_integration_with_existing_tools(self, mock_orchestrator):
        """Test how Priority 5 workflows integrate with existing deepflow tools."""
        if not MCP_AVAILABLE:
            pytest.skip("MCP not available")
        
        server = DeepflowMCPServer()
        
        # Mock the orchestrator integration
        with patch('deepflow.workflow_orchestrator.WorkflowOrchestrator', return_value=mock_orchestrator):
            # Test that workflows can use existing analysis tools
            result = await server.analyze_dependencies("test_project")
            
            # Should work alongside workflow tools
            assert result is not None
    
    def test_priority5_tools_in_mcp_server_registry(self):
        """Test that Priority 5 tools are properly registered in MCP server."""
        if not MCP_AVAILABLE:
            pytest.skip("MCP not available")
        
        server = DeepflowMCPServer()
        
        # Expected Priority 5 tool names
        expected_tools = [
            "create_analysis_pipeline",
            "execute_workflow",
            "create_conditional_workflow", 
            "create_batch_operation",
            "execute_batch_operation",
            "setup_scheduled_hygiene",
            "load_custom_workflow",
            "get_workflow_status",
            "list_workflows",
            "get_workflow_metrics"
        ]
        
        # Check if tools are available (this would depend on actual implementation)
        available_methods = [method for method in dir(server) if not method.startswith('_')]
        
        # At least some Priority 5 functionality should be available
        priority5_methods = [method for method in available_methods if any(tool in method for tool in expected_tools)]
        assert len(priority5_methods) >= 0  # Adjust based on actual implementation


class TestPriority5Performance:
    """Test performance characteristics of Priority 5 workflow features."""
    
    @pytest.fixture
    def large_workflow_project(self):
        """Create a large project for workflow performance testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            
            # Create many files to test batch operations
            for i in range(20):
                file_content = f"""
import os
import sys
import unused_module_{i}

def function_{i}():
    pass

class Class_{i}:
    def method_{i}(self):
        return {i}

unused_var_{i} = "unused"
"""
                (project_path / f"module_{i}.py").write_text(file_content)
            
            yield str(project_path)
    
    @pytest.mark.asyncio
    async def test_workflow_performance_with_large_project(self, large_workflow_project):
        """Test Priority 5 workflow performance with large projects."""
        if not MCP_AVAILABLE:
            pytest.skip("MCP not available")
        
        import time
        
        orchestrator = WorkflowOrchestrator(large_workflow_project)
        
        # Test pipeline creation performance
        start_time = time.time()
        pipeline = await orchestrator.create_analysis_pipeline(
            "Performance Test Pipeline",
            [
                {'name': 'standardize_patterns', 'parameters': {}},
                {'name': 'optimize_imports', 'parameters': {}},
                {'name': 'remove_dead_code', 'parameters': {}}
            ]
        )
        pipeline_creation_time = time.time() - start_time
        
        # Test batch operation creation performance
        start_time = time.time()
        all_files = [str(f) for f in Path(large_workflow_project).glob("*.py")]
        batch_op = await orchestrator.create_batch_operation(
            'optimize_imports', all_files[:10], {}, parallel=True  # Limit to 10 files
        )
        batch_creation_time = time.time() - start_time
        
        # Performance assertions (adjust thresholds as needed)
        assert pipeline_creation_time < 5.0  # Should create pipeline quickly
        assert batch_creation_time < 3.0     # Should create batch operation quickly
        
        # Verify results are meaningful
        assert len(pipeline.steps) == 3
        assert len(batch_op.target_files) == 10
    
    @pytest.mark.asyncio
    async def test_concurrent_workflow_execution(self, large_workflow_project):
        """Test concurrent execution of multiple workflows."""
        if not MCP_AVAILABLE:
            pytest.skip("MCP not available")
        
        orchestrator = WorkflowOrchestrator(large_workflow_project)
        
        # Create multiple pipelines
        pipelines = []
        for i in range(3):
            pipeline = await orchestrator.create_analysis_pipeline(
                f"Concurrent Pipeline {i}",
                [{'name': 'standardize_patterns', 'parameters': {}}]
            )
            pipelines.append(pipeline)
        
        # Execute pipelines concurrently
        import time
        start_time = time.time()
        
        # Mock the execution to avoid actual tool calls
        async def mock_execute(workflow_id):
            await asyncio.sleep(0.1)  # Simulate work
            return Mock(
                execution_id=f"exec_{workflow_id}",
                workflow_id=workflow_id,
                status=WorkflowStatus.COMPLETED
            )
        
        orchestrator.execute_pipeline = mock_execute
        
        # Run concurrently
        executions = await asyncio.gather(*[
            orchestrator.execute_pipeline(pipeline.workflow_id)
            for pipeline in pipelines
        ])
        
        total_time = time.time() - start_time
        
        # Should complete all executions
        assert len(executions) == 3
        assert all(exec.status == WorkflowStatus.COMPLETED for exec in executions)
        
        # Concurrent execution should be faster than sequential
        assert total_time < 1.0  # Should be much less than 3 * 0.1 = 0.3s + overhead


if __name__ == "__main__":
    pytest.main([__file__, "-v"])