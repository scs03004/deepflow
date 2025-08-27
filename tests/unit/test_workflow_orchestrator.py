"""
Test suite for Workflow Orchestrator (Priority 5).

Comprehensive tests for analysis pipelines, conditional workflows, 
batch operations, custom workflow definitions, and scheduled code hygiene.
"""

import asyncio
import json
import os
import tempfile
import pytest
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Import the workflow orchestrator
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from deepflow.workflow_orchestrator import (
    WorkflowOrchestrator,
    WorkflowDefinition,
    WorkflowStep,
    WorkflowExecution,
    WorkflowCondition,
    BatchOperation,
    WorkflowStepType,
    WorkflowStatus,
    ConditionType
)


class TestWorkflowOrchestrator:
    """Test cases for WorkflowOrchestrator class."""
    
    @pytest.fixture
    def temp_project(self):
        """Create a temporary project directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            
            # Create sample Python files
            (project_path / "main.py").write_text("""
import os
import unused_module

def test_function():
    pass

def unused_function():
    return "unused"
""")
            
            (project_path / "utils.py").write_text("""
import json
import json  # Duplicate

def utility():
    pass
""")
            
            yield str(project_path)
    
    @pytest.fixture
    def orchestrator(self, temp_project):
        """Create a WorkflowOrchestrator instance."""
        return WorkflowOrchestrator(temp_project)
    
    def test_initialization(self, temp_project):
        """Test orchestrator initialization."""
        orchestrator = WorkflowOrchestrator(temp_project)
        
        assert orchestrator.project_path == Path(temp_project)
        assert len(orchestrator.workflows) == 0
        assert len(orchestrator.executions) == 0
        assert len(orchestrator.tool_registry) > 0  # Should have default tools
        assert orchestrator.max_concurrent_workflows == 5
        
    def test_tool_registry_setup(self, orchestrator):
        """Test that tool registry is properly set up."""
        # Should have smart refactoring tools
        expected_tools = [
            'standardize_patterns',
            'optimize_imports', 
            'suggest_file_splits',
            'remove_dead_code',
            'generate_docstrings',
            'comprehensive_refactor'
        ]
        
        for tool in expected_tools:
            assert tool in orchestrator.tool_registry


class TestAnalysisPipelines:
    """Test analysis pipeline functionality."""
    
    @pytest.fixture
    def orchestrator(self, temp_project):
        return WorkflowOrchestrator(temp_project)
    
    @pytest.mark.asyncio
    async def test_create_analysis_pipeline(self, orchestrator):
        """Test creating an analysis pipeline."""
        tools = [
            {'name': 'analyze_code_quality', 'parameters': {}},
            {'name': 'optimize_imports', 'parameters': {'apply_changes': False}},
            {'name': 'standardize_patterns', 'parameters': {}}
        ]
        
        workflow = await orchestrator.create_analysis_pipeline(
            "Test Pipeline", tools
        )
        
        assert workflow.name == "Test Pipeline"
        assert len(workflow.steps) == 3
        assert 'pipeline' in workflow.tags
        assert 'analysis' in workflow.tags
        
        # Check step dependencies
        steps = workflow.steps
        assert len(steps[0].depends_on) == 0  # First step has no dependencies
        assert steps[0].step_id in steps[1].depends_on  # Second step depends on first
        assert steps[1].step_id in steps[2].depends_on  # Third step depends on second
    
    @pytest.mark.asyncio
    async def test_execute_pipeline(self, orchestrator):
        """Test executing an analysis pipeline."""
        # Mock tool functions to avoid actual execution
        mock_results = {
            'analyze_code_quality': {'score': 0.8, 'issues': 5},
            'optimize_imports': {'unused': 2, 'duplicates': 1},
            'standardize_patterns': {'consistency': 0.7, 'violations': 3}
        }
        
        async def mock_tool_func(params):
            tool_name = params.get('_tool_name', 'unknown')
            return mock_results.get(tool_name, {'status': 'completed'})
        
        # Replace tool registry with mocks
        for tool_name in ['analyze_code_quality', 'optimize_imports', 'standardize_patterns']:
            orchestrator.tool_registry[tool_name] = mock_tool_func
        
        # Create and execute pipeline
        tools = [
            {'name': 'analyze_code_quality', 'parameters': {'_tool_name': 'analyze_code_quality'}},
            {'name': 'optimize_imports', 'parameters': {'_tool_name': 'optimize_imports'}},
            {'name': 'standardize_patterns', 'parameters': {'_tool_name': 'standardize_patterns'}}
        ]
        
        workflow = await orchestrator.create_analysis_pipeline("Test Pipeline", tools)
        execution = await orchestrator.execute_pipeline(workflow.workflow_id)
        
        assert execution.status == WorkflowStatus.COMPLETED
        assert len(execution.step_results) == 3
        assert execution.start_time is not None
        assert execution.end_time is not None
        
        # Check metrics updated
        assert orchestrator.metrics['workflows_executed'] == 1
        assert orchestrator.metrics['successful_executions'] == 1


class TestConditionalWorkflows:
    """Test conditional workflow functionality."""
    
    @pytest.fixture
    def orchestrator(self, temp_project):
        return WorkflowOrchestrator(temp_project)
    
    @pytest.mark.asyncio
    async def test_create_conditional_workflow(self, orchestrator):
        """Test creating a conditional workflow."""
        conditional_steps = [
            {
                'tool': 'analyze_code_quality',
                'parameters': {},
                'conditions': [
                    {
                        'field_path': 'result.score',
                        'type': 'lt',
                        'value': 0.8,
                        'description': 'Code quality below threshold'
                    }
                ],
                'description': 'Analyze code quality'
            },
            {
                'tool': 'comprehensive_refactor',
                'parameters': {'apply_changes': True},
                'conditions': [
                    {
                        'field_path': 'step_results.conditional_step_1.score',
                        'type': 'lt', 
                        'value': 0.6,
                        'description': 'Trigger refactor for very low quality'
                    }
                ],
                'depends_on': ['conditional_step_1'],
                'description': 'Comprehensive refactor if needed'
            }
        ]
        
        workflow = await orchestrator.create_conditional_workflow(
            "Quality-Based Refactor", conditional_steps
        )
        
        assert workflow.name == "Quality-Based Refactor"
        assert len(workflow.steps) == 2
        assert 'conditional' in workflow.tags
        
        # Check conditions
        first_step = workflow.steps[0]
        assert len(first_step.conditions) == 1
        assert first_step.conditions[0].field_path == 'result.score'
        assert first_step.conditions[0].condition_type == ConditionType.LESS_THAN
        assert first_step.conditions[0].expected_value == 0.8
    
    def test_workflow_condition_evaluation(self):
        """Test workflow condition evaluation."""
        condition = WorkflowCondition(
            field_path="analysis.consistency_score",
            condition_type=ConditionType.LESS_THAN,
            expected_value=0.8
        )
        
        # Test with matching condition
        context = {
            'analysis': {
                'consistency_score': 0.6
            }
        }
        assert condition.evaluate(context) == True
        
        # Test with non-matching condition
        context = {
            'analysis': {
                'consistency_score': 0.9
            }
        }
        assert condition.evaluate(context) == False
        
        # Test with missing field
        context = {'analysis': {}}
        assert condition.evaluate(context) == False
    
    def test_condition_types(self):
        """Test different condition types."""
        context = {'value': 10, 'text': 'hello world', 'flag': True}
        
        # Greater than
        cond = WorkflowCondition('value', ConditionType.GREATER_THAN, 5)
        assert cond.evaluate(context) == True
        
        # Equals
        cond = WorkflowCondition('flag', ConditionType.EQUALS, True)
        assert cond.evaluate(context) == True
        
        # Contains
        cond = WorkflowCondition('text', ConditionType.CONTAINS, 'world')
        assert cond.evaluate(context) == True
        
        # Not contains
        cond = WorkflowCondition('text', ConditionType.NOT_CONTAINS, 'python')
        assert cond.evaluate(context) == True
        
        # Exists
        cond = WorkflowCondition('value', ConditionType.EXISTS, None)
        assert cond.evaluate(context) == True
        
        # Not exists
        cond = WorkflowCondition('missing', ConditionType.NOT_EXISTS, None)
        assert cond.evaluate(context) == True


class TestBatchOperations:
    """Test batch operation functionality."""
    
    @pytest.fixture
    def orchestrator(self, temp_project):
        return WorkflowOrchestrator(temp_project)
    
    @pytest.fixture
    def multiple_projects(self):
        """Create multiple temporary project directories."""
        temp_dirs = []
        try:
            for i in range(3):
                temp_dir = tempfile.mkdtemp(prefix=f"batch_test_{i}_")
                project_path = Path(temp_dir)
                
                # Create test files
                (project_path / "main.py").write_text(f"""
import os
import unused_module_{i}

def test_function_{i}():
    pass
""")
                temp_dirs.append(temp_dir)
            
            yield temp_dirs
        finally:
            # Cleanup
            for temp_dir in temp_dirs:
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                except:
                    pass
    
    @pytest.mark.asyncio
    async def test_create_batch_operation(self, orchestrator, multiple_projects):
        """Test creating a batch operation."""
        batch_op = await orchestrator.create_batch_operation(
            operation_type='optimize_imports',
            targets=multiple_projects,
            parameters={'apply_changes': False},
            parallel=True
        )
        
        assert batch_op.operation_type == 'optimize_imports'
        assert len(batch_op.target_projects) == 3
        assert batch_op.parallel == True
        assert batch_op.batch_id.startswith('batch_')
    
    @pytest.mark.asyncio
    async def test_execute_batch_operation(self, orchestrator, multiple_projects):
        """Test executing a batch operation."""
        # Mock the batch execution
        mock_result = {'unused_imports': 1, 'duplicates': 0}
        
        async def mock_batch_item(operation_type, target, params):
            return mock_result
        
        orchestrator._execute_batch_item_async = mock_batch_item
        
        # Create and execute batch operation
        batch_op = await orchestrator.create_batch_operation(
            'optimize_imports', multiple_projects, {}, parallel=False
        )
        
        result = await orchestrator.execute_batch_operation(batch_op.batch_id)
        
        assert len(result.results) == 3
        assert result.summary['total_targets'] == 3
        assert result.summary['successful'] == 3
        assert result.summary['failed'] == 0
        assert result.summary['success_rate'] == 1.0
    
    @pytest.mark.asyncio
    async def test_batch_operation_with_failures(self, orchestrator, multiple_projects):
        """Test batch operation handling failures."""
        call_count = 0
        
        async def mock_batch_item_with_failure(operation_type, target, params):
            nonlocal call_count
            call_count += 1
            if call_count == 2:  # Fail the second operation
                raise Exception("Simulated failure")
            return {'status': 'success'}
        
        orchestrator._execute_batch_item_async = mock_batch_item_with_failure
        
        batch_op = await orchestrator.create_batch_operation(
            'test_operation', multiple_projects, {}, parallel=False
        )
        
        result = await orchestrator.execute_batch_operation(batch_op.batch_id)
        
        assert result.summary['successful'] == 2
        assert result.summary['failed'] == 1
        assert len(result.failed_items) == 1
        assert 'Simulated failure' in result.failed_items[0]['error']


class TestCustomWorkflowDefinition:
    """Test custom workflow definition functionality."""
    
    @pytest.fixture
    def orchestrator(self, temp_project):
        return WorkflowOrchestrator(temp_project)
    
    @pytest.fixture
    def sample_workflow_data(self):
        """Sample workflow definition data."""
        return {
            'id': 'custom_test_workflow',
            'name': 'Custom Test Workflow',
            'description': 'A test workflow for validation',
            'tags': ['test', 'custom'],
            'parallel_execution': False,
            'max_concurrent_steps': 2,
            'steps': [
                {
                    'id': 'step_1',
                    'type': 'analysis',
                    'tool': 'analyze_code_quality',
                    'parameters': {'analysis_type': 'all'},
                    'conditions': [],
                    'depends_on': [],
                    'timeout_seconds': 300,
                    'retry_count': 1,
                    'description': 'Analyze code quality'
                },
                {
                    'id': 'step_2', 
                    'type': 'condition',
                    'tool': 'optimize_imports',
                    'parameters': {'apply_changes': True},
                    'conditions': [
                        {
                            'field_path': 'step_results.step_1.issues_found',
                            'type': 'gt',
                            'value': 5,
                            'description': 'Only optimize if many issues found'
                        }
                    ],
                    'depends_on': ['step_1'],
                    'timeout_seconds': 180,
                    'retry_count': 0,
                    'description': 'Conditional import optimization'
                }
            ]
        }
    
    def test_load_workflow_from_dict(self, orchestrator, sample_workflow_data):
        """Test loading workflow from dictionary."""
        workflow = orchestrator.load_workflow_from_dict(sample_workflow_data)
        
        assert workflow.workflow_id == 'custom_test_workflow'
        assert workflow.name == 'Custom Test Workflow'
        assert len(workflow.steps) == 2
        assert 'test' in workflow.tags
        assert 'custom' in workflow.tags
        assert workflow.parallel_execution == False
        assert workflow.max_concurrent_steps == 2
        
        # Check first step
        step1 = workflow.steps[0]
        assert step1.step_id == 'step_1'
        assert step1.step_type == WorkflowStepType.ANALYSIS
        assert step1.tool_name == 'analyze_code_quality'
        assert step1.timeout_seconds == 300
        assert step1.retry_count == 1
        
        # Check second step
        step2 = workflow.steps[1]
        assert step2.step_id == 'step_2'
        assert step2.step_type == WorkflowStepType.CONDITION
        assert len(step2.conditions) == 1
        assert 'step_1' in step2.depends_on
    
    def test_save_workflow_to_yaml(self, orchestrator, sample_workflow_data, tmp_path):
        """Test saving workflow to YAML file."""
        # Load workflow
        workflow = orchestrator.load_workflow_from_dict(sample_workflow_data)
        
        # Save to YAML
        yaml_file = tmp_path / "test_workflow.yaml"
        orchestrator.save_workflow_to_yaml(workflow.workflow_id, str(yaml_file))
        
        # Verify file was created
        assert yaml_file.exists()
        
        # Load and verify content
        with open(yaml_file, 'r') as f:
            saved_data = yaml.safe_load(f)
        
        assert saved_data['id'] == 'custom_test_workflow'
        assert saved_data['name'] == 'Custom Test Workflow'
        assert len(saved_data['steps']) == 2
        assert saved_data['steps'][0]['tool'] == 'analyze_code_quality'
    
    def test_load_workflow_from_yaml(self, orchestrator, sample_workflow_data, tmp_path):
        """Test loading workflow from YAML file."""
        # Create YAML file
        yaml_file = tmp_path / "test_workflow.yaml"
        with open(yaml_file, 'w') as f:
            yaml.dump(sample_workflow_data, f)
        
        # Load workflow
        workflow = orchestrator.load_workflow_from_yaml(str(yaml_file))
        
        assert workflow.workflow_id == 'custom_test_workflow'
        assert workflow.name == 'Custom Test Workflow'
        assert len(workflow.steps) == 2


class TestScheduledCodeHygiene:
    """Test scheduled code hygiene functionality."""
    
    @pytest.fixture
    def orchestrator(self, temp_project):
        return WorkflowOrchestrator(temp_project)
    
    @pytest.mark.asyncio
    async def test_setup_scheduled_hygiene(self, orchestrator):
        """Test setting up scheduled code hygiene."""
        schedule_config = {
            'interval_minutes': 60,
            'project_path': str(orchestrator.project_path)
        }
        
        workflow_id = await orchestrator.setup_scheduled_hygiene(schedule_config)
        
        assert workflow_id in orchestrator.scheduled_workflows
        
        workflow = orchestrator.scheduled_workflows[workflow_id]
        assert workflow.name == "Scheduled Code Hygiene"
        assert workflow.schedule_interval_minutes == 60
        assert workflow.next_run_time is not None
        assert len(workflow.steps) == 4  # Should have 4 hygiene tools
    
    def test_scheduler_start_stop(self, orchestrator):
        """Test starting and stopping the scheduler."""
        assert orchestrator.scheduler_running == False
        
        orchestrator.start_scheduler()
        assert orchestrator.scheduler_running == True
        
        orchestrator.stop_scheduler()
        assert orchestrator.scheduler_running == False
    
    @pytest.mark.asyncio
    async def test_scheduled_workflow_execution(self, orchestrator):
        """Test that scheduled workflows can be executed."""
        # Mock the execution to avoid actual tool calls
        original_execute = orchestrator.execute_pipeline
        execution_called = False
        
        async def mock_execute(workflow_id):
            nonlocal execution_called
            execution_called = True
            # Return a mock execution
            return WorkflowExecution(
                execution_id="mock_exec",
                workflow_id=workflow_id,
                status=WorkflowStatus.COMPLETED
            )
        
        orchestrator.execute_pipeline = mock_execute
        
        # Set up scheduled workflow
        schedule_config = {'interval_minutes': 1}
        workflow_id = await orchestrator.setup_scheduled_hygiene(schedule_config)
        
        # Manually trigger the scheduled workflow
        orchestrator._run_scheduled_workflow(workflow_id)
        
        # Give some time for async execution
        await asyncio.sleep(0.1)
        
        # Restore original method
        orchestrator.execute_pipeline = original_execute


class TestWorkflowManagement:
    """Test workflow management and query functionality."""
    
    @pytest.fixture
    def orchestrator(self, temp_project):
        return WorkflowOrchestrator(temp_project)
    
    @pytest.mark.asyncio
    async def test_workflow_listing(self, orchestrator):
        """Test listing workflows."""
        # Create multiple workflows
        await orchestrator.create_analysis_pipeline("Pipeline 1", [
            {'name': 'analyze_code_quality', 'parameters': {}}
        ])
        
        await orchestrator.create_analysis_pipeline("Pipeline 2", [
            {'name': 'optimize_imports', 'parameters': {}}
        ])
        
        # Test listing all workflows
        all_workflows = orchestrator.list_workflows()
        assert len(all_workflows) == 2
        
        # Test filtering by tags
        pipeline_workflows = orchestrator.list_workflows(tags={'pipeline'})
        assert len(pipeline_workflows) == 2
        
        analysis_workflows = orchestrator.list_workflows(tags={'analysis'})
        assert len(analysis_workflows) == 2
    
    def test_workflow_status_queries(self, orchestrator):
        """Test workflow and execution status queries."""
        # Initially empty
        assert orchestrator.get_workflow_status('nonexistent') is None
        assert orchestrator.get_execution_status('nonexistent') is None
        
        # Test with actual workflow
        workflow_data = {
            'id': 'test_workflow',
            'name': 'Test Workflow',
            'description': 'Test',
            'steps': []
        }
        
        workflow = orchestrator.load_workflow_from_dict(workflow_data)
        
        # Should be able to retrieve
        retrieved = orchestrator.get_workflow_status('test_workflow')
        assert retrieved is not None
        assert retrieved.name == 'Test Workflow'
    
    def test_metrics_tracking(self, orchestrator):
        """Test metrics tracking."""
        initial_metrics = orchestrator.get_metrics()
        
        assert 'workflows_executed' in initial_metrics
        assert 'successful_executions' in initial_metrics
        assert 'failed_executions' in initial_metrics
        assert 'average_execution_time' in initial_metrics
        assert 'batch_operations_completed' in initial_metrics
        
        # All should start at 0
        assert initial_metrics['workflows_executed'] == 0
        assert initial_metrics['successful_executions'] == 0
    
    def test_execution_cleanup(self, orchestrator):
        """Test execution cleanup functionality."""
        # Create mock old execution
        old_execution = WorkflowExecution(
            execution_id="old_exec",
            workflow_id="test_workflow"
        )
        old_execution.end_time = datetime.now() - timedelta(hours=48)  # 48 hours old
        
        # Create recent execution
        recent_execution = WorkflowExecution(
            execution_id="recent_exec", 
            workflow_id="test_workflow"
        )
        recent_execution.end_time = datetime.now() - timedelta(hours=1)  # 1 hour old
        
        orchestrator.executions["old_exec"] = old_execution
        orchestrator.executions["recent_exec"] = recent_execution
        
        # Cleanup executions older than 24 hours
        orchestrator.cleanup_executions(max_age_hours=24)
        
        # Old execution should be removed, recent should remain
        assert "old_exec" not in orchestrator.executions
        assert "recent_exec" in orchestrator.executions


class TestWorkflowIntegration:
    """Integration tests for complex workflow scenarios."""
    
    @pytest.fixture
    def orchestrator(self, temp_project):
        return WorkflowOrchestrator(temp_project)
    
    @pytest.mark.asyncio
    async def test_complex_conditional_pipeline(self, orchestrator):
        """Test a complex pipeline with multiple conditions."""
        # Mock tool functions
        mock_results = {
            'analyze_code_quality': {'score': 0.6, 'issues': 15},
            'optimize_imports': {'optimized': 5},
            'comprehensive_refactor': {'improvements': 8}
        }
        
        async def mock_tool_func(params):
            tool_name = params.get('_tool_name', 'unknown')
            return mock_results.get(tool_name, {})
        
        for tool in mock_results.keys():
            orchestrator.tool_registry[tool] = mock_tool_func
        
        # Create complex workflow
        conditional_steps = [
            {
                'tool': 'analyze_code_quality',
                'parameters': {'_tool_name': 'analyze_code_quality'},
                'conditions': [],
                'description': 'Initial quality analysis'
            },
            {
                'tool': 'optimize_imports',
                'parameters': {'_tool_name': 'optimize_imports'},
                'conditions': [
                    {
                        'field_path': 'step_results.conditional_step_1.issues',
                        'type': 'gt',
                        'value': 10,
                        'description': 'Many issues found'
                    }
                ],
                'depends_on': ['conditional_step_1'],
                'description': 'Import optimization if needed'
            },
            {
                'tool': 'comprehensive_refactor',
                'parameters': {'_tool_name': 'comprehensive_refactor'},
                'conditions': [
                    {
                        'field_path': 'step_results.conditional_step_1.score',
                        'type': 'lt',
                        'value': 0.7,
                        'description': 'Low quality score'
                    }
                ],
                'depends_on': ['conditional_step_2'],
                'description': 'Full refactor for low quality'
            }
        ]
        
        workflow = await orchestrator.create_conditional_workflow(
            "Complex Quality Workflow", conditional_steps
        )
        
        # Execute the workflow
        execution = await orchestrator.execute_pipeline(workflow.workflow_id)
        
        assert execution.status == WorkflowStatus.COMPLETED
        # All steps should execute based on our mock conditions
        assert len(execution.step_results) == 3
    
    @pytest.mark.asyncio
    async def test_workflow_error_handling(self, orchestrator):
        """Test workflow error handling and retry logic."""
        call_count = 0
        
        async def failing_tool_func(params):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:  # Fail first two attempts
                raise Exception(f"Attempt {call_count} failed")
            return {'status': 'success', 'attempt': call_count}
        
        orchestrator.tool_registry['failing_tool'] = failing_tool_func
        
        # Create workflow with retry
        tools = [{'name': 'failing_tool', 'parameters': {}}]
        workflow = await orchestrator.create_analysis_pipeline("Retry Test", tools)
        
        # Set retry count on the step
        workflow.steps[0].retry_count = 2
        
        # Execute
        execution = await orchestrator.execute_pipeline(workflow.workflow_id)
        
        assert execution.status == WorkflowStatus.COMPLETED
        assert call_count == 3  # Initial + 2 retries
        assert execution.step_results['step_1_failing_tool']['attempt'] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])