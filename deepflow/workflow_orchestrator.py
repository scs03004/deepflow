"""
Workflow Orchestrator - Priority 5 Implementation

Advanced workflow system for chaining MCP tools, conditional execution,
batch operations, custom workflows, and scheduled code hygiene.

This module provides the core engine for Priority 5: Tool Workflows & Chaining features.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Set
import yaml
import os
import schedule
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkflowStepType(Enum):
    """Types of workflow steps."""
    ANALYSIS = "analysis"
    CONDITION = "condition"
    ACTION = "action"
    BATCH = "batch"
    PIPELINE = "pipeline"
    SCHEDULE = "schedule"


class WorkflowStatus(Enum):
    """Workflow execution statuses."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class ConditionType(Enum):
    """Types of conditions for conditional workflows."""
    GREATER_THAN = "gt"
    LESS_THAN = "lt"
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"


@dataclass
class WorkflowCondition:
    """Represents a condition for conditional workflow execution."""
    field_path: str  # e.g., "analysis_result.consistency_score"
    condition_type: ConditionType
    expected_value: Any
    description: str = ""
    
    def evaluate(self, context: Dict[str, Any]) -> bool:
        """Evaluate the condition against the provided context."""
        try:
            # Navigate to the field using dot notation
            value = context
            for key in self.field_path.split('.'):
                if isinstance(value, dict):
                    value = value.get(key)
                elif hasattr(value, key):
                    value = getattr(value, key)
                else:
                    return False
            
            # Evaluate condition
            if self.condition_type == ConditionType.GREATER_THAN:
                return value > self.expected_value
            elif self.condition_type == ConditionType.LESS_THAN:
                return value < self.expected_value
            elif self.condition_type == ConditionType.EQUALS:
                return value == self.expected_value
            elif self.condition_type == ConditionType.NOT_EQUALS:
                return value != self.expected_value
            elif self.condition_type == ConditionType.CONTAINS:
                return self.expected_value in str(value)
            elif self.condition_type == ConditionType.NOT_CONTAINS:
                return self.expected_value not in str(value)
            elif self.condition_type == ConditionType.EXISTS:
                return value is not None
            elif self.condition_type == ConditionType.NOT_EXISTS:
                return value is None
                
        except Exception as e:
            logger.warning(f"Condition evaluation failed: {e}")
            return False
        
        return False


@dataclass
class WorkflowStep:
    """Represents a single step in a workflow."""
    step_id: str
    step_type: WorkflowStepType
    tool_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    conditions: List[WorkflowCondition] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    timeout_seconds: int = 300
    retry_count: int = 0
    description: str = ""
    
    # Execution state
    status: WorkflowStatus = WorkflowStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retries_attempted: int = 0


@dataclass
class WorkflowDefinition:
    """Defines a complete workflow with steps and metadata."""
    workflow_id: str
    name: str
    description: str
    steps: List[WorkflowStep] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    created_by: str = "system"
    version: str = "1.0"
    tags: Set[str] = field(default_factory=set)
    
    # Execution settings
    parallel_execution: bool = False
    max_concurrent_steps: int = 3
    global_timeout_seconds: int = 1800  # 30 minutes
    
    # Schedule settings (for scheduled workflows)
    schedule_cron: Optional[str] = None
    schedule_interval_minutes: Optional[int] = None
    next_run_time: Optional[datetime] = None


@dataclass
class WorkflowExecution:
    """Represents a workflow execution instance."""
    execution_id: str
    workflow_id: str
    status: WorkflowStatus = WorkflowStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    context: Dict[str, Any] = field(default_factory=dict)
    step_results: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchOperation:
    """Represents a batch operation across multiple files/projects."""
    batch_id: str
    operation_type: str
    target_files: List[str] = field(default_factory=list)
    target_projects: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    parallel: bool = True
    max_workers: int = 4
    
    # Results
    results: List[Dict[str, Any]] = field(default_factory=list)
    failed_items: List[Dict[str, Any]] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)


class WorkflowOrchestrator:
    """
    Main orchestrator for Priority 5: Tool Workflows & Chaining.
    
    Provides advanced workflow capabilities including:
    - Analysis pipelines that chain multiple tools
    - Conditional workflows based on results
    - Batch operations across files/projects
    - Custom workflow definitions
    - Scheduled code hygiene checks
    """
    
    def __init__(self, project_path: str = "."):
        """Initialize the workflow orchestrator."""
        self.project_path = Path(project_path)
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.scheduled_workflows: Dict[str, WorkflowDefinition] = {}
        self.batch_operations: Dict[str, BatchOperation] = {}
        
        # Tool registry for available tools
        self.tool_registry: Dict[str, Callable] = {}
        self._register_default_tools()
        
        # Execution settings
        self.max_concurrent_workflows = 5
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Scheduler for code hygiene
        self.scheduler_running = False
        
        # Metrics and monitoring
        self.metrics = {
            'workflows_executed': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_execution_time': 0.0,
            'batch_operations_completed': 0
        }
    
    def _register_default_tools(self):
        """Register default tools available for workflows."""
        # Import tools dynamically to avoid circular imports
        try:
            from .smart_refactoring_engine import SmartRefactoringEngine
            
            self.tool_registry.update({
                'standardize_patterns': self._create_tool_wrapper('standardize_patterns'),
                'optimize_imports': self._create_tool_wrapper('optimize_imports'),
                'suggest_file_splits': self._create_tool_wrapper('suggest_file_splits'),
                'remove_dead_code': self._create_tool_wrapper('remove_dead_code'),
                'generate_docstrings': self._create_tool_wrapper('generate_docstrings'),
                'comprehensive_refactor': self._create_tool_wrapper('comprehensive_refactor')
            })
            
        except ImportError:
            logger.warning("Smart Refactoring Engine not available for workflows")
        
        # Add dependency analysis tools
        try:
            import sys
            tools_dir = Path(__file__).parent.parent / "tools"
            if str(tools_dir) not in sys.path:
                sys.path.insert(0, str(tools_dir))
            
            self.tool_registry.update({
                'analyze_dependencies': self._create_dependency_tool_wrapper('analyze_dependencies'),
                'analyze_code_quality': self._create_dependency_tool_wrapper('analyze_code_quality'),
                'validate_commit': self._create_dependency_tool_wrapper('validate_commit'),
                'generate_documentation': self._create_dependency_tool_wrapper('generate_documentation')
            })
            
        except ImportError:
            logger.warning("Dependency analysis tools not available for workflows")
    
    def _create_tool_wrapper(self, tool_name: str) -> Callable:
        """Create a wrapper for smart refactoring tools."""
        async def tool_wrapper(parameters: Dict[str, Any]) -> Dict[str, Any]:
            from .smart_refactoring_engine import SmartRefactoringEngine
            
            project_path = parameters.get('project_path', str(self.project_path))
            engine = SmartRefactoringEngine(project_path)
            
            if tool_name == 'standardize_patterns':
                result = engine.standardize_patterns(parameters.get('target_files'))
                return {
                    'pattern_type': result.pattern_type,
                    'consistency_score': result.consistency_score,
                    'violations': result.violations,
                    'files_affected': result.files_affected
                }
            elif tool_name == 'optimize_imports':
                result = engine.optimize_imports(parameters.get('target_files'))
                return {
                    'unused_imports': result.unused_imports,
                    'duplicate_imports': result.duplicate_imports,
                    'circular_imports': result.circular_imports,
                    'optimization_suggestions': result.optimization_suggestions
                }
            elif tool_name == 'suggest_file_splits':
                result = engine.suggest_file_splits(parameters.get('target_files'))
                return {
                    'split_recommendations': [
                        {
                            'file_path': r.file_path,
                            'size_score': r.size_score,
                            'complexity_score': r.complexity_score,
                            'recommendations': r.split_recommendations
                        }
                        for r in result
                    ]
                }
            elif tool_name == 'remove_dead_code':
                result = engine.detect_dead_code(parameters.get('target_files'))
                return {
                    'unused_functions': result.unused_functions,
                    'unused_classes': result.unused_classes,
                    'unused_variables': result.unused_variables,
                    'unreachable_code': result.unreachable_code
                }
            elif tool_name == 'generate_docstrings':
                result = engine.generate_documentation(parameters.get('target_files'))
                return {
                    'missing_docstrings': result.missing_docstrings,
                    'generated_docstrings': result.generated_docstrings
                }
            elif tool_name == 'comprehensive_refactor':
                # Run comprehensive analysis
                pattern_analysis = engine.standardize_patterns()
                import_analysis = engine.optimize_imports()
                file_splits = engine.suggest_file_splits()
                dead_code = engine.detect_dead_code()
                doc_analysis = engine.generate_documentation()
                
                return {
                    'pattern_analysis': {
                        'consistency_score': pattern_analysis.consistency_score,
                        'violations': len(pattern_analysis.violations)
                    },
                    'import_analysis': {
                        'unused_count': len(import_analysis.unused_imports),
                        'duplicate_count': len(import_analysis.duplicate_imports)
                    },
                    'file_analysis': {
                        'files_needing_splits': len([f for f in file_splits if f.split_recommendations])
                    },
                    'dead_code_analysis': {
                        'total_dead_items': len(dead_code.unused_functions) + len(dead_code.unused_classes)
                    },
                    'documentation_analysis': {
                        'missing_docs': len(doc_analysis.missing_docstrings)
                    }
                }
        
        return tool_wrapper
    
    def _create_dependency_tool_wrapper(self, tool_name: str) -> Callable:
        """Create a wrapper for dependency analysis tools."""
        async def tool_wrapper(parameters: Dict[str, Any]) -> Dict[str, Any]:
            # Simplified implementation - in a real system this would call actual tools
            project_path = parameters.get('project_path', str(self.project_path))
            
            # Mock results for demonstration
            if tool_name == 'analyze_dependencies':
                return {
                    'nodes_count': 25,
                    'edges_count': 48,
                    'circular_dependencies': 0,
                    'orphan_modules': 2
                }
            elif tool_name == 'analyze_code_quality':
                return {
                    'overall_score': 0.85,
                    'issues_found': 12,
                    'technical_debt_hours': 8.5
                }
            elif tool_name == 'validate_commit':
                return {
                    'valid': True,
                    'warnings': 2,
                    'blockers': 0
                }
            elif tool_name == 'generate_documentation':
                return {
                    'files_documented': 5,
                    'coverage_improvement': 0.25
                }
            
            return {'status': 'completed'}
        
        return tool_wrapper
    
    # 1. Analysis Pipelines Implementation
    
    async def create_analysis_pipeline(self, pipeline_name: str, tools: List[Dict[str, Any]], 
                                     project_path: str = None) -> WorkflowDefinition:
        """
        Create an analysis pipeline that chains multiple MCP tools in sequence.
        
        Args:
            pipeline_name: Name for the pipeline
            tools: List of tools to chain, each with 'name' and 'parameters'
            project_path: Optional project path override
            
        Returns:
            WorkflowDefinition for the created pipeline
        """
        logger.info(f"Creating analysis pipeline: {pipeline_name}")
        
        workflow_id = f"pipeline_{uuid.uuid4().hex[:8]}"
        workflow = WorkflowDefinition(
            workflow_id=workflow_id,
            name=pipeline_name,
            description=f"Analysis pipeline chaining {len(tools)} tools",
            tags={'pipeline', 'analysis'}
        )
        
        # Create sequential steps
        previous_step_id = None
        for i, tool_config in enumerate(tools):
            step_id = f"step_{i+1}_{tool_config['name']}"
            
            parameters = tool_config.get('parameters', {})
            if project_path:
                parameters['project_path'] = project_path
            
            step = WorkflowStep(
                step_id=step_id,
                step_type=WorkflowStepType.ANALYSIS,
                tool_name=tool_config['name'],
                parameters=parameters,
                depends_on=[previous_step_id] if previous_step_id else [],
                description=f"Execute {tool_config['name']} analysis"
            )
            
            workflow.steps.append(step)
            previous_step_id = step_id
        
        self.workflows[workflow_id] = workflow
        logger.info(f"Created analysis pipeline with {len(workflow.steps)} steps")
        
        return workflow
    
    async def execute_pipeline(self, workflow_id: str) -> WorkflowExecution:
        """Execute an analysis pipeline."""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        execution = WorkflowExecution(
            execution_id=f"exec_{uuid.uuid4().hex[:8]}",
            workflow_id=workflow_id,
            start_time=datetime.now()
        )
        
        self.executions[execution.execution_id] = execution
        execution.status = WorkflowStatus.RUNNING
        
        logger.info(f"Executing pipeline: {workflow.name}")
        
        try:
            # Execute steps in dependency order
            executed_steps = set()
            step_results = {}
            
            while len(executed_steps) < len(workflow.steps):
                ready_steps = [
                    step for step in workflow.steps
                    if step.step_id not in executed_steps and
                    all(dep in executed_steps for dep in step.depends_on)
                ]
                
                if not ready_steps:
                    break
                
                # Execute ready steps (could be parallel if workflow allows)
                for step in ready_steps:
                    step_result = await self._execute_step(step, step_results)
                    step_results[step.step_id] = step_result
                    executed_steps.add(step.step_id)
            
            execution.step_results = step_results
            execution.status = WorkflowStatus.COMPLETED
            execution.end_time = datetime.now()
            
            # Update metrics
            self.metrics['workflows_executed'] += 1
            self.metrics['successful_executions'] += 1
            
            duration = (execution.end_time - execution.start_time).total_seconds()
            self._update_average_execution_time(duration)
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error = str(e)
            execution.end_time = datetime.now()
            
            self.metrics['workflows_executed'] += 1
            self.metrics['failed_executions'] += 1
            
            logger.error(f"Pipeline execution failed: {e}")
        
        return execution
    
    # 2. Conditional Workflows Implementation
    
    async def create_conditional_workflow(self, workflow_name: str, 
                                        conditional_steps: List[Dict[str, Any]]) -> WorkflowDefinition:
        """
        Create a conditional workflow that executes different actions based on analysis results.
        
        Args:
            workflow_name: Name for the workflow
            conditional_steps: List of steps with conditions
            
        Returns:
            WorkflowDefinition for the conditional workflow
        """
        logger.info(f"Creating conditional workflow: {workflow_name}")
        
        workflow_id = f"conditional_{uuid.uuid4().hex[:8]}"
        workflow = WorkflowDefinition(
            workflow_id=workflow_id,
            name=workflow_name,
            description=f"Conditional workflow with {len(conditional_steps)} conditional steps",
            tags={'conditional', 'workflow'}
        )
        
        for i, step_config in enumerate(conditional_steps):
            step_id = f"conditional_step_{i+1}"
            
            # Parse conditions
            conditions = []
            for condition_config in step_config.get('conditions', []):
                condition = WorkflowCondition(
                    field_path=condition_config['field_path'],
                    condition_type=ConditionType(condition_config['type']),
                    expected_value=condition_config['value'],
                    description=condition_config.get('description', '')
                )
                conditions.append(condition)
            
            step = WorkflowStep(
                step_id=step_id,
                step_type=WorkflowStepType.CONDITION,
                tool_name=step_config['tool'],
                parameters=step_config.get('parameters', {}),
                conditions=conditions,
                depends_on=step_config.get('depends_on', []),
                description=step_config.get('description', f"Conditional step {i+1}")
            )
            
            workflow.steps.append(step)
        
        self.workflows[workflow_id] = workflow
        logger.info(f"Created conditional workflow with {len(workflow.steps)} steps")
        
        return workflow
    
    # 3. Batch Operations Implementation
    
    async def create_batch_operation(self, operation_type: str, targets: List[str], 
                                   parameters: Dict[str, Any], parallel: bool = True) -> BatchOperation:
        """
        Create a batch operation to apply fixes across multiple files simultaneously.
        
        Args:
            operation_type: Type of operation (e.g., 'optimize_imports', 'remove_dead_code')
            targets: List of file paths or project paths
            parameters: Parameters for the operation
            parallel: Whether to execute in parallel
            
        Returns:
            BatchOperation instance
        """
        logger.info(f"Creating batch operation: {operation_type} on {len(targets)} targets")
        
        batch_id = f"batch_{uuid.uuid4().hex[:8]}"
        batch_op = BatchOperation(
            batch_id=batch_id,
            operation_type=operation_type,
            parameters=parameters,
            parallel=parallel
        )
        
        # Classify targets as files or projects
        for target in targets:
            target_path = Path(target)
            if target_path.is_file():
                batch_op.target_files.append(target)
            else:
                batch_op.target_projects.append(target)
        
        self.batch_operations[batch_id] = batch_op
        
        return batch_op
    
    async def execute_batch_operation(self, batch_id: str) -> BatchOperation:
        """Execute a batch operation."""
        if batch_id not in self.batch_operations:
            raise ValueError(f"Batch operation {batch_id} not found")
        
        batch_op = self.batch_operations[batch_id]
        logger.info(f"Executing batch operation: {batch_op.operation_type}")
        
        all_targets = batch_op.target_files + batch_op.target_projects
        
        if batch_op.parallel and len(all_targets) > 1:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=batch_op.max_workers) as executor:
                future_to_target = {
                    executor.submit(self._execute_batch_item, batch_op.operation_type, 
                                  target, batch_op.parameters): target
                    for target in all_targets
                }
                
                for future in as_completed(future_to_target):
                    target = future_to_target[future]
                    try:
                        result = future.result()
                        batch_op.results.append({
                            'target': target,
                            'result': result,
                            'status': 'success'
                        })
                    except Exception as e:
                        batch_op.failed_items.append({
                            'target': target,
                            'error': str(e),
                            'status': 'failed'
                        })
        else:
            # Sequential execution
            for target in all_targets:
                try:
                    result = await self._execute_batch_item_async(batch_op.operation_type, 
                                                                target, batch_op.parameters)
                    batch_op.results.append({
                        'target': target,
                        'result': result,
                        'status': 'success'
                    })
                except Exception as e:
                    batch_op.failed_items.append({
                        'target': target,
                        'error': str(e),
                        'status': 'failed'
                    })
        
        # Generate summary
        batch_op.summary = {
            'total_targets': len(all_targets),
            'successful': len(batch_op.results),
            'failed': len(batch_op.failed_items),
            'success_rate': len(batch_op.results) / len(all_targets) if all_targets else 0
        }
        
        self.metrics['batch_operations_completed'] += 1
        logger.info(f"Batch operation completed: {batch_op.summary}")
        
        return batch_op
    
    # 4. Custom Workflow Definition Implementation
    
    def load_workflow_from_yaml(self, yaml_file: str) -> WorkflowDefinition:
        """Load a custom workflow definition from YAML file."""
        with open(yaml_file, 'r') as f:
            workflow_data = yaml.safe_load(f)
        
        return self._parse_workflow_definition(workflow_data)
    
    def load_workflow_from_dict(self, workflow_data: Dict[str, Any]) -> WorkflowDefinition:
        """Load a custom workflow definition from dictionary."""
        return self._parse_workflow_definition(workflow_data)
    
    def _parse_workflow_definition(self, data: Dict[str, Any]) -> WorkflowDefinition:
        """Parse workflow definition from data."""
        workflow_id = data.get('id', f"custom_{uuid.uuid4().hex[:8]}")
        
        workflow = WorkflowDefinition(
            workflow_id=workflow_id,
            name=data['name'],
            description=data.get('description', ''),
            tags=set(data.get('tags', [])),
            parallel_execution=data.get('parallel_execution', False),
            max_concurrent_steps=data.get('max_concurrent_steps', 3)
        )
        
        # Parse steps
        for step_data in data.get('steps', []):
            conditions = []
            for cond_data in step_data.get('conditions', []):
                condition = WorkflowCondition(
                    field_path=cond_data['field_path'],
                    condition_type=ConditionType(cond_data['type']),
                    expected_value=cond_data['value'],
                    description=cond_data.get('description', '')
                )
                conditions.append(condition)
            
            step = WorkflowStep(
                step_id=step_data['id'],
                step_type=WorkflowStepType(step_data['type']),
                tool_name=step_data['tool'],
                parameters=step_data.get('parameters', {}),
                conditions=conditions,
                depends_on=step_data.get('depends_on', []),
                timeout_seconds=step_data.get('timeout_seconds', 300),
                retry_count=step_data.get('retry_count', 0),
                description=step_data.get('description', '')
            )
            workflow.steps.append(step)
        
        self.workflows[workflow_id] = workflow
        return workflow
    
    def save_workflow_to_yaml(self, workflow_id: str, output_file: str):
        """Save a workflow definition to YAML file."""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        
        # Convert to serializable format
        workflow_data = {
            'id': workflow.workflow_id,
            'name': workflow.name,
            'description': workflow.description,
            'tags': list(workflow.tags),
            'parallel_execution': workflow.parallel_execution,
            'max_concurrent_steps': workflow.max_concurrent_steps,
            'steps': []
        }
        
        for step in workflow.steps:
            step_data = {
                'id': step.step_id,
                'type': step.step_type.value,
                'tool': step.tool_name,
                'parameters': step.parameters,
                'depends_on': step.depends_on,
                'timeout_seconds': step.timeout_seconds,
                'retry_count': step.retry_count,
                'description': step.description,
                'conditions': []
            }
            
            for condition in step.conditions:
                cond_data = {
                    'field_path': condition.field_path,
                    'type': condition.condition_type.value,
                    'value': condition.expected_value,
                    'description': condition.description
                }
                step_data['conditions'].append(cond_data)
            
            workflow_data['steps'].append(step_data)
        
        with open(output_file, 'w') as f:
            yaml.dump(workflow_data, f, default_flow_style=False, indent=2)
        
        logger.info(f"Saved workflow {workflow_id} to {output_file}")
    
    # 5. Scheduled Code Hygiene Implementation
    
    async def setup_scheduled_hygiene(self, schedule_config: Dict[str, Any]) -> str:
        """
        Set up scheduled code hygiene checks.
        
        Args:
            schedule_config: Configuration for scheduled hygiene
            
        Returns:
            Workflow ID for the scheduled hygiene workflow
        """
        logger.info("Setting up scheduled code hygiene")
        
        # Create hygiene workflow
        hygiene_tools = [
            {'name': 'analyze_code_quality', 'parameters': {}},
            {'name': 'optimize_imports', 'parameters': {'apply_changes': True}},
            {'name': 'remove_dead_code', 'parameters': {'apply_changes': True, 'safe_mode': True}},
            {'name': 'standardize_patterns', 'parameters': {'apply_changes': False}}  # Analyze only
        ]
        
        workflow = await self.create_analysis_pipeline(
            "Scheduled Code Hygiene",
            hygiene_tools,
            schedule_config.get('project_path')
        )
        
        # Configure scheduling
        if 'interval_minutes' in schedule_config:
            workflow.schedule_interval_minutes = schedule_config['interval_minutes']
        
        if 'cron' in schedule_config:
            workflow.schedule_cron = schedule_config['cron']
        
        # Calculate next run time
        if workflow.schedule_interval_minutes:
            workflow.next_run_time = datetime.now() + timedelta(minutes=workflow.schedule_interval_minutes)
        
        self.scheduled_workflows[workflow.workflow_id] = workflow
        
        # Set up the actual schedule
        if workflow.schedule_interval_minutes:
            schedule.every(workflow.schedule_interval_minutes).minutes.do(
                self._run_scheduled_workflow, workflow.workflow_id
            )
        
        logger.info(f"Scheduled hygiene workflow created: {workflow.workflow_id}")
        return workflow.workflow_id
    
    def start_scheduler(self):
        """Start the scheduled workflow runner."""
        if self.scheduler_running:
            return
        
        self.scheduler_running = True
        logger.info("Starting workflow scheduler")
        
        async def scheduler_loop():
            while self.scheduler_running:
                schedule.run_pending()
                await asyncio.sleep(60)  # Check every minute
        
        # Run scheduler in background
        asyncio.create_task(scheduler_loop())
    
    def stop_scheduler(self):
        """Stop the scheduled workflow runner."""
        self.scheduler_running = False
        schedule.clear()
        logger.info("Stopped workflow scheduler")
    
    def _run_scheduled_workflow(self, workflow_id: str):
        """Execute a scheduled workflow."""
        logger.info(f"Running scheduled workflow: {workflow_id}")
        asyncio.create_task(self.execute_pipeline(workflow_id))
    
    # Helper methods
    
    async def _execute_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Any:
        """Execute a single workflow step."""
        logger.info(f"Executing step: {step.step_id}")
        
        step.status = WorkflowStatus.RUNNING
        step.start_time = time.time()
        
        try:
            # Check conditions
            if step.conditions:
                for condition in step.conditions:
                    if not condition.evaluate(context):
                        logger.info(f"Step {step.step_id} skipped due to condition: {condition.description}")
                        step.status = WorkflowStatus.COMPLETED
                        step.result = {'skipped': True, 'reason': 'condition_not_met'}
                        step.end_time = time.time()
                        return step.result
            
            # Execute tool
            if step.tool_name in self.tool_registry:
                tool_func = self.tool_registry[step.tool_name]
                result = await tool_func(step.parameters)
                
                step.status = WorkflowStatus.COMPLETED
                step.result = result
                step.end_time = time.time()
                
                return result
            else:
                raise ValueError(f"Tool {step.tool_name} not found in registry")
                
        except Exception as e:
            step.status = WorkflowStatus.FAILED
            step.error = str(e)
            step.end_time = time.time()
            
            # Retry logic
            if step.retries_attempted < step.retry_count:
                step.retries_attempted += 1
                logger.info(f"Retrying step {step.step_id} (attempt {step.retries_attempted})")
                await asyncio.sleep(1)  # Brief delay before retry
                return await self._execute_step(step, context)
            
            raise e
    
    async def _execute_batch_item_async(self, operation_type: str, target: str, parameters: Dict[str, Any]) -> Any:
        """Execute a single batch item asynchronously."""
        if operation_type in self.tool_registry:
            tool_func = self.tool_registry[operation_type]
            params = parameters.copy()
            params['project_path'] = target
            return await tool_func(params)
        else:
            raise ValueError(f"Operation {operation_type} not supported")
    
    def _execute_batch_item(self, operation_type: str, target: str, parameters: Dict[str, Any]) -> Any:
        """Execute a single batch item synchronously."""
        # This would be called from thread pool
        return asyncio.run(self._execute_batch_item_async(operation_type, target, parameters))
    
    def _update_average_execution_time(self, duration: float):
        """Update the average execution time metric."""
        current_avg = self.metrics['average_execution_time']
        total_executions = self.metrics['workflows_executed']
        
        if total_executions == 1:
            self.metrics['average_execution_time'] = duration
        else:
            # Calculate running average
            self.metrics['average_execution_time'] = (
                (current_avg * (total_executions - 1) + duration) / total_executions
            )
    
    # Query and management methods
    
    def get_workflow_status(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        """Get workflow definition by ID."""
        return self.workflows.get(workflow_id)
    
    def get_execution_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get execution status by ID."""
        return self.executions.get(execution_id)
    
    def list_workflows(self, tags: Optional[Set[str]] = None) -> List[WorkflowDefinition]:
        """List all workflows, optionally filtered by tags."""
        workflows = list(self.workflows.values())
        
        if tags:
            workflows = [w for w in workflows if tags.intersection(w.tags)]
        
        return workflows
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics."""
        return self.metrics.copy()
    
    def cleanup_executions(self, max_age_hours: int = 24):
        """Clean up old execution records."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        expired_executions = [
            exec_id for exec_id, execution in self.executions.items()
            if execution.end_time and execution.end_time < cutoff_time
        ]
        
        for exec_id in expired_executions:
            del self.executions[exec_id]
        
        logger.info(f"Cleaned up {len(expired_executions)} expired executions")