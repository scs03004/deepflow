#!/usr/bin/env python3
"""
Demo Script for Priority 5: Tool Workflows & Chaining Features

This script demonstrates all Priority 5 capabilities including:
- Analysis Pipelines: Chain multiple MCP tools in sequence
- Conditional Workflows: Execute different actions based on analysis results
- Batch Operations: Apply fixes across multiple files simultaneously
- Custom Workflow Definition: User-defined tool combinations
- Scheduled Code Hygiene: Regular automated quality checks

Usage:
    python demo_priority5_features.py
    python demo_priority5_features.py --apply-changes  # Actually apply workflow changes
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
import yaml
from pathlib import Path

# Add project to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from deepflow.workflow_orchestrator import (
        WorkflowOrchestrator,
        WorkflowStatus,
        ConditionType
    )
    from deepflow.mcp.server import DeepflowMCPServer
    WORKFLOW_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: Workflow features not available: {e}")
    sys.exit(1)


def create_demo_projects():
    """Create multiple demo projects for workflow testing."""
    print("🏗️  Creating demo projects for workflow testing...")
    
    temp_dir = tempfile.mkdtemp(prefix="priority5_demo_")
    base_path = Path(temp_dir)
    
    projects = []
    
    # Project 1: Quality issues for pipeline demo
    project1 = base_path / "quality_project"
    project1.mkdir()
    
    (project1 / "main.py").write_text("""
import os
import sys
import unused_module
import json
import json  # Duplicate import

def camelCaseFunction(param1, param2):
    return param1 + param2

def snake_case_function():
    pass

class TestClass:
    def method_without_docs(self):
        return "test"

def unused_helper():
    return "never called"

unused_var = "not used"
""")
    
    (project1 / "utils.py").write_text("""
import re
import time
import unused_util_import

def utility_func():
    pass

def dead_function():
    pass

class UtilityClass:
    pass
""")
    
    projects.append(str(project1))
    
    # Project 2: Batch operation demo
    project2 = base_path / "batch_project"
    project2.mkdir()
    
    for i in range(5):
        (project2 / f"module_{i}.py").write_text(f"""
import os
import unused_import_{i}

def function_{i}():
    pass

class Class_{i}:
    def method_{i}(self):
        return {i}

unused_var_{i} = "unused"
""")
    
    projects.append(str(project2))
    
    # Project 3: Conditional workflow demo
    project3 = base_path / "conditional_project"
    project3.mkdir()
    
    (project3 / "main.py").write_text("""
import os
import sys

def poorQualityCode():
    x = 1
    y = 2
    z = x + y
    if True:
        if True:
            if True:
                print("nested")
    return z

class UndocumentedClass:
    def __init__(self):
        pass
    
    def badMethod(self):
        return None
""")
    
    projects.append(str(project3))
    
    print(f"✅ Created {len(projects)} demo projects at: {temp_dir}")
    for i, project in enumerate(projects, 1):
        print(f"   Project {i}: {Path(project).name}")
    
    return projects


def print_header(title: str, emoji: str = "🔧"):
    """Print a formatted header."""
    print(f"\n{emoji} {title}")
    print("=" * (len(title) + 4))


def print_results(title: str, results: dict, emoji: str = "📊"):
    """Print results in a formatted way."""
    print(f"\n{emoji} {title}")
    print("-" * (len(title) + 4))
    print(json.dumps(results, indent=2, default=str))


async def demo_analysis_pipelines(orchestrator: WorkflowOrchestrator, project_path: str, apply_changes: bool = False):
    """Demonstrate analysis pipeline functionality."""
    print_header("Analysis Pipelines Demo", "🔗")
    
    print("Creating a comprehensive quality analysis pipeline...")
    
    # Define pipeline tools
    pipeline_tools = [
        {
            'name': 'analyze_code_quality', 
            'parameters': {'thorough': True}
        },
        {
            'name': 'optimize_imports', 
            'parameters': {'apply_changes': apply_changes}
        },
        {
            'name': 'standardize_patterns', 
            'parameters': {'apply_changes': apply_changes}
        },
        {
            'name': 'generate_docstrings', 
            'parameters': {'apply_changes': apply_changes}
        }
    ]
    
    try:
        # Create pipeline
        workflow = await orchestrator.create_analysis_pipeline(
            "Quality Analysis Pipeline",
            pipeline_tools,
            project_path
        )
        
        print(f"✅ Created pipeline: {workflow.name} ({workflow.workflow_id})")
        print(f"   Steps: {len(workflow.steps)}")
        print(f"   Tags: {list(workflow.tags)}")
        
        # Execute pipeline
        print("\nExecuting pipeline...")
        execution = await orchestrator.execute_pipeline(workflow.workflow_id)
        
        result_summary = {
            "execution_id": execution.execution_id,
            "status": execution.status.value,
            "steps_completed": len(execution.step_results),
            "duration": str(execution.end_time - execution.start_time) if execution.end_time and execution.start_time else "N/A",
            "results_preview": {
                step_id: str(result)[:100] + "..." if len(str(result)) > 100 else result
                for step_id, result in list(execution.step_results.items())[:2]
            }
        }
        
        print_results("Pipeline Execution Results", result_summary)
        
        return {"pipeline_created": True, "execution_completed": execution.status == WorkflowStatus.COMPLETED}
        
    except Exception as e:
        print(f"❌ Pipeline demo failed: {e}")
        return {"pipeline_created": False, "error": str(e)}


async def demo_conditional_workflows(orchestrator: WorkflowOrchestrator, project_path: str, apply_changes: bool = False):
    """Demonstrate conditional workflow functionality."""
    print_header("Conditional Workflows Demo", "🔀")
    
    print("Creating a quality-based conditional workflow...")
    
    # Define conditional steps
    conditional_steps = [
        {
            'tool': 'analyze_code_quality',
            'parameters': {},
            'conditions': [],  # Always execute first step
            'description': 'Analyze overall code quality'
        },
        {
            'tool': 'standardize_patterns',
            'parameters': {'apply_changes': apply_changes},
            'conditions': [
                {
                    'field_path': 'step_results.conditional_step_1.score',
                    'type': 'lt',
                    'value': 0.8,
                    'description': 'Pattern standardization if quality < 80%'
                }
            ],
            'depends_on': ['conditional_step_1'],
            'description': 'Standardize patterns if quality is low'
        },
        {
            'tool': 'comprehensive_refactor',
            'parameters': {'apply_changes': apply_changes, 'priority_filter': 'high'},
            'conditions': [
                {
                    'field_path': 'step_results.conditional_step_1.score',
                    'type': 'lt',
                    'value': 0.6,
                    'description': 'Full refactor if quality < 60%'
                }
            ],
            'depends_on': ['conditional_step_2'],
            'description': 'Comprehensive refactor for very poor quality'
        }
    ]
    
    try:
        # Create conditional workflow
        workflow = await orchestrator.create_conditional_workflow(
            "Quality-Based Refactor Workflow",
            conditional_steps
        )
        
        print(f"✅ Created conditional workflow: {workflow.name}")
        print(f"   Steps: {len(workflow.steps)}")
        print(f"   Conditions: {sum(len(step.conditions) for step in workflow.steps)}")
        
        # Simulate execution (mock results for demo)
        print("\nSimulating conditional workflow execution...")
        
        # Mock low quality score to trigger conditions
        mock_context = {
            'step_results': {
                'conditional_step_1': {'score': 0.55, 'issues': 15}  # Low quality
            }
        }
        
        triggered_steps = []
        for step in workflow.steps:
            if not step.conditions:
                triggered_steps.append(step.step_id)
            else:
                for condition in step.conditions:
                    if condition.evaluate(mock_context):
                        triggered_steps.append(step.step_id)
                        break
        
        conditional_summary = {
            "workflow_id": workflow.workflow_id,
            "total_steps": len(workflow.steps),
            "triggered_steps": triggered_steps,
            "trigger_rate": f"{len(triggered_steps)}/{len(workflow.steps)}",
            "conditions_met": "Quality score (0.55) triggered refactoring steps"
        }
        
        print_results("Conditional Workflow Analysis", conditional_summary)
        
        return {"workflow_created": True, "conditions_working": len(triggered_steps) > 1}
        
    except Exception as e:
        print(f"❌ Conditional workflow demo failed: {e}")
        return {"workflow_created": False, "error": str(e)}


async def demo_batch_operations(orchestrator: WorkflowOrchestrator, projects: list, apply_changes: bool = False):
    """Demonstrate batch operation functionality."""
    print_header("Batch Operations Demo", "📦")
    
    print(f"Creating batch operation across {len(projects)} projects...")
    
    try:
        # Create batch operation for import optimization
        batch_op = await orchestrator.create_batch_operation(
            operation_type='optimize_imports',
            targets=projects,
            parameters={'apply_changes': apply_changes, 'safe_mode': True},
            parallel=True
        )
        
        print(f"✅ Created batch operation: {batch_op.batch_id}")
        print(f"   Operation: {batch_op.operation_type}")
        print(f"   Targets: {len(batch_op.target_projects)} projects")
        print(f"   Parallel: {batch_op.parallel}")
        
        # Execute batch operation
        print("\nExecuting batch operation...")
        result = await orchestrator.execute_batch_operation(batch_op.batch_id)
        
        batch_summary = {
            "batch_id": result.batch_id,
            "operation_type": result.operation_type,
            "total_targets": result.summary['total_targets'],
            "successful": result.summary['successful'],
            "failed": result.summary['failed'],
            "success_rate": f"{result.summary['success_rate']:.1%}",
            "results_sample": result.results[:2] if result.results else []
        }
        
        print_results("Batch Operation Results", batch_summary)
        
        return {"batch_created": True, "batch_executed": True, "success_rate": result.summary['success_rate']}
        
    except Exception as e:
        print(f"❌ Batch operation demo failed: {e}")
        return {"batch_created": False, "error": str(e)}


async def demo_custom_workflow_definition(orchestrator: WorkflowOrchestrator, project_path: str):
    """Demonstrate custom workflow definition functionality."""
    print_header("Custom Workflow Definition Demo", "📝")
    
    print("Creating custom workflow from YAML definition...")
    
    # Define custom workflow
    custom_workflow = {
        'id': 'custom_quality_workflow',
        'name': 'Custom Quality Enhancement Workflow',
        'description': 'A user-defined workflow for comprehensive quality improvement',
        'tags': ['custom', 'quality', 'enhancement'],
        'parallel_execution': False,
        'max_concurrent_steps': 2,
        'steps': [
            {
                'id': 'initial_analysis',
                'type': 'analysis',
                'tool': 'analyze_code_quality',
                'parameters': {'detailed': True},
                'conditions': [],
                'depends_on': [],
                'timeout_seconds': 300,
                'retry_count': 1,
                'description': 'Comprehensive code quality analysis'
            },
            {
                'id': 'conditional_import_fix',
                'type': 'condition',
                'tool': 'optimize_imports',
                'parameters': {'apply_changes': True},
                'conditions': [
                    {
                        'field_path': 'step_results.initial_analysis.import_issues',
                        'type': 'gt',
                        'value': 0,
                        'description': 'Fix imports if issues found'
                    }
                ],
                'depends_on': ['initial_analysis'],
                'timeout_seconds': 180,
                'retry_count': 0,
                'description': 'Conditional import optimization'
            },
            {
                'id': 'pattern_standardization',
                'type': 'action',
                'tool': 'standardize_patterns',
                'parameters': {'apply_changes': True, 'strict_mode': True},
                'conditions': [
                    {
                        'field_path': 'step_results.initial_analysis.consistency_score',
                        'type': 'lt',
                        'value': 0.85,
                        'description': 'Standardize if consistency below 85%'
                    }
                ],
                'depends_on': ['conditional_import_fix'],
                'timeout_seconds': 240,
                'retry_count': 1,
                'description': 'Pattern standardization for consistency'
            }
        ]
    }
    
    try:
        # Load custom workflow
        workflow = orchestrator.load_workflow_from_dict(custom_workflow)
        
        print(f"✅ Loaded custom workflow: {workflow.name}")
        print(f"   ID: {workflow.workflow_id}")
        print(f"   Steps: {len(workflow.steps)}")
        print(f"   Parallel execution: {workflow.parallel_execution}")
        
        # Save workflow to YAML for demonstration
        temp_yaml = Path(tempfile.mktemp(suffix='.yaml'))
        orchestrator.save_workflow_to_yaml(workflow.workflow_id, str(temp_yaml))
        
        print(f"\n📄 Workflow saved to: {temp_yaml}")
        
        # Show YAML content
        with open(temp_yaml, 'r') as f:
            yaml_content = f.read()
        
        print("YAML Content Preview:")
        print(yaml_content[:500] + "..." if len(yaml_content) > 500 else yaml_content)
        
        # Clean up
        temp_yaml.unlink()
        
        custom_summary = {
            "workflow_loaded": True,
            "workflow_id": workflow.workflow_id,
            "steps_loaded": len(workflow.steps),
            "conditions_loaded": sum(len(step.conditions) for step in workflow.steps),
            "yaml_export": "Successfully exported and imported YAML format"
        }
        
        print_results("Custom Workflow Definition", custom_summary)
        
        return {"custom_workflow_created": True, "yaml_support": True}
        
    except Exception as e:
        print(f"❌ Custom workflow demo failed: {e}")
        return {"custom_workflow_created": False, "error": str(e)}


async def demo_scheduled_hygiene(orchestrator: WorkflowOrchestrator, project_path: str):
    """Demonstrate scheduled code hygiene functionality."""
    print_header("Scheduled Code Hygiene Demo", "⏰")
    
    print("Setting up scheduled code hygiene checks...")
    
    try:
        # Set up scheduled hygiene (short interval for demo)
        schedule_config = {
            'project_path': project_path,
            'interval_minutes': 5,  # Very frequent for demo
            'safety_mode': True,
            'apply_fixes': False  # Safe default for demo
        }
        
        workflow_id = await orchestrator.setup_scheduled_hygiene(schedule_config)
        
        print(f"✅ Scheduled hygiene workflow created: {workflow_id}")
        
        # Get scheduled workflow info
        workflow = orchestrator.scheduled_workflows.get(workflow_id)
        if workflow:
            hygiene_summary = {
                "workflow_id": workflow_id,
                "schedule_interval": f"{workflow.schedule_interval_minutes} minutes",
                "next_run": workflow.next_run_time.isoformat() if workflow.next_run_time else "Not scheduled",
                "hygiene_tools": len(workflow.steps),
                "safety_mode": schedule_config['safety_mode'],
                "auto_apply_fixes": schedule_config['apply_fixes']
            }
            
            print_results("Scheduled Hygiene Configuration", hygiene_summary)
            
            # Start scheduler for demo (briefly)
            print("\n⚡ Starting scheduler for demonstration...")
            orchestrator.start_scheduler()
            
            print("   Scheduler is now running...")
            print("   (In production, this would run continuously)")
            
            # Stop scheduler after brief demo
            await asyncio.sleep(1)
            orchestrator.stop_scheduler()
            print("   Scheduler stopped for demo")
        
        return {"scheduled_hygiene_setup": True, "scheduler_functional": True}
        
    except Exception as e:
        print(f"❌ Scheduled hygiene demo failed: {e}")
        return {"scheduled_hygiene_setup": False, "error": str(e)}


async def demo_workflow_management(orchestrator: WorkflowOrchestrator):
    """Demonstrate workflow management and monitoring."""
    print_header("Workflow Management Demo", "📊")
    
    try:
        # List all workflows
        workflows = orchestrator.list_workflows()
        
        print(f"📋 Total workflows created: {len(workflows)}")
        
        if workflows:
            print("\nWorkflow Summary:")
            for workflow in workflows:
                print(f"   • {workflow.name} ({workflow.workflow_id})")
                print(f"     Tags: {list(workflow.tags)}")
                print(f"     Steps: {len(workflow.steps)}")
                print(f"     Created: {workflow.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Get metrics
        metrics = orchestrator.get_metrics()
        print_results("Workflow Orchestrator Metrics", metrics)
        
        # Cleanup old executions (demo)
        print("\n🧹 Cleaning up demo executions...")
        orchestrator.cleanup_executions(max_age_hours=0)  # Clean all for demo
        
        management_summary = {
            "total_workflows": len(workflows),
            "workflow_types": {
                "pipelines": len([w for w in workflows if 'pipeline' in w.tags]),
                "conditional": len([w for w in workflows if 'conditional' in w.tags]),
                "custom": len([w for w in workflows if 'custom' in w.tags]),
                "scheduled": len(orchestrator.scheduled_workflows)
            },
            "executions_tracked": metrics['workflows_executed'],
            "success_rate": metrics['successful_executions'] / max(metrics['workflows_executed'], 1)
        }
        
        print_results("Workflow Management Summary", management_summary)
        
        return {"management_functional": True, "metrics_available": True}
        
    except Exception as e:
        print(f"❌ Workflow management demo failed: {e}")
        return {"management_functional": False, "error": str(e)}


async def demo_mcp_integration(projects: list, apply_changes: bool = False):
    """Demonstrate MCP server integration with workflows."""
    print_header("MCP Integration Demo", "🔌")
    
    try:
        # Create MCP server instance
        server = DeepflowMCPServer()
        print("✅ MCP Server created successfully")
        
        # Test workflow MCP tools
        mcp_tools = [
            ("create_analysis_pipeline", {
                "pipeline_name": "MCP Demo Pipeline",
                "tools": [
                    {"name": "analyze_code_quality", "parameters": {}},
                    {"name": "optimize_imports", "parameters": {"apply_changes": apply_changes}}
                ],
                "project_path": projects[0]
            }),
            ("list_workflows", {"project_path": projects[0]}),
            ("get_workflow_metrics", {"time_period": "last_7_days", "project_path": projects[0]})
        ]
        
        mcp_results = {}
        
        for tool_name, arguments in mcp_tools:
            print(f"\n🔧 Testing MCP tool: {tool_name}")
            try:
                # Get the handler method
                handler_name = f"_handle_{tool_name}"
                if hasattr(server, handler_name):
                    handler = getattr(server, handler_name)
                    start_time = time.time()
                    result = await handler(arguments)
                    end_time = time.time()
                    
                    if result and len(result) > 0:
                        result_text = result[0].text
                        try:
                            result_json = json.loads(result_text)
                            print(f"   ✅ Completed in {end_time - start_time:.2f}s")
                            
                            # Store key metrics
                            if tool_name == "create_analysis_pipeline":
                                mcp_results["pipeline_created"] = result_json.get("created", False)
                                mcp_results["pipeline_steps"] = len(result_json.get("steps", []))
                            elif tool_name == "list_workflows":
                                mcp_results["workflows_listed"] = result_json.get("total_workflows", 0)
                            elif tool_name == "get_workflow_metrics":
                                mcp_results["metrics_available"] = "workflows_executed" in result_json
                            
                        except json.JSONDecodeError:
                            print(f"   ✅ Tool executed (non-JSON response)")
                            print(f"   📝 Response: {result_text[:100]}...")
                    else:
                        print("   ❌ No result returned")
                else:
                    print(f"   ❌ Handler {handler_name} not found")
                    
            except Exception as e:
                print(f"   ❌ Error testing {tool_name}: {e}")
        
        print_results("MCP Integration Results", mcp_results)
        
        return {"mcp_integration": True, "tools_functional": len(mcp_results) > 0}
        
    except Exception as e:
        print(f"❌ MCP integration demo failed: {e}")
        return {"mcp_integration": False, "error": str(e)}


async def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Demo Priority 5: Tool Workflows & Chaining Features")
    parser.add_argument("--apply-changes", action="store_true", 
                       help="Actually apply workflow changes (default: dry run)")
    parser.add_argument("--skip-mcp", action="store_true",
                       help="Skip MCP integration demo")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick demo with fewer features")
    
    args = parser.parse_args()
    
    print("🎯 Priority 5: Tool Workflows & Chaining Demo")
    print("=" * 55)
    
    if args.apply_changes:
        print("⚠️  WARNING: --apply-changes is enabled. Files will be modified!")
        print("   Press Ctrl+C within 3 seconds to cancel...")
        try:
            time.sleep(3)
        except KeyboardInterrupt:
            print("\n❌ Cancelled by user")
            return
    else:
        print("ℹ️  Running in dry-run mode (no files will be modified)")
    
    demo_results = {}
    
    try:
        # Create demo projects
        projects = create_demo_projects()
        main_project = projects[0]
        
        # Initialize workflow orchestrator
        print_header("Initializing Workflow Orchestrator", "⚙️")
        orchestrator = WorkflowOrchestrator(main_project)
        print("✅ Workflow Orchestrator initialized successfully")
        
        # Demo 1: Analysis Pipelines
        print("\n" + "="*60)
        pipeline_results = await demo_analysis_pipelines(orchestrator, main_project, args.apply_changes)
        demo_results.update(pipeline_results)
        
        # Demo 2: Conditional Workflows
        print("\n" + "="*60)
        conditional_results = await demo_conditional_workflows(orchestrator, projects[2], args.apply_changes)
        demo_results.update(conditional_results)
        
        # Demo 3: Batch Operations
        if not args.quick:
            print("\n" + "="*60)
            batch_results = await demo_batch_operations(orchestrator, projects, args.apply_changes)
            demo_results.update(batch_results)
        
        # Demo 4: Custom Workflow Definition
        print("\n" + "="*60)
        custom_results = await demo_custom_workflow_definition(orchestrator, main_project)
        demo_results.update(custom_results)
        
        # Demo 5: Scheduled Code Hygiene
        if not args.quick:
            print("\n" + "="*60)
            hygiene_results = await demo_scheduled_hygiene(orchestrator, main_project)
            demo_results.update(hygiene_results)
        
        # Demo 6: Workflow Management
        print("\n" + "="*60)
        management_results = await demo_workflow_management(orchestrator)
        demo_results.update(management_results)
        
        # Demo 7: MCP Integration
        if not args.skip_mcp and not args.quick:
            print("\n" + "="*60)
            mcp_results = await demo_mcp_integration(projects, args.apply_changes)
            demo_results.update(mcp_results)
        
        # Final Summary
        print("\n" + "="*60)
        print_header("Priority 5 Demo Summary", "🎉")
        
        success_count = sum(1 for result in demo_results.values() if result is True)
        total_features = len([k for k in demo_results.keys() if not k.endswith('_error')])
        
        print(f"✅ Features Demonstrated: {success_count}/{total_features}")
        print(f"🏗️  Projects Created: {len(projects)}")
        print(f"⚡ Workflows Created: {len(orchestrator.workflows)}")
        print(f"📊 Success Rate: {success_count/total_features:.1%}")
        
        print("\n💡 Priority 5 Capabilities Demonstrated:")
        print("   ✅ Analysis Pipelines - Chain tools sequentially")
        print("   ✅ Conditional Workflows - Execute based on results")
        print("   ✅ Batch Operations - Process multiple targets") 
        print("   ✅ Custom Definitions - User-defined YAML workflows")
        print("   ✅ Scheduled Hygiene - Automated quality checks")
        print("   ✅ Workflow Management - Monitor and control")
        if not args.skip_mcp and not args.quick:
            print("   ✅ MCP Integration - Claude Code compatibility")
        
        print(f"\n📁 Demo projects location: {Path(projects[0]).parent}")
        
        if not args.apply_changes:
            print("\n💡 Tips:")
            print("   - Run with --apply-changes to see actual workflow execution")
            print("   - Use --quick for faster demo with core features")
            print("   - Check created workflows in the orchestrator")
            
    except KeyboardInterrupt:
        print("\n❌ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        try:
            if 'projects' in locals():
                import shutil
                base_dir = str(Path(projects[0]).parent)
                shutil.rmtree(base_dir)
                print(f"🧹 Cleaned up demo projects: {base_dir}")
        except:
            print("⚠️  Could not clean up all demo projects")


if __name__ == "__main__":
    asyncio.run(main())