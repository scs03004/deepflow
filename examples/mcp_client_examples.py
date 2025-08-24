#!/usr/bin/env python3
"""
Deepflow MCP Client Examples
============================

Example scripts showing how to use Deepflow MCP tools programmatically.
These examples demonstrate various integration patterns for different use cases.

Installation:
    pip install deepflow[mcp]

Usage:
    python examples/mcp_client_examples.py
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    from mcp.client.session import ClientSession
    from mcp.client.stdio import stdio_client
    import mcp.types as types
    MCP_AVAILABLE = True
except ImportError:
    print("ERROR: MCP client libraries not available. Install with: pip install mcp")
    MCP_AVAILABLE = False
    exit(1)

class DeepflowMCPClient:
    """Simple MCP client for Deepflow tools."""
    
    def __init__(self, server_command: str = "deepflow-mcp-server"):
        """Initialize MCP client."""
        self.server_command = server_command
        self.session: Optional[ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        # Start server process and create client session
        self.session = await stdio_client(self.server_command)
        await self.session.__aenter__()
        
        # Initialize the session
        result = await self.session.initialize()
        print(f"Connected to server: {result.server_info.name}")
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.__aexit__(exc_type, exc_val, exc_tb)
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available MCP tools."""
        response = await self.session.list_tools()
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "schema": tool.inputSchema
            }
            for tool in response.tools
        ]
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> List[Any]:
        """Call an MCP tool with given arguments."""
        response = await self.session.call_tool(tool_name, arguments)
        return response.content

# Example 1: Basic Project Analysis
async def basic_analysis_example(project_path: str = "."):
    """Example: Basic project dependency analysis."""
    print("\n🔍 Example 1: Basic Project Analysis")
    print("=" * 50)
    
    async with DeepflowMCPClient() as client:
        # Analyze dependencies
        print("Analyzing project dependencies...")
        result = await client.call_tool("analyze_dependencies", {
            "project_path": project_path,
            "format": "text",
            "ai_awareness": True
        })
        
        print("Dependencies Analysis Result:")
        for content in result:
            if hasattr(content, 'text'):
                print(content.text[:500] + "..." if len(content.text) > 500 else content.text)

# Example 2: Code Quality Assessment
async def code_quality_example(project_path: str = "."):
    """Example: Comprehensive code quality analysis."""
    print("\n🔍 Example 2: Code Quality Assessment")
    print("=" * 50)
    
    async with DeepflowMCPClient() as client:
        # Run full code quality analysis
        print("Running comprehensive code quality analysis...")
        result = await client.call_tool("analyze_code_quality", {
            "project_path": project_path,
            "analysis_type": "all",
            "fix_imports": False
        })
        
        print("Code Quality Analysis Result:")
        for content in result:
            if hasattr(content, 'text'):
                try:
                    # Try to parse as JSON for better formatting
                    data = json.loads(content.text)
                    print(json.dumps(data, indent=2))
                except json.JSONDecodeError:
                    print(content.text)

# Example 3: Pre-Commit Validation
async def commit_validation_example(project_path: str = "."):
    """Example: Validate changes before commit."""
    print("\n✅ Example 3: Pre-Commit Validation")
    print("=" * 50)
    
    async with DeepflowMCPClient() as client:
        print("Validating current changes...")
        result = await client.call_tool("validate_commit", {
            "project_path": project_path,
            "check_dependencies": True,
            "check_patterns": True
        })
        
        print("Validation Result:")
        for content in result:
            if hasattr(content, 'text'):
                try:
                    data = json.loads(content.text)
                    if data.get("valid"):
                        print("✅ All validations passed!")
                    else:
                        print("❌ Validation issues found:")
                    print(json.dumps(data, indent=2))
                except json.JSONDecodeError:
                    print(content.text)

# Example 4: Documentation Generation
async def documentation_example(project_path: str = ".", output_path: str = None):
    """Example: Generate project documentation."""
    print("\n📚 Example 4: Documentation Generation")
    print("=" * 50)
    
    async with DeepflowMCPClient() as client:
        print("Generating dependency map documentation...")
        result = await client.call_tool("generate_documentation", {
            "project_path": project_path,
            "doc_type": "dependency_map",
            "output_path": output_path
        })
        
        print("Documentation Generation Result:")
        for content in result:
            if hasattr(content, 'text'):
                print(content.text)

# Example 5: Performance Monitoring
async def performance_monitoring_example():
    """Example: Monitor MCP tool performance."""
    print("\n📊 Example 5: Performance Monitoring")
    print("=" * 50)
    
    async with DeepflowMCPClient() as client:
        # Time multiple operations
        operations = [
            ("analyze_dependencies", {"project_path": ".", "format": "text"}),
            ("analyze_code_quality", {"project_path": ".", "analysis_type": "imports"}),
            ("validate_commit", {"project_path": "."}),
        ]
        
        performance_data = []
        
        for tool_name, args in operations:
            start_time = time.time()
            try:
                result = await client.call_tool(tool_name, args)
                end_time = time.time()
                duration = end_time - start_time
                
                performance_data.append({
                    "tool": tool_name,
                    "duration": duration,
                    "success": True,
                    "result_size": len(str(result))
                })
                
                print(f"✅ {tool_name}: {duration:.2f}s")
                
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                
                performance_data.append({
                    "tool": tool_name,
                    "duration": duration,
                    "success": False,
                    "error": str(e)
                })
                
                print(f"❌ {tool_name}: {duration:.2f}s (ERROR: {e})")
        
        print("\nPerformance Summary:")
        total_time = sum(p["duration"] for p in performance_data)
        success_count = sum(1 for p in performance_data if p["success"])
        print(f"Total time: {total_time:.2f}s")
        print(f"Successful operations: {success_count}/{len(operations)}")

# Example 6: Batch Analysis
async def batch_analysis_example(project_paths: List[str]):
    """Example: Analyze multiple projects in batch."""
    print("\n🔄 Example 6: Batch Analysis")
    print("=" * 50)
    
    async with DeepflowMCPClient() as client:
        results = {}
        
        for project_path in project_paths:
            if not Path(project_path).exists():
                print(f"❌ Skipping {project_path} (not found)")
                continue
                
            print(f"Analyzing {project_path}...")
            
            try:
                # Quick dependency analysis for each project
                result = await client.call_tool("analyze_dependencies", {
                    "project_path": project_path,
                    "format": "json",
                    "ai_awareness": False  # Faster for batch processing
                })
                
                # Parse result
                for content in result:
                    if hasattr(content, 'text'):
                        try:
                            data = json.loads(content.text)
                            results[project_path] = {
                                "nodes": len(data.get("nodes", [])),
                                "status": "success"
                            }
                        except json.JSONDecodeError:
                            results[project_path] = {
                                "status": "parse_error",
                                "raw_result": content.text[:100]
                            }
                
            except Exception as e:
                results[project_path] = {
                    "status": "error",
                    "error": str(e)
                }
        
        print("\nBatch Analysis Results:")
        for project, result in results.items():
            if result["status"] == "success":
                print(f"✅ {project}: {result['nodes']} modules")
            else:
                print(f"❌ {project}: {result['status']}")

# Example 7: Custom Analysis Workflow
async def custom_workflow_example(project_path: str = "."):
    """Example: Custom analysis workflow combining multiple tools."""
    print("\n🔧 Example 7: Custom Analysis Workflow")
    print("=" * 50)
    
    async with DeepflowMCPClient() as client:
        workflow_results = {}
        
        # Step 1: Basic dependency analysis
        print("Step 1: Analyzing dependencies...")
        deps_result = await client.call_tool("analyze_dependencies", {
            "project_path": project_path,
            "format": "json",
            "ai_awareness": True
        })
        workflow_results["dependencies"] = deps_result
        
        # Step 2: Code quality check
        print("Step 2: Checking code quality...")
        quality_result = await client.call_tool("analyze_code_quality", {
            "project_path": project_path,
            "analysis_type": "all",
            "fix_imports": False
        })
        workflow_results["quality"] = quality_result
        
        # Step 3: Validation
        print("Step 3: Validating current state...")
        validation_result = await client.call_tool("validate_commit", {
            "project_path": project_path,
            "check_dependencies": True,
            "check_patterns": True
        })
        workflow_results["validation"] = validation_result
        
        # Generate summary report
        print("\n📋 Workflow Summary Report")
        print("=" * 30)
        
        # Parse and summarize results
        for step, result in workflow_results.items():
            print(f"\n{step.title()} Results:")
            for content in result:
                if hasattr(content, 'text'):
                    try:
                        data = json.loads(content.text)
                        if step == "dependencies":
                            print(f"  Modules analyzed: {len(data.get('nodes', []))}")
                        elif step == "validation":
                            print(f"  Valid: {'✅' if data.get('valid') else '❌'}")
                        else:
                            print(f"  Data keys: {list(data.keys())}")
                    except json.JSONDecodeError:
                        print(f"  Text result: {len(content.text)} characters")

# Example 8: Error Handling and Retry Logic
async def robust_analysis_example(project_path: str = "."):
    """Example: Robust analysis with error handling and retries."""
    print("\n🛡️ Example 8: Robust Analysis with Error Handling")
    print("=" * 50)
    
    max_retries = 3
    retry_delay = 2
    
    tools_to_test = [
        ("analyze_dependencies", {"project_path": project_path, "format": "text"}),
        ("analyze_code_quality", {"project_path": project_path, "analysis_type": "imports"}),
        ("validate_commit", {"project_path": project_path}),
    ]
    
    async with DeepflowMCPClient() as client:
        for tool_name, args in tools_to_test:
            print(f"\nTesting {tool_name}...")
            
            for attempt in range(max_retries):
                try:
                    result = await client.call_tool(tool_name, args)
                    print(f"✅ {tool_name} succeeded on attempt {attempt + 1}")
                    
                    # Process result
                    for content in result:
                        if hasattr(content, 'text'):
                            result_preview = content.text[:100].replace('\n', ' ')
                            print(f"   Result preview: {result_preview}...")
                    
                    break  # Success, exit retry loop
                    
                except Exception as e:
                    print(f"❌ {tool_name} failed on attempt {attempt + 1}: {e}")
                    
                    if attempt < max_retries - 1:
                        print(f"   Retrying in {retry_delay} seconds...")
                        await asyncio.sleep(retry_delay)
                    else:
                        print(f"   Giving up after {max_retries} attempts")

# Main execution function
async def main():
    """Run all examples."""
    if not MCP_AVAILABLE:
        return
    
    print("🚀 Deepflow MCP Client Examples")
    print("=" * 50)
    print("These examples demonstrate various ways to use Deepflow MCP tools programmatically.")
    print("Make sure the deepflow-mcp-server is running before executing these examples.\n")
    
    # Default project path (current directory)
    project_path = "."
    
    try:
        # Run examples
        await basic_analysis_example(project_path)
        await code_quality_example(project_path)
        await commit_validation_example(project_path)
        await documentation_example(project_path)
        await performance_monitoring_example()
        
        # Batch analysis with multiple paths
        batch_paths = [".", "./tests", "./examples"]  # Adjust as needed
        await batch_analysis_example(batch_paths)
        
        await custom_workflow_example(project_path)
        await robust_analysis_example(project_path)
        
    except Exception as e:
        print(f"\n❌ Example execution failed: {e}")
        print("Make sure the deepflow-mcp-server is running and accessible.")
        print("Start it with: deepflow-mcp-server")

if __name__ == "__main__":
    # Run the examples
    asyncio.run(main())