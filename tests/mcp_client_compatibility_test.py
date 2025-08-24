#!/usr/bin/env python3
"""
MCP Client Compatibility Testing
================================

Test suite for verifying Deepflow MCP server compatibility with various MCP clients.
Tests protocol compliance, performance, and functionality across different clients.

Usage:
    python tests/mcp_client_compatibility_test.py
    pytest tests/mcp_client_compatibility_test.py -v
"""

import asyncio
import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import pytest

try:
    from mcp.client.session import ClientSession
    from mcp.client.stdio import stdio_client
    import mcp.types as types
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    pytest.skip("MCP client libraries not available", allow_module_level=True)

class MCPClientTester:
    """Base class for testing MCP client compatibility."""
    
    def __init__(self, server_command: str = "deepflow-mcp-server"):
        """Initialize the MCP client tester."""
        self.server_command = server_command
        self.session: Optional[ClientSession] = None
        self.server_process: Optional[subprocess.Popen] = None
    
    async def setup(self):
        """Set up the test environment."""
        # Create session
        self.session = await stdio_client(self.server_command)
        await self.session.__aenter__()
        
        # Initialize session
        result = await self.session.initialize()
        return result
    
    async def teardown(self):
        """Clean up the test environment."""
        if self.session:
            await self.session.__aexit__(None, None, None)
    
    async def test_basic_connection(self) -> Dict[str, Any]:
        """Test basic MCP connection and handshake."""
        result = await self.setup()
        
        return {
            "success": True,
            "server_name": result.server_info.name,
            "server_version": result.server_info.version,
            "protocol_version": result.protocol_version
        }
    
    async def test_list_tools(self) -> Dict[str, Any]:
        """Test listing available tools."""
        response = await self.session.list_tools()
        
        tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "has_schema": bool(tool.inputSchema)
            }
            for tool in response.tools
        ]
        
        expected_tools = [
            "analyze_dependencies",
            "analyze_code_quality", 
            "validate_commit",
            "generate_documentation"
        ]
        
        missing_tools = [tool for tool in expected_tools if tool not in [t["name"] for t in tools]]
        
        return {
            "success": len(missing_tools) == 0,
            "tools": tools,
            "expected_tools": expected_tools,
            "missing_tools": missing_tools,
            "tool_count": len(tools)
        }
    
    async def test_tool_execution(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Test executing a specific tool."""
        start_time = time.time()
        
        try:
            response = await self.session.call_tool(tool_name, arguments)
            end_time = time.time()
            
            # Analyze response
            content_types = []
            total_content_length = 0
            
            for content in response.content:
                if hasattr(content, 'type'):
                    content_types.append(content.type)
                if hasattr(content, 'text'):
                    total_content_length += len(content.text)
            
            return {
                "success": True,
                "tool_name": tool_name,
                "execution_time": end_time - start_time,
                "content_types": content_types,
                "content_length": total_content_length,
                "response_count": len(response.content)
            }
            
        except Exception as e:
            end_time = time.time()
            return {
                "success": False,
                "tool_name": tool_name,
                "execution_time": end_time - start_time,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    async def test_concurrent_requests(self, max_concurrent: int = 3) -> Dict[str, Any]:
        """Test handling of concurrent requests."""
        # Create multiple concurrent requests
        tasks = []
        for i in range(max_concurrent):
            task = self.test_tool_execution("analyze_dependencies", {
                "project_path": ".",
                "format": "text",
                "ai_awareness": False  # Faster for testing
            })
            tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        end_time = time.time()
        
        successful = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
        failed = len(results) - successful
        
        return {
            "success": failed == 0,
            "total_requests": max_concurrent,
            "successful": successful,
            "failed": failed,
            "total_time": end_time - start_time,
            "avg_time_per_request": (end_time - start_time) / max_concurrent,
            "results": results
        }
    
    async def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling with invalid inputs."""
        error_tests = [
            {
                "name": "invalid_tool",
                "tool": "nonexistent_tool",
                "args": {}
            },
            {
                "name": "invalid_path",
                "tool": "analyze_dependencies",
                "args": {"project_path": "/nonexistent/path"}
            },
            {
                "name": "invalid_format",
                "tool": "analyze_dependencies", 
                "args": {"project_path": ".", "format": "invalid_format"}
            }
        ]
        
        results = []
        
        for test in error_tests:
            try:
                result = await self.test_tool_execution(test["tool"], test["args"])
                results.append({
                    "test": test["name"],
                    "handled_gracefully": not result.get("success", True),
                    "result": result
                })
            except Exception as e:
                results.append({
                    "test": test["name"],
                    "handled_gracefully": False,
                    "exception": str(e)
                })
        
        graceful_handling = sum(1 for r in results if r.get("handled_gracefully"))
        
        return {
            "success": graceful_handling == len(error_tests),
            "total_tests": len(error_tests),
            "gracefully_handled": graceful_handling,
            "results": results
        }

class ProtocolComplianceTest:
    """Test MCP protocol compliance."""
    
    @staticmethod
    def validate_tool_schema(tool_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that tool schema follows JSON Schema specification."""
        required_fields = ["type", "properties"]
        issues = []
        
        for field in required_fields:
            if field not in tool_schema:
                issues.append(f"Missing required field: {field}")
        
        if tool_schema.get("type") != "object":
            issues.append(f"Expected type 'object', got '{tool_schema.get('type')}'")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "schema": tool_schema
        }
    
    @staticmethod
    def validate_response_format(response_content: List[Any]) -> Dict[str, Any]:
        """Validate MCP response format compliance."""
        issues = []
        
        if not isinstance(response_content, list):
            issues.append("Response content must be a list")
        
        for i, content in enumerate(response_content):
            if not hasattr(content, 'type'):
                issues.append(f"Content item {i} missing 'type' field")
            
            if hasattr(content, 'type') and content.type == 'text':
                if not hasattr(content, 'text'):
                    issues.append(f"Text content item {i} missing 'text' field")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "content_count": len(response_content)
        }

# Pytest test cases
@pytest.fixture
async def mcp_client():
    """Fixture to provide MCP client for testing."""
    client = MCPClientTester()
    await client.setup()
    yield client
    await client.teardown()

@pytest.mark.asyncio
async def test_connection_establishment():
    """Test that MCP connection can be established."""
    client = MCPClientTester()
    
    try:
        result = await client.test_basic_connection()
        assert result["success"], "Failed to establish MCP connection"
        assert "server_name" in result, "Server name not provided"
        
    finally:
        await client.teardown()

@pytest.mark.asyncio
async def test_tool_discovery(mcp_client):
    """Test that all expected tools are available."""
    result = await mcp_client.test_list_tools()
    
    assert result["success"], f"Missing tools: {result.get('missing_tools', [])}"
    assert result["tool_count"] >= 4, f"Expected at least 4 tools, found {result['tool_count']}"
    
    # Verify each tool has required properties
    for tool in result["tools"]:
        assert tool["name"], "Tool missing name"
        assert tool["description"], "Tool missing description"
        assert tool["has_schema"], f"Tool {tool['name']} missing input schema"

@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name,args", [
    ("analyze_dependencies", {"project_path": ".", "format": "text"}),
    ("analyze_code_quality", {"project_path": ".", "analysis_type": "imports"}),
    ("validate_commit", {"project_path": "."}),
])
async def test_individual_tool_execution(mcp_client, tool_name, args):
    """Test execution of individual tools."""
    result = await mcp_client.test_tool_execution(tool_name, args)
    
    if not result["success"]:
        # Allow certain failures in test environment
        if "not a git repository" in result.get("error", "").lower():
            pytest.skip("Test environment not a git repository")
        if "no python files" in result.get("error", "").lower():
            pytest.skip("Test environment has no Python files to analyze")
    
    assert result["success"], f"Tool {tool_name} failed: {result.get('error')}"
    assert result["execution_time"] < 30.0, f"Tool {tool_name} took too long: {result['execution_time']}s"
    assert result["content_length"] > 0, f"Tool {tool_name} returned empty content"

@pytest.mark.asyncio
async def test_concurrent_execution(mcp_client):
    """Test concurrent request handling."""
    result = await mcp_client.test_concurrent_requests(max_concurrent=3)
    
    # Allow some failures in test environment
    if result["failed"] > 0:
        failure_reasons = []
        for res in result.get("results", []):
            if isinstance(res, dict) and not res.get("success"):
                failure_reasons.append(res.get("error", "unknown"))
        
        # Skip if failures are due to test environment
        if any("git repository" in reason.lower() for reason in failure_reasons):
            pytest.skip("Test environment issues")
    
    assert result["successful"] >= 2, f"Too many concurrent failures: {result['failed']}/{result['total_requests']}"
    assert result["total_time"] < 60.0, f"Concurrent execution took too long: {result['total_time']}s"

@pytest.mark.asyncio
async def test_error_handling_compliance(mcp_client):
    """Test that errors are handled gracefully."""
    result = await mcp_client.test_error_handling()
    
    assert result["gracefully_handled"] >= result["total_tests"] // 2, \
        f"Poor error handling: {result['gracefully_handled']}/{result['total_tests']} handled gracefully"

@pytest.mark.asyncio
async def test_protocol_compliance(mcp_client):
    """Test MCP protocol compliance."""
    # Test tool listing compliance
    tools_result = await mcp_client.test_list_tools()
    assert tools_result["success"], "Tool listing failed"
    
    # Test schema compliance for each tool
    for tool in tools_result["tools"]:
        # This would require access to the raw tool schema
        # For now, just check that tools have expected properties
        assert tool["name"], f"Tool missing name"
        assert tool["has_schema"], f"Tool {tool['name']} missing schema"
    
    # Test response format compliance
    test_result = await mcp_client.test_tool_execution("analyze_dependencies", {
        "project_path": ".",
        "format": "text"
    })
    
    if test_result["success"]:
        assert test_result["content_types"], "Response missing content types"
        assert "text" in test_result["content_types"], "Expected text content in response"

# Performance benchmarks
@pytest.mark.benchmark
@pytest.mark.asyncio
async def test_performance_benchmarks():
    """Performance benchmark tests."""
    client = MCPClientTester()
    
    try:
        await client.setup()
        
        # Benchmark individual tools
        benchmarks = {}
        
        tools_to_benchmark = [
            ("analyze_dependencies", {"project_path": ".", "format": "text", "ai_awareness": False}),
            ("analyze_code_quality", {"project_path": ".", "analysis_type": "imports"}),
        ]
        
        for tool_name, args in tools_to_benchmark:
            times = []
            
            # Run multiple times for average
            for _ in range(3):
                result = await client.test_tool_execution(tool_name, args)
                if result["success"]:
                    times.append(result["execution_time"])
            
            if times:
                benchmarks[tool_name] = {
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "runs": len(times)
                }
        
        # Print benchmark results
        print("\n" + "="*50)
        print("PERFORMANCE BENCHMARKS")
        print("="*50)
        for tool, metrics in benchmarks.items():
            print(f"{tool}:")
            print(f"  Average: {metrics['avg_time']:.3f}s")
            print(f"  Range: {metrics['min_time']:.3f}s - {metrics['max_time']:.3f}s")
        print("="*50)
        
        # Assert reasonable performance
        for tool, metrics in benchmarks.items():
            assert metrics["avg_time"] < 10.0, f"{tool} too slow: {metrics['avg_time']}s"
    
    finally:
        await client.teardown()

if __name__ == "__main__":
    # Run tests directly
    async def run_compatibility_tests():
        """Run all compatibility tests."""
        print("🧪 Running MCP Client Compatibility Tests")
        print("=" * 50)
        
        client = MCPClientTester()
        
        try:
            # Test 1: Basic connection
            print("1. Testing basic connection...")
            connection_result = await client.test_basic_connection()
            print(f"   {'✅' if connection_result['success'] else '❌'} Connection: {connection_result.get('server_name', 'Unknown')}")
            
            # Test 2: Tool discovery
            print("2. Testing tool discovery...")
            tools_result = await client.test_list_tools()
            print(f"   {'✅' if tools_result['success'] else '❌'} Tools: {tools_result['tool_count']} found")
            if tools_result.get("missing_tools"):
                print(f"   Missing: {tools_result['missing_tools']}")
            
            # Test 3: Individual tool execution
            print("3. Testing tool execution...")
            test_tools = [
                ("analyze_dependencies", {"project_path": ".", "format": "text"}),
                ("analyze_code_quality", {"project_path": ".", "analysis_type": "imports"}),
            ]
            
            for tool_name, args in test_tools:
                result = await client.test_tool_execution(tool_name, args)
                status = "✅" if result["success"] else "❌"
                time_str = f"({result['execution_time']:.2f}s)"
                print(f"   {status} {tool_name} {time_str}")
                if not result["success"]:
                    print(f"      Error: {result.get('error', 'Unknown')}")
            
            # Test 4: Concurrent execution
            print("4. Testing concurrent execution...")
            concurrent_result = await client.test_concurrent_requests(max_concurrent=3)
            success_rate = concurrent_result["successful"] / concurrent_result["total_requests"]
            status = "✅" if success_rate >= 0.66 else "❌"  # Allow 1/3 failures
            print(f"   {status} Concurrent: {concurrent_result['successful']}/{concurrent_result['total_requests']} succeeded")
            
            # Test 5: Error handling
            print("5. Testing error handling...")
            error_result = await client.test_error_handling()
            status = "✅" if error_result["success"] else "❌"
            print(f"   {status} Error handling: {error_result['gracefully_handled']}/{error_result['total_tests']} handled gracefully")
            
            print("\n" + "=" * 50)
            print("✅ MCP Client Compatibility Testing Complete")
            
        except Exception as e:
            print(f"❌ Testing failed: {e}")
            raise
        
        finally:
            await client.teardown()
    
    # Run the tests
    if MCP_AVAILABLE:
        asyncio.run(run_compatibility_tests())
    else:
        print("❌ MCP libraries not available. Install with: pip install mcp")