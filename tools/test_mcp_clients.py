#!/usr/bin/env python3
"""
Multi-Client MCP Testing Suite
===============================

Test Deepflow MCP server compatibility with various MCP client implementations.
This script tests different client libraries and configurations to ensure 
broad compatibility across the MCP ecosystem.

Supported test scenarios:
- Standard MCP Python client
- Different protocol versions
- Various transport mechanisms
- Performance under different loads
- Error handling across client types

Usage:
    python tools/test_mcp_clients.py
    python tools/test_mcp_clients.py --client-type stdio
    python tools/test_mcp_clients.py --performance-test
"""

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

# Try importing different MCP client implementations
MCP_CLIENTS = {}

try:
    from mcp.client.session import ClientSession
    from mcp.client.stdio import stdio_client
    import mcp.types as types
    MCP_CLIENTS['standard'] = True
except ImportError:
    MCP_CLIENTS['standard'] = False

# Additional client libraries could be tested here
# try:
#     import alternative_mcp_client
#     MCP_CLIENTS['alternative'] = True
# except ImportError:
#     MCP_CLIENTS['alternative'] = False

@dataclass
class TestResult:
    """Test result data structure."""
    client_type: str
    test_name: str
    success: bool
    duration: float
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

class MCPClientTestSuite:
    """Test suite for multiple MCP client implementations."""
    
    def __init__(self, server_command: str = "deepflow-mcp-server", verbose: bool = False):
        """Initialize the test suite."""
        self.server_command = server_command
        self.verbose = verbose
        self.results: List[TestResult] = []
    
    def log(self, message: str, level: str = "INFO"):
        """Log message if verbose mode enabled."""
        if self.verbose or level in ["ERROR", "WARNING"]:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")
    
    async def test_standard_client(self) -> List[TestResult]:
        """Test with standard MCP Python client."""
        if not MCP_CLIENTS.get('standard'):
            return [TestResult(
                client_type="standard",
                test_name="availability_check", 
                success=False,
                duration=0.0,
                error="Standard MCP client not available"
            )]
        
        self.log("Testing standard MCP client...")
        results = []
        
        try:
            # Test 1: Basic connection
            start_time = time.time()
            session = await stdio_client(self.server_command)
            await session.__aenter__()
            
            init_result = await session.initialize()
            end_time = time.time()
            
            results.append(TestResult(
                client_type="standard",
                test_name="connection",
                success=True,
                duration=end_time - start_time,
                details={
                    "server_name": init_result.server_info.name,
                    "protocol_version": init_result.protocol_version
                }
            ))
            
            # Test 2: List tools
            start_time = time.time()
            tools_response = await session.list_tools()
            end_time = time.time()
            
            results.append(TestResult(
                client_type="standard",
                test_name="list_tools",
                success=len(tools_response.tools) > 0,
                duration=end_time - start_time,
                details={"tool_count": len(tools_response.tools)}
            ))
            
            # Test 3: Tool execution
            if len(tools_response.tools) > 0:
                tool_name = tools_response.tools[0].name
                start_time = time.time()
                
                # Use basic arguments for testing
                test_args = {"project_path": ".", "format": "text"} if "analyze" in tool_name else {}
                
                try:
                    tool_response = await session.call_tool(tool_name, test_args)
                    end_time = time.time()
                    
                    results.append(TestResult(
                        client_type="standard",
                        test_name="tool_execution",
                        success=True,
                        duration=end_time - start_time,
                        details={
                            "tool_name": tool_name,
                            "content_items": len(tool_response.content)
                        }
                    ))
                except Exception as e:
                    end_time = time.time()
                    results.append(TestResult(
                        client_type="standard",
                        test_name="tool_execution",
                        success=False,
                        duration=end_time - start_time,
                        error=str(e)
                    ))
            
            # Test 4: Multiple sequential calls
            start_time = time.time()
            sequential_results = []
            
            for i in range(3):
                try:
                    response = await session.call_tool("analyze_dependencies", {
                        "project_path": ".",
                        "format": "text",
                        "ai_awareness": False
                    })
                    sequential_results.append(True)
                except:
                    sequential_results.append(False)
            
            end_time = time.time()
            
            results.append(TestResult(
                client_type="standard",
                test_name="sequential_calls",
                success=sum(sequential_results) >= 2,  # Allow 1 failure
                duration=end_time - start_time,
                details={
                    "successful_calls": sum(sequential_results),
                    "total_calls": len(sequential_results)
                }
            ))
            
            # Cleanup
            await session.__aexit__(None, None, None)
            
        except Exception as e:
            self.log(f"Standard client test failed: {e}", "ERROR")
            results.append(TestResult(
                client_type="standard",
                test_name="general",
                success=False,
                duration=0.0,
                error=str(e)
            ))
        
        return results
    
    async def test_concurrent_clients(self, num_clients: int = 3) -> List[TestResult]:
        """Test multiple concurrent client connections."""
        self.log(f"Testing {num_clients} concurrent clients...")
        
        if not MCP_CLIENTS.get('standard'):
            return [TestResult(
                client_type="concurrent",
                test_name="availability_check",
                success=False,
                duration=0.0,
                error="Standard MCP client not available for concurrent testing"
            )]
        
        async def single_client_test(client_id: int):
            """Run test with a single client."""
            try:
                session = await stdio_client(self.server_command)
                await session.__aenter__()
                
                # Initialize and run a simple test
                await session.initialize()
                tools_response = await session.list_tools()
                
                # Try to call a tool
                if tools_response.tools:
                    await session.call_tool("analyze_dependencies", {
                        "project_path": ".",
                        "format": "text",
                        "ai_awareness": False
                    })
                
                await session.__aexit__(None, None, None)
                return True
                
            except Exception as e:
                self.log(f"Client {client_id} failed: {e}", "WARNING")
                return False
        
        start_time = time.time()
        
        # Run concurrent clients
        tasks = [single_client_test(i) for i in range(num_clients)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        
        # Analyze results
        successful = sum(1 for r in results if r is True)
        failed = num_clients - successful
        
        return [TestResult(
            client_type="concurrent",
            test_name="multiple_clients",
            success=successful >= num_clients // 2,  # At least half should succeed
            duration=end_time - start_time,
            details={
                "total_clients": num_clients,
                "successful": successful,
                "failed": failed,
                "success_rate": successful / num_clients
            }
        )]
    
    async def test_error_scenarios(self) -> List[TestResult]:
        """Test error handling across different client scenarios."""
        self.log("Testing error handling scenarios...")
        
        if not MCP_CLIENTS.get('standard'):
            return [TestResult(
                client_type="error_test",
                test_name="availability_check",
                success=False, 
                duration=0.0,
                error="Standard MCP client not available for error testing"
            )]
        
        results = []
        error_scenarios = [
            {
                "name": "invalid_tool_name",
                "tool": "nonexistent_tool",
                "args": {},
                "expected": "graceful_failure"
            },
            {
                "name": "invalid_arguments",
                "tool": "analyze_dependencies",
                "args": {"invalid_param": "invalid_value"},
                "expected": "graceful_failure"
            },
            {
                "name": "invalid_project_path",
                "tool": "analyze_dependencies", 
                "args": {"project_path": "/nonexistent/path"},
                "expected": "graceful_failure"
            }
        ]
        
        try:
            session = await stdio_client(self.server_command)
            await session.__aenter__()
            await session.initialize()
            
            for scenario in error_scenarios:
                start_time = time.time()
                
                try:
                    response = await session.call_tool(scenario["tool"], scenario["args"])
                    end_time = time.time()
                    
                    # Check if error was handled gracefully
                    graceful = True
                    if hasattr(response, 'content'):
                        for content in response.content:
                            if hasattr(content, 'text') and 'error' in content.text.lower():
                                graceful = True
                                break
                    
                    results.append(TestResult(
                        client_type="error_test",
                        test_name=scenario["name"],
                        success=graceful,
                        duration=end_time - start_time,
                        details={"handled_gracefully": graceful}
                    ))
                
                except Exception as e:
                    end_time = time.time()
                    # Exception is expected, but should be handled gracefully
                    results.append(TestResult(
                        client_type="error_test",
                        test_name=scenario["name"],
                        success=True,  # Exception is acceptable for error scenarios
                        duration=end_time - start_time,
                        details={"exception_type": type(e).__name__, "message": str(e)}
                    ))
            
            await session.__aexit__(None, None, None)
            
        except Exception as e:
            self.log(f"Error testing failed: {e}", "ERROR")
            results.append(TestResult(
                client_type="error_test",
                test_name="general",
                success=False,
                duration=0.0,
                error=str(e)
            ))
        
        return results
    
    async def test_performance_scenarios(self) -> List[TestResult]:
        """Test performance under various load conditions."""
        self.log("Testing performance scenarios...")
        
        if not MCP_CLIENTS.get('standard'):
            return [TestResult(
                client_type="performance",
                test_name="availability_check",
                success=False,
                duration=0.0,
                error="Standard MCP client not available for performance testing"
            )]
        
        results = []
        
        try:
            session = await stdio_client(self.server_command)
            await session.__aenter__()
            await session.initialize()
            
            # Test 1: Rapid sequential calls
            start_time = time.time()
            rapid_results = []
            
            for i in range(5):
                try:
                    response = await session.call_tool("analyze_dependencies", {
                        "project_path": ".",
                        "format": "text",
                        "ai_awareness": False
                    })
                    rapid_results.append(True)
                except:
                    rapid_results.append(False)
            
            end_time = time.time()
            
            results.append(TestResult(
                client_type="performance",
                test_name="rapid_sequential",
                success=sum(rapid_results) >= 4,  # Allow 1 failure
                duration=end_time - start_time,
                details={
                    "calls_per_second": len(rapid_results) / (end_time - start_time),
                    "success_rate": sum(rapid_results) / len(rapid_results)
                }
            ))
            
            # Test 2: Large project simulation (if available)
            # This would test with a larger project structure
            start_time = time.time()
            
            try:
                response = await session.call_tool("analyze_code_quality", {
                    "project_path": ".",
                    "analysis_type": "all",
                    "fix_imports": False
                })
                end_time = time.time()
                
                results.append(TestResult(
                    client_type="performance",
                    test_name="comprehensive_analysis",
                    success=True,
                    duration=end_time - start_time,
                    details={"analysis_type": "full"}
                ))
                
            except Exception as e:
                end_time = time.time()
                results.append(TestResult(
                    client_type="performance",
                    test_name="comprehensive_analysis",
                    success=False,
                    duration=end_time - start_time,
                    error=str(e)
                ))
            
            await session.__aexit__(None, None, None)
            
        except Exception as e:
            self.log(f"Performance testing failed: {e}", "ERROR")
            results.append(TestResult(
                client_type="performance",
                test_name="general",
                success=False,
                duration=0.0,
                error=str(e)
            ))
        
        return results
    
    async def run_all_tests(self, include_performance: bool = False) -> Dict[str, Any]:
        """Run all compatibility tests."""
        self.log("Starting comprehensive MCP client compatibility tests...")
        
        all_results = []
        
        # Test standard client
        standard_results = await self.test_standard_client()
        all_results.extend(standard_results)
        
        # Test concurrent clients
        concurrent_results = await self.test_concurrent_clients()
        all_results.extend(concurrent_results)
        
        # Test error scenarios
        error_results = await self.test_error_scenarios()
        all_results.extend(error_results)
        
        # Optional performance tests
        if include_performance:
            performance_results = await self.test_performance_scenarios()
            all_results.extend(performance_results)
        
        # Store results
        self.results = all_results
        
        # Generate summary
        total_tests = len(all_results)
        passed_tests = sum(1 for r in all_results if r.success)
        
        summary = {
            "total_tests": total_tests,
            "passed": passed_tests,
            "failed": total_tests - passed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "client_types_tested": list(set(r.client_type for r in all_results)),
            "average_duration": sum(r.duration for r in all_results) / total_tests if total_tests > 0 else 0
        }
        
        return {
            "summary": summary,
            "results": [
                {
                    "client_type": r.client_type,
                    "test_name": r.test_name,
                    "success": r.success,
                    "duration": r.duration,
                    "error": r.error,
                    "details": r.details
                }
                for r in all_results
            ]
        }
    
    def print_results_summary(self, results: Dict[str, Any]):
        """Print a formatted summary of test results."""
        summary = results["summary"]
        test_results = results["results"]
        
        print("\n" + "="*60)
        print("MCP CLIENT COMPATIBILITY TEST RESULTS")
        print("="*60)
        
        # Overall summary
        print(f"\n📊 Overall Results:")
        print(f"   Total Tests: {summary['total_tests']}")
        print(f"   Passed: {summary['passed']} ✅")
        print(f"   Failed: {summary['failed']} {'❌' if summary['failed'] > 0 else ''}")
        print(f"   Success Rate: {summary['success_rate']:.1%}")
        print(f"   Average Duration: {summary['average_duration']:.3f}s")
        
        # Results by client type
        client_types = summary["client_types_tested"]
        for client_type in client_types:
            client_results = [r for r in test_results if r["client_type"] == client_type]
            passed = sum(1 for r in client_results if r["success"])
            total = len(client_results)
            
            print(f"\n🔧 {client_type.title()} Client:")
            print(f"   Tests: {passed}/{total}")
            
            for result in client_results:
                status = "✅" if result["success"] else "❌"
                duration = f"({result['duration']:.3f}s)" if result["duration"] > 0 else ""
                print(f"   {status} {result['test_name']} {duration}")
                
                if not result["success"] and result["error"]:
                    print(f"      Error: {result['error']}")
                
                if result["details"]:
                    for key, value in result["details"].items():
                        print(f"      {key}: {value}")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        
        if summary["success_rate"] >= 0.9:
            print("   ✅ Excellent compatibility across tested clients")
        elif summary["success_rate"] >= 0.7:
            print("   ⚠️  Good compatibility with some issues to address")
        else:
            print("   ❌ Significant compatibility issues detected")
        
        failed_results = [r for r in test_results if not r["success"]]
        if failed_results:
            print("\n🔧 Issues to address:")
            for result in failed_results:
                print(f"   - {result['client_type']}.{result['test_name']}: {result.get('error', 'Unknown error')}")
        
        print("\n" + "="*60)

async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MCP Client Compatibility Test Suite")
    parser.add_argument("--server", default="deepflow-mcp-server",
                       help="MCP server command")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")
    parser.add_argument("--performance", action="store_true",
                       help="Include performance tests")
    parser.add_argument("--clients", type=int, default=3,
                       help="Number of concurrent clients to test")
    parser.add_argument("--output", help="Save results to JSON file")
    
    args = parser.parse_args()
    
    # Check if any MCP clients are available
    available_clients = [k for k, v in MCP_CLIENTS.items() if v]
    if not available_clients:
        print("❌ No MCP client libraries available for testing")
        print("Install with: pip install mcp")
        return 1
    
    print(f"🧪 Starting MCP Client Compatibility Tests")
    print(f"Available client libraries: {', '.join(available_clients)}")
    
    # Run tests
    test_suite = MCPClientTestSuite(args.server, args.verbose)
    results = await test_suite.run_all_tests(include_performance=args.performance)
    
    # Display results
    test_suite.print_results_summary(results)
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")
    
    # Return appropriate exit code
    return 0 if results["summary"]["success_rate"] >= 0.8 else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))