#!/usr/bin/env python3
"""
Deepflow MCP Test Client
========================

Standalone test client for validating Deepflow MCP server functionality.
Can be used to test compatibility with various MCP client implementations.

This tool provides:
- Manual testing capabilities
- Client implementation examples  
- Protocol validation
- Performance testing
- Cross-platform compatibility testing

Usage:
    python tools/mcp_test_client.py --help
    python tools/mcp_test_client.py test-connection
    python tools/mcp_test_client.py run-analysis --tool analyze_dependencies
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    from mcp.client.session import ClientSession
    from mcp.client.stdio import stdio_client
    import mcp.types as types
    MCP_AVAILABLE = True
except ImportError:
    print("ERROR: MCP client libraries not available.")
    print("Install with: pip install mcp")
    MCP_AVAILABLE = False

class DeepflowMCPTestClient:
    """Test client for Deepflow MCP server."""
    
    def __init__(self, server_command: str = "deepflow-mcp-server", verbose: bool = False):
        """Initialize the test client."""
        self.server_command = server_command
        self.verbose = verbose
        self.session: Optional[ClientSession] = None
    
    def log(self, message: str, level: str = "INFO"):
        """Log message if verbose mode is enabled."""
        if self.verbose or level in ["ERROR", "WARNING"]:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] {level}: {message}")
    
    async def connect(self) -> Dict[str, Any]:
        """Establish connection to MCP server."""
        try:
            self.log("Connecting to MCP server...")
            self.session = await stdio_client(self.server_command)
            await self.session.__aenter__()
            
            self.log("Initializing MCP session...")
            result = await self.session.initialize()
            
            connection_info = {
                "success": True,
                "server_name": result.server_info.name,
                "server_version": result.server_info.version,
                "protocol_version": result.protocol_version,
                "capabilities": result.capabilities.__dict__ if hasattr(result, 'capabilities') else {}
            }
            
            self.log(f"Connected to server: {connection_info['server_name']}")
            return connection_info
            
        except Exception as e:
            self.log(f"Connection failed: {e}", "ERROR")
            return {"success": False, "error": str(e)}
    
    async def disconnect(self):
        """Disconnect from MCP server."""
        if self.session:
            self.log("Disconnecting from MCP server...")
            await self.session.__aexit__(None, None, None)
            self.session = None
    
    async def list_tools(self) -> Dict[str, Any]:
        """List all available MCP tools."""
        try:
            self.log("Requesting tool list...")
            response = await self.session.list_tools()
            
            tools = []
            for tool in response.tools:
                tool_info = {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                }
                tools.append(tool_info)
            
            self.log(f"Found {len(tools)} tools")
            return {"success": True, "tools": tools}
            
        except Exception as e:
            self.log(f"Failed to list tools: {e}", "ERROR")
            return {"success": False, "error": str(e)}
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Call a specific MCP tool."""
        try:
            self.log(f"Calling tool: {tool_name}")
            self.log(f"Arguments: {json.dumps(arguments, indent=2)}")
            
            start_time = time.time()
            response = await self.session.call_tool(tool_name, arguments)
            end_time = time.time()
            
            # Process response content
            content_data = []
            total_text_length = 0
            
            for content in response.content:
                content_info = {
                    "type": getattr(content, 'type', 'unknown'),
                }
                
                if hasattr(content, 'text'):
                    content_info["text"] = content.text
                    content_info["length"] = len(content.text)
                    total_text_length += len(content.text)
                
                content_data.append(content_info)
            
            result = {
                "success": True,
                "tool_name": tool_name,
                "execution_time": end_time - start_time,
                "content": content_data,
                "total_text_length": total_text_length,
                "content_count": len(content_data)
            }
            
            self.log(f"Tool completed in {result['execution_time']:.2f}s")
            self.log(f"Response: {result['content_count']} content items, {total_text_length} total characters")
            
            return result
            
        except Exception as e:
            self.log(f"Tool call failed: {e}", "ERROR")
            return {"success": False, "error": str(e), "tool_name": tool_name}

class InteractiveTester:
    """Interactive testing interface."""
    
    def __init__(self, client: DeepflowMCPTestClient):
        self.client = client
        self.tools_cache: Optional[List[Dict[str, Any]]] = None
    
    async def start_interactive_session(self):
        """Start interactive testing session."""
        print("🧪 Deepflow MCP Interactive Test Client")
        print("=" * 50)
        
        # Connect to server
        connection = await self.client.connect()
        if not connection["success"]:
            print(f"❌ Failed to connect: {connection['error']}")
            return
        
        print(f"✅ Connected to {connection['server_name']}")
        
        try:
            # Load available tools
            tools_result = await self.client.list_tools()
            if tools_result["success"]:
                self.tools_cache = tools_result["tools"]
                print(f"📋 Available tools: {len(self.tools_cache)}")
                for i, tool in enumerate(self.tools_cache):
                    print(f"  {i+1}. {tool['name']} - {tool['description']}")
            
            # Interactive loop
            while True:
                print("\n" + "=" * 30)
                print("Commands:")
                print("  1. List tools")
                print("  2. Call tool") 
                print("  3. Test all tools")
                print("  4. Performance benchmark")
                print("  5. Exit")
                
                choice = input("\nEnter choice (1-5): ").strip()
                
                if choice == "1":
                    await self.show_tools()
                elif choice == "2":
                    await self.interactive_tool_call()
                elif choice == "3":
                    await self.test_all_tools()
                elif choice == "4":
                    await self.run_benchmark()
                elif choice == "5":
                    break
                else:
                    print("Invalid choice. Please enter 1-5.")
        
        finally:
            await self.client.disconnect()
    
    async def show_tools(self):
        """Display detailed tool information."""
        if not self.tools_cache:
            result = await self.client.list_tools()
            if not result["success"]:
                print(f"❌ Failed to get tools: {result['error']}")
                return
            self.tools_cache = result["tools"]
        
        print("\n📋 Available Tools:")
        print("=" * 50)
        
        for i, tool in enumerate(self.tools_cache):
            print(f"\n{i+1}. {tool['name']}")
            print(f"   Description: {tool['description']}")
            
            # Show schema properties
            schema = tool.get('input_schema', {})
            properties = schema.get('properties', {})
            if properties:
                print("   Parameters:")
                for param, details in properties.items():
                    param_type = details.get('type', 'unknown')
                    default = details.get('default', 'N/A')
                    description = details.get('description', 'No description')
                    print(f"     - {param} ({param_type}): {description}")
                    if default != 'N/A':
                        print(f"       Default: {default}")
    
    async def interactive_tool_call(self):
        """Interactive tool calling."""
        if not self.tools_cache:
            print("❌ No tools available")
            return
        
        print("\nSelect a tool to call:")
        for i, tool in enumerate(self.tools_cache):
            print(f"  {i+1}. {tool['name']}")
        
        try:
            choice = int(input("Enter tool number: ")) - 1
            if choice < 0 or choice >= len(self.tools_cache):
                print("❌ Invalid tool selection")
                return
        except ValueError:
            print("❌ Invalid input")
            return
        
        selected_tool = self.tools_cache[choice]
        tool_name = selected_tool['name']
        
        print(f"\nCalling tool: {tool_name}")
        
        # Get parameters
        schema = selected_tool.get('input_schema', {})
        properties = schema.get('properties', {})
        arguments = {}
        
        for param, details in properties.items():
            param_type = details.get('type', 'string')
            default = details.get('default')
            description = details.get('description', '')
            
            if default is not None:
                prompt = f"{param} ({param_type}) [{default}]: {description}\n> "
            else:
                prompt = f"{param} ({param_type}): {description}\n> "
            
            value = input(prompt).strip()
            
            if not value and default is not None:
                value = default
            elif not value:
                continue  # Skip optional parameters
            
            # Type conversion
            if param_type == 'boolean':
                value = value.lower() in ('true', 'yes', '1', 'on')
            elif param_type == 'integer':
                try:
                    value = int(value)
                except ValueError:
                    print(f"⚠️  Invalid integer, using string: {value}")
            
            arguments[param] = value
        
        # Call the tool
        print(f"\n🔧 Calling {tool_name} with arguments:")
        print(json.dumps(arguments, indent=2))
        
        result = await self.client.call_tool(tool_name, arguments)
        
        if result["success"]:
            print(f"✅ Tool completed in {result['execution_time']:.2f}s")
            
            # Display results
            for i, content in enumerate(result["content"]):
                print(f"\nContent {i+1} ({content['type']}):")
                if content.get('text'):
                    text = content['text']
                    if len(text) > 500:
                        print(text[:500] + f"... ({len(text)} total characters)")
                    else:
                        print(text)
        else:
            print(f"❌ Tool failed: {result['error']}")
    
    async def test_all_tools(self):
        """Test all available tools with default parameters."""
        if not self.tools_cache:
            print("❌ No tools available")
            return
        
        print("\n🧪 Testing all tools with default parameters...")
        
        # Default test cases for each tool
        test_cases = {
            "analyze_dependencies": {"project_path": ".", "format": "text", "ai_awareness": False},
            "analyze_code_quality": {"project_path": ".", "analysis_type": "imports", "fix_imports": False},
            "validate_commit": {"project_path": ".", "check_dependencies": True, "check_patterns": False},
            "generate_documentation": {"project_path": ".", "doc_type": "dependency_map"}
        }
        
        results = []
        
        for tool in self.tools_cache:
            tool_name = tool['name']
            test_args = test_cases.get(tool_name, {"project_path": "."})
            
            print(f"\n🔧 Testing {tool_name}...")
            result = await self.client.call_tool(tool_name, test_args)
            
            if result["success"]:
                print(f"  ✅ Passed ({result['execution_time']:.2f}s)")
                status = "PASS"
            else:
                print(f"  ❌ Failed: {result['error']}")
                status = "FAIL"
            
            results.append({
                "tool": tool_name,
                "status": status,
                "time": result.get('execution_time', 0),
                "error": result.get('error')
            })
        
        # Summary
        passed = sum(1 for r in results if r['status'] == 'PASS')
        total = len(results)
        
        print(f"\n📊 Test Summary: {passed}/{total} passed")
        
        if passed < total:
            print("\nFailed tests:")
            for result in results:
                if result['status'] == 'FAIL':
                    print(f"  ❌ {result['tool']}: {result['error']}")
    
    async def run_benchmark(self):
        """Run performance benchmarks."""
        if not self.tools_cache:
            print("❌ No tools available")
            return
        
        print("\n⏱️  Running performance benchmarks...")
        
        # Benchmark parameters
        iterations = 3
        test_cases = {
            "analyze_dependencies": {"project_path": ".", "format": "text", "ai_awareness": False},
            "analyze_code_quality": {"project_path": ".", "analysis_type": "imports"}
        }
        
        benchmarks = {}
        
        for tool_name, args in test_cases.items():
            if not any(t['name'] == tool_name for t in self.tools_cache):
                continue
            
            print(f"\n🏃 Benchmarking {tool_name} ({iterations} runs)...")
            
            times = []
            success_count = 0
            
            for i in range(iterations):
                result = await self.client.call_tool(tool_name, args)
                if result["success"]:
                    times.append(result['execution_time'])
                    success_count += 1
                    print(f"  Run {i+1}: {result['execution_time']:.3f}s")
                else:
                    print(f"  Run {i+1}: FAILED - {result['error']}")
            
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                
                benchmarks[tool_name] = {
                    "success_rate": success_count / iterations,
                    "avg_time": avg_time,
                    "min_time": min_time,
                    "max_time": max_time,
                    "times": times
                }
                
                print(f"  Average: {avg_time:.3f}s")
                print(f"  Range: {min_time:.3f}s - {max_time:.3f}s")
                print(f"  Success rate: {success_count}/{iterations}")
        
        # Summary
        print(f"\n📊 Benchmark Summary:")
        print("=" * 40)
        for tool, metrics in benchmarks.items():
            print(f"{tool}:")
            print(f"  Success rate: {metrics['success_rate']:.1%}")
            print(f"  Average time: {metrics['avg_time']:.3f}s")
            print(f"  Performance: {'✅ Good' if metrics['avg_time'] < 5.0 else '⚠️  Slow'}")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Deepflow MCP Test Client")
    parser.add_argument("--server", default="deepflow-mcp-server", 
                       help="MCP server command (default: deepflow-mcp-server)")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Enable verbose logging")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Test connection command
    subparsers.add_parser("test-connection", help="Test MCP server connection")
    
    # List tools command
    subparsers.add_parser("list-tools", help="List available MCP tools")
    
    # Run analysis command
    analysis_parser = subparsers.add_parser("run-analysis", help="Run analysis tool")
    analysis_parser.add_argument("--tool", required=True, help="Tool name to run")
    analysis_parser.add_argument("--args", type=json.loads, default={}, 
                                help="Tool arguments as JSON")
    
    # Interactive command
    subparsers.add_parser("interactive", help="Start interactive session")
    
    # Benchmark command
    subparsers.add_parser("benchmark", help="Run performance benchmarks")
    
    args = parser.parse_args()
    
    if not MCP_AVAILABLE:
        return 1
    
    if not args.command:
        parser.print_help()
        return 1
    
    client = DeepflowMCPTestClient(args.server, args.verbose)
    
    async def run_command():
        if args.command == "test-connection":
            connection = await client.connect()
            if connection["success"]:
                print(f"✅ Connected to {connection['server_name']} (protocol: {connection['protocol_version']})")
                await client.disconnect()
                return 0
            else:
                print(f"❌ Connection failed: {connection['error']}")
                return 1
        
        elif args.command == "list-tools":
            await client.connect()
            try:
                result = await client.list_tools()
                if result["success"]:
                    print(json.dumps(result["tools"], indent=2))
                    return 0
                else:
                    print(f"❌ Failed to list tools: {result['error']}")
                    return 1
            finally:
                await client.disconnect()
        
        elif args.command == "run-analysis":
            await client.connect()
            try:
                result = await client.call_tool(args.tool, args.args)
                if result["success"]:
                    print(json.dumps({
                        "success": True,
                        "execution_time": result["execution_time"],
                        "content": result["content"]
                    }, indent=2))
                    return 0
                else:
                    print(f"❌ Analysis failed: {result['error']}")
                    return 1
            finally:
                await client.disconnect()
        
        elif args.command == "interactive":
            tester = InteractiveTester(client)
            await tester.start_interactive_session()
            return 0
        
        elif args.command == "benchmark":
            await client.connect()
            try:
                tester = InteractiveTester(client)
                await tester.run_benchmark()
                return 0
            finally:
                await client.disconnect()
    
    return asyncio.run(run_command())

if __name__ == "__main__":
    sys.exit(main())