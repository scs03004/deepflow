#!/usr/bin/env python3
"""
Real-Time Intelligence Demo for Deepflow
========================================

This demo showcases the new Real-Time Intelligence features:
- Live file monitoring with watchdog
- Incremental dependency graph updates
- AI context window monitoring
- Architectural violation detection
- MCP integration for Claude Code

Usage:
    python demo_realtime_intelligence.py
"""

import asyncio
import json
import time
import tempfile
from pathlib import Path

# Import our real-time intelligence system
from deepflow.mcp.realtime_intelligence import RealTimeIntelligenceEngine
from deepflow.mcp.server import DeepflowMCPServer

async def demo_realtime_intelligence():
    """Demonstrate the Real-Time Intelligence capabilities."""
    
    print("Deepflow Real-Time Intelligence Demo")
    print("=" * 50)
    
    # Create a temporary project for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        
        # Create some demo Python files
        demo_files = {
            "main.py": """
import os
import sys
from utils import helper_function

def main():
    print("Hello, Real-Time Intelligence!")
    helper_function()

if __name__ == "__main__":
    main()
""",
            "utils.py": """
import json
import requests

def helper_function():
    data = {"message": "Real-time monitoring is working!"}
    return json.dumps(data, indent=2)

def unused_function():
    # This function will trigger unused import detection
    pass
""",
            "large_file.py": """
# This file will trigger AI context alerts
""" + "\n".join([f"def function_{i}():\n    pass\n" for i in range(200)])
        }
        
        print(f"Creating demo project at: {project_path}")
        for filename, content in demo_files.items():
            (project_path / filename).write_text(content)
        
        print(f"Created {len(demo_files)} demo files")
        print()
        
        # Initialize Real-Time Intelligence Engine
        print("Initializing Real-Time Intelligence Engine...")
        engine = RealTimeIntelligenceEngine(str(project_path), ai_awareness=True)
        
        # Set up notification callback
        notifications = []
        def notification_callback(data):
            notifications.append(data)
            print(f"Real-time notification: {data['type']} - {data['event']['file_path']}")
        
        engine.add_notification_callback(notification_callback)
        
        # Start monitoring
        print("Starting real-time monitoring...")
        monitoring_started = await engine.start_monitoring()
        
        if monitoring_started:
            print("Real-time monitoring active!")
            
            # Get initial stats
            stats = engine.get_real_time_stats()
            print(f"Monitoring {stats['stats']['files_monitored']} files")
            print()
            
            # Demonstrate file changes
            print("Making changes to files to trigger monitoring...")
            
            # Modify main.py to add an import
            main_file = project_path / "main.py"
            modified_content = """
import os
import sys
import datetime  # New import added!
from utils import helper_function

def main():
    print(f"Hello at {datetime.datetime.now()}!")
    helper_function()

if __name__ == "__main__":
    main()
"""
            main_file.write_text(modified_content)
            print("   - Modified main.py (added datetime import)")
            
            # Create a new file
            new_file = project_path / "new_module.py"
            new_file.write_text("""
import numpy as np  # This will be detected as a new dependency

def new_function():
    return "Created by real-time intelligence demo"
""")
            print("   - Created new_module.py")
            
            # Wait for file system events to be processed
            await asyncio.sleep(2)
            
            # Get updated stats and activity
            updated_stats = engine.get_real_time_stats()
            activity = engine.get_recent_activity(limit=10)
            
            print()
            print("Real-time Analysis Results:")
            print(f"   - Changes processed: {updated_stats['stats']['changes_processed']}")
            print(f"   - Dependency updates: {updated_stats['stats']['incremental_updates']}")
            print(f"   - AI alerts generated: {updated_stats['stats']['alerts_generated']}")
            
            if activity['changes']:
                print("   - Recent file changes:")
                for change in activity['changes']:
                    print(f"     - {change['event_type']}: {Path(change['file_path']).name}")
            
            if activity['ai_alerts']:
                print("   - AI context alerts:")
                for alert in activity['ai_alerts']:
                    print(f"     - {alert['type']}: {Path(alert['file_path']).name}")
            
            # Stop monitoring
            await engine.stop_monitoring()
            print("Stopped real-time monitoring")
            
        else:
            print("Could not start real-time monitoring (watchdog may not be available)")
    
    print()
    print("Testing MCP Integration...")
    
    # Test MCP server with real-time tools
    server = DeepflowMCPServer()
    tools = server.get_tools()
    realtime_tools = [tool for tool in tools if 'realtime' in tool.name]
    
    print(f"MCP Server has {len(realtime_tools)} real-time tools:")
    for tool in realtime_tools:
        print(f"   - {tool.name}: {tool.description}")
    
    # Test a real-time tool via MCP
    print()
    print("Testing MCP real-time tool...")
    
    try:
        result = await server._handle_start_realtime_monitoring({
            "project_path": ".",
            "ai_awareness": True
        })
        
        response = json.loads(result[0].text)
        print(f"   MCP Tool Response: {response['status']}")
        
        # Get stats
        stats_result = await server._handle_get_realtime_stats({})
        stats = json.loads(stats_result[0].text)
        print(f"   Monitoring Status: {stats.get('monitoring', False)}")
        
        # Stop monitoring
        await server._handle_stop_realtime_monitoring({})
        print("   Stopped MCP monitoring")
        
    except Exception as e:
        print(f"   MCP test error: {e}")
    
    print()
    print("Real-Time Intelligence Demo Complete!")
    print()
    print("Key Features Demonstrated:")
    print("- Live file monitoring with watchdog")
    print("- Incremental dependency graph updates") 
    print("- Real-time notifications")
    print("- AI context window monitoring")
    print("- MCP integration for Claude Code")
    print()
    print("Ready for Claude Code integration!")
    print("   Use: deepflow-mcp-server")

if __name__ == "__main__":
    asyncio.run(demo_realtime_intelligence())