#!/usr/bin/env python3
"""
Demo script to test real-time intelligence features.
"""

import asyncio
import tempfile
import time
from pathlib import Path

from deepflow.mcp.realtime_intelligence import RealTimeIntelligenceEngine

async def demo_realtime_intelligence():
    """Demonstrate real-time intelligence capabilities."""
    # Create a temporary project directory
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        print(f">>> Testing Real-Time Intelligence in: {project_path}")
        
        # Create a simple Python project
        (project_path / "main.py").write_text("""
import sys
import json
from utils import helper

def main():
    data = {'status': 'ok', 'version': sys.version_info[:2]}
    return helper.process_data(data)

if __name__ == '__main__':
    print(json.dumps(main()))
""")
        
        (project_path / "utils.py").write_text("""
import os
from pathlib import Path

def helper():
    return {'cwd': os.getcwd()}

def process_data(data):
    return {**data, 'processed': True, 'timestamp': time.time()}
""")
        
        # Initialize the real-time intelligence engine
        engine = RealTimeIntelligenceEngine(str(project_path), ai_awareness=True)
        
        # Add notification callback
        def notification_callback(data):
            print(f">> Notification: {data['type']}")
            if 'event' in data:
                event = data['event']
                print(f"   File: {Path(event['file_path']).name}")
                print(f"   Action: {event['event_type']}")
                if event['estimated_tokens'] > 0:
                    print(f"   Tokens: ~{event['estimated_tokens']}")
        
        engine.add_notification_callback(notification_callback)
        
        # Start monitoring
        print("\n>> Starting real-time monitoring...")
        success = await engine.start_monitoring()
        
        if not success:
            print("!! Failed to start monitoring")
            return
        
        print("** Real-time monitoring started!")
        
        # Show initial stats
        stats = engine.get_real_time_stats()
        print(f"\n>> Initial Stats:")
        print(f"   Files monitored: {stats['stats']['files_monitored']}")
        print(f"   AI awareness: {stats['ai_awareness']}")
        
        # Test AI context stats
        context_stats = engine.get_ai_context_stats()
        print(f"\n>> AI Context Analysis:")
        print(f"   Python files: {context_stats['total_python_files']}")
        print(f"   Total tokens: ~{context_stats['total_estimated_tokens']}")
        print(f"   Oversized files: {context_stats['optimal_context_violations']}")
        
        # Start AI session
        session_id = engine.start_ai_session(
            session_name="Demo Session",
            session_description="Testing real-time intelligence features",
            session_tags={"demo", "testing"}
        )
        print(f"\n>> Started AI Session: {session_id}")
        
        # Simulate file changes
        print("\n>> Simulating file changes...")
        
        # Wait a bit for initial setup
        await asyncio.sleep(0.5)
        
        # Modify utils.py
        utils_file = project_path / "utils.py"
        utils_content = utils_file.read_text()
        utils_file.write_text(utils_content + """
import time  # Added import

def format_output(data):
    \"\"\"Format data for output display.\"\"\"
    return json.dumps(data, indent=2)
""")
        print("   Modified utils.py - added new function")
        
        # Wait for processing
        await asyncio.sleep(1.0)
        
        # Create a new file
        new_file = project_path / "config.py"
        new_file.write_text("""
# Configuration settings
DATABASE_URL = 'sqlite:///demo.db'
DEBUG = True
VERSION = '1.0.0'

class Config:
    def __init__(self):
        self.database_url = DATABASE_URL
        self.debug = DEBUG
        self.version = VERSION
""")
        print("   Created config.py")
        
        # Wait for processing
        await asyncio.sleep(1.0)
        
        # Perform impact analysis
        print("\n>> Performing change impact analysis...")
        impact = await engine.analyze_change_impact(
            str(utils_file), 
            "modification",
            {"description": "Added format_output function and time import"}
        )
        
        print(f"   Risk Assessment: {impact.risk_assessment}")
        print(f"   Impact Score: {impact.impact_score:.2f}")
        print(f"   Affected Dependencies: {len(impact.dependency_impacts)}")
        
        # Show recent activity
        print("\n>> Recent Activity:")
        activity = engine.get_recent_activity(limit=5)
        
        for change in activity['changes']:
            print(f"   {Path(change['file_path']).name}: {change['event_type']}")
        
        for update in activity['dependency_updates']:
            if update['added_deps'] or update['removed_deps']:
                print(f"   {Path(update['file_path']).name}: deps changed")
        
        if activity['ai_alerts']:
            for alert in activity['ai_alerts']:
                print(f"   {Path(alert['file_path']).name}: {alert['type']} ({alert['priority']})")
        
        # Show pattern deviations if any
        if activity['pattern_deviations']:
            print("\n>> Pattern Deviations Detected:")
            for deviation in activity['pattern_deviations'][-3:]:
                print(f"   {Path(deviation['file_path']).name}: {deviation['deviation_type']}")
                print(f"     Suggestion: {deviation['suggestion']}")
        
        # End AI session
        print(f"\n>> Ending AI session...")
        completed_session = engine.end_ai_session([
            "Successfully tested real-time monitoring",
            "Validated file change detection",
            "Confirmed pattern analysis works"
        ])
        
        if completed_session:
            duration = completed_session.end_time - completed_session.start_time
            print(f"   Session completed in {duration:.1f} seconds")
            print(f"   Files modified: {len(completed_session.files_modified)}")
            print(f"   Goals achieved: {len(completed_session.goals_achieved)}")
        
        # Get final stats
        final_stats = engine.get_real_time_stats()
        print(f"\n>> Final Statistics:")
        print(f"   Changes processed: {final_stats['stats']['changes_processed']}")
        print(f"   Incremental updates: {final_stats['stats']['incremental_updates']}")
        print(f"   Pattern deviations: {final_stats['stats']['pattern_deviations']}")
        print(f"   AI alerts generated: {final_stats['stats']['alerts_generated']}")
        print(f"   Sessions tracked: {final_stats['stats']['sessions_tracked']}")
        print(f"   Impact analyses: {final_stats['stats']['impact_analyses_performed']}")
        
        # Stop monitoring
        await engine.stop_monitoring()
        print("\n** Real-time monitoring stopped successfully!")
        print("\n** Real-Time Intelligence Demo Complete!")

if __name__ == "__main__":
    print("*** Deepflow Real-Time Intelligence Demo ***")
    print("="*50)
    
    try:
        asyncio.run(demo_realtime_intelligence())
    except KeyboardInterrupt:
        print("\n>>> Demo interrupted by user")
    except Exception as e:
        print(f"\n!!! Demo failed: {e}")
        import traceback
        traceback.print_exc()