#!/usr/bin/env python3
"""
Priority 2: Proactive AI Development Assistance Demo
====================================================

This demo showcases the new Priority 2 features:
- Pattern Deviation Detection for AI-generated code
- Circular Dependency Prevention
- File Split Suggestions based on AI comprehension  
- Duplicate Pattern Identification

Usage:
    python demo_priority2_features.py
"""

import asyncio
import tempfile
from pathlib import Path
import json

# Import our Priority 2 features
from deepflow.mcp.realtime_intelligence import (
    RealTimeIntelligenceEngine,
    FileChangeEvent
)
import time

async def demo_priority2_features():
    """Demonstrate Priority 2: Proactive AI Development Assistance capabilities."""
    
    print("Priority 2: Proactive AI Development Assistance Demo")
    print("=" * 60)
    
    # Create temporary project for demo
    with tempfile.TemporaryDirectory() as temp_dir:
        project_path = Path(temp_dir)
        
        print(f"Creating demo project at: {project_path}")
        
        # Initialize Real-Time Intelligence Engine
        engine = RealTimeIntelligenceEngine(str(project_path), ai_awareness=True)
        
        print("Real-Time Intelligence Engine initialized with Priority 2 features")
        print()
        
        # Demo 1: Pattern Deviation Detection
        print("DEMO 1: Pattern Deviation Detection")
        print("-" * 40)
        
        bad_patterns_file = project_path / "bad_patterns.py"
        bad_patterns_file.write_text("""
# This file contains various pattern violations

import os
from os import path  # Mixed import styles

def BadFunctionName():  # Should be snake_case
    return "Bad naming"

class badClassName:  # Should be PascalCase
    def AnotherBadMethod(self):  # Should be snake_case
        pass

def VeryLongFunctionThatShouldBeBrokenUp():  # Long function name + will be long
    # This function will be very long to trigger structure warnings
""" + "\n    ".join([f"line_{i} = {i}" for i in range(55)]) + """
    return "Very long function"

def good_function():  # This is correct
    return "Good naming"

class GoodClass:  # This is correct
    pass
""")
        
        # Test pattern deviation detection
        change_event = FileChangeEvent(
            file_path=str(bad_patterns_file),
            event_type="modified",
            timestamp=time.time(),
            is_python=True
        )
        
        await engine._check_pattern_deviations(change_event)
        
        print(f"Pattern deviations detected: {len(engine._pattern_deviations)}")
        for deviation in engine._pattern_deviations:
            print(f"  WARNING: {deviation.deviation_type.upper()}: {deviation.actual_pattern}")
            print(f"      Expected: {deviation.expected_pattern}")
            print(f"      Suggestion: {deviation.suggestion}")
            print(f"      Confidence: {deviation.confidence:.1%}")
            print()
        
        # Demo 2: Duplicate Pattern Detection
        print("DEMO 2: Duplicate Pattern Detection")
        print("-" * 40)
        
        duplicate_file = project_path / "duplicates.py"
        duplicate_file.write_text("""
def calculate_area_v1(width, height):
    result = width * height
    return result

def process_data(data):
    cleaned = data.strip().lower()
    return cleaned

def calculate_area_v2(w, h):  # Different names, same logic
    result = w * h
    return result

def clean_input(input_data):  # Different name, same logic as process_data
    cleaned = input_data.strip().lower() 
    return cleaned

def calculate_area_exact_duplicate(width, height):  # Exact duplicate
    result = width * height
    return result
""")
        
        change_event = FileChangeEvent(
            file_path=str(duplicate_file),
            event_type="modified",
            timestamp=time.time(),
            is_python=True
        )
        
        await engine._detect_duplicate_patterns(change_event)
        
        print(f"Duplicate patterns found: {len(engine._duplicate_patterns)}")
        for duplicate in engine._duplicate_patterns:
            print(f"  DUPLICATE {duplicate.pattern_type.upper()}: {duplicate.similarity_score:.0%} similarity")
            locations = duplicate.duplicate_locations
            if len(locations) >= 2:
                print(f"      Functions: {locations[0]['function']} and {locations[1]['function']}")
            print(f"      Suggestion: {duplicate.consolidation_suggestion}")
            print(f"      Savings: {duplicate.estimated_savings}")
            print()
        
        # Demo 3: File Split Suggestions
        print("DEMO 3: File Split Suggestions")
        print("-" * 40)
        
        large_file_content = """
# This is a large file that should be split for better AI comprehension

class UserManager:
    def create_user(self):
        pass
    
    def delete_user(self):
        pass
    
    def update_user(self):
        pass

class OrderManager: 
    def create_order(self):
        pass
    
    def process_order(self):
        pass
    
    def cancel_order(self):
        pass

class PaymentProcessor:
    def process_payment(self):
        pass
    
    def refund_payment(self):
        pass

class EmailService:
    def send_email(self):
        pass
    
    def send_bulk_email(self):
        pass

""" + "\n".join([f"""
def utility_function_{i}():
    '''Utility function {i}'''
    return f"utility_{i}"
""" for i in range(20)])  # Many utility functions
        
        large_file = project_path / "large_monolith.py"
        large_file.write_text(large_file_content)
        
        # Calculate estimated tokens
        estimated_tokens = len(large_file_content) // 4
        
        change_event = FileChangeEvent(
            file_path=str(large_file),
            event_type="modified",
            timestamp=time.time(),
            is_python=True,
            estimated_tokens=estimated_tokens
        )
        
        await engine._suggest_file_splits(change_event)
        
        print(f"File split suggestions: {len(engine._file_split_suggestions)}")
        for suggestion in engine._file_split_suggestions:
            print(f"  FILE: {Path(suggestion.file_path).name}")
            print(f"      Current size: {suggestion.current_size_tokens} tokens")
            print(f"      Priority: {suggestion.priority}")
            print(f"      Rationale: {suggestion.split_rationale}")
            print(f"      Improvement: {suggestion.estimated_improvement}")
            print("      Suggested splits:")
            for split in suggestion.suggested_splits:
                print(f"        - {split['type']}: {split['name']} -> {split['suggested_filename']}")
                print(f"          Reason: {split['rationale']}")
            print()
        
        # Demo 4: Circular Dependency Prevention
        print("DEMO 4: Circular Dependency Prevention")
        print("-" * 40)
        
        # Create files that would create circular dependencies
        file_a = project_path / "module_a.py"
        file_a.write_text("""
import module_b

def function_a():
    return module_b.function_b()
""")
        
        file_b = project_path / "module_b.py"
        file_b.write_text("""
import module_a  # This creates a circular dependency

def function_b():
    return "Hello from B"
""")
        
        # Mock dependency graph for this demo
        from types import SimpleNamespace
        mock_graph = SimpleNamespace()
        mock_graph.nodes = {
            str(file_a): SimpleNamespace(imports=['module_b']),
            str(file_b): SimpleNamespace(imports=['module_a'])
        }
        engine._dependency_graph = mock_graph
        
        change_event = FileChangeEvent(
            file_path=str(file_b),
            event_type="modified",
            timestamp=time.time(),
            is_python=True
        )
        
        await engine._check_circular_dependencies(change_event)
        
        print(f"Circular dependency alerts: {len(engine._circular_dependency_alerts)}")
        for alert in engine._circular_dependency_alerts:
            print(f"  RISK LEVEL: {alert.risk_level.upper()}")
            print(f"      Involved files: {[Path(f).name for f in alert.involved_files]}")
            print(f"      Dependency chain: {alert.dependency_chain}")
            print(f"      Impact: {alert.impact_assessment}")
            print(f"      Prevention: {alert.prevention_suggestion}")
            print()
        
        # Demo 5: Comprehensive Statistics
        print("DEMO 5: Enhanced Real-Time Statistics")
        print("-" * 40)
        
        stats = engine.get_real_time_stats()
        print("Priority 2 Feature Statistics:")
        print(f"  • Pattern deviations detected: {stats['pattern_deviations']}")
        print(f"  • Circular dependencies prevented: {stats['circular_dependency_alerts']}")
        print(f"  • File split suggestions made: {stats['file_split_suggestions']}")
        print(f"  • Duplicate patterns found: {stats['duplicate_patterns']}")
        print()
        
        print("Performance Metrics:")
        for key, value in stats['stats'].items():
            print(f"  • {key.replace('_', ' ').title()}: {value}")
        print()
        
        print("Pattern Learning Progress:")
        learned = stats['learned_patterns']
        print(f"  • Function naming patterns: {learned['naming_functions']}")
        print(f"  • Class naming patterns: {learned['naming_classes']}")
        print(f"  • Import patterns: {learned['import_patterns']}")
        print()
        
        # Demo 6: Recent Activity Report
        print("DEMO 6: Recent Activity Report")
        print("-" * 40)
        
        activity = engine.get_recent_activity(limit=5)
        
        if activity['pattern_deviations']:
            print("Recent Pattern Deviations:")
            for deviation in activity['pattern_deviations'][-3:]:
                print(f"  • {deviation['deviation_type']} in {Path(deviation['file_path']).name}")
                print(f"    Expected: {deviation['expected_pattern']}")
                print(f"    Actual: {deviation['actual_pattern']}")
        print()
        
        if activity['file_split_suggestions']:
            print("Recent File Split Suggestions:")
            for suggestion in activity['file_split_suggestions']:
                print(f"  • {Path(suggestion['file_path']).name}: {suggestion['current_size_tokens']} tokens")
                print(f"    Priority: {suggestion['priority']}")
        print()
        
        if activity['duplicate_patterns']:
            print("Recent Duplicate Patterns:")
            for duplicate in activity['duplicate_patterns']:
                print(f"  • {duplicate['pattern_type']}: {duplicate['similarity_score']:.0%} similarity")
        print()
        
        if activity['circular_dependency_alerts']:
            print("Recent Circular Dependency Alerts:")
            for alert in activity['circular_dependency_alerts']:
                files = [Path(f).name for f in alert['involved_files']]
                print(f"  • {alert['risk_level'].upper()}: {' <-> '.join(files)}")
        print()
    
    print("Priority 2: Proactive AI Development Assistance Demo Complete!")
    print()
    print("Key Features Demonstrated:")
    print("- Pattern Deviation Detection - Identifies inconsistent AI-generated patterns")
    print("- Circular Dependency Prevention - Prevents import cycles before they occur")  
    print("- File Split Suggestions - Optimizes files for AI comprehension")
    print("- Duplicate Pattern Identification - Finds consolidation opportunities")
    print("- Enhanced Real-time Statistics - Comprehensive monitoring metrics")
    print("- AI Development Intelligence - Purpose-built for AI coding workflows")
    print()
    print("Ready for Claude Code integration with MCP tools!")

if __name__ == "__main__":
    asyncio.run(demo_priority2_features())