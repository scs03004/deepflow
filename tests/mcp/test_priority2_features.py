#!/usr/bin/env python3
"""
Comprehensive Test Suite for Priority 2: Proactive AI Development Assistance
============================================================================

Tests for:
- Pattern Deviation Detection
- Circular Dependency Prevention  
- File Split Suggestions
- Duplicate Pattern Identification
- AI Development Intelligence Features
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
import json
import time

# Test imports with graceful fallbacks
try:
    from deepflow.mcp.realtime_intelligence import (
        RealTimeIntelligenceEngine,
        PatternDeviationAlert,
        CircularDependencyAlert,
        FileSplitSuggestion,
        DuplicatePatternAlert,
        FileChangeEvent
    )
    REALTIME_AVAILABLE = True
except ImportError:
    REALTIME_AVAILABLE = False
    pytest.skip("Real-time intelligence not available", allow_module_level=True)

class TestPatternDeviationDetection:
    """Test pattern deviation detection for AI-generated code."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.engine = RealTimeIntelligenceEngine(
            project_path=str(self.test_dir),
            ai_awareness=True
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, 'engine') and self.engine._is_monitoring:
            asyncio.run(self.engine.stop_monitoring())
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    @pytest.mark.asyncio
    async def test_naming_convention_detection(self):
        """Test detection of naming convention violations."""
        # Create test file with naming violations
        test_file = self.test_dir / "bad_naming.py"
        test_file.write_text("""
def BadFunctionName():  # Should be snake_case
    pass

class badClassName:  # Should be PascalCase
    pass

def another_Bad_Function():  # Mixed case
    pass

class GoodClassName:  # This is correct
    pass

def good_function_name():  # This is correct
    pass
""")
        
        # Create file change event
        change_event = FileChangeEvent(
            file_path=str(test_file),
            event_type="modified",
            timestamp=time.time(),
            is_python=True
        )
        
        # Test pattern deviation detection
        await self.engine._check_pattern_deviations(change_event)
        
        # Check results
        assert len(self.engine._pattern_deviations) > 0
        
        # Should detect function naming violations
        function_violations = [
            alert for alert in self.engine._pattern_deviations 
            if alert.deviation_type == 'naming' and 'Function' in alert.actual_pattern
        ]
        assert len(function_violations) >= 2  # BadFunctionName and another_Bad_Function
        
        # Should detect class naming violations
        class_violations = [
            alert for alert in self.engine._pattern_deviations 
            if alert.deviation_type == 'naming' and 'Class' in alert.actual_pattern
        ]
        assert len(class_violations) >= 1  # badClassName
        
        # Check violation details
        for violation in self.engine._pattern_deviations:
            assert violation.confidence > 0.0
            assert violation.suggestion is not None
            assert violation.severity in ['low', 'medium', 'high']
    
    @pytest.mark.asyncio
    async def test_import_pattern_detection(self):
        """Test detection of inconsistent import patterns."""
        test_file = self.test_dir / "mixed_imports.py"
        test_file.write_text("""
import os
import sys
from os import path  # Mixed with import os above
from sys import argv  # Mixed with import sys above
import json
from json import loads  # Mixed with import json above
""")
        
        change_event = FileChangeEvent(
            file_path=str(test_file),
            event_type="modified",
            timestamp=time.time(),
            is_python=True
        )
        
        await self.engine._check_pattern_deviations(change_event)
        
        # Should detect mixed import styles
        import_violations = [
            alert for alert in self.engine._pattern_deviations 
            if alert.deviation_type == 'imports'
        ]
        assert len(import_violations) > 0
        
        for violation in import_violations:
            assert 'Mixed import styles' in violation.actual_pattern
            assert violation.severity == 'low'
    
    @pytest.mark.asyncio
    async def test_structure_pattern_detection(self):
        """Test detection of structural issues like overly long functions."""
        # Create a file with a very long function
        long_function_lines = ["def very_long_function():"] + [f"    line_{i} = {i}" for i in range(60)]
        
        test_file = self.test_dir / "long_function.py"
        test_file.write_text("\n".join(long_function_lines))
        
        change_event = FileChangeEvent(
            file_path=str(test_file),
            event_type="modified",
            timestamp=time.time(),
            is_python=True
        )
        
        await self.engine._check_pattern_deviations(change_event)
        
        # Should detect overly long function
        structure_violations = [
            alert for alert in self.engine._pattern_deviations 
            if alert.deviation_type == 'structure'
        ]
        assert len(structure_violations) >= 1
        
        violation = structure_violations[0]
        assert 'very_long_function' in violation.actual_pattern
        assert 'breaking down' in violation.suggestion.lower()
    
    @pytest.mark.asyncio
    async def test_pattern_learning(self):
        """Test that the system learns patterns from existing code."""
        # Create files with consistent patterns
        file1 = self.test_dir / "pattern1.py"
        file1.write_text("""
def good_function_one():
    pass

class GoodClassOne:
    pass
""")
        
        file2 = self.test_dir / "pattern2.py"
        file2.write_text("""
def good_function_two():
    pass

class GoodClassTwo:
    pass
""")
        
        # Process both files
        for file_path in [file1, file2]:
            change_event = FileChangeEvent(
                file_path=str(file_path),
                event_type="created",
                timestamp=time.time(),
                is_python=True
            )
            await self.engine._check_pattern_deviations(change_event)
        
        # Check that patterns were learned
        assert len(self.engine._learned_patterns['naming_conventions']['functions']) >= 2
        assert len(self.engine._learned_patterns['naming_conventions']['classes']) >= 2
        assert 'good_function_one' in self.engine._learned_patterns['naming_conventions']['functions']
        assert 'GoodClassOne' in self.engine._learned_patterns['naming_conventions']['classes']

class TestCircularDependencyPrevention:
    """Test circular dependency detection and prevention."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.engine = RealTimeIntelligenceEngine(
            project_path=str(self.test_dir),
            ai_awareness=True
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    @pytest.mark.asyncio
    async def test_circular_dependency_detection(self):
        """Test detection of potential circular dependencies."""
        # Mock dependency graph with circular dependency
        from types import SimpleNamespace
        
        # Create mock dependency graph
        mock_graph = SimpleNamespace()
        mock_graph.nodes = {
            str(self.test_dir / "module_a.py"): SimpleNamespace(imports=['module_b']),
            str(self.test_dir / "module_b.py"): SimpleNamespace(imports=['module_a'])
        }
        
        self.engine._dependency_graph = mock_graph
        
        # Create change event for module_a
        change_event = FileChangeEvent(
            file_path=str(self.test_dir / "module_a.py"),
            event_type="modified",
            timestamp=time.time(),
            is_python=True
        )
        
        await self.engine._check_circular_dependencies(change_event)
        
        # Should detect circular dependency
        assert len(self.engine._circular_dependency_alerts) >= 1
        
        alert = self.engine._circular_dependency_alerts[0]
        assert len(alert.involved_files) == 2
        assert alert.risk_level in ['potential', 'likely', 'confirmed']
        assert 'module_a' in str(alert.involved_files)
        assert 'module_b' in str(alert.involved_files)
    
    @pytest.mark.asyncio 
    async def test_no_false_positive_circular_dependencies(self):
        """Test that legitimate dependencies don't trigger false alarms."""
        # Mock dependency graph without circular dependencies
        from types import SimpleNamespace
        
        mock_graph = SimpleNamespace()
        mock_graph.nodes = {
            str(self.test_dir / "utils.py"): SimpleNamespace(imports=['json', 'os']),
            str(self.test_dir / "main.py"): SimpleNamespace(imports=['utils'])
        }
        
        self.engine._dependency_graph = mock_graph
        
        change_event = FileChangeEvent(
            file_path=str(self.test_dir / "main.py"),
            event_type="modified",
            timestamp=time.time(),
            is_python=True
        )
        
        await self.engine._check_circular_dependencies(change_event)
        
        # Should not detect any circular dependencies
        assert len(self.engine._circular_dependency_alerts) == 0

class TestFileSplitSuggestions:
    """Test file split suggestions for better AI comprehension."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.engine = RealTimeIntelligenceEngine(
            project_path=str(self.test_dir),
            ai_awareness=True
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    @pytest.mark.asyncio
    async def test_large_file_split_suggestion(self):
        """Test that large files trigger split suggestions."""
        # Create a large file with multiple classes
        large_file_content = """
class FirstClass:
    def method_one(self):
        pass
    
    def method_two(self):
        pass

class SecondClass:
    def method_one(self):
        pass
    
    def method_two(self):
        pass

class ThirdClass:
    def method_one(self):
        pass
    
    def method_two(self):
        pass

""" + "\n".join([f"def utility_function_{i}():\n    pass\n" for i in range(15)])
        
        test_file = self.test_dir / "large_file.py"
        test_file.write_text(large_file_content)
        
        # Create change event with large token count
        change_event = FileChangeEvent(
            file_path=str(test_file),
            event_type="modified",
            timestamp=time.time(),
            is_python=True,
            estimated_tokens=3000  # Large enough to trigger suggestion
        )
        
        await self.engine._suggest_file_splits(change_event)
        
        # Should generate split suggestions
        assert len(self.engine._file_split_suggestions) >= 1
        
        suggestion = self.engine._file_split_suggestions[0]
        assert suggestion.current_size_tokens == 3000
        assert len(suggestion.suggested_splits) > 0
        assert suggestion.priority in ['medium', 'high']
        
        # Should suggest splitting classes
        class_splits = [s for s in suggestion.suggested_splits if s['type'] == 'class']
        assert len(class_splits) >= 3  # FirstClass, SecondClass, ThirdClass
        
        # Should suggest splitting utility functions
        function_splits = [s for s in suggestion.suggested_splits if s['type'] == 'functions']
        assert len(function_splits) >= 1
    
    @pytest.mark.asyncio
    async def test_small_file_no_split_suggestion(self):
        """Test that small files don't trigger unnecessary split suggestions."""
        small_file_content = """
def simple_function():
    return "Hello, World!"

class SimpleClass:
    pass
"""
        
        test_file = self.test_dir / "small_file.py"
        test_file.write_text(small_file_content)
        
        change_event = FileChangeEvent(
            file_path=str(test_file),
            event_type="modified",
            timestamp=time.time(),
            is_python=True,
            estimated_tokens=100  # Small file
        )
        
        await self.engine._suggest_file_splits(change_event)
        
        # Should not generate split suggestions
        assert len(self.engine._file_split_suggestions) == 0
    
    @pytest.mark.asyncio
    async def test_critical_priority_for_huge_files(self):
        """Test that very large files get high priority split suggestions."""
        change_event = FileChangeEvent(
            file_path=str(self.test_dir / "huge_file.py"),
            event_type="modified",
            timestamp=time.time(),
            is_python=True,
            estimated_tokens=5000  # Very large
        )
        
        # Mock file content
        with patch('builtins.open', mock_open_with_content("class HugeClass:\n    pass")):
            await self.engine._suggest_file_splits(change_event)
        
        if self.engine._file_split_suggestions:
            suggestion = self.engine._file_split_suggestions[0]
            assert suggestion.priority == 'high'

def mock_open_with_content(content):
    """Helper to mock file reading."""
    from unittest.mock import mock_open
    return mock_open(read_data=content)

class TestDuplicatePatternDetection:
    """Test duplicate pattern identification and consolidation suggestions."""
    
    def setup_method(self):
        """Set up test fixtures.""" 
        self.test_dir = Path(tempfile.mkdtemp())
        self.engine = RealTimeIntelligenceEngine(
            project_path=str(self.test_dir),
            ai_awareness=True
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    @pytest.mark.asyncio
    async def test_duplicate_function_detection(self):
        """Test detection of duplicate function implementations."""
        test_file_content = """
def calculate_area_v1(width, height):
    result = width * height
    return result

def some_other_function():
    print("Different function")

def calculate_area_v2(w, h):  # Same logic, different names
    result = w * h
    return result

def calculate_area_duplicate(width, height):  # Exact duplicate
    result = width * height
    return result
"""
        
        test_file = self.test_dir / "duplicates.py"
        test_file.write_text(test_file_content)
        
        change_event = FileChangeEvent(
            file_path=str(test_file),
            event_type="modified",
            timestamp=time.time(),
            is_python=True
        )
        
        await self.engine._detect_duplicate_patterns(change_event)
        
        # Should detect duplicate functions
        assert len(self.engine._duplicate_patterns) >= 1
        
        duplicate = self.engine._duplicate_patterns[0]
        assert duplicate.pattern_type == 'function'
        assert len(duplicate.duplicate_locations) == 2
        assert duplicate.similarity_score == 1.0  # Exact match
        assert 'identical implementations' in duplicate.consolidation_suggestion
    
    @pytest.mark.asyncio
    async def test_unique_functions_no_duplicates(self):
        """Test that unique functions don't trigger false duplicate alerts."""
        test_file_content = """
def function_one():
    return "one"

def function_two():
    return "two"

def function_three():
    print("three")
    return 3
"""
        
        test_file = self.test_dir / "unique.py"
        test_file.write_text(test_file_content)
        
        change_event = FileChangeEvent(
            file_path=str(test_file),
            event_type="modified",
            timestamp=time.time(),
            is_python=True
        )
        
        await self.engine._detect_duplicate_patterns(change_event)
        
        # Should not detect any duplicates
        assert len(self.engine._duplicate_patterns) == 0

class TestPriority2Integration:
    """Test integration of all Priority 2 features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.engine = RealTimeIntelligenceEngine(
            project_path=str(self.test_dir),
            ai_awareness=True
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    @pytest.mark.asyncio
    async def test_comprehensive_file_analysis(self):
        """Test that all Priority 2 features work together on a complex file."""
        # Create a complex file that should trigger multiple alerts
        complex_file_content = """
import os
from os import path  # Mixed imports

class badClassName:  # Bad naming
    def VeryLongFunctionThatShouldBeSplit(self):  # Bad naming + structure
        # Very long function (50+ lines)
""" + "\n        ".join([f"line_{i} = {i}" for i in range(55)]) + """
        return "done"

def duplicate_logic():
    result = 1 + 2
    return result

def another_duplicate():  # Same logic as above
    result = 1 + 2
    return result

# Many more classes to trigger file split
""" + "\n".join([f"""
class Class{i}:
    def method(self):
        pass
""" for i in range(5)])
        
        test_file = self.test_dir / "complex_file.py"
        test_file.write_text(complex_file_content)
        
        # Estimate large token count
        estimated_tokens = len(complex_file_content) // 4
        
        change_event = FileChangeEvent(
            file_path=str(test_file),
            event_type="modified",
            timestamp=time.time(),
            is_python=True,
            estimated_tokens=estimated_tokens
        )
        
        # Run all Priority 2 analysis
        await self.engine._check_pattern_deviations(change_event)
        await self.engine._suggest_file_splits(change_event)
        await self.engine._detect_duplicate_patterns(change_event)
        
        # Should detect multiple issues
        assert len(self.engine._pattern_deviations) > 0, "Should detect pattern deviations"
        assert len(self.engine._duplicate_patterns) > 0, "Should detect duplicate patterns"
        
        # If file is large enough, should suggest splits
        if estimated_tokens > 2000:
            assert len(self.engine._file_split_suggestions) > 0, "Should suggest file splits"
        
        # Check statistics
        stats = self.engine.get_real_time_stats()
        assert stats['pattern_deviations'] > 0
        assert stats['duplicate_patterns'] > 0
        assert stats['stats']['pattern_deviations'] > 0
        assert stats['stats']['duplicate_patterns_found'] > 0
    
    def test_stats_and_activity_reporting(self):
        """Test that Priority 2 features are included in stats and activity."""
        # Add some test data
        self.engine._pattern_deviations.append(PatternDeviationAlert(
            file_path="test.py",
            deviation_type="naming",
            expected_pattern="snake_case",
            actual_pattern="PascalCase",
            confidence=0.8,
            suggestion="Use snake_case"
        ))
        
        self.engine._file_split_suggestions.append(FileSplitSuggestion(
            file_path="large.py",
            current_size_tokens=3000,
            suggested_splits=[{'type': 'class', 'name': 'TestClass'}],
            split_rationale="Too large",
            estimated_improvement="Better comprehension"
        ))
        
        # Test stats
        stats = self.engine.get_real_time_stats()
        assert stats['pattern_deviations'] == 1
        assert stats['file_split_suggestions'] == 1
        assert 'learned_patterns' in stats
        
        # Test activity
        activity = self.engine.get_recent_activity(10)
        assert 'pattern_deviations' in activity
        assert 'file_split_suggestions' in activity
        assert 'duplicate_patterns' in activity
        assert len(activity['pattern_deviations']) == 1
        assert len(activity['file_split_suggestions']) == 1

class TestErrorHandling:
    """Test error handling in Priority 2 features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = RealTimeIntelligenceEngine("/nonexistent/path")
    
    @pytest.mark.asyncio
    async def test_pattern_detection_with_invalid_file(self):
        """Test pattern detection handles invalid files gracefully."""
        change_event = FileChangeEvent(
            file_path="/nonexistent/file.py",
            event_type="modified",
            timestamp=time.time(),
            is_python=True
        )
        
        # Should not crash
        await self.engine._check_pattern_deviations(change_event)
        
        # Should not generate spurious alerts
        assert len(self.engine._pattern_deviations) == 0
    
    @pytest.mark.asyncio
    async def test_circular_dependency_with_no_graph(self):
        """Test circular dependency detection with no dependency graph."""
        change_event = FileChangeEvent(
            file_path="test.py",
            event_type="modified",
            timestamp=time.time(),
            is_python=True
        )
        
        # Should not crash when dependency graph is None
        await self.engine._check_circular_dependencies(change_event)
        assert len(self.engine._circular_dependency_alerts) == 0

class TestPriority2Performance:
    """Test performance characteristics of Priority 2 features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.engine = RealTimeIntelligenceEngine(
            project_path=str(self.test_dir),
            ai_awareness=True
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    @pytest.mark.asyncio
    async def test_large_file_performance(self):
        """Test performance with large files."""
        # Create a very large file
        large_content = "def func():\n    pass\n" * 1000  # 1000 functions
        
        test_file = self.test_dir / "large_performance_test.py"
        test_file.write_text(large_content)
        
        change_event = FileChangeEvent(
            file_path=str(test_file),
            event_type="modified",
            timestamp=time.time(),
            is_python=True,
            estimated_tokens=len(large_content) // 4
        )
        
        start_time = time.time()
        
        # Run all Priority 2 analysis
        await self.engine._check_pattern_deviations(change_event)
        await self.engine._suggest_file_splits(change_event)
        await self.engine._detect_duplicate_patterns(change_event)
        
        end_time = time.time()
        analysis_time = end_time - start_time
        
        # Should complete within reasonable time (5 seconds for 1000 functions)
        assert analysis_time < 5.0, f"Analysis took {analysis_time:.2f} seconds, should be < 5.0"
    
    def test_memory_usage_with_many_alerts(self):
        """Test memory management with many alerts."""
        # Generate many alerts to test trimming
        for i in range(200):
            self.engine._pattern_deviations.append(PatternDeviationAlert(
                file_path=f"test_{i}.py",
                deviation_type="naming",
                expected_pattern="snake_case",
                actual_pattern="PascalCase",
                confidence=0.8,
                suggestion="Use snake_case"
            ))
        
        # Should trim to maximum size
        assert len(self.engine._pattern_deviations) == 100  # Max size as defined in code
        
        # Should keep most recent alerts
        last_alert = self.engine._pattern_deviations[-1]
        assert "test_199.py" in last_alert.file_path

if __name__ == "__main__":
    pytest.main([__file__, "-v"])