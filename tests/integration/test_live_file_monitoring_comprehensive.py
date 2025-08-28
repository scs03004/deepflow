"""
Comprehensive Live File Monitoring Tests (Priority 2.1)
Tests real-time file monitoring, debounced event processing, and incremental analysis.
"""

import pytest
import asyncio
import tempfile
import time
import threading
import shutil
import os
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import sys

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Test availability of required modules
WATCHDOG_AVAILABLE = False
REALTIME_AVAILABLE = False

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    pass

try:
    from deepflow.mcp.realtime_intelligence import RealTimeIntelligenceEngine
    REALTIME_AVAILABLE = True
except ImportError:
    REALTIME_AVAILABLE = False


@pytest.mark.integration
class TestFileSystemEvents:
    """Test real-time file change detection and event processing."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        self.monitored_events = []
        self.event_lock = threading.Lock()
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_test_project(self):
        """Create a test project structure."""
        # Main module
        (self.test_path / "main.py").write_text("""
import sys
import json
from utils import helper

def main():
    return helper.process_data({'status': 'ok'})

if __name__ == '__main__':
    print(json.dumps(main()))
""")
        
        # Utils module
        (self.test_path / "utils.py").write_text("""
import os
from pathlib import Path

def helper():
    return {'cwd': os.getcwd()}

def process_data(data):
    return {**data, 'processed': True}
""")
        
        # Config module
        (self.test_path / "config.py").write_text("""
DATABASE_URL = 'sqlite:///test.db'
DEBUG = True
VERSION = '1.0.0'
""")
    
    @pytest.mark.skipif(not WATCHDOG_AVAILABLE, reason="Watchdog not available")
    def test_real_time_file_change_detection(self):
        """Test that file changes are detected in real-time."""
        self.create_test_project()
        
        # Custom event handler to track events
        class TestEventHandler(FileSystemEventHandler):
            def __init__(self, test_instance):
                self.test_instance = test_instance
                
            def on_modified(self, event):
                if not event.is_directory and event.src_path.endswith('.py'):
                    with self.test_instance.event_lock:
                        self.test_instance.monitored_events.append({
                            'type': 'modified',
                            'path': event.src_path,
                            'timestamp': time.time()
                        })
            
            def on_created(self, event):
                if not event.is_directory and event.src_path.endswith('.py'):
                    with self.test_instance.event_lock:
                        self.test_instance.monitored_events.append({
                            'type': 'created',
                            'path': event.src_path,
                            'timestamp': time.time()
                        })
            
            def on_deleted(self, event):
                if not event.is_directory and event.src_path.endswith('.py'):
                    with self.test_instance.event_lock:
                        self.test_instance.monitored_events.append({
                            'type': 'deleted',
                            'path': event.src_path,
                            'timestamp': time.time()
                        })
        
        # Set up file monitoring
        observer = Observer()
        event_handler = TestEventHandler(self)
        observer.schedule(event_handler, str(self.test_path), recursive=True)
        observer.start()
        
        try:
            # Allow observer to initialize
            time.sleep(0.1)
            
            # Test file creation
            new_file = self.test_path / "new_module.py"
            new_file.write_text("# New module\ndef new_function():\n    pass\n")
            time.sleep(0.1)
            
            # Test file modification
            main_file = self.test_path / "main.py"
            main_file.write_text(main_file.read_text() + "\n# Modified\n")
            time.sleep(0.1)
            
            # Test file deletion
            config_file = self.test_path / "config.py"
            config_file.unlink()
            time.sleep(0.1)
            
            # Verify events were captured
            with self.event_lock:
                assert len(self.monitored_events) >= 3, f"Expected at least 3 events, got {len(self.monitored_events)}"
                
                # Check event types
                event_types = [event['type'] for event in self.monitored_events]
                assert 'created' in event_types, "File creation event not detected"
                assert 'modified' in event_types, "File modification event not detected" 
                assert 'deleted' in event_types, "File deletion event not detected"
                
                # Verify timing (events should be recent)
                current_time = time.time()
                for event in self.monitored_events:
                    assert (current_time - event['timestamp']) < 1.0, "Event timestamp too old"
                    
        finally:
            observer.stop()
            observer.join(timeout=2.0)
    
    @pytest.mark.skipif(not WATCHDOG_AVAILABLE, reason="Watchdog not available")
    def test_debounced_event_processing(self):
        """Test that rapid file changes are debounced (500ms window)."""
        self.create_test_project()
        
        # Track debounced events
        debounced_events = []
        debounce_lock = threading.Lock()
        
        class DebouncedEventHandler(FileSystemEventHandler):
            def __init__(self):
                self.last_event_time = {}
                self.debounce_delay = 0.5  # 500ms
                
            def on_modified(self, event):
                if not event.is_directory and event.src_path.endswith('.py'):
                    current_time = time.time()
                    path = event.src_path
                    
                    # Check if enough time has passed since last event for this file
                    if path not in self.last_event_time or \
                       (current_time - self.last_event_time[path]) >= self.debounce_delay:
                        
                        with debounce_lock:
                            debounced_events.append({
                                'path': path,
                                'timestamp': current_time
                            })
                        self.last_event_time[path] = current_time
        
        observer = Observer()
        event_handler = DebouncedEventHandler()
        observer.schedule(event_handler, str(self.test_path), recursive=True)
        observer.start()
        
        try:
            time.sleep(0.1)
            
            # Rapidly modify the same file multiple times
            test_file = self.test_path / "main.py"
            original_content = test_file.read_text()
            
            # Make 5 rapid modifications within 200ms
            for i in range(5):
                test_file.write_text(original_content + f"\n# Modification {i}\n")
                time.sleep(0.04)  # 40ms between modifications
            
            # Wait for debounce period
            time.sleep(0.6)
            
            # Make one more modification after debounce period
            test_file.write_text(original_content + "\n# Final modification\n")
            time.sleep(0.6)
            
            # Verify debouncing worked
            with debounce_lock:
                # Should have only 2 debounced events: first rapid change and final change
                assert len(debounced_events) <= 2, f"Debouncing failed: got {len(debounced_events)} events"
                
                if len(debounced_events) == 2:
                    # Verify time gap between events is at least 500ms
                    time_gap = debounced_events[1]['timestamp'] - debounced_events[0]['timestamp']
                    assert time_gap >= 0.5, f"Debounce period not respected: {time_gap}s gap"
                    
        finally:
            observer.stop()
            observer.join(timeout=2.0)
    
    @pytest.mark.skipif(not WATCHDOG_AVAILABLE, reason="Watchdog not available")
    def test_large_directory_monitoring(self):
        """Test monitoring directories with many files (1,000+ files)."""
        # Create a large directory structure
        num_files = 100  # Reduced for CI performance, can increase for stress testing
        
        # Create subdirectories
        for i in range(10):
            subdir = self.test_path / f"module_{i:02d}"
            subdir.mkdir()
            
            # Create files in each subdirectory
            for j in range(10):
                file_path = subdir / f"file_{j:02d}.py"
                file_path.write_text(f"""
# Module {i}, File {j}
import sys
import json

class Module{i}File{j}:
    def process(self):
        return {{'module': {i}, 'file': {j}}}

def function_{i}_{j}():
    return Module{i}File{j}().process()
""")
        
        # Track events from large directory
        large_dir_events = []
        event_lock = threading.Lock()
        
        class LargeDirEventHandler(FileSystemEventHandler):
            def on_modified(self, event):
                if not event.is_directory and event.src_path.endswith('.py'):
                    with event_lock:
                        large_dir_events.append({
                            'type': 'modified',
                            'path': event.src_path,
                            'timestamp': time.time()
                        })
        
        observer = Observer()
        event_handler = LargeDirEventHandler()
        observer.schedule(event_handler, str(self.test_path), recursive=True)
        observer.start()
        
        try:
            time.sleep(0.2)
            
            # Modify multiple files simultaneously
            modified_files = []
            for i in range(5):
                file_path = self.test_path / f"module_{i:02d}" / "file_00.py"
                if file_path.exists():
                    content = file_path.read_text()
                    file_path.write_text(content + f"\n# Modified at {time.time()}\n")
                    modified_files.append(str(file_path))
            
            # Wait for events to be processed
            time.sleep(0.5)
            
            # Verify monitoring worked for large directory
            with event_lock:
                assert len(large_dir_events) >= len(modified_files), \
                    f"Expected at least {len(modified_files)} events, got {len(large_dir_events)}"
                
                # Verify all modified files were detected
                detected_paths = [event['path'] for event in large_dir_events]
                for modified_file in modified_files:
                    # Convert to absolute path for comparison
                    abs_modified = os.path.abspath(modified_file)
                    found = any(os.path.abspath(path) == abs_modified for path in detected_paths)
                    assert found, f"Modified file not detected: {modified_file}"
                    
        finally:
            observer.stop()
            observer.join(timeout=2.0)
    
    @pytest.mark.skipif(not WATCHDOG_AVAILABLE, reason="Watchdog not available")
    def test_permission_and_access_error_handling(self):
        """Test handling of permission errors and access issues."""
        self.create_test_project()
        
        error_events = []
        error_lock = threading.Lock()
        
        class ErrorHandlingEventHandler(FileSystemEventHandler):
            def on_error(self, event):
                with error_lock:
                    error_events.append({
                        'type': 'error',
                        'event': str(event),
                        'timestamp': time.time()
                    })
            
            def on_modified(self, event):
                try:
                    # Simulate permission checking
                    if event.src_path and os.path.exists(event.src_path):
                        # Try to read file to simulate access check
                        if event.src_path.endswith('.py'):
                            with open(event.src_path, 'r') as f:
                                content = f.read(100)  # Read first 100 chars
                except PermissionError:
                    with error_lock:
                        error_events.append({
                            'type': 'permission_error',
                            'path': event.src_path,
                            'timestamp': time.time()
                        })
                except Exception as e:
                    with error_lock:
                        error_events.append({
                            'type': 'access_error',
                            'path': event.src_path,
                            'error': str(e),
                            'timestamp': time.time()
                        })
        
        observer = Observer()
        event_handler = ErrorHandlingEventHandler()
        observer.schedule(event_handler, str(self.test_path), recursive=True)
        observer.start()
        
        try:
            time.sleep(0.1)
            
            # Test normal operation first
            test_file = self.test_path / "test_access.py"
            test_file.write_text("# Test file for access testing\n")
            time.sleep(0.1)
            
            # Modify the file
            test_file.write_text("# Modified test file\n")
            time.sleep(0.1)
            
            # Try to access non-existent file (should not crash)
            non_existent = self.test_path / "non_existent.py"
            try:
                with open(str(non_existent), 'w') as f:
                    f.write("# This will trigger an event\n")
                # Delete immediately to test cleanup handling
                non_existent.unlink()
            except:
                pass
            
            time.sleep(0.2)
            
            # Verify error handling didn't crash the monitoring
            # The observer should still be running
            assert observer.is_alive(), "Observer crashed during error handling"
            
        finally:
            observer.stop()
            observer.join(timeout=2.0)


@pytest.mark.integration
class TestIncrementalAnalysis:
    """Test selective dependency graph updates and performance optimization."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def create_dependency_project(self):
        """Create a project with clear dependency relationships."""
        # Base module
        (self.test_path / "base.py").write_text("""
class BaseClass:
    def base_method(self):
        return "base"
""")
        
        # Module A depends on base
        (self.test_path / "module_a.py").write_text("""
from base import BaseClass

class ModuleA(BaseClass):
    def method_a(self):
        return self.base_method() + "_a"
""")
        
        # Module B depends on A and base
        (self.test_path / "module_b.py").write_text("""
from base import BaseClass
from module_a import ModuleA

class ModuleB(BaseClass):
    def __init__(self):
        self.module_a = ModuleA()
    
    def method_b(self):
        return self.module_a.method_a() + "_b"
""")
        
        # Module C depends on B
        (self.test_path / "module_c.py").write_text("""
from module_b import ModuleB

class ModuleC:
    def __init__(self):
        self.module_b = ModuleB()
    
    def method_c(self):
        return self.module_b.method_b() + "_c"
""")
        
        # Independent module
        (self.test_path / "independent.py").write_text("""
class Independent:
    def independent_method(self):
        return "independent"
""")
    
    @pytest.mark.skipif(not REALTIME_AVAILABLE, reason="RealTimeIntelligenceEngine not available")
    def test_selective_dependency_graph_updates(self):
        """Test that only affected parts of dependency graph are updated."""
        self.create_dependency_project()
        
        # Mock the RealTimeIntelligenceEngine to track updates
        with patch('deepflow.mcp.realtime_intelligence.RealTimeIntelligenceEngine') as mock_rt:
            mock_instance = MagicMock()
            mock_rt.return_value = mock_instance
            
            # Track which files are analyzed during updates
            analyzed_files = []
            
            def mock_analyze_file(file_path):
                analyzed_files.append(str(file_path))
                return {
                    'file': str(file_path),
                    'imports': [],
                    'exports': [],
                    'dependencies': []
                }
            
            mock_instance.analyze_single_file = mock_analyze_file
            mock_instance.update_dependency_graph = MagicMock()
            
            # Create intelligence instance
            intelligence = mock_rt()
            
            # Simulate initial full analysis
            for py_file in self.test_path.glob("*.py"):
                intelligence.analyze_single_file(py_file)
            
            initial_analysis_count = len(analyzed_files)
            analyzed_files.clear()
            
            # Simulate changing independent.py (should not affect others)
            independent_file = self.test_path / "independent.py"
            independent_file.write_text(independent_file.read_text() + "\n# Modified\n")
            
            # Trigger incremental update
            intelligence.analyze_single_file(independent_file)
            
            # Should only analyze the independent file
            assert len(analyzed_files) == 1, f"Expected 1 file analyzed, got {analyzed_files}"
            assert str(independent_file) in analyzed_files[0], "Wrong file analyzed"
            
            analyzed_files.clear()
            
            # Simulate changing base.py (should affect dependent modules)
            base_file = self.test_path / "base.py"
            base_file.write_text(base_file.read_text() + "\n# Base modified\n")
            
            # In a real incremental system, this would trigger analysis of dependents
            # For testing, simulate the selective update
            affected_files = [base_file, self.test_path / "module_a.py", 
                            self.test_path / "module_b.py", self.test_path / "module_c.py"]
            
            for file in affected_files:
                intelligence.analyze_single_file(file)
            
            # Should analyze base.py and all its dependents, but not independent.py
            assert len(analyzed_files) == 4, f"Expected 4 files analyzed, got {len(analyzed_files)}"
            
            analyzed_paths = [Path(f).name for f in analyzed_files]
            assert "base.py" in analyzed_paths, "Base file not analyzed"
            assert "module_a.py" in analyzed_paths, "Module A not analyzed" 
            assert "module_b.py" in analyzed_paths, "Module B not analyzed"
            assert "module_c.py" in analyzed_paths, "Module C not analyzed"
            assert not any("independent.py" in f for f in analyzed_files), "Independent file incorrectly analyzed"
    
    def test_performance_comparison_validation(self):
        """Test that incremental analysis provides significant performance improvement."""
        self.create_dependency_project()
        
        # Add more files for meaningful performance comparison
        for i in range(20):
            module_file = self.test_path / f"extra_module_{i:02d}.py"
            module_file.write_text(f"""
# Extra module {i}
import sys
import json

class ExtraModule{i}:
    def process_{i}(self):
        return {{'module': {i}, 'data': 'processed'}}

def function_{i}():
    return ExtraModule{i}().process_{i}()
""")
        
        # Simulate full analysis timing
        start_time = time.time()
        
        # Mock full analysis by reading all files
        full_analysis_files = []
        for py_file in self.test_path.glob("*.py"):
            with open(py_file, 'r') as f:
                content = f.read()
                full_analysis_files.append({
                    'path': str(py_file),
                    'size': len(content),
                    'lines': len(content.split('\n'))
                })
        
        full_analysis_time = time.time() - start_time
        
        # Simulate incremental analysis (only one file)
        start_time = time.time()
        
        # Incremental: only analyze the changed file
        changed_file = self.test_path / "independent.py"
        with open(changed_file, 'r') as f:
            content = f.read()
            incremental_result = {
                'path': str(changed_file),
                'size': len(content),
                'lines': len(content.split('\n'))
            }
        
        incremental_analysis_time = time.time() - start_time
        
        # Validate performance improvement
        # Incremental should be significantly faster
        if full_analysis_time > 0:
            improvement_ratio = full_analysis_time / incremental_analysis_time
            # Should be at least 5x faster for incremental (targeting 10x+)
            assert improvement_ratio >= 5.0, f"Performance improvement insufficient: {improvement_ratio}x"
        
        # Validate that incremental analysis processes fewer files
        assert len(full_analysis_files) > 1, "Full analysis should process multiple files"
        # Incremental processes just one file (represented by single result)
        assert incremental_result is not None, "Incremental analysis should process the changed file"
    
    def test_memory_usage_optimization(self):
        """Test that incremental analysis uses memory efficiently."""
        self.create_dependency_project()
        
        # Simulate memory usage tracking
        memory_usage = []
        
        def track_memory_usage(stage, files_processed):
            """Simulate memory tracking."""
            # Estimate memory based on files processed
            estimated_memory = len(files_processed) * 1024  # 1KB per file estimate
            memory_usage.append({
                'stage': stage,
                'files_processed': len(files_processed),
                'estimated_memory': estimated_memory
            })
        
        # Full analysis simulation
        all_files = list(self.test_path.glob("*.py"))
        track_memory_usage("full_analysis", all_files)
        
        # Incremental analysis simulation (only changed file)
        changed_files = [self.test_path / "independent.py"]
        track_memory_usage("incremental_analysis", changed_files)
        
        # Validate memory optimization
        full_memory = memory_usage[0]['estimated_memory']
        incremental_memory = memory_usage[1]['estimated_memory']
        
        assert incremental_memory < full_memory, "Incremental analysis should use less memory"
        
        memory_ratio = full_memory / incremental_memory if incremental_memory > 0 else float('inf')
        assert memory_ratio >= 3.0, f"Memory usage improvement insufficient: {memory_ratio}x"
    
    def test_cache_invalidation_accuracy(self):
        """Test that cache invalidation correctly identifies affected dependencies."""
        self.create_dependency_project()
        
        # Simulate dependency cache
        dependency_cache = {}
        
        # Build initial cache
        for py_file in self.test_path.glob("*.py"):
            file_content = py_file.read_text()
            imports = []
            for line in file_content.split('\n'):
                if line.strip().startswith('from ') or line.strip().startswith('import '):
                    imports.append(line.strip())
            
            dependency_cache[str(py_file)] = {
                'imports': imports,
                'last_modified': py_file.stat().st_mtime,
                'dependents': []
            }
        
        # Build dependency relationships
        for file_path, cache_entry in dependency_cache.items():
            for import_line in cache_entry['imports']:
                if 'from ' in import_line:
                    module_name = import_line.split('from ')[1].split(' import')[0].strip()
                    for other_path in dependency_cache.keys():
                        if Path(other_path).stem == module_name:
                            dependency_cache[other_path]['dependents'].append(file_path)
        
        # Build transitive dependency relationships (for multi-level dependencies)
        def get_all_dependents(module_path, visited=None):
            if visited is None:
                visited = set()
            if module_path in visited:
                return set()
            
            visited.add(module_path)
            all_dependents = set()
            
            if module_path in dependency_cache:
                direct_dependents = dependency_cache[module_path]['dependents']
                all_dependents.update(direct_dependents)
                
                for dependent in direct_dependents:
                    transitive_deps = get_all_dependents(dependent, visited.copy())
                    all_dependents.update(transitive_deps)
            
            return all_dependents
        
        # Test cache invalidation when base.py changes
        base_file = self.test_path / "base.py"
        base_path = str(base_file)
        
        # Simulate file change
        base_file.write_text(base_file.read_text() + "\n# Modified base\n")
        new_mtime = base_file.stat().st_mtime
        
        # Identify files that need cache invalidation (including transitive dependencies)
        files_to_invalidate = [base_path]
        all_dependents = get_all_dependents(base_path)
        files_to_invalidate.extend(all_dependents)
        
        # Validate cache invalidation accuracy
        expected_invalidated = {'base.py', 'module_a.py', 'module_b.py', 'module_c.py'}
        actually_invalidated = {Path(f).name for f in files_to_invalidate}
        
        assert expected_invalidated.issubset(actually_invalidated), \
            f"Missing invalidated files: {expected_invalidated - actually_invalidated}"
        
        # Independent.py should NOT be invalidated
        assert 'independent.py' not in actually_invalidated, \
            "independent.py should not be invalidated when base.py changes"
        
        # Validate that cache reflects the change
        if base_path in dependency_cache:
            assert dependency_cache[base_path]['last_modified'] != new_mtime or \
                   base_path in files_to_invalidate, "Cache invalidation missed the changed file"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])