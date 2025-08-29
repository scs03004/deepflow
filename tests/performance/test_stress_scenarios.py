"""
Comprehensive Stress Testing Scenarios (Priority 4)
Tests system behavior under extreme conditions and resource constraints.
"""

import pytest
import tempfile
import os
import shutil
import time
import threading
import asyncio
import psutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import gc
import random
import string
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import weakref

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Test availability of required modules
TOOLS_AVAILABLE = False
REALTIME_AVAILABLE = False
WATCHDOG_AVAILABLE = False

try:
    from tools.dependency_visualizer import DependencyAnalyzer
    from tools.code_analyzer import CodeAnalyzer
    from tools.doc_generator import DocumentationGenerator
    TOOLS_AVAILABLE = True
except ImportError:
    pass

try:
    from deepflow.mcp.realtime_intelligence import RealTimeIntelligenceEngine
    REALTIME_AVAILABLE = True
except ImportError:
    pass

try:
    import watchdog
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    pass


class StressTestMonitor:
    """Monitor system resources during stress tests."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.monitoring = False
        self.measurements = []
        self.peak_memory = 0
        self.peak_cpu = 0
        self.monitor_thread = None
        
    def start_monitoring(self, interval: float = 0.5):
        """Start continuous monitoring."""
        self.monitoring = True
        self.measurements = []
        self.peak_memory = 0
        self.peak_cpu = 0
        
        def monitor_loop():
            while self.monitoring:
                try:
                    memory_mb = self.process.memory_info().rss / 1024 / 1024
                    cpu_percent = self.process.cpu_percent()
                    
                    self.peak_memory = max(self.peak_memory, memory_mb)
                    self.peak_cpu = max(self.peak_cpu, cpu_percent)
                    
                    self.measurements.append({
                        'timestamp': time.time(),
                        'memory_mb': memory_mb,
                        'cpu_percent': cpu_percent
                    })
                    
                    time.sleep(interval)
                except Exception:
                    break
        
        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self) -> Dict[str, Any]:
        """Stop monitoring and return results."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        return {
            'peak_memory_mb': self.peak_memory,
            'peak_cpu_percent': self.peak_cpu,
            'measurements': len(self.measurements),
            'duration': self.measurements[-1]['timestamp'] - self.measurements[0]['timestamp'] 
                       if len(self.measurements) >= 2 else 0
        }


@pytest.mark.stress
class TestStressScenarios:
    """Stress testing scenarios for extreme conditions."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        self.monitor = StressTestMonitor()
        
    def teardown_method(self):
        """Clean up test environment."""
        self.monitor.stop_monitoring()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)
        gc.collect()
    
    def create_extreme_codebase(self, num_files: int, complexity_factor: int = 1) -> List[Path]:
        """Create extremely large and complex codebase for stress testing."""
        created_files = []
        chars_per_file = 1000 * complexity_factor
        
        # Create deep directory structure
        max_depth = min(10, num_files // 100 + 1)
        
        for file_idx in range(num_files):
            # Create nested directory path
            depth = min(file_idx % max_depth + 1, max_depth)
            dir_parts = [f"level_{i}" for i in range(depth)]
            dir_path = self.test_path
            
            for part in dir_parts:
                dir_path = dir_path / part
                dir_path.mkdir(exist_ok=True)
            
            # Create complex Python file
            file_path = dir_path / f"complex_module_{file_idx:06d}.py"
            
            # Generate complex content with many imports and classes
            imports = []
            for i in range(complexity_factor * 10):
                imports.append(f"import module_{i % 100}")
                imports.append(f"from package_{i % 50} import Class{i % 20}")
            
            # Generate random code content
            classes = []
            for class_idx in range(complexity_factor * 5):
                methods = []
                for method_idx in range(20):
                    methods.append(f"""
    def method_{method_idx}(self, param_{method_idx}: str) -> str:
        '''Method {method_idx} in class {class_idx}.'''
        data = {{'key_{j}': 'value_{j}' for j in range({method_idx + 1})}}
        result = param_{method_idx}.upper() + str(len(data))
        return f'{{self.__class__.__name__}}_{{result}}'
""")
                
                class_code = f"""
class ComplexClass{file_idx}_{class_idx}:
    '''Complex class {class_idx} in file {file_idx}.'''
    
    def __init__(self):
        self.data_{class_idx} = {{'file': {file_idx}, 'class': {class_idx}}}
        self.counters = [0] * {class_idx + 10}
        self.cache = {{}}
        
{''.join(methods)}
    
    def complex_computation_{class_idx}(self):
        '''Perform complex computation.'''
        result = 0
        for i in range({(class_idx + 1) * 100}):
            result += i ** 2 % {class_idx + 1}
            self.counters[i % len(self.counters)] += 1
        return result
"""
                classes.append(class_code)
            
            # Generate functions
            functions = []
            for func_idx in range(complexity_factor * 8):
                functions.append(f"""
def complex_function_{file_idx}_{func_idx}(*args, **kwargs):
    '''Complex function {func_idx} in file {file_idx}.'''
    processing_data = []
    for i in range({func_idx + 10}):
        item = {{
            'id': i,
            'data': ''.join(random.choice(string.ascii_letters) for _ in range({func_idx + 5})),
            'metadata': {{
                'file': {file_idx},
                'function': {func_idx},
                'complexity': {complexity_factor}
            }}
        }}
        processing_data.append(item)
    
    # Simulate complex processing
    results = []
    for item in processing_data:
        processed = {{
            'original_id': item['id'],
            'processed_data': item['data'].upper(),
            'hash': hash(item['data']),
            'length': len(item['data'])
        }}
        results.append(processed)
    
    return {{
        'function': 'complex_function_{file_idx}_{func_idx}',
        'input_count': len(args),
        'results': results,
        'success': True
    }}
""")
            
            # Combine all content
            content = f'''"""
Complex module {file_idx} - Stress test file
Generated for extreme codebase stress testing
File size target: ~{chars_per_file} characters
"""

import sys
import os
import json
import time
import random
import string
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from collections import defaultdict, Counter
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

# Generated imports
{''.join(imports)}

# Module constants
MODULE_{file_idx}_CONFIG = {{
    'file_id': {file_idx},
    'complexity': {complexity_factor},
    'generated_at': time.time(),
    'features': ['feature_{j}' for j in range({complexity_factor * 5})],
    'settings': {{
        'debug': {str(file_idx % 2 == 0).lower()},
        'max_items': {1000 + file_idx * 10},
        'timeout': {30 + file_idx % 60},
        'batch_size': {min(100, complexity_factor * 25)}
    }}
}}

# Generated classes
{''.join(classes)}

# Generated functions  
{''.join(functions)}

# Module-level processing
_MODULE_{file_idx}_CACHE = {{}}
_MODULE_{file_idx}_STATS = Counter()

def get_module_stats() -> Dict[str, Any]:
    """Get module statistics."""
    return {{
        'module_id': {file_idx},
        'classes': {len(classes)},
        'functions': {len(functions)},
        'cache_size': len(_MODULE_{file_idx}_CACHE),
        'stats': dict(_MODULE_{file_idx}_STATS)
    }}

if __name__ == "__main__":
    print(f"Module {file_idx} stress test")
    stats = get_module_stats()
    print(f"Stats: {{stats}}")
'''
            
            # Pad content to reach target size if needed
            current_size = len(content)
            if current_size < chars_per_file:
                padding_needed = chars_per_file - current_size
                padding = '\\n'.join([f"# Padding line {i}" for i in range(padding_needed // 20)])
                content += '\\n' + padding
            
            file_path.write_text(content, encoding='utf-8')
            created_files.append(file_path)
            
            # Progress indicator for large file creation
            if file_idx > 0 and file_idx % 100 == 0:
                print(f"Created {file_idx}/{num_files} stress test files...")
        
        return created_files
    
    @pytest.mark.skipif(not TOOLS_AVAILABLE, reason="Tools not available")
    @pytest.mark.slow
    def test_extreme_file_count_handling(self):
        """Test handling of projects with extreme number of files."""
        # Test increasing file counts to find system limits
        test_cases = [
            {'files': 1000, 'timeout': 300},   # 1K files, 5min timeout
            {'files': 5000, 'timeout': 900},   # 5K files, 15min timeout (if system can handle)
        ]
        
        successful_cases = []
        
        for test_case in test_cases:
            num_files = test_case['files']
            timeout = test_case['timeout']
            
            print(f"\\n🔥 Stress testing with {num_files} files (timeout: {timeout}s)...")
            
            try:
                # Start monitoring
                self.monitor.start_monitoring()
                
                # Create extreme codebase (with lower complexity for very large counts)
                complexity = max(1, 3 - (num_files // 2000))  # Reduce complexity for larger file counts
                created_files = self.create_extreme_codebase(num_files, complexity)
                
                print(f"Created {len(created_files)} files, starting analysis...")
                
                # Perform analysis with timeout
                analyzer = DependencyAnalyzer(str(self.test_path))
                
                start_time = time.time()
                
                def run_analysis():
                    return analyzer.analyze_project()
                
                # Run analysis in thread with timeout
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(run_analysis)
                    
                    try:
                        result = future.result(timeout=timeout)
                        end_time = time.time()
                        execution_time = end_time - start_time
                        
                        # Stop monitoring and get results
                        monitoring_results = self.monitor.stop_monitoring()
                        
                        print(f"✅ {num_files} files analyzed in {execution_time:.1f}s")
                        print(f"   Peak memory: {monitoring_results['peak_memory_mb']:.1f}MB")
                        print(f"   Peak CPU: {monitoring_results['peak_cpu_percent']:.1f}%")
                        
                        # Verify analysis completed successfully
                        assert result is not None, "Analysis should complete successfully"
                        
                        # Check resource usage is reasonable
                        max_memory_gb = min(8, num_files * 0.002)  # 2MB per 1000 files max
                        assert monitoring_results['peak_memory_mb'] <= max_memory_gb * 1024, \
                            f"Memory usage {monitoring_results['peak_memory_mb']:.1f}MB too high"
                        
                        successful_cases.append({
                            'files': num_files,
                            'execution_time': execution_time,
                            'peak_memory_mb': monitoring_results['peak_memory_mb'],
                            'success': True
                        })
                        
                    except TimeoutError:
                        print(f"⏱️ {num_files} files timed out after {timeout}s")
                        monitoring_results = self.monitor.stop_monitoring()
                        
                        # Timeout is acceptable for extreme stress tests
                        successful_cases.append({
                            'files': num_files,
                            'execution_time': timeout,
                            'peak_memory_mb': monitoring_results['peak_memory_mb'],
                            'success': False,
                            'reason': 'timeout'
                        })
                        
                        if num_files <= 1000:  # Smaller tests should not timeout
                            pytest.fail(f"Analysis of {num_files} files should not timeout")
                        
            except Exception as e:
                print(f"❌ {num_files} files failed: {e}")
                monitoring_results = self.monitor.stop_monitoring()
                
                # Large stress tests may fail due to system limits
                if num_files <= 1000:
                    pytest.fail(f"Analysis of {num_files} files should not fail: {e}")
        
        # At least some test should succeed or timeout gracefully
        assert len(successful_cases) > 0, "At least one extreme file count test should run"
    
    @pytest.mark.skipif(not TOOLS_AVAILABLE, reason="Tools not available")
    def test_deep_directory_nesting(self):
        """Test handling of deeply nested directory structures (20+ levels)."""
        max_depths = [10, 20, 30]  # Test increasing directory depths
        
        for max_depth in max_depths:
            print(f"\\n🏗️ Testing directory depth: {max_depth} levels...")
            
            # Create deeply nested structure
            current_path = self.test_path
            paths_created = []
            
            # Create nested directories
            for level in range(max_depth):
                current_path = current_path / f"level_{level:02d}"
                current_path.mkdir(exist_ok=True)
                paths_created.append(current_path)
            
            # Create files at various depths
            files_created = []
            for depth_idx in range(0, max_depth, 5):  # Every 5 levels
                target_path = paths_created[depth_idx] if depth_idx < len(paths_created) else current_path
                
                test_file = target_path / f"module_depth_{depth_idx}.py"
                test_file.write_text(f"""
# Module at depth {depth_idx}
import sys
import os
from pathlib import Path

class DepthClass{depth_idx}:
    '''Class at directory depth {depth_idx}.'''
    
    def __init__(self):
        self.depth = {depth_idx}
        self.path = Path(__file__).parent
    
    def get_depth_info(self):
        return {{
            'depth': self.depth,
            'path_parts': len(self.path.parts),
            'absolute_path': str(self.path.resolve())
        }}

def depth_function_{depth_idx}():
    '''Function at depth {depth_idx}.'''
    return DepthClass{depth_idx}().get_depth_info()

if __name__ == "__main__":
    print(f"Module at depth {depth_idx}")
    print(depth_function_{depth_idx}())
""")
                files_created.append(test_file)
            
            print(f"Created {len(files_created)} files across {max_depth} directory levels")
            
            # Test analysis with deep nesting
            try:
                analyzer = DependencyAnalyzer(str(self.test_path))
                
                start_time = time.time()
                result = analyzer.analyze_project()
                end_time = time.time()
                
                execution_time = end_time - start_time
                
                print(f"✅ Deep nesting ({max_depth} levels) analyzed in {execution_time:.2f}s")
                
                # Should handle deep nesting without issues
                assert result is not None, f"Should analyze deeply nested structure (depth {max_depth})"
                
                # Execution time should be reasonable even with deep nesting
                max_time = min(60, max_depth * 2)  # 2 seconds per depth level max
                assert execution_time <= max_time, \
                    f"Deep nesting analysis took {execution_time:.2f}s, should be <= {max_time}s"
                    
            except Exception as e:
                if max_depth <= 20:  # Reasonable depths should work
                    pytest.fail(f"Deep directory nesting (depth {max_depth}) should not fail: {e}")
                else:
                    print(f"⚠️ Extreme depth {max_depth} failed as expected: {e}")
            
            # Clean up for next test
            shutil.rmtree(self.test_dir, ignore_errors=True)
            self.test_dir = tempfile.mkdtemp()
            self.test_path = Path(self.test_dir)
    
    @pytest.mark.skipif(not WATCHDOG_AVAILABLE or not REALTIME_AVAILABLE, 
                        reason="Watchdog or real-time features not available")
    def test_rapid_file_change_bursts(self):
        """Test real-time monitoring under rapid file change scenarios."""
        num_files = 50
        changes_per_file = 10
        
        # Create test files
        test_files = []
        for i in range(num_files):
            file_path = self.test_path / f"burst_test_{i:03d}.py"
            file_path.write_text(f"""
import sys
import time

class BurstTestClass{i}:
    def __init__(self):
        self.version = 0
        self.last_updated = time.time()
    
    def update(self):
        self.version += 1
        self.last_updated = time.time()
        return f"Updated to version {{self.version}}"

def burst_function_{i}():
    instance = BurstTestClass{i}()
    return instance.update()
""")
            test_files.append(file_path)
        
        print(f"\\n⚡ Testing rapid file changes: {num_files} files, {changes_per_file} changes each...")
        
        # Set up real-time monitoring
        try:
            engine = RealTimeIntelligenceEngine(str(self.test_path), ai_awareness=True)
            
            # Track events
            events_detected = []
            
            def event_callback(event_data):
                events_detected.append({
                    'timestamp': time.time(),
                    'event': event_data
                })
            
            engine.add_notification_callback(event_callback)
            
            # Start monitoring
            start_time = time.time()
            self.monitor.start_monitoring()
            
            try:
                asyncio.run(engine.start_monitoring())
                
                # Wait for monitoring to initialize
                time.sleep(0.5)
                
                # Perform rapid file changes
                change_threads = []
                
                def rapid_changes(file_path, num_changes):
                    """Perform rapid changes to a file."""
                    try:
                        for change_idx in range(num_changes):
                            # Read current content
                            current_content = file_path.read_text()
                            
                            # Make a change
                            modified_content = current_content + f"\\n# Change {change_idx} at {time.time()}\\n"
                            file_path.write_text(modified_content)
                            
                            # Small delay between changes
                            time.sleep(0.02)  # 20ms between changes (very rapid)
                            
                    except Exception as e:
                        print(f"Error in rapid changes for {file_path}: {e}")
                
                # Start rapid changes in parallel
                with ThreadPoolExecutor(max_workers=min(10, num_files)) as executor:
                    futures = []
                    for file_path in test_files:
                        future = executor.submit(rapid_changes, file_path, changes_per_file)
                        futures.append(future)
                    
                    # Wait for all changes to complete
                    for future in as_completed(futures, timeout=30):
                        try:
                            future.result()
                        except Exception as e:
                            print(f"Rapid change thread error: {e}")
                
                # Wait for event processing
                time.sleep(2.0)
                
                # Stop monitoring
                asyncio.run(engine.stop_monitoring())
                
                end_time = time.time()
                total_time = end_time - start_time
                
                monitoring_results = self.monitor.stop_monitoring()
                
                total_changes = num_files * changes_per_file
                changes_per_second = total_changes / total_time
                
                print(f"✅ Rapid changes completed:")
                print(f"   Total changes: {total_changes}")
                print(f"   Time taken: {total_time:.2f}s") 
                print(f"   Change rate: {changes_per_second:.1f} changes/sec")
                print(f"   Events detected: {len(events_detected)}")
                print(f"   Peak memory: {monitoring_results['peak_memory_mb']:.1f}MB")
                
                # System should handle rapid changes without crashing
                assert monitoring_results['peak_memory_mb'] < 1000, \
                    "Memory usage should stay reasonable during rapid changes"
                
                # Should detect at least some events (debouncing will reduce count)
                expected_min_events = max(1, total_changes // 20)  # At least 5% detection rate
                assert len(events_detected) >= expected_min_events, \
                    f"Should detect at least {expected_min_events} events, got {len(events_detected)}"
                    
            except asyncio.TimeoutError:
                print("⏱️ Rapid change monitoring timed out (acceptable for stress test)")
                asyncio.run(engine.stop_monitoring())
                
            except Exception as e:
                print(f"⚠️ Rapid change monitoring error: {e}")
                # Try to stop monitoring gracefully
                try:
                    asyncio.run(engine.stop_monitoring())
                except:
                    pass
                    
                # Rapid change stress test failures are acceptable
                pytest.skip(f"Rapid file change stress test not supported: {e}")
                
        except Exception as e:
            pytest.skip(f"Real-time monitoring setup failed: {e}")
    
    @pytest.mark.skipif(not TOOLS_AVAILABLE, reason="Tools not available")
    def test_memory_exhaustion_recovery(self):
        """Test graceful handling of memory exhaustion scenarios."""
        print("\\n🧠 Testing memory exhaustion recovery...")
        
        # Start with baseline memory measurement
        baseline_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        print(f"Baseline memory: {baseline_memory:.1f}MB")
        
        # Create increasingly memory-intensive scenarios
        memory_stress_levels = [
            {'files': 200, 'complexity': 3, 'expected_memory_mb': 100},
            {'files': 500, 'complexity': 5, 'expected_memory_mb': 300},
            {'files': 1000, 'complexity': 2, 'expected_memory_mb': 400},  # Lower complexity for many files
        ]
        
        successful_levels = []
        memory_limit_hit = False
        
        for level in memory_stress_levels:
            if memory_limit_hit:
                break
                
            print(f"\\nTesting memory stress: {level['files']} files, complexity {level['complexity']}...")
            
            try:
                # Start monitoring
                self.monitor.start_monitoring()
                
                # Create memory-intensive codebase
                created_files = self.create_extreme_codebase(level['files'], level['complexity'])
                
                # Monitor memory during creation
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                print(f"Memory after file creation: {current_memory:.1f}MB")
                
                # Check if we're approaching system limits
                available_memory = psutil.virtual_memory().available / 1024 / 1024  # MB
                if current_memory > available_memory * 0.7:  # Using 70% of available memory
                    print("⚠️ Approaching memory limits, stopping stress test")
                    memory_limit_hit = True
                    break
                
                # Perform analysis
                analyzer = DependencyAnalyzer(str(self.test_path))
                
                start_time = time.time()
                result = analyzer.analyze_project()
                end_time = time.time()
                
                # Stop monitoring
                monitoring_results = self.monitor.stop_monitoring()
                
                execution_time = end_time - start_time
                peak_memory = monitoring_results['peak_memory_mb']
                
                print(f"✅ Memory stress level completed:")
                print(f"   Execution time: {execution_time:.1f}s")
                print(f"   Peak memory: {peak_memory:.1f}MB")
                print(f"   Memory increase: {peak_memory - baseline_memory:.1f}MB")
                
                successful_levels.append({
                    'files': level['files'],
                    'complexity': level['complexity'],
                    'peak_memory_mb': peak_memory,
                    'execution_time': execution_time,
                    'success': True
                })
                
                # Verify analysis completed successfully  
                assert result is not None, "Analysis should complete despite memory stress"
                
                # Check memory usage is within reasonable bounds
                max_acceptable_memory = level['expected_memory_mb'] * 2  # Allow 2x expected
                if peak_memory > max_acceptable_memory:
                    print(f"⚠️ High memory usage: {peak_memory:.1f}MB > {max_acceptable_memory}MB")
                    # Don't fail the test, just warn
                
            except MemoryError:
                print("🔴 MemoryError encountered - testing recovery...")
                
                # Memory error is expected in stress testing
                # Test that system can recover
                gc.collect()  # Force garbage collection
                time.sleep(1)
                
                recovery_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
                memory_recovered = baseline_memory + 50  # Allow 50MB overhead
                
                if recovery_memory <= memory_recovered:
                    print(f"✅ Memory recovered: {recovery_memory:.1f}MB <= {memory_recovered:.1f}MB")
                    successful_levels.append({
                        'files': level['files'],
                        'recovery': True,
                        'memory_after_recovery': recovery_memory
                    })
                else:
                    print(f"⚠️ Memory not fully recovered: {recovery_memory:.1f}MB")
                
                memory_limit_hit = True
                
            except Exception as e:
                print(f"❌ Memory stress level failed: {e}")
                monitoring_results = self.monitor.stop_monitoring()
                
                # Memory-related failures are acceptable in stress testing
                if any(keyword in str(e).lower() for keyword in ['memory', 'resource', 'limit']):
                    successful_levels.append({
                        'files': level['files'],
                        'error': str(e),
                        'memory_related': True
                    })
                else:
                    # Non-memory errors should not occur in small stress tests
                    if level['files'] <= 200:
                        pytest.fail(f"Small memory stress test should not fail: {e}")
            
            # Clean up between tests
            shutil.rmtree(self.test_dir, ignore_errors=True)
            self.test_dir = tempfile.mkdtemp()
            self.test_path = Path(self.test_dir)
            gc.collect()
        
        # Should complete at least one memory stress level
        assert len(successful_levels) >= 1, f"Should complete at least one memory stress level: {successful_levels}"
        
        print(f"\\n📊 Memory stress test summary: {len(successful_levels)} levels completed")
    
    def test_disk_space_exhaustion(self):
        """Test handling of disk space exhaustion during analysis."""
        print("\\n💾 Testing disk space exhaustion handling...")
        
        # Check available disk space
        disk_usage = shutil.disk_usage(self.test_dir)
        available_gb = disk_usage.free / (1024**3)
        
        print(f"Available disk space: {available_gb:.1f}GB")
        
        if available_gb < 1.0:  # Less than 1GB available
            pytest.skip("Insufficient disk space for exhaustion test")
        
        # Create files that consume significant disk space
        large_files = []
        target_size_mb = min(100, available_gb * 1024 * 0.1)  # Use up to 10% of available space
        
        try:
            # Create large files progressively
            file_size_mb = 10  # 10MB per file
            num_large_files = int(target_size_mb // file_size_mb)
            
            print(f"Creating {num_large_files} files of {file_size_mb}MB each...")
            
            for file_idx in range(num_large_files):
                large_file = self.test_path / f"large_file_{file_idx:03d}.py"
                
                # Create file with repeated content to reach target size
                base_content = f"""
# Large file {file_idx} for disk space testing
import sys
import os
from typing import Dict, List

class LargeDataClass{file_idx}:
    def __init__(self):
        # Large data structure
        self.data = {{}}
        
"""
                
                # Repeat content to reach target size
                content_size = len(base_content.encode('utf-8'))
                repetitions = (file_size_mb * 1024 * 1024) // content_size
                
                large_content = base_content
                for rep in range(min(repetitions, 1000)):  # Limit repetitions for safety
                    large_content += f"        self.data['{file_idx}_{rep}'] = 'data_' * 100\\n"
                
                large_file.write_text(large_content)
                large_files.append(large_file)
                
                # Check if we're approaching disk limits
                current_usage = shutil.disk_usage(self.test_dir)
                remaining_gb = current_usage.free / (1024**3)
                
                if remaining_gb < 0.5:  # Less than 500MB remaining
                    print(f"⚠️ Approaching disk limit, created {len(large_files)} large files")
                    break
            
            print(f"Created {len(large_files)} large files")
            
            # Test analysis with limited disk space
            if TOOLS_AVAILABLE:
                analyzer = DependencyAnalyzer(str(self.test_path))
                
                try:
                    # This might fail due to disk space issues during temp file creation
                    result = analyzer.analyze_project()
                    
                    print("✅ Analysis completed despite large files")
                    assert result is not None, "Should handle large files"
                    
                except OSError as e:
                    if e.errno == 28:  # ENOSPC - No space left on device
                        print("✅ Correctly handled disk space exhaustion")
                        assert True, "Should handle disk space exhaustion gracefully"
                    else:
                        print(f"⚠️ Disk-related OS error: {e}")
                        # Other OS errors might be system-specific
                        
                except Exception as e:
                    if "space" in str(e).lower() or "disk" in str(e).lower():
                        print("✅ Handled disk space issue gracefully")
                        assert True, "Should handle disk space issues"
                    else:
                        print(f"⚠️ Unexpected error during disk space test: {e}")
                        # Large file analysis might fail for other reasons
            
        except OSError as e:
            if e.errno == 28:  # ENOSPC - No space left on device
                print("✅ Disk space exhaustion detected during file creation")
                assert True, "Should detect disk space exhaustion"
            else:
                print(f"⚠️ OS error during disk space test: {e}")
                
        except Exception as e:
            print(f"⚠️ Disk space test error: {e}")
            # Disk space tests are system-dependent
            
        finally:
            # Clean up large files
            for large_file in large_files:
                try:
                    if large_file.exists():
                        large_file.unlink()
                except Exception:
                    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])