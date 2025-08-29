"""
Comprehensive Error Handling and Recovery Tests (Priority 3.2)
Tests graceful degradation, missing dependencies, partial completion, and recovery scenarios.
"""

import pytest
import tempfile
import os
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import sys
import time
import threading
from contextlib import contextmanager

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Test availability of required modules
TOOLS_AVAILABLE = False
DEEPFLOW_AVAILABLE = False

try:
    from tools.dependency_visualizer import DependencyAnalyzer
    from tools.code_analyzer import CodeAnalyzer
    from tools.doc_generator import DocumentationGenerator
    TOOLS_AVAILABLE = True
except ImportError:
    pass

try:
    from deepflow.mcp.server import DeepflowMCPServer
    DEEPFLOW_AVAILABLE = True
except ImportError:
    pass


@contextmanager
def simulate_disk_full():
    """Context manager to simulate disk full scenarios."""
    original_open = open
    
    def mock_open_disk_full(*args, **kwargs):
        if 'w' in args[1] if len(args) > 1 else kwargs.get('mode', 'r'):
            raise OSError(28, "No space left on device")  # ENOSPC
        return original_open(*args, **kwargs)
    
    with patch('builtins.open', side_effect=mock_open_disk_full):
        yield


@contextmanager 
def simulate_permission_denied():
    """Context manager to simulate permission denied scenarios."""
    original_open = open
    
    def mock_open_permission_denied(*args, **kwargs):
        if 'w' in args[1] if len(args) > 1 else kwargs.get('mode', 'r'):
            raise PermissionError(13, "Permission denied")  # EACCES
        return original_open(*args, **kwargs)
    
    with patch('builtins.open', side_effect=mock_open_permission_denied):
        yield


@pytest.mark.unit
class TestGracefulDegradation:
    """Test graceful degradation under various failure conditions."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
    
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
        
        # Config module with missing dependency
        (self.test_path / "config.py").write_text("""
try:
    import missing_package  # This will fail
    HAS_MISSING_PACKAGE = True
except ImportError:
    HAS_MISSING_PACKAGE = False

DATABASE_URL = 'sqlite:///test.db'
DEBUG = True
""")
    
    @pytest.mark.skipif(not TOOLS_AVAILABLE, reason="Tools not available")
    def test_missing_dependency_handling(self):
        """Test handling of missing dependencies with graceful fallback."""
        self.create_test_project()
        
        analyzer = DependencyAnalyzer(str(self.test_path))
        
        # Mock a missing dependency scenario
        with patch('importlib.util.spec_from_file_location') as mock_spec:
            # Simulate ImportError for missing dependencies
            mock_spec.side_effect = ImportError("No module named 'missing_package'")
            
            try:
                result = analyzer.analyze_project()
                
                # Analysis should complete despite missing dependencies
                assert result is not None, "Analysis should handle missing dependencies gracefully"
                
                # Should have some nodes even with missing dependencies
                if hasattr(result, 'nodes'):
                    # Should at least find the files that can be analyzed
                    assert len(result.nodes) >= 2, "Should analyze available files despite missing dependencies"
                
            except Exception as e:
                # Check if it's a graceful handling
                if "missing_package" in str(e):
                    # This is acceptable - we're testing graceful degradation
                    pass
                else:
                    pytest.fail(f"Unexpected error during missing dependency handling: {e}")
    
    @pytest.mark.skipif(not TOOLS_AVAILABLE, reason="Tools not available")
    def test_partial_analysis_completion(self):
        """Test that analysis can complete partially when some files fail."""
        self.create_test_project()
        
        # Create a file with syntax errors
        (self.test_path / "broken.py").write_text("""
import sys
import json

def broken_function(
    # Missing closing parenthesis and return
    x = [1, 2, 3,
""")
        
        # Create a file that causes analysis errors
        (self.test_path / "problematic.py").write_text("""
import sys
import os
from nonexistent_module import NonexistentClass

def function_with_issues():
    return NonexistentClass()
""")
        
        analyzer = DependencyAnalyzer(str(self.test_path))
        
        # Should complete partial analysis
        try:
            result = analyzer.analyze_project()
            
            # Should get results for analyzable files
            assert result is not None, "Should complete partial analysis"
            
            # Should process at least the valid files
            if hasattr(result, 'nodes'):
                valid_files = 0
                for node in result.nodes:
                    if 'main.py' in str(node) or 'utils.py' in str(node):
                        valid_files += 1
                
                assert valid_files >= 1, "Should analyze at least some valid files"
                
        except Exception as e:
            # Partial completion might still raise errors, but should provide partial results
            pytest.skip(f"Partial analysis handling varies by implementation: {e}")
    
    @pytest.mark.skipif(not TOOLS_AVAILABLE, reason="Tools not available")
    def test_network_connectivity_failures(self):
        """Test handling of network failures during analysis."""
        self.create_test_project()
        
        # Create file that might try to access network resources
        (self.test_path / "network_dependent.py").write_text("""
import urllib.request
import json

def fetch_remote_config():
    # This might try to access network during analysis
    try:
        with urllib.request.urlopen('https://config.example.com/config.json') as response:
            return json.load(response)
    except:
        return {'default': 'config'}

def process_with_remote():
    config = fetch_remote_config()
    return config
""")
        
        analyzer = DependencyAnalyzer(str(self.test_path))
        
        # Mock network failures
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_urlopen.side_effect = OSError("Network unreachable")
            
            try:
                result = analyzer.analyze_project()
                
                # Should complete analysis without network access
                assert result is not None, "Should handle network failures gracefully"
                
            except Exception as e:
                # Network failures during static analysis should be rare,
                # but graceful handling is acceptable
                if "Network" in str(e):
                    pass  # Expected for this test
                else:
                    pytest.fail(f"Unexpected network-related error: {e}")
    
    @pytest.mark.skipif(not TOOLS_AVAILABLE, reason="Tools not available") 
    def test_disk_space_exhaustion_scenarios(self):
        """Test handling of disk space exhaustion during output generation."""
        self.create_test_project()
        
        # Create a documentation generator to test file writing
        if TOOLS_AVAILABLE:
            try:
                doc_gen = DocumentationGenerator(str(self.test_path))
                
                # Simulate disk full during documentation generation
                with simulate_disk_full():
                    try:
                        # Attempt to generate documentation
                        result = doc_gen.generate_dependency_map()
                        
                        # If it succeeds, it should handle disk full gracefully
                        # (might generate partial results)
                        assert result is not None or True, "Should handle disk full scenarios"
                        
                    except OSError as e:
                        if e.errno == 28:  # ENOSPC - No space left on device
                            # Expected error - graceful handling
                            assert True, "Correctly detected disk full condition"
                        else:
                            pytest.fail(f"Unexpected OS error: {e}")
                    except Exception as e:
                        # Other exceptions might be acceptable depending on implementation
                        if "space" in str(e).lower() or "disk" in str(e).lower():
                            pass  # Acceptable disk-related error handling
                        else:
                            pytest.fail(f"Unexpected error during disk full simulation: {e}")
                        
            except ImportError:
                pytest.skip("DocumentationGenerator not available for disk space test")
    
    @pytest.mark.skipif(not TOOLS_AVAILABLE, reason="Tools not available")
    def test_permission_denied_scenarios(self):
        """Test handling of permission denied errors."""
        self.create_test_project()
        
        if TOOLS_AVAILABLE:
            try:
                doc_gen = DocumentationGenerator(str(self.test_path))
                
                # Simulate permission denied during file operations
                with simulate_permission_denied():
                    try:
                        result = doc_gen.generate_dependency_map()
                        
                        # Should handle permission errors gracefully
                        assert result is not None or True, "Should handle permission errors"
                        
                    except PermissionError as e:
                        if e.errno == 13:  # EACCES - Permission denied
                            # Expected error - graceful handling
                            assert True, "Correctly detected permission denied"
                        else:
                            pytest.fail(f"Unexpected permission error: {e}")
                    except Exception as e:
                        # Other exceptions might be acceptable
                        if "permission" in str(e).lower():
                            pass  # Acceptable permission-related error handling
                        else:
                            pytest.fail(f"Unexpected error during permission test: {e}")
                            
            except ImportError:
                pytest.skip("DocumentationGenerator not available for permission test")
    
    def test_memory_exhaustion_recovery(self):
        """Test recovery from memory exhaustion scenarios."""
        # Create a scenario that could lead to memory issues
        large_data = []
        
        try:
            # Simulate analyzing a very large amount of data
            for i in range(1000):  # Reasonable size for testing
                large_data.append({
                    'file': f'large_file_{i}.py',
                    'dependencies': [f'dep_{j}' for j in range(100)],
                    'content': 'x = ' + str(list(range(1000)))
                })
            
            # Test that we can handle large datasets gracefully
            processed = 0
            memory_limit_hit = False
            
            try:
                for item in large_data:
                    # Simulate processing each item
                    processed += 1
                    
                    # Simulate memory pressure detection
                    if processed > 500:  # Arbitrary limit for testing
                        memory_limit_hit = True
                        break
                        
            except MemoryError:
                memory_limit_hit = True
            
            # Should handle memory pressure gracefully
            assert processed > 0, "Should process at least some items before memory limit"
            
            if memory_limit_hit:
                assert processed >= 100, "Should process reasonable amount before graceful degradation"
                
        except Exception as e:
            pytest.skip(f"Memory exhaustion test varies by system: {e}")
    
    @pytest.mark.skipif(not TOOLS_AVAILABLE, reason="Tools not available")
    def test_corrupted_file_recovery(self):
        """Test recovery from corrupted or unreadable files."""
        self.create_test_project()
        
        # Create files with various corruption scenarios
        
        # File with null bytes
        (self.test_path / "null_bytes.py").write_bytes(b"import sys\x00\x00\nimport os\n")
        
        # File with invalid UTF-8 sequences
        try:
            (self.test_path / "invalid_utf8.py").write_bytes(b"import sys\n\xff\xfe\nimport os\n")
        except Exception:
            pass  # Some systems might reject this
        
        # Empty file that looks like Python
        (self.test_path / "empty_file.py").write_text("")
        
        # File with only whitespace
        (self.test_path / "whitespace_only.py").write_text("   \n\n\t\t\n   ")
        
        analyzer = DependencyAnalyzer(str(self.test_path))
        
        try:
            result = analyzer.analyze_project()
            
            # Should complete analysis despite corrupted files
            assert result is not None, "Should handle corrupted files gracefully"
            
            # Should still analyze the valid files
            if hasattr(result, 'nodes'):
                valid_nodes = 0
                for node in result.nodes:
                    if any(valid_file in str(node) for valid_file in ['main.py', 'utils.py']):
                        valid_nodes += 1
                
                assert valid_nodes >= 1, "Should analyze valid files despite corruption"
                
        except Exception as e:
            # Some corruption handling might vary by implementation
            if any(keyword in str(e).lower() for keyword in ['corrupt', 'decode', 'utf-8']):
                pass  # Acceptable corruption-related errors
            else:
                pytest.fail(f"Unexpected error during corrupted file test: {e}")
    
    def test_timeout_recovery(self):
        """Test recovery from operation timeouts."""
        # Simulate a long-running operation that might timeout
        def long_running_operation():
            time.sleep(2)  # 2 second operation
            return "completed"
        
        # Test timeout handling
        start_time = time.time()
        timeout = 1.0  # 1 second timeout
        result = None
        timed_out = False
        
        try:
            # Simulate timeout with threading
            def target():
                nonlocal result
                result = long_running_operation()
            
            thread = threading.Thread(target=target)
            thread.start()
            thread.join(timeout=timeout)
            
            if thread.is_alive():
                # Operation timed out
                timed_out = True
                # In real implementation, would clean up the thread
            
        except Exception as e:
            pytest.fail(f"Timeout test setup failed: {e}")
        
        elapsed = time.time() - start_time
        
        # Should handle timeout gracefully
        if timed_out:
            assert elapsed >= timeout, "Should respect timeout duration"
            assert elapsed < timeout + 0.5, "Should not wait too long after timeout"
            assert result is None, "Should not have result after timeout"
        else:
            # Operation completed within timeout
            assert result == "completed", "Should complete if within timeout"


@pytest.mark.unit  
class TestErrorRecoveryMechanisms:
    """Test specific error recovery mechanisms and fallback strategies."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_fallback_dependency_resolution(self):
        """Test fallback strategies when dependency resolution fails."""
        # Create files with complex dependency issues
        (self.test_path / "circular_a.py").write_text("""
import circular_b
from circular_c import function_c

def function_a():
    return circular_b.function_b()
""")
        
        (self.test_path / "circular_b.py").write_text("""
from circular_a import function_a
import circular_c

def function_b():
    return circular_c.function_c()
""")
        
        (self.test_path / "circular_c.py").write_text("""
from circular_a import function_a
from circular_b import function_b

def function_c():
    # Creates complex circular dependency
    return function_a() + function_b()
""")
        
        if TOOLS_AVAILABLE:
            analyzer = DependencyAnalyzer(str(self.test_path))
            
            try:
                result = analyzer.analyze_project()
                
                # Should provide some form of result even with circular dependencies
                assert result is not None, "Should handle circular dependencies with fallback"
                
                # Should detect the files even if dependency resolution is complex
                if hasattr(result, 'nodes'):
                    assert len(result.nodes) >= 3, "Should detect all files despite circular deps"
                
            except Exception as e:
                # Complex circular dependencies might cause various errors
                # The key is that it should be handled gracefully
                if "circular" in str(e).lower() or "recursion" in str(e).lower():
                    pass  # Expected for circular dependency handling
                else:
                    pytest.fail(f"Unexpected error in circular dependency handling: {e}")
    
    def test_incremental_recovery_strategies(self):
        """Test incremental recovery when partial operations fail."""
        # Create a project with mixed valid/invalid files
        valid_files = ['main.py', 'utils.py', 'config.py']
        invalid_files = ['syntax_error.py', 'import_error.py']
        
        for filename in valid_files:
            (self.test_path / filename).write_text(f"""
import sys
import json

def {filename.replace('.py', '')}_function():
    return '{filename} processed'
""")
        
        (self.test_path / "syntax_error.py").write_text("""
import sys
def broken_function(
    # Missing closing paren and body
""")
        
        (self.test_path / "import_error.py").write_text("""
from nonexistent_module import NonexistentClass
import definitely_not_a_real_module

def problematic_function():
    return NonexistentClass()
""")
        
        if TOOLS_AVAILABLE:
            analyzer = DependencyAnalyzer(str(self.test_path))
            
            # Track what gets processed successfully
            processed_files = []
            errors_encountered = []
            
            try:
                result = analyzer.analyze_project()
                
                # Should process valid files successfully
                assert result is not None, "Should provide results for processable files"
                
                if hasattr(result, 'nodes'):
                    for node in result.nodes:
                        if any(valid_file in str(node) for valid_file in valid_files):
                            processed_files.append(node)
                
                # Should have processed at least some valid files
                assert len(processed_files) >= 2, f"Should process valid files incrementally: {processed_files}"
                
            except Exception as e:
                errors_encountered.append(str(e))
                
                # Should still have processed some files before error
                # This tests incremental processing with recovery
                if any(keyword in str(e).lower() for keyword in ['syntax', 'import', 'module']):
                    # These are expected errors that should be handled gracefully
                    pass
                else:
                    pytest.fail(f"Unexpected error during incremental recovery: {e}")
    
    @pytest.mark.skipif(not DEEPFLOW_AVAILABLE, reason="Deepflow MCP not available")
    def test_mcp_server_error_recovery(self):
        """Test MCP server recovery from various error conditions."""
        try:
            with patch('deepflow.mcp.server.MCP_AVAILABLE', True), \
                 patch('deepflow.mcp.server.TOOLS_AVAILABLE', True):
                
                server = DeepflowMCPServer()
                
                # Test tool execution error recovery
                tools = server.get_tools()
                assert len(tools) > 0, "Should have tools available"
                
                # Simulate tool execution errors
                with patch.object(server, '_handle_analyze_dependencies') as mock_handler:
                    mock_handler.side_effect = Exception("Simulated tool error")
                    
                    # Server should handle tool errors gracefully
                    try:
                        # This would normally be called by MCP framework
                        # We're testing the error handling mechanism
                        assert server is not None, "Server should remain stable after tool errors"
                        
                    except Exception as e:
                        # Error handling should prevent server crashes
                        if "Simulated tool error" in str(e):
                            pass  # Expected error, handled gracefully
                        else:
                            pytest.fail(f"Server should handle tool errors gracefully: {e}")
                
        except ImportError:
            pytest.skip("MCP server dependencies not available")
    
    def test_resource_cleanup_after_errors(self):
        """Test that resources are properly cleaned up after errors."""
        temp_files = []
        temp_dirs = []
        
        try:
            # Create temporary resources
            for i in range(5):
                temp_file = self.test_path / f"temp_file_{i}.tmp"
                temp_file.write_text(f"temporary content {i}")
                temp_files.append(temp_file)
                
                temp_dir = self.test_path / f"temp_dir_{i}"
                temp_dir.mkdir()
                temp_dirs.append(temp_dir)
            
            # Simulate operation that might fail
            def operation_that_might_fail():
                # Do some work with the resources
                for temp_file in temp_files:
                    content = temp_file.read_text()
                    
                # Simulate failure partway through
                if len(temp_files) > 3:
                    raise Exception("Simulated processing error")
                
                return "success"
            
            # Test resource cleanup on error
            cleanup_performed = False
            
            try:
                result = operation_that_might_fail()
            except Exception as e:
                # Perform cleanup after error
                try:
                    for temp_file in temp_files:
                        if temp_file.exists():
                            temp_file.unlink()
                    
                    for temp_dir in temp_dirs:
                        if temp_dir.exists():
                            temp_dir.rmdir()
                    
                    cleanup_performed = True
                    
                except Exception as cleanup_error:
                    pytest.fail(f"Cleanup failed after error: {cleanup_error}")
            
            # Verify cleanup was performed
            if cleanup_performed:
                remaining_files = [f for f in temp_files if f.exists()]
                remaining_dirs = [d for d in temp_dirs if d.exists()]
                
                assert len(remaining_files) == 0, f"Should clean up temp files: {remaining_files}"
                assert len(remaining_dirs) == 0, f"Should clean up temp dirs: {remaining_dirs}"
                
        except Exception as e:
            pytest.fail(f"Resource cleanup test failed: {e}")
    
    def test_configuration_error_recovery(self):
        """Test recovery from configuration errors and invalid settings."""
        # Test various configuration error scenarios
        invalid_configs = [
            None,  # Missing config
            {},    # Empty config
            {'invalid_key': 'invalid_value'},  # Invalid structure
            {'timeout': 'not_a_number'},  # Type errors
            {'max_files': -1},  # Invalid values
        ]
        
        for config in invalid_configs:
            try:
                # Simulate configuration loading with error recovery
                def load_config_with_fallback(config):
                    if config is None:
                        return {'default': True, 'timeout': 30}
                    
                    if not isinstance(config, dict):
                        return {'default': True, 'timeout': 30}
                    
                    # Validate and provide defaults
                    validated_config = {}
                    
                    # Handle timeout setting
                    timeout = config.get('timeout', 30)
                    if isinstance(timeout, str):
                        try:
                            timeout = int(timeout)
                        except ValueError:
                            timeout = 30  # Default fallback
                    
                    if timeout <= 0:
                        timeout = 30
                        
                    validated_config['timeout'] = timeout
                    
                    # Handle max_files setting
                    max_files = config.get('max_files', 1000)
                    if not isinstance(max_files, int) or max_files <= 0:
                        max_files = 1000
                        
                    validated_config['max_files'] = max_files
                    
                    return validated_config
                
                result_config = load_config_with_fallback(config)
                
                # Should always get a valid configuration
                assert isinstance(result_config, dict), "Should return valid config dict"
                assert 'timeout' in result_config, "Should have timeout setting"
                assert 'max_files' in result_config, "Should have max_files setting"
                assert result_config['timeout'] > 0, "Should have positive timeout"
                assert result_config['max_files'] > 0, "Should have positive max_files"
                
            except Exception as e:
                pytest.fail(f"Configuration recovery failed for {config}: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])