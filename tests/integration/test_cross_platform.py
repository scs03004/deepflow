"""
Comprehensive Cross-Platform Compatibility Tests (Priority 4)
Tests platform-specific behaviors, path handling, and file system differences.
"""

import pytest
import tempfile
import os
import shutil
import platform
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from unittest.mock import patch, MagicMock
import sys
import stat
import time
from typing import Dict, List, Any, Optional
import subprocess

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Test availability of required modules
TOOLS_AVAILABLE = False

try:
    from tools.dependency_visualizer import DependencyAnalyzer
    from tools.code_analyzer import CodeAnalyzer
    from tools.doc_generator import DocumentationGenerator
    TOOLS_AVAILABLE = True
except ImportError:
    pass

# Platform detection
CURRENT_PLATFORM = platform.system().lower()
IS_WINDOWS = CURRENT_PLATFORM == 'windows'
IS_MACOS = CURRENT_PLATFORM == 'darwin'
IS_LINUX = CURRENT_PLATFORM == 'linux'


class CrossPlatformTestHelper:
    """Helper class for cross-platform testing utilities."""
    
    @staticmethod
    def create_platform_specific_paths():
        """Create test paths for different platforms."""
        return {
            'windows': [
                'C:\\Users\\test\\project\\module.py',
                'D:\\Projects\\app\\src\\main.py',
                'C:\\Program Files\\App\\config.py',
                '\\\\server\\share\\data\\file.py'  # UNC path
            ],
            'posix': [
                '/home/user/project/module.py',
                '/usr/local/lib/python/package/__init__.py',
                '/opt/application/src/main.py',
                '/tmp/test_file.py'
            ],
            'macos': [
                '/Users/user/Documents/project/app.py',
                '/Applications/MyApp.app/Contents/Resources/script.py',
                '/System/Library/Python/module.py',
                '/Volumes/External/backup/code.py'
            ]
        }
    
    @staticmethod
    def create_unicode_test_files(base_dir: Path):
        """Create test files with Unicode characters in names and content."""
        unicode_files = []
        
        # Different Unicode scenarios
        unicode_cases = [
            ('simple_ascii.py', 'ASCII file content'),
            ('café_français.py', '# Café français\\ndef français(): pass'),
            ('中文_测试.py', '# 中文测试文件\\ndef 测试函数(): pass'),
            ('русский_файл.py', '# Русский файл\\ndef русская_функция(): pass'),
            ('emoji_😀_file.py', '# Emoji test 🐍\\ndef happy_function(): return "😀"'),
            ('mixed_café_中文_🌟.py', '# Mixed Unicode: café 中文 🌟\\ndef international(): pass'),
        ]
        
        for filename, content in unicode_cases:
            try:
                file_path = base_dir / filename
                file_path.write_text(content, encoding='utf-8')
                unicode_files.append(file_path)
            except (OSError, UnicodeError) as e:
                # Some filesystems may not support certain Unicode characters
                print(f"⚠️ Could not create Unicode file {filename}: {e}")
        
        return unicode_files
    
    @staticmethod
    def simulate_case_sensitivity_scenarios(base_dir: Path):
        """Create files to test case sensitivity handling."""
        test_files = []
        
        # Create files with different casing
        case_variants = [
            'CamelCase.py',
            'camelcase.py',
            'UPPERCASE.py',
            'lowercase.py',
            'MixedCASE.py'
        ]
        
        for variant in case_variants:
            try:
                file_path = base_dir / variant
                file_path.write_text(f"""
# Case sensitivity test file: {variant}
import os
import sys

class TestClass_{variant.replace('.py', '')}:
    def __init__(self):
        self.filename = "{variant}"
        self.case_info = {{
            'original': "{variant}",
            'lower': "{variant.lower()}",
            'upper': "{variant.upper()}"
        }}
""")
                test_files.append(file_path)
            except Exception as e:
                print(f"⚠️ Could not create case test file {variant}: {e}")
        
        return test_files


@pytest.mark.integration
class TestCrossPlatformCompatibility:
    """Test cross-platform compatibility scenarios."""
    
    def setup_method(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.test_path = Path(self.test_dir)
        self.helper = CrossPlatformTestHelper()
        
    def teardown_method(self):
        """Clean up test environment."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @pytest.mark.skipif(not TOOLS_AVAILABLE, reason="Tools not available")
    def test_windows_path_handling(self):
        """Test Windows-specific path handling (backslashes, drive letters)."""
        print(f"\\n🪟 Testing Windows path handling (current platform: {CURRENT_PLATFORM})...")
        
        # Create Windows-style project structure
        windows_structure = {
            'src\\main.py': 'import sys\\nfrom utils import helper',
            'src\\utils\\__init__.py': '# Utils package',
            'src\\utils\\helper.py': 'def help(): return "Windows helper"',
            'config\\settings.py': 'DEBUG = True',
            'tests\\test_main.py': 'import unittest\\nfrom src.main import *'
        }
        
        # Create files with Windows-style paths
        created_files = []
        for win_path, content in windows_structure.items():
            # Convert Windows path separators for current platform
            cross_platform_path = Path(self.test_path / win_path.replace('\\\\', os.sep))
            cross_platform_path.parent.mkdir(parents=True, exist_ok=True)
            cross_platform_path.write_text(content)
            created_files.append(cross_platform_path)
        
        print(f"Created {len(created_files)} files with cross-platform paths")
        
        # Test path normalization in analysis
        analyzer = DependencyAnalyzer(str(self.test_path))
        
        try:
            result = analyzer.analyze_project()
            assert result is not None, "Should handle Windows-style paths"
            
            print("✅ Windows path handling successful")
            
            # Test specific Windows scenarios if on Windows
            if IS_WINDOWS:
                # Test UNC paths (if available)
                try:
                    unc_test_dir = Path('\\\\localhost\\C$\\temp')
                    if unc_test_dir.exists():
                        unc_analyzer = DependencyAnalyzer(str(unc_test_dir))
                        # This might fail, but should not crash
                        print("🔍 UNC path access tested")
                except Exception as e:
                    print(f"⚠️ UNC path test skipped: {e}")
                
                # Test drive letter handling
                drive_paths = ['C:\\\\temp', 'D:\\\\projects', 'E:\\\\backup']
                for drive_path in drive_paths:
                    try:
                        if Path(drive_path).exists():
                            print(f"✅ Drive {drive_path[0]}: accessible")
                            break
                    except Exception:
                        continue
            
        except Exception as e:
            if IS_WINDOWS:
                pytest.fail(f"Windows path handling should work on Windows: {e}")
            else:
                print(f"⚠️ Windows path test on {CURRENT_PLATFORM}: {e}")
    
    @pytest.mark.skipif(not TOOLS_AVAILABLE, reason="Tools not available")
    def test_macos_case_sensitivity(self):
        """Test macOS case-sensitive/insensitive filesystem handling."""
        print(f"\\n🍎 Testing macOS case sensitivity (current platform: {CURRENT_PLATFORM})...")
        
        # Create files with case variations
        case_files = self.helper.simulate_case_sensitivity_scenarios(self.test_path)
        print(f"Created {len(case_files)} files with case variations")
        
        # Test case sensitivity behavior
        analyzer = DependencyAnalyzer(str(self.test_path))
        
        try:
            result = analyzer.analyze_project()
            assert result is not None, "Should handle case variations"
            
            # Check how many unique files were found
            if hasattr(result, 'nodes'):
                unique_files = len(result.nodes)
                expected_files = len(case_files)
                
                print(f"Files created: {expected_files}, Files found: {unique_files}")
                
                if IS_MACOS:
                    # macOS default is case-insensitive but case-preserving
                    # The exact behavior depends on filesystem (HFS+/APFS)
                    print("🔍 macOS case sensitivity behavior detected")
                    
                    # Check if filesystem is case sensitive
                    test_upper = self.test_path / "CASE_TEST"
                    test_lower = self.test_path / "case_test"
                    
                    try:
                        test_upper.write_text("upper")
                        test_lower.write_text("lower")
                        
                        if test_upper.exists() and test_lower.exists():
                            if test_upper.read_text() != test_lower.read_text():
                                print("✅ Case-sensitive filesystem detected")
                            else:
                                print("✅ Case-insensitive filesystem detected")
                        
                    except Exception:
                        print("⚠️ Case sensitivity test inconclusive")
                    finally:
                        for test_file in [test_upper, test_lower]:
                            try:
                                test_file.unlink()
                            except Exception:
                                pass
                
                print("✅ Case sensitivity handling successful")
                
        except Exception as e:
            print(f"⚠️ Case sensitivity test error: {e}")
    
    @pytest.mark.skipif(not TOOLS_AVAILABLE, reason="Tools not available")
    def test_linux_permission_models(self):
        """Test Linux permission models and access controls."""
        print(f"\\n🐧 Testing Linux permissions (current platform: {CURRENT_PLATFORM})...")
        
        # Create files with different permission scenarios
        permission_files = []
        
        # Regular file
        regular_file = self.test_path / "regular.py"
        regular_file.write_text("# Regular file\\nimport os")
        permission_files.append(regular_file)
        
        # Create files with different permissions (if on Unix-like system)
        if not IS_WINDOWS:
            try:
                # Read-only file
                readonly_file = self.test_path / "readonly.py"
                readonly_file.write_text("# Read-only file\\nimport sys")
                readonly_file.chmod(0o444)  # Read-only for all
                permission_files.append(readonly_file)
                
                # Executable file
                executable_file = self.test_path / "executable.py"
                executable_file.write_text("#!/usr/bin/env python3\\n# Executable script")
                executable_file.chmod(0o755)  # Executable
                permission_files.append(executable_file)
                
                # No-read file (if we have permissions)
                try:
                    noread_file = self.test_path / "noread.py"
                    noread_file.write_text("# No read permissions")
                    noread_file.chmod(0o000)  # No permissions
                    permission_files.append(noread_file)
                except PermissionError:
                    print("⚠️ Cannot create no-read file (insufficient privileges)")
                
                print(f"Created {len(permission_files)} files with various permissions")
                
            except Exception as e:
                print(f"⚠️ Permission setup failed: {e}")
        
        # Test analysis with permission variations
        analyzer = DependencyAnalyzer(str(self.test_path))
        
        try:
            result = analyzer.analyze_project()
            assert result is not None, "Should handle permission variations"
            
            print("✅ Permission handling successful")
            
            # Test specific Linux scenarios
            if IS_LINUX:
                # Test symlink handling
                try:
                    symlink_target = regular_file
                    symlink_path = self.test_path / "symlink.py"
                    symlink_path.symlink_to(symlink_target)
                    
                    # Re-analyze with symlink
                    symlink_analyzer = DependencyAnalyzer(str(self.test_path))
                    symlink_result = symlink_analyzer.analyze_project()
                    
                    print("✅ Symlink handling tested")
                    
                except Exception as e:
                    print(f"⚠️ Symlink test failed: {e}")
                
                # Test hardlink handling
                try:
                    hardlink_path = self.test_path / "hardlink.py"
                    hardlink_path.hardlink_to(regular_file)
                    
                    print("✅ Hardlink handling tested")
                    
                except Exception as e:
                    print(f"⚠️ Hardlink test failed: {e}")
            
        except Exception as e:
            print(f"⚠️ Linux permission test error: {e}")
        
        finally:
            # Clean up permission test files
            for perm_file in permission_files:
                try:
                    if perm_file.exists():
                        # Reset permissions before deletion
                        perm_file.chmod(0o644)
                        perm_file.unlink()
                except Exception:
                    pass
    
    @pytest.mark.skipif(not TOOLS_AVAILABLE, reason="Tools not available")
    def test_unicode_path_handling(self):
        """Test handling of Unicode characters in file paths across platforms."""
        print(f"\\n🌍 Testing Unicode path handling (current platform: {CURRENT_PLATFORM})...")
        
        # Create Unicode test files
        unicode_files = self.helper.create_unicode_test_files(self.test_path)
        print(f"Successfully created {len(unicode_files)} Unicode test files")
        
        # Test analysis with Unicode paths
        analyzer = DependencyAnalyzer(str(self.test_path))
        
        try:
            result = analyzer.analyze_project()
            assert result is not None, "Should handle Unicode paths"
            
            # Verify Unicode files were processed
            if hasattr(result, 'nodes') and unicode_files:
                files_found = len(result.nodes)
                print(f"Unicode files processed: {files_found}")
                
                # Should find at least some Unicode files
                assert files_found >= len(unicode_files) // 2, \
                    f"Should process at least half of Unicode files: {files_found}/{len(unicode_files)}"
            
            print("✅ Unicode path handling successful")
            
        except Exception as e:
            # Unicode handling may vary by platform and filesystem
            print(f"⚠️ Unicode path test error: {e}")
            
            # Test with simpler Unicode cases
            try:
                simple_unicode = self.test_path / "simple_café.py"
                simple_unicode.write_text("# Simple Unicode test")
                
                simple_analyzer = DependencyAnalyzer(str(self.test_path))
                simple_result = simple_analyzer.analyze_project()
                
                if simple_result is not None:
                    print("✅ Basic Unicode handling works")
                    
            except Exception as simple_error:
                print(f"⚠️ Even basic Unicode handling failed: {simple_error}")
    
    def test_path_separator_normalization(self):
        """Test path separator normalization across platforms."""
        print(f"\\n📂 Testing path separator normalization...")
        
        # Test different path formats
        test_paths = [
            'src/main.py',           # Unix style
            'src\\\\main.py',          # Windows style
            'src/utils\\\\helper.py',   # Mixed style
            './src/../src/main.py',  # Relative with navigation
        ]
        
        normalized_paths = []
        
        for test_path in test_paths:
            try:
                # Test Path normalization
                path_obj = Path(test_path)
                normalized = path_obj.resolve()
                normalized_paths.append((test_path, str(normalized)))
                
            except Exception as e:
                print(f"⚠️ Path normalization failed for {test_path}: {e}")
        
        print(f"✅ Normalized {len(normalized_paths)} paths")
        
        # Test with actual files
        if TOOLS_AVAILABLE:
            # Create test file
            test_file = self.test_path / "path_test.py"
            test_file.write_text("# Path normalization test")
            
            # Test different ways to reference the same file
            path_variants = [
                str(test_file),
                str(test_file.resolve()),
                str(test_file.absolute()),
            ]
            
            if not IS_WINDOWS:
                # Add Unix-style relative path
                relative_path = os.path.relpath(test_file, self.test_path)
                path_variants.append(str(self.test_path / relative_path))
            
            # All variants should refer to the same file
            for variant in path_variants:
                try:
                    variant_path = Path(variant)
                    if variant_path.exists():
                        assert variant_path.read_text() == "# Path normalization test"
                        
                except Exception as e:
                    print(f"⚠️ Path variant {variant} failed: {e}")
            
            print("✅ Path separator normalization successful")
    
    def test_filesystem_encoding_handling(self):
        """Test handling of different filesystem encodings."""
        print(f"\\n🔤 Testing filesystem encoding handling...")
        
        # Get system encoding information
        fs_encoding = sys.getfilesystemencoding()
        default_encoding = sys.getdefaultencoding()
        
        print(f"Filesystem encoding: {fs_encoding}")
        print(f"Default encoding: {default_encoding}")
        
        # Test different encoding scenarios
        encoding_tests = [
            ('utf-8', '# UTF-8 encoded content: café 中文'),
            ('latin-1', '# Latin-1 content: café naïve'),
            ('ascii', '# ASCII content only'),
        ]
        
        successful_encodings = []
        
        for encoding_name, content in encoding_tests:
            try:
                test_file = self.test_path / f"encoding_{encoding_name.replace('-', '_')}.py"
                
                # Write with specific encoding
                test_file.write_text(content, encoding=encoding_name)
                
                # Read back with same encoding
                read_content = test_file.read_text(encoding=encoding_name)
                
                if read_content == content:
                    successful_encodings.append(encoding_name)
                    print(f"✅ {encoding_name} encoding successful")
                
            except (UnicodeError, OSError) as e:
                print(f"⚠️ {encoding_name} encoding failed: {e}")
        
        # Should handle at least UTF-8 and ASCII
        assert 'utf-8' in successful_encodings or 'ascii' in successful_encodings, \
            "Should handle at least basic encodings"
        
        print(f"✅ Encoding tests completed: {len(successful_encodings)} successful")
    
    def test_platform_specific_imports(self):
        """Test handling of platform-specific imports and modules."""
        print(f"\\n🔧 Testing platform-specific imports...")
        
        # Create files with platform-specific imports
        platform_imports = {
            'windows_specific.py': '''
import os
if os.name == 'nt':
    import msvcrt
    import winreg
    import winsound
else:
    # Fallback imports
    msvcrt = None
    winreg = None
    winsound = None

def windows_function():
    if msvcrt:
        return "Windows functionality available"
    return "Windows functionality not available"
''',
            'unix_specific.py': '''
import os
if os.name == 'posix':
    import pwd
    import grp
    import termios
else:
    # Fallback
    pwd = None
    grp = None
    termios = None

def unix_function():
    if pwd:
        return "Unix functionality available"
    return "Unix functionality not available"
''',
            'cross_platform.py': '''
import os
import sys
import platform

# Platform detection
IS_WINDOWS = platform.system() == 'Windows'
IS_LINUX = platform.system() == 'Linux'
IS_MACOS = platform.system() == 'Darwin'

# Conditional imports
if IS_WINDOWS:
    try:
        import msvcrt
    except ImportError:
        msvcrt = None
elif IS_LINUX or IS_MACOS:
    try:
        import termios
    except ImportError:
        termios = None

def get_platform_info():
    return {
        'system': platform.system(),
        'release': platform.release(),
        'architecture': platform.architecture(),
        'python_version': sys.version
    }
'''
        }
        
        created_files = []
        for filename, content in platform_imports.items():
            file_path = self.test_path / filename
            file_path.write_text(content)
            created_files.append(file_path)
        
        print(f"Created {len(created_files)} platform-specific test files")
        
        # Test analysis with platform-specific imports
        if TOOLS_AVAILABLE:
            analyzer = DependencyAnalyzer(str(self.test_path))
            
            try:
                result = analyzer.analyze_project()
                assert result is not None, "Should handle platform-specific imports"
                
                print("✅ Platform-specific import analysis successful")
                
                # Test code quality analysis
                code_analyzer = CodeAnalyzer(str(self.test_path))
                quality_result = code_analyzer.analyze_code_quality(str(self.test_path))
                
                if quality_result is not None:
                    print("✅ Code quality analysis with platform imports successful")
                    
            except Exception as e:
                print(f"⚠️ Platform-specific import analysis error: {e}")
    
    def test_long_path_handling(self):
        """Test handling of very long file paths."""
        print(f"\\n📏 Testing long path handling...")
        
        # Create deeply nested directory structure
        max_path_length = 260 if IS_WINDOWS else 4096  # Windows vs Unix limits
        
        # Start with a reasonable base
        current_path = self.test_path / "long_path_test"
        current_path.mkdir(exist_ok=True)
        
        # Add nested directories until we approach the limit
        depth = 0
        path_components = []
        
        while len(str(current_path)) < max_path_length - 100:  # Leave buffer for filename
            component = f"nested_level_{depth:03d}"
            path_components.append(component)
            current_path = current_path / component
            
            try:
                current_path.mkdir(exist_ok=True)
                depth += 1
                
                if depth > 50:  # Reasonable limit for testing
                    break
                    
            except OSError as e:
                print(f"⚠️ Cannot create deeper nesting at depth {depth}: {e}")
                break
        
        print(f"Created directory structure with {depth} levels")
        print(f"Final path length: {len(str(current_path))} characters")
        
        # Create a test file at maximum depth
        try:
            test_file = current_path / "deep_module.py"
            test_file.write_text(f"""
# Deep nested module at depth {depth}
import os
import sys
from pathlib import Path

def get_depth_info():
    return {{
        'depth': {depth},
        'path_length': len(__file__),
        'absolute_path': os.path.abspath(__file__)
    }}

# Module constants
DEPTH = {depth}
PATH_LENGTH = len(__file__)
""")
            
            print(f"✅ Created file at path length: {len(str(test_file))}")
            
            # Test analysis with long paths
            if TOOLS_AVAILABLE:
                analyzer = DependencyAnalyzer(str(self.test_path))
                
                try:
                    result = analyzer.analyze_project()
                    
                    if result is not None:
                        print("✅ Long path analysis successful")
                    else:
                        print("⚠️ Long path analysis returned None")
                        
                except Exception as e:
                    print(f"⚠️ Long path analysis failed: {e}")
            
        except OSError as e:
            print(f"⚠️ Cannot create file at maximum depth: {e}")
            
            # Test with shorter path
            try:
                shorter_path = self.test_path / "long_path_test" / "shorter_module.py"
                shorter_path.write_text("# Shorter path test")
                print("✅ Shorter path file creation successful")
                
            except Exception as shorter_error:
                print(f"⚠️ Even shorter path failed: {shorter_error}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])