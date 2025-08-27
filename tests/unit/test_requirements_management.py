"""
Tests for Requirements Management functionality in SmartRefactoringEngine.

Tests the AI coding workflow helper that detects missing packages from imports
and intelligently updates requirements.txt.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, mock_open, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from deepflow.smart_refactoring_engine import (
        SmartRefactoringEngine, 
        RequirementsAnalysis
    )
    SMART_REFACTORING_AVAILABLE = True
except ImportError:
    SMART_REFACTORING_AVAILABLE = False

# Skip all tests if smart refactoring engine is not available
pytestmark = pytest.mark.skipif(
    not SMART_REFACTORING_AVAILABLE, 
    reason="SmartRefactoringEngine not available"
)


@pytest.fixture
def temp_project():
    """Create a temporary project directory for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_python_files(temp_project):
    """Create sample Python files with various imports."""
    # Main application file
    (temp_project / "main.py").write_text("""
import os
import sys
import json
import requests
from flask import Flask, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
""")
    
    # Utility file
    (temp_project / "utils.py").write_text("""
import datetime
import numpy as np
import yaml
from rich.console import Console
import asyncio
""")
    
    # Data science file
    (temp_project / "analysis.py").write_text("""
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
import tensorflow as tf
""")
    
    return temp_project


@pytest.fixture
def sample_requirements(temp_project):
    """Create a sample requirements.txt file."""
    requirements_content = """fastapi==0.104.1
uvicorn==0.24.0
unused-package==1.0.0
"""
    (temp_project / "requirements.txt").write_text(requirements_content)
    return temp_project


class TestRequirementsAnalysis:
    """Test requirements analysis functionality."""
    
    def test_init_with_comprehensive_mapping(self, temp_project):
        """Test that engine initializes with comprehensive package mapping."""
        engine = SmartRefactoringEngine(str(temp_project))
        
        # Check some key mappings are present
        assert engine.import_to_package['flask'] == 'flask'
        assert engine.import_to_package['sklearn'] == 'scikit-learn'
        assert engine.import_to_package['yaml'] == 'pyyaml'
        assert engine.import_to_package['cv2'] == 'opencv-python'
        assert engine.import_to_package['PIL'] == 'Pillow'
        assert engine.import_to_package['bs4'] == 'beautifulsoup4'
        
        # Check built-in modules are marked correctly
        assert engine.import_to_package['os'] == ''
        assert engine.import_to_package['sys'] == ''
        assert engine.import_to_package['json'] == ''
    
    def test_parse_requirements_file_exists(self, sample_requirements):
        """Test parsing existing requirements.txt file."""
        engine = SmartRefactoringEngine(str(sample_requirements))
        requirements = engine._parse_requirements_file()
        
        expected = ['fastapi==0.104.1', 'uvicorn==0.24.0', 'unused-package==1.0.0']
        assert requirements == expected
    
    def test_parse_requirements_file_missing(self, temp_project):
        """Test parsing when requirements.txt doesn't exist."""
        engine = SmartRefactoringEngine(str(temp_project))
        requirements = engine._parse_requirements_file()
        
        assert requirements == []
    
    def test_parse_requirements_with_comments(self, temp_project):
        """Test parsing requirements.txt with comments and empty lines."""
        requirements_content = """# Main dependencies
fastapi==0.104.1

# Development dependencies
pytest==7.0.0
# This is a comment
black==23.0.0

# End of file
"""
        (temp_project / "requirements.txt").write_text(requirements_content)
        
        engine = SmartRefactoringEngine(str(temp_project))
        requirements = engine._parse_requirements_file()
        
        expected = ['fastapi==0.104.1', 'pytest==7.0.0', 'black==23.0.0']
        assert requirements == expected
    
    def test_detect_all_imports(self, sample_python_files):
        """Test detection of imports across multiple files."""
        engine = SmartRefactoringEngine(str(sample_python_files))
        python_files = engine._get_python_files()
        detected_imports = engine._detect_all_imports(python_files)
        
        # Convert to dict for easier testing
        import_dict = {imp['import_name']: imp for imp in detected_imports}
        
        # Check that major imports are detected
        assert 'requests' in import_dict
        assert 'flask' in import_dict
        assert 'pandas' in import_dict
        assert 'numpy' in import_dict
        assert 'sklearn' in import_dict
        assert 'matplotlib' in import_dict
        assert 'tensorflow' in import_dict
        
        # Check built-ins are also detected
        assert 'os' in import_dict
        assert 'sys' in import_dict
        assert 'json' in import_dict
        
        # Check file usage tracking
        assert 'main.py' in import_dict['requests']['files'][0]
        assert 'utils.py' in import_dict['numpy']['files'][0]
    
    def test_map_import_to_package_known_mappings(self, temp_project):
        """Test mapping imports to packages for known mappings."""
        engine = SmartRefactoringEngine(str(temp_project))
        
        # Test exact mappings
        assert engine._map_import_to_package('sklearn') == 'scikit-learn'
        assert engine._map_import_to_package('yaml') == 'pyyaml'
        assert engine._map_import_to_package('cv2') == 'opencv-python'
        assert engine._map_import_to_package('PIL') == 'Pillow'
        assert engine._map_import_to_package('bs4') == 'beautifulsoup4'
        
        # Test built-in modules
        assert engine._map_import_to_package('os') == ''
        assert engine._map_import_to_package('sys') == ''
        assert engine._map_import_to_package('json') == ''
    
    def test_map_import_to_package_unknown(self, temp_project):
        """Test mapping unknown imports falls back to direct name."""
        engine = SmartRefactoringEngine(str(temp_project))
        
        # Unknown imports should map to themselves
        assert engine._map_import_to_package('unknown_package') == 'unknown_package'
        assert engine._map_import_to_package('my_custom_lib') == 'my_custom_lib'
    
    def test_get_mapping_confidence(self, temp_project):
        """Test confidence scoring for import mappings."""
        engine = SmartRefactoringEngine(str(temp_project))
        
        # Known mappings should have high confidence
        assert engine._get_mapping_confidence('flask') == 1.0
        assert engine._get_mapping_confidence('sklearn') == 1.0
        
        # Unknown mappings should have medium confidence
        assert engine._get_mapping_confidence('unknown_package') == 0.7
    
    def test_find_unused_packages(self, temp_project):
        """Test detection of unused packages."""
        engine = SmartRefactoringEngine(str(temp_project))
        
        current_packages = {'flask', 'pandas', 'unused-package', 'fastapi'}
        detected_imports = [
            {'import_name': 'flask', 'files': ['main.py'], 'usage_count': 1},
            {'import_name': 'pandas', 'files': ['main.py'], 'usage_count': 1},
        ]
        
        unused = engine._find_unused_packages(current_packages, detected_imports)
        
        # unused-package should be detected as unused
        # fastapi might be flagged (conservative detection)
        assert 'unused-package' in unused
    
    def test_is_likely_indirect_dependency(self, temp_project):
        """Test identification of indirect dependencies."""
        engine = SmartRefactoringEngine(str(temp_project))
        
        # Common indirect dependencies should be recognized
        assert engine._is_likely_indirect_dependency('setuptools') == True
        assert engine._is_likely_indirect_dependency('pip') == True
        assert engine._is_likely_indirect_dependency('wheel') == True
        assert engine._is_likely_indirect_dependency('urllib3') == True
        
        # Regular packages should not be flagged
        assert engine._is_likely_indirect_dependency('flask') == False
        assert engine._is_likely_indirect_dependency('pandas') == False
    
    def test_generate_update_recommendations(self, temp_project):
        """Test generation of intelligent update recommendations."""
        engine = SmartRefactoringEngine(str(temp_project))
        
        missing_packages = [
            {
                'package_name': 'flask',
                'confidence': 1.0,
                'is_standard_library': False
            },
            {
                'package_name': 'requests',
                'confidence': 0.8,
                'is_standard_library': False
            },
            {
                'package_name': 'os',
                'confidence': 0.7,
                'is_standard_library': True
            }
        ]
        unused_packages = ['unused-package']
        version_conflicts = []
        
        recommendations = engine._generate_update_recommendations(
            missing_packages, unused_packages, version_conflicts
        )
        
        # Should have recommendations for high confidence, medium confidence, and unused
        assert len(recommendations) == 3
        
        # Check high confidence recommendation
        high_rec = next(r for r in recommendations if r['priority'] == 'high')
        assert 'flask' in high_rec['packages']
        assert high_rec['type'] == 'add_packages'
        
        # Check medium confidence recommendation  
        medium_rec = next(r for r in recommendations if r['priority'] == 'medium')
        assert 'requests' in medium_rec['packages']
        
        # Check unused packages recommendation
        unused_rec = next(r for r in recommendations if r['type'] == 'review_unused')
        assert 'unused-package' in unused_rec['packages']


class TestRequirementsAnalysisIntegration:
    """Test the main analyze_requirements method."""
    
    def test_analyze_requirements_complete_flow(self, sample_python_files, sample_requirements):
        """Test complete requirements analysis flow."""
        engine = SmartRefactoringEngine(str(sample_requirements))
        
        analysis = engine.analyze_requirements()
        
        # Check analysis structure
        assert isinstance(analysis, RequirementsAnalysis)
        assert isinstance(analysis.missing_packages, list)
        assert isinstance(analysis.unused_packages, list)
        assert isinstance(analysis.update_recommendations, list)
        assert isinstance(analysis.current_requirements, list)
        assert isinstance(analysis.detected_imports, list)
        
        # Should detect missing packages
        assert len(analysis.missing_packages) > 0
        
        # Should have current requirements
        assert len(analysis.current_requirements) == 3  # From fixture
        
        # Should have recommendations
        assert len(analysis.update_recommendations) > 0
    
    def test_analyze_requirements_with_target_files(self, sample_python_files):
        """Test analysis with specific target files."""
        engine = SmartRefactoringEngine(str(sample_python_files))
        
        target_files = [str(sample_python_files / "main.py")]
        analysis = engine.analyze_requirements(target_files=target_files)
        
        # Should only analyze the specified file
        main_imports = set()
        for imp in analysis.detected_imports:
            main_imports.add(imp['import_name'])
        
        # Should have imports from main.py
        assert 'requests' in main_imports
        assert 'flask' in main_imports
        assert 'pandas' in main_imports
        
        # Should not have imports from other files
        # (this is harder to test directly, but fewer imports overall)
        assert len(analysis.detected_imports) < 15  # Rough check


class TestRequirementsUpdating:
    """Test requirements.txt updating functionality."""
    
    def test_update_requirements_dry_run(self, sample_python_files, sample_requirements):
        """Test dry run requirements updating."""
        engine = SmartRefactoringEngine(str(sample_requirements))
        analysis = engine.analyze_requirements()
        
        results = engine.update_requirements_file(analysis, dry_run=True)
        
        # Check dry run results
        assert results['dry_run'] == True
        assert results['packages_added'] > 0
        assert results['original_count'] == 3
        assert results['new_count'] > results['original_count']
        assert isinstance(results['new_requirements'], list)
        
        # Original file should be unchanged
        original_content = (sample_requirements / "requirements.txt").read_text()
        assert "fastapi==0.104.1" in original_content
        assert "pandas" not in original_content  # Should not be added in dry run
    
    def test_update_requirements_apply_changes(self, sample_python_files, sample_requirements):
        """Test actually applying requirements changes."""
        engine = SmartRefactoringEngine(str(sample_requirements))
        analysis = engine.analyze_requirements()
        
        # Apply changes
        results = engine.update_requirements_file(analysis, dry_run=False, backup=True)
        
        # Check results
        assert results['dry_run'] == False
        assert results['backup_created'] == True
        assert results['packages_added'] > 0
        
        # Check backup was created
        backup_path = sample_requirements / "requirements.txt.backup"
        assert backup_path.exists()
        
        # Check original requirements.txt was updated
        updated_content = (sample_requirements / "requirements.txt").read_text()
        assert len(updated_content.split('\n')) > 5  # Should have more packages
        
        # Should contain some new packages
        high_confidence_packages = [
            pkg['package_name'] for pkg in analysis.missing_packages
            if pkg['confidence'] >= 0.8 and not pkg['is_standard_library']
        ]
        
        for pkg in high_confidence_packages[:3]:  # Check first few
            assert pkg in updated_content
    
    def test_update_requirements_no_backup(self, sample_python_files, sample_requirements):
        """Test updating without backup creation."""
        engine = SmartRefactoringEngine(str(sample_requirements))
        analysis = engine.analyze_requirements()
        
        results = engine.update_requirements_file(analysis, dry_run=False, backup=False)
        
        assert results['backup_created'] == False
        backup_path = sample_requirements / "requirements.txt.backup"
        assert not backup_path.exists()


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_project(self, temp_project):
        """Test analysis of empty project."""
        engine = SmartRefactoringEngine(str(temp_project))
        analysis = engine.analyze_requirements()
        
        assert len(analysis.missing_packages) == 0
        assert len(analysis.unused_packages) == 0
        assert len(analysis.current_requirements) == 0
        assert len(analysis.detected_imports) == 0
    
    def test_invalid_python_file(self, temp_project):
        """Test handling of invalid Python files."""
        # Create invalid Python file
        (temp_project / "invalid.py").write_text("import invalid syntax here!")
        
        engine = SmartRefactoringEngine(str(temp_project))
        
        # Should handle gracefully without crashing
        analysis = engine.analyze_requirements()
        assert isinstance(analysis, RequirementsAnalysis)
    
    def test_missing_requirements_file_update(self, sample_python_files):
        """Test updating when requirements.txt doesn't exist."""
        engine = SmartRefactoringEngine(str(sample_python_files))
        analysis = engine.analyze_requirements()
        
        # Should work even without existing requirements.txt
        results = engine.update_requirements_file(analysis, dry_run=False)
        
        assert results['original_count'] == 0
        assert results['new_count'] > 0
        assert (sample_python_files / "requirements.txt").exists()
    
    @patch('builtins.open', side_effect=PermissionError("Access denied"))
    def test_file_permission_error(self, mock_open, temp_project):
        """Test handling of file permission errors."""
        engine = SmartRefactoringEngine(str(temp_project))
        
        # Should handle permission errors gracefully
        requirements = engine._parse_requirements_file()
        assert requirements == []


class TestPerformance:
    """Test performance aspects of requirements analysis."""
    
    def test_large_project_simulation(self, temp_project):
        """Test performance with many files."""
        # Create many Python files
        for i in range(20):
            (temp_project / f"module_{i}.py").write_text(f"""
import os
import sys
import requests
import pandas as pd
from flask import Flask

def function_{i}():
    pass
""")
        
        engine = SmartRefactoringEngine(str(temp_project))
        
        import time
        start_time = time.time()
        analysis = engine.analyze_requirements()
        end_time = time.time()
        
        # Should complete in reasonable time (less than 5 seconds)
        assert end_time - start_time < 5.0
        
        # Should still produce valid results
        assert len(analysis.detected_imports) > 0
        assert len(analysis.missing_packages) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])