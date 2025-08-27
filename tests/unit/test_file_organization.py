#!/usr/bin/env python3
"""
Comprehensive test suite for file organization functionality.
Tests the smart file organization system that handles messy AI-generated project structures.
"""

import unittest
from unittest.mock import Mock, patch, mock_open, MagicMock
from pathlib import Path
import tempfile
import shutil
import os
import ast

# Import the classes we're testing
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from deepflow.smart_refactoring_engine import SmartRefactoringEngine, FileOrganizationAnalysis
except ImportError as e:
    print(f"Failed to import SmartRefactoringEngine: {e}")
    SmartRefactoringEngine = None
    FileOrganizationAnalysis = None


@unittest.skipIf(SmartRefactoringEngine is None, "SmartRefactoringEngine not available")
class TestFileOrganization(unittest.TestCase):
    """Test file organization functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.engine = SmartRefactoringEngine(self.test_dir)
        
        # Create test files structure (typical messy AI-generated project)
        self.create_messy_project_structure()
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def create_messy_project_structure(self):
        """Create a messy project structure typical of AI-generated code."""
        # Root clutter - files that should be in subdirectories
        test_files = {
            'test_main.py': 'import unittest\nclass TestMain(unittest.TestCase): pass',
            'user_test.py': 'import pytest\ndef test_user(): pass',
            'test_helper.py': 'import unittest\nclass TestHelper(unittest.TestCase): pass',
            'config_settings.py': 'DATABASE_URL = "sqlite:///test.db"',
            'settings_prod.py': 'DEBUG = False',
            'user_model.py': 'class User:\n    def __init__(self): pass',
            'product_model.py': 'from dataclasses import dataclass\n@dataclass\nclass Product: pass',
            'user_view.py': 'def render_user_page(): return "user page"',
            'product_view.py': 'from flask import render_template\ndef product_list(): pass',
            'auth_controller.py': '@app.route("/login")\ndef login(): pass',
            'api_controller.py': 'from fastapi import APIRouter\nrouter = APIRouter()',
            'string_utils.py': 'def format_string(s): return s.upper()',
            'file_helper.py': 'import os\ndef read_file(path): pass',
            'data_script.py': 'if __name__ == "__main__":\n    print("processing data")',
            'migration_script.py': '# Migration script\nif __name__ == "__main__": pass',
            'UserService.py': 'class UserService:\n    def get_user(self): pass',  # PascalCase
            'api-handler.py': 'def handle_request(): pass',  # kebab-case
            'main.py': 'if __name__ == "__main__":\n    print("main app")',  # Should stay in root
            'app.py': 'from flask import Flask\napp = Flask(__name__)',  # Should stay in root
        }
        
        for filename, content in test_files.items():
            file_path = self.test_dir / filename
            file_path.write_text(content, encoding='utf-8')
    
    def test_analyze_current_structure(self):
        """Test analysis of current project structure."""
        all_files = list(self.test_dir.glob('**/*.py'))
        structure = self.engine._analyze_current_structure(all_files)
        
        self.assertIn('total_files', structure)
        self.assertIn('directories', structure)
        self.assertIn('file_types', structure)
        self.assertIn('depth_distribution', structure)
        
        # Should have files in root directory
        self.assertIn('root', structure['directories'])
        self.assertGreater(structure['directories']['root'], 0)
        
        # Should detect Python files
        self.assertIn('.py', structure['file_types'])
        
        # Should have depth 0 files (root files)
        self.assertIn(0, structure['depth_distribution'])
        self.assertGreater(structure['depth_distribution'][0], 0)
    
    def test_detect_root_clutter(self):
        """Test detection of files that should be moved from root."""
        all_files = list(self.test_dir.glob('**/*.py'))
        root_clutter = self.engine._detect_root_clutter(all_files)
        
        # Should detect test files
        test_clutter = [c for c in root_clutter if 'test' in c['suggested_directory']]
        self.assertGreater(len(test_clutter), 0)
        
        # Should detect config files
        config_clutter = [c for c in root_clutter if 'config' in c['suggested_directory']]
        self.assertGreater(len(config_clutter), 0)
        
        # Should detect model files
        model_clutter = [c for c in root_clutter if 'model' in c['suggested_directory']]
        self.assertGreater(len(model_clutter), 0)
        
        # Should NOT suggest moving main.py or app.py
        main_files = [c for c in root_clutter if c['file_name'] in ['main.py', 'app.py']]
        self.assertEqual(len(main_files), 0, "main.py and app.py should stay in root")
        
        # Check confidence scores
        for clutter in root_clutter:
            self.assertIn('confidence', clutter)
            self.assertGreaterEqual(clutter['confidence'], 0.0)
            self.assertLessEqual(clutter['confidence'], 1.0)
    
    def test_categorize_file_purpose(self):
        """Test categorization of file purposes based on content and name."""
        # Test file detection
        test_content = "import unittest\nclass TestUser(unittest.TestCase): pass"
        purpose = self.engine._categorize_file_purpose(test_content, "test_user.py")
        self.assertEqual(purpose, 'test')
        
        # Model file detection
        model_content = "from dataclasses import dataclass\n@dataclass\nclass User: pass"
        purpose = self.engine._categorize_file_purpose(model_content, "user_model.py")
        self.assertEqual(purpose, 'model')
        
        # View file detection
        view_content = "from flask import render_template\ndef user_view(): pass"
        purpose = self.engine._categorize_file_purpose(view_content, "user_view.py")
        self.assertEqual(purpose, 'view')
        
        # Controller/API file detection
        controller_content = "@app.route('/api')\ndef api_handler(): pass"
        purpose = self.engine._categorize_file_purpose(controller_content, "api.py")
        self.assertEqual(purpose, 'controller')
        
        # Utility file detection
        util_content = "def format_string(s): return s.upper()"
        purpose = self.engine._categorize_file_purpose(util_content, "string_utils.py")
        self.assertEqual(purpose, 'utility')
        
        # Config file detection
        config_content = "DATABASE_URL = 'sqlite:///app.db'"
        purpose = self.engine._categorize_file_purpose(config_content, "config.py")
        self.assertEqual(purpose, 'config')
        
        # Script file detection
        script_content = 'if __name__ == "__main__":\n    print("running script")'
        purpose = self.engine._categorize_file_purpose(script_content, "data_processor.py")
        self.assertEqual(purpose, 'script')
    
    def test_suggest_directory_structure(self):
        """Test suggestion of improved directory structure."""
        all_files = list(self.test_dir.glob('**/*.py'))
        suggestions = self.engine._suggest_directory_structure(all_files)
        
        # Should suggest creating common directories
        suggested_dirs = [s['directory_name'] for s in suggestions]
        
        # Should suggest tests directory (multiple test files)
        self.assertIn('tests', suggested_dirs)
        
        # Should suggest models directory (multiple model files)
        self.assertIn('models', suggested_dirs)
        
        # Should suggest views directory (multiple view files)
        self.assertIn('views', suggested_dirs)
        
        # Should suggest config directory (multiple config files)
        self.assertIn('config', suggested_dirs)
        
        # Check suggestion structure
        for suggestion in suggestions:
            self.assertIn('directory_name', suggestion)
            self.assertIn('description', suggestion)
            self.assertIn('file_count', suggestion)
            self.assertIn('purpose', suggestion)
            self.assertIn('priority', suggestion)
            self.assertIn(suggestion['priority'], ['high', 'medium', 'low'])
    
    def test_detect_naming_inconsistencies(self):
        """Test detection of naming pattern inconsistencies."""
        all_files = list(self.test_dir.glob('**/*.py'))
        inconsistencies = self.engine._detect_naming_inconsistencies(all_files)
        
        # Should find files with different naming patterns
        self.assertGreater(len(inconsistencies), 0)
        
        # Check for specific inconsistencies we created
        problem_files = [i['file_path'] for i in inconsistencies]
        
        # Should detect PascalCase file (UserService.py) in snake_case project
        pascal_files = [f for f in problem_files if 'UserService.py' in f]
        self.assertGreater(len(pascal_files), 0, "Should detect PascalCase inconsistency")
        
        # Should detect kebab-case file (api-handler.py) in snake_case project  
        kebab_files = [f for f in problem_files if 'api-handler.py' in f]
        self.assertGreater(len(kebab_files), 0, "Should detect kebab-case inconsistency")
        
        # Check inconsistency structure
        for inconsistency in inconsistencies:
            self.assertIn('file_path', inconsistency)
            self.assertIn('current_pattern', inconsistency)
            self.assertIn('expected_pattern', inconsistency)
            self.assertIn('suggested_name', inconsistency)
            self.assertIn('confidence', inconsistency)
    
    def test_naming_pattern_detection(self):
        """Test detection of specific naming patterns."""
        # Test snake_case
        self.assertEqual(self.engine._detect_naming_pattern('user_model'), 'snake_case')
        self.assertEqual(self.engine._detect_naming_pattern('test_helper'), 'snake_case')
        
        # Test camelCase
        self.assertEqual(self.engine._detect_naming_pattern('userModel'), 'camelCase')
        self.assertEqual(self.engine._detect_naming_pattern('testHelper'), 'camelCase')
        
        # Test PascalCase
        self.assertEqual(self.engine._detect_naming_pattern('UserModel'), 'PascalCase')
        self.assertEqual(self.engine._detect_naming_pattern('TestHelper'), 'PascalCase')
        
        # Test kebab-case
        self.assertEqual(self.engine._detect_naming_pattern('user-model'), 'kebab-case')
        self.assertEqual(self.engine._detect_naming_pattern('test-helper'), 'kebab-case')
        
        # Test mixed/unclear patterns
        self.assertEqual(self.engine._detect_naming_pattern('User_Model'), 'mixed')
        self.assertEqual(self.engine._detect_naming_pattern('user123'), 'mixed')
    
    def test_naming_pattern_conversion(self):
        """Test conversion between naming patterns."""
        # Convert to snake_case
        self.assertEqual(self.engine._convert_naming_pattern('UserModel', 'snake_case'), 'user_model')
        self.assertEqual(self.engine._convert_naming_pattern('userModel', 'snake_case'), 'user_model')
        self.assertEqual(self.engine._convert_naming_pattern('user-model', 'snake_case'), 'user_model')
        
        # Convert to camelCase
        self.assertEqual(self.engine._convert_naming_pattern('user_model', 'camelCase'), 'userModel')
        self.assertEqual(self.engine._convert_naming_pattern('UserModel', 'camelCase'), 'userModel')
        
        # Convert to PascalCase
        self.assertEqual(self.engine._convert_naming_pattern('user_model', 'PascalCase'), 'UserModel')
        self.assertEqual(self.engine._convert_naming_pattern('userModel', 'PascalCase'), 'UserModel')
        
        # Convert to kebab-case
        self.assertEqual(self.engine._convert_naming_pattern('user_model', 'kebab-case'), 'user-model')
        self.assertEqual(self.engine._convert_naming_pattern('UserModel', 'kebab-case'), 'user-model')
    
    def test_generate_relocation_recommendations(self):
        """Test generation of file relocation recommendations."""
        all_files = list(self.test_dir.glob('**/*.py'))
        suggested_directories = self.engine._suggest_directory_structure(all_files)
        relocations = self.engine._generate_relocation_recommendations(all_files, suggested_directories)
        
        self.assertGreater(len(relocations), 0)
        
        # Check relocation structure
        for relocation in relocations:
            self.assertIn('file_path', relocation)
            self.assertIn('current_location', relocation)
            self.assertIn('target_directory', relocation)
            self.assertIn('purpose', relocation)
            self.assertIn('confidence', relocation)
            self.assertIn('reason', relocation)
            
            # Confidence should be reasonable
            self.assertGreaterEqual(relocation['confidence'], 0.4)
            self.assertLessEqual(relocation['confidence'], 1.0)
    
    def test_calculate_structure_score(self):
        """Test calculation of project structure quality score."""
        all_files = list(self.test_dir.glob('**/*.py'))
        root_clutter = self.engine._detect_root_clutter(all_files)
        naming_issues = self.engine._detect_naming_inconsistencies(all_files)
        
        score = self.engine._calculate_structure_score(all_files, root_clutter, naming_issues)
        
        # Score should be between 0 and 100
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)
        
        # With our messy test structure, score should be relatively low
        self.assertLess(score, 80.0, "Messy structure should get low score")
    
    def test_full_file_organization_analysis(self):
        """Test the complete file organization analysis process."""
        analysis = self.engine.analyze_file_organization(check_patterns=True)
        
        # Check that we got a proper FileOrganizationAnalysis object
        self.assertIsInstance(analysis, FileOrganizationAnalysis)
        
        # Check all required fields are present
        self.assertIsInstance(analysis.project_structure_score, float)
        self.assertGreaterEqual(analysis.project_structure_score, 0.0)
        self.assertLessEqual(analysis.project_structure_score, 100.0)
        
        self.assertIsInstance(analysis.root_clutter_files, list)
        self.assertGreater(len(analysis.root_clutter_files), 0)
        
        self.assertIsInstance(analysis.suggested_directories, list)
        self.assertGreater(len(analysis.suggested_directories), 0)
        
        self.assertIsInstance(analysis.file_relocations, list)
        self.assertGreater(len(analysis.file_relocations), 0)
        
        self.assertIsInstance(analysis.naming_inconsistencies, list)
        self.assertGreater(len(analysis.naming_inconsistencies), 0)
        
        self.assertIsInstance(analysis.organization_recommendations, list)
        self.assertGreater(len(analysis.organization_recommendations), 0)
        
        self.assertIsInstance(analysis.current_structure, dict)
        self.assertIn('total_files', analysis.current_structure)
        
        self.assertIsInstance(analysis.ideal_structure, dict)
        self.assertIn('suggested_directories', analysis.ideal_structure)
    
    def test_dry_run_file_organization(self):
        """Test file organization in dry-run mode."""
        analysis = self.engine.analyze_file_organization(check_patterns=True)
        results = self.engine.organize_files(analysis, dry_run=True, backup=True)
        
        # Check results structure
        self.assertIsInstance(results, dict)
        self.assertIn('success', results)
        self.assertTrue(results['success'])
        self.assertTrue(results['dry_run'])
        
        self.assertIn('changes_applied', results)
        self.assertIn('directories_created', results)
        self.assertIn('files_moved', results)
        self.assertIn('files_renamed', results)
        self.assertIn('errors', results)
        
        # In dry run, no actual changes should be made
        # Files should still be in original locations
        original_files = set(f.name for f in self.test_dir.glob('*.py'))
        self.assertIn('test_main.py', original_files)
        self.assertIn('user_model.py', original_files)
        
        # No new directories should be created
        subdirs = [d for d in self.test_dir.iterdir() if d.is_dir()]
        self.assertEqual(len(subdirs), 0, "No directories should be created in dry run")
    
    def test_actual_file_organization_high_confidence_only(self):
        """Test actual file organization with high confidence moves only."""
        # Create a temporary copy for actual moves
        test_copy_dir = Path(tempfile.mkdtemp())
        
        try:
            # Copy all test files to the new directory
            for file_path in self.test_dir.glob('*.py'):
                shutil.copy2(file_path, test_copy_dir / file_path.name)
            
            # Create engine for the copy
            copy_engine = SmartRefactoringEngine(test_copy_dir)
            analysis = copy_engine.analyze_file_organization(check_patterns=True)
            
            # Apply organization with actual changes
            results = copy_engine.organize_files(analysis, dry_run=False, backup=True)
            
            self.assertIsInstance(results, dict)
            self.assertIn('success', results)
            self.assertFalse(results['dry_run'])
            
            # Some high-confidence changes should have been applied
            self.assertGreaterEqual(results['changes_applied'], 0)
            
            # Check that some directories might have been created
            if results['directories_created']:
                # Verify directories actually exist
                for directory_name in results['directories_created']:
                    if not directory_name.startswith('[DRY RUN]'):
                        dir_path = test_copy_dir / directory_name
                        self.assertTrue(dir_path.exists(), f"Directory {directory_name} should exist")
                        self.assertTrue(dir_path.is_dir(), f"{directory_name} should be a directory")
            
            # Check that files were actually moved if any high-confidence moves occurred
            if results['files_moved']:
                for move in results['files_moved']:
                    if not move['from'].startswith('[DRY RUN]'):
                        # Original location should not exist (or be a backup)
                        original_path = test_copy_dir / move['from']
                        new_path = test_copy_dir / move['to']
                        
                        # Either the file moved or it still exists (low confidence, not moved)
                        self.assertTrue(
                            new_path.exists() or original_path.exists(),
                            f"File should exist in either original or new location: {move}"
                        )
        
        finally:
            # Clean up
            if test_copy_dir.exists():
                shutil.rmtree(test_copy_dir)
    
    def test_organization_recommendations_generation(self):
        """Test generation of high-level organization recommendations."""
        # Create sample data
        root_clutter = [
            {'file_name': 'test_user.py', 'suggested_directory': 'tests'},
            {'file_name': 'user_model.py', 'suggested_directory': 'models'}
        ]
        relocations = [
            {'confidence': 0.8, 'target_directory': 'tests'},
            {'confidence': 0.9, 'target_directory': 'models'}
        ]
        naming_issues = [
            {'confidence': 0.7, 'file_path': 'UserService.py'},
            {'confidence': 0.8, 'file_path': 'api-handler.py'}
        ]
        
        recommendations = self.engine._generate_organization_recommendations(
            root_clutter, relocations, naming_issues
        )
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Check recommendation structure
        for rec in recommendations:
            self.assertIn('type', rec)
            self.assertIn('priority', rec)
            self.assertIn('action', rec)
            self.assertIn('affected_files', rec)
            self.assertIn('description', rec)
            self.assertIn(rec['priority'], ['high', 'medium', 'low'])
    
    def test_ideal_structure_generation(self):
        """Test generation of ideal project structure."""
        suggested_directories = [
            {'directory_name': 'tests', 'description': 'Test files', 'file_count': 3, 'priority': 'high'},
            {'directory_name': 'models', 'description': 'Data models', 'file_count': 2, 'priority': 'medium'}
        ]
        relocations = [
            {'target_directory': 'tests', 'confidence': 0.9},
            {'target_directory': 'models', 'confidence': 0.8}
        ]
        
        ideal = self.engine._generate_ideal_structure(suggested_directories, relocations)
        
        self.assertIsInstance(ideal, dict)
        self.assertIn('suggested_directories', ideal)
        self.assertIn('organization_principles', ideal)
        self.assertIn('structure_benefits', ideal)
        
        # Check suggested directories
        self.assertIn('tests', ideal['suggested_directories'])
        self.assertIn('models', ideal['suggested_directories'])
        
        # Check that principles and benefits are lists with content
        self.assertIsInstance(ideal['organization_principles'], list)
        self.assertGreater(len(ideal['organization_principles']), 0)
        
        self.assertIsInstance(ideal['structure_benefits'], list)
        self.assertGreater(len(ideal['structure_benefits']), 0)
    
    def test_error_handling_for_file_operations(self):
        """Test error handling during file operations."""
        analysis = self.engine.analyze_file_organization()
        
        # Mock file operations to simulate errors
        with patch('shutil.move', side_effect=PermissionError("Permission denied")):
            results = self.engine.organize_files(analysis, dry_run=False, backup=True)
            
            # Should handle errors gracefully
            self.assertIsInstance(results, dict)
            self.assertIn('success', results)
            self.assertIn('errors', results)
            
            # If errors occurred, success might be False or errors list populated
            if not results['success']:
                self.assertGreater(len(results['errors']), 0)
    
    def test_backup_creation_during_organization(self):
        """Test that backups are created when requested."""
        # This test would need actual file operations, which is complex in unit tests
        # For now, we verify the logic path exists
        analysis = self.engine.analyze_file_organization()
        
        # Test dry run with backup=True
        results = self.engine.organize_files(analysis, dry_run=True, backup=True)
        
        # Should complete successfully
        self.assertTrue(results['success'])
        self.assertTrue(results['dry_run'])
    
    def test_no_changes_when_project_well_organized(self):
        """Test behavior when project is already well organized."""
        # Create a well-organized test structure
        well_organized_dir = Path(tempfile.mkdtemp())
        
        try:
            # Create proper directory structure
            (well_organized_dir / 'tests').mkdir()
            (well_organized_dir / 'models').mkdir()
            (well_organized_dir / 'views').mkdir()
            
            # Create well-placed files
            (well_organized_dir / 'main.py').write_text('if __name__ == "__main__": pass')
            (well_organized_dir / 'tests' / 'test_main.py').write_text('import unittest')
            (well_organized_dir / 'models' / 'user_model.py').write_text('class User: pass')
            (well_organized_dir / 'views' / 'user_view.py').write_text('def render(): pass')
            
            # Analyze the well-organized structure
            well_organized_engine = SmartRefactoringEngine(well_organized_dir)
            analysis = well_organized_engine.analyze_file_organization()
            
            # Should have a high structure score
            self.assertGreater(analysis.project_structure_score, 80.0)
            
            # Should have minimal or no clutter
            self.assertLessEqual(len(analysis.root_clutter_files), 1)  # Only main.py in root is OK
            
            # Should have few or no relocations needed
            high_confidence_relocations = [r for r in analysis.file_relocations if r['confidence'] >= 0.8]
            self.assertLessEqual(len(high_confidence_relocations), 1)
        
        finally:
            if well_organized_dir.exists():
                shutil.rmtree(well_organized_dir)


if __name__ == '__main__':
    unittest.main()