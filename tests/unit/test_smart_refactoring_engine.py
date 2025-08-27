"""
Test suite for Smart Refactoring Engine (Priority 4).

Comprehensive tests for pattern standardization, import optimization, 
file splitting, dead code removal, and documentation generation.
"""

import ast
import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

# Import the smart refactoring engine
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from deepflow.smart_refactoring_engine import (
    SmartRefactoringEngine,
    PatternAnalysis,
    ImportAnalysis,
    FileSplitAnalysis,
    DeadCodeAnalysis,
    DocumentationAnalysis
)


class TestSmartRefactoringEngine:
    """Test cases for SmartRefactoringEngine class."""
    
    @pytest.fixture
    def temp_project(self):
        """Create a temporary project directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            
            # Create sample Python files
            (project_path / "main.py").write_text("""
import os
import sys
import unused_module

def camelCaseFunction():
    pass

def snake_case_function():
    pass

class MyClass:
    def __init__(self):
        pass
    
    def method_without_docstring(self):
        return "test"

def unused_function():
    \"\"\"This function is never called.\"\"\"
    return "unused"

def documented_function():
    \"\"\"This function has proper documentation.
    
    Returns:
        str: A test string.
    \"\"\"
    return "documented"
""")
            
            (project_path / "utils.py").write_text("""
import json
import json  # Duplicate import

def utility_function():
    pass

class UtilityClass:
    pass

unused_variable = "never used"
""")
            
            (project_path / "large_file.py").write_text("""
# Large file with many classes and functions
""" + "\n".join([f"""
class Class{i}:
    def __init__(self):
        pass
    
    def method_{i}(self):
        return {i}
""" for i in range(15)]) + "\n".join([f"""
def function_{i}():
    return {i}
""" for i in range(20)]))
            
            yield project_path
    
    @pytest.fixture
    def engine(self, temp_project):
        """Create a SmartRefactoringEngine instance."""
        return SmartRefactoringEngine(str(temp_project))
    
    def test_initialization(self, temp_project):
        """Test engine initialization."""
        engine = SmartRefactoringEngine(str(temp_project))
        assert engine.project_path == temp_project
        assert engine.pattern_cache == {}
        assert engine.import_cache == {}
        assert engine.analysis_cache == {}
    
    def test_get_python_files(self, engine, temp_project):
        """Test getting Python files from project."""
        files = engine._get_python_files()
        assert len(files) == 3
        file_names = {f.name for f in files}
        assert file_names == {"main.py", "utils.py", "large_file.py"}
    
    def test_standardize_patterns(self, engine):
        """Test pattern standardization analysis."""
        result = engine.standardize_patterns()
        
        assert isinstance(result, PatternAnalysis)
        assert result.pattern_type == "comprehensive"
        assert 0 <= result.consistency_score <= 1
        assert isinstance(result.violations, list)
        assert isinstance(result.files_affected, list)
        
        # Should detect mixed naming conventions
        violations = result.violations
        naming_violations = [v for v in violations if v['type'] == 'inconsistent_naming']
        assert len(naming_violations) > 0
    
    def test_optimize_imports(self, engine):
        """Test import optimization analysis."""
        result = engine.optimize_imports()
        
        assert isinstance(result, ImportAnalysis)
        assert isinstance(result.unused_imports, list)
        assert isinstance(result.duplicate_imports, list)
        assert isinstance(result.circular_imports, list)
        assert isinstance(result.optimization_suggestions, list)
        
        # Should detect unused and duplicate imports
        assert len(result.unused_imports) > 0  # unused_module in main.py
        assert len(result.duplicate_imports) > 0  # duplicate json import in utils.py
    
    def test_suggest_file_splits(self, engine):
        """Test file splitting analysis."""
        result = engine.suggest_file_splits()
        
        assert isinstance(result, list)
        assert all(isinstance(analysis, FileSplitAnalysis) for analysis in result)
        
        # Large file should have split recommendations
        large_file_analysis = next((a for a in result if "large_file.py" in a.file_path), None)
        assert large_file_analysis is not None
        assert large_file_analysis.complexity_score > 0.5
        assert len(large_file_analysis.split_recommendations) > 0
    
    def test_detect_dead_code(self, engine):
        """Test dead code detection."""
        result = engine.detect_dead_code()
        
        assert isinstance(result, DeadCodeAnalysis)
        assert isinstance(result.unused_functions, list)
        assert isinstance(result.unused_classes, list)
        assert isinstance(result.unused_variables, list)
        assert isinstance(result.unreachable_code, list)
        
        # Should detect unused function and variable
        assert any("unused_function" in func for func in result.unused_functions)
    
    def test_generate_documentation(self, engine):
        """Test documentation generation."""
        result = engine.generate_documentation()
        
        assert isinstance(result, DocumentationAnalysis)
        assert isinstance(result.missing_docstrings, list)
        assert isinstance(result.incomplete_docstrings, list)
        assert isinstance(result.generated_docstrings, dict)
        
        # Should detect functions without docstrings
        assert len(result.missing_docstrings) > 0
        assert len(result.generated_docstrings) > 0
        
        # Check generated docstring format
        sample_docstring = next(iter(result.generated_docstrings.values()))
        assert '"""' in sample_docstring
        assert "function" in sample_docstring or "class" in sample_docstring
    
    def test_extract_patterns(self, engine, temp_project):
        """Test pattern extraction from AST."""
        main_file = temp_project / "main.py"
        with open(main_file, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content, filename=str(main_file))
        patterns = engine._extract_patterns(tree, str(main_file))
        
        assert 'function' in patterns
        assert 'class' in patterns
        assert 'import' in patterns
        
        # Check function patterns
        functions = patterns['function']
        function_names = {f['name'] for f in functions}
        assert 'camelCaseFunction' in function_names
        assert 'snake_case_function' in function_names
    
    def test_detect_pattern_violations(self, engine, temp_project):
        """Test pattern violation detection."""
        main_file = temp_project / "main.py"
        with open(main_file, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content, filename=str(main_file))
        patterns = engine._extract_patterns(tree, str(main_file))
        violations = engine._detect_pattern_violations(patterns, str(main_file))
        
        # Should detect inconsistent naming
        assert len(violations) > 0
        naming_violation = next((v for v in violations if v['type'] == 'inconsistent_naming'), None)
        assert naming_violation is not None
    
    def test_analyze_file_imports(self, engine, temp_project):
        """Test import analysis for a single file."""
        utils_file = temp_project / "utils.py"
        with open(utils_file, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content, filename=str(utils_file))
        analysis = engine._analyze_file_imports(tree, str(utils_file), content)
        
        assert 'unused' in analysis
        assert 'duplicates' in analysis
        assert 'suggestions' in analysis
        
        # Should detect duplicate json import
        assert len(analysis['duplicates']) > 0
    
    def test_calculate_size_score(self, engine):
        """Test file size scoring."""
        small_content = "import os\n\ndef test():\n    pass"
        large_content = "\n".join([f"def function_{i}(): pass" for i in range(100)])
        
        small_score = engine._calculate_size_score(small_content)
        large_score = engine._calculate_size_score(large_content)
        
        assert 0 <= small_score <= 1
        assert 0 <= large_score <= 1
        assert small_score < large_score
    
    def test_calculate_complexity_score(self, engine):
        """Test complexity scoring."""
        simple_content = "def simple(): return 1"
        complex_content = """
def complex_function():
    if True:
        for i in range(10):
            try:
                with open('file') as f:
                    if f:
                        while True:
                            break
            except:
                pass
    return 1
"""
        
        simple_tree = ast.parse(simple_content)
        complex_tree = ast.parse(complex_content)
        
        simple_score = engine._calculate_complexity_score(simple_tree)
        complex_score = engine._calculate_complexity_score(complex_tree)
        
        assert 0 <= simple_score <= 1
        assert 0 <= complex_score <= 1
        assert simple_score < complex_score
    
    def test_generate_docstring(self, engine):
        """Test docstring generation."""
        function_item = {
            'type': 'function',
            'name': 'test_function',
            'args': ['arg1', 'arg2']
        }
        
        class_item = {
            'type': 'class',
            'name': 'TestClass',
            'args': []
        }
        
        func_docstring = engine._generate_docstring(function_item)
        class_docstring = engine._generate_docstring(class_item)
        
        assert 'test_function function' in func_docstring
        assert 'arg1' in func_docstring
        assert 'arg2' in func_docstring
        assert 'Args:' in func_docstring
        assert 'Returns:' in func_docstring
        
        assert 'TestClass class' in class_docstring
        assert 'Attributes:' in class_docstring
    
    def test_apply_refactoring_dry_run(self, engine):
        """Test applying refactoring in dry run mode."""
        # Generate analysis results
        pattern_analysis = engine.standardize_patterns()
        import_analysis = engine.optimize_imports()
        
        analysis_results = {
            'pattern_analysis': pattern_analysis,
            'import_analysis': import_analysis
        }
        
        result = engine.apply_refactoring(analysis_results, dry_run=True)
        
        assert isinstance(result, dict)
        assert 'patterns_standardized' in result
        assert 'imports_optimized' in result
        assert 'files_split' in result
        assert 'dead_code_removed' in result
        assert 'documentation_added' in result
        assert 'errors' in result
        
        # In dry run, nothing should be applied
        assert result['patterns_standardized'] == 0
        assert result['imports_optimized'] == 0
    
    def test_comprehensive_analysis(self, engine):
        """Test running all analysis methods together."""
        pattern_analysis = engine.standardize_patterns()
        import_analysis = engine.optimize_imports()
        file_split_analysis = engine.suggest_file_splits()
        dead_code_analysis = engine.detect_dead_code()
        doc_analysis = engine.generate_documentation()
        
        # All analyses should return appropriate types
        assert isinstance(pattern_analysis, PatternAnalysis)
        assert isinstance(import_analysis, ImportAnalysis)
        assert isinstance(file_split_analysis, list)
        assert isinstance(dead_code_analysis, DeadCodeAnalysis)
        assert isinstance(doc_analysis, DocumentationAnalysis)
        
        # Should detect various issues
        assert len(pattern_analysis.violations) > 0
        assert len(import_analysis.unused_imports) > 0
        assert len(file_split_analysis) > 0
        assert len(doc_analysis.missing_docstrings) > 0


class TestPatternAnalysis:
    """Test PatternAnalysis dataclass."""
    
    def test_pattern_analysis_creation(self):
        """Test creating PatternAnalysis instance."""
        analysis = PatternAnalysis(
            pattern_type="test",
            consistency_score=0.85,
            violations=[{'type': 'naming', 'count': 3}],
            recommended_pattern="snake_case",
            files_affected=["file1.py", "file2.py"]
        )
        
        assert analysis.pattern_type == "test"
        assert analysis.consistency_score == 0.85
        assert len(analysis.violations) == 1
        assert analysis.recommended_pattern == "snake_case"
        assert len(analysis.files_affected) == 2


class TestImportAnalysis:
    """Test ImportAnalysis dataclass."""
    
    def test_import_analysis_creation(self):
        """Test creating ImportAnalysis instance."""
        analysis = ImportAnalysis(
            unused_imports=["module1", "module2"],
            duplicate_imports=["json"],
            circular_imports=[("file1", "file2")],
            optimization_suggestions=[{'type': 'merge', 'imports': ['os', 'sys']}]
        )
        
        assert len(analysis.unused_imports) == 2
        assert len(analysis.duplicate_imports) == 1
        assert len(analysis.circular_imports) == 1
        assert len(analysis.optimization_suggestions) == 1


class TestFileSplitAnalysis:
    """Test FileSplitAnalysis dataclass."""
    
    def test_file_split_analysis_creation(self):
        """Test creating FileSplitAnalysis instance."""
        analysis = FileSplitAnalysis(
            file_path="large_file.py",
            size_score=0.9,
            complexity_score=0.8,
            split_recommendations=[{'type': 'class_grouping'}],
            suggested_modules=["classes.py", "utils.py"]
        )
        
        assert analysis.file_path == "large_file.py"
        assert analysis.size_score == 0.9
        assert analysis.complexity_score == 0.8
        assert len(analysis.split_recommendations) == 1
        assert len(analysis.suggested_modules) == 2


class TestDeadCodeAnalysis:
    """Test DeadCodeAnalysis dataclass."""
    
    def test_dead_code_analysis_creation(self):
        """Test creating DeadCodeAnalysis instance."""
        analysis = DeadCodeAnalysis(
            unused_functions=["func1", "func2"],
            unused_classes=["Class1"],
            unused_variables=["var1"],
            unreachable_code=[{'line': 50, 'reason': 'after return'}]
        )
        
        assert len(analysis.unused_functions) == 2
        assert len(analysis.unused_classes) == 1
        assert len(analysis.unused_variables) == 1
        assert len(analysis.unreachable_code) == 1


class TestDocumentationAnalysis:
    """Test DocumentationAnalysis dataclass."""
    
    def test_documentation_analysis_creation(self):
        """Test creating DocumentationAnalysis instance."""
        analysis = DocumentationAnalysis(
            missing_docstrings=[{'name': 'func1', 'type': 'function'}],
            incomplete_docstrings=[{'name': 'func2', 'type': 'function'}],
            generated_docstrings={'func1': '"""Generated docstring."""'}
        )
        
        assert len(analysis.missing_docstrings) == 1
        assert len(analysis.incomplete_docstrings) == 1
        assert len(analysis.generated_docstrings) == 1


class TestErrorHandling:
    """Test error handling in SmartRefactoringEngine."""
    
    @pytest.fixture
    def engine_with_bad_files(self):
        """Create engine with problematic files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            
            # Create file with syntax error
            (project_path / "syntax_error.py").write_text("def invalid syntax:")
            
            # Create empty file
            (project_path / "empty.py").write_text("")
            
            yield SmartRefactoringEngine(str(project_path))
    
    def test_handles_syntax_errors(self, engine_with_bad_files):
        """Test that engine handles files with syntax errors."""
        # Should not crash on syntax errors
        result = engine_with_bad_files.standardize_patterns()
        assert isinstance(result, PatternAnalysis)
    
    def test_handles_empty_files(self, engine_with_bad_files):
        """Test that engine handles empty files."""
        result = engine_with_bad_files.generate_documentation()
        assert isinstance(result, DocumentationAnalysis)
    
    def test_handles_nonexistent_directory(self):
        """Test that engine handles nonexistent directory."""
        engine = SmartRefactoringEngine("/nonexistent/path")
        files = engine._get_python_files()
        assert len(files) == 0


class TestIntegration:
    """Integration tests for Priority 4 features."""
    
    @pytest.fixture
    def realistic_project(self):
        """Create a realistic project structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            project_path = Path(temp_dir)
            
            # Create main module
            (project_path / "__init__.py").write_text("")
            
            (project_path / "models.py").write_text("""
import json
import sys
import os
import unused_import

class UserModel:
    def __init__(self, name):
        self.name = name
    
    def get_name(self):
        return self.name

class ProductModel:
    def __init__(self, name, price):
        self.name = name
        self.price = price

def helper_function():
    pass

def unused_helper():
    return "never called"
""")
            
            (project_path / "views.py").write_text("""
from models import UserModel, ProductModel
import json

def displayUser(user):
    print(f"User: {user.get_name()}")

def display_product(product):
    print(f"Product: {product.name} - ${product.price}")

class ViewManager:
    def render(self):
        pass
""")
            
            yield project_path
    
    def test_end_to_end_analysis(self, realistic_project):
        """Test complete analysis pipeline."""
        engine = SmartRefactoringEngine(str(realistic_project))
        
        # Run all analyses
        pattern_analysis = engine.standardize_patterns()
        import_analysis = engine.optimize_imports()
        file_splits = engine.suggest_file_splits()
        dead_code = engine.detect_dead_code()
        docs = engine.generate_documentation()
        
        # Verify comprehensive analysis
        assert pattern_analysis.consistency_score < 1.0  # Mixed patterns detected
        assert len(import_analysis.unused_imports) > 0  # unused_import detected
        assert len(dead_code.unused_functions) > 0  # unused_helper detected
        assert len(docs.missing_docstrings) > 0  # Missing docstrings detected
    
    def test_refactoring_suggestions_quality(self, realistic_project):
        """Test the quality of refactoring suggestions."""
        engine = SmartRefactoringEngine(str(realistic_project))
        
        pattern_analysis = engine.standardize_patterns()
        
        # Should detect mixed naming conventions (displayUser vs display_product)
        naming_violations = [v for v in pattern_analysis.violations 
                           if v['type'] == 'inconsistent_naming']
        assert len(naming_violations) > 0
        
        doc_analysis = engine.generate_documentation()
        
        # Should generate proper docstrings
        for docstring in doc_analysis.generated_docstrings.values():
            assert '"""' in docstring
            assert len(docstring.strip()) > 10  # Substantial content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])