"""
Unit tests for code_analyzer.py
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from dataclasses import asdict

# Mock imports before importing the module
with patch.dict('sys.modules', {
    'rich': MagicMock(),
    'networkx': MagicMock()
}):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tools"))
    import code_analyzer


class TestCodeAnalyzer:
    """Test cases for CodeAnalyzer class."""
    
    def test_init(self, mock_project_structure):
        """Test CodeAnalyzer initialization."""
        analyzer = code_analyzer.CodeAnalyzer(str(mock_project_structure))
        
        assert analyzer.project_path == str(mock_project_structure)
        assert hasattr(analyzer, 'console')
    
    def test_init_with_nonexistent_path(self):
        """Test CodeAnalyzer initialization with nonexistent path."""
        with pytest.raises(FileNotFoundError):
            code_analyzer.CodeAnalyzer("/nonexistent/path")
    
    def test_analyze_unused_imports(self, mock_project_structure):
        """Test unused import analysis.""" 
        # Create test file with unused import
        test_file = mock_project_structure / "test_unused.py"
        test_file.write_text("""
import os
import sys
import unused_module
from pathlib import Path

def main():
    print(os.getcwd())
    return Path.cwd()
""")
        
        analyzer = code_analyzer.CodeAnalyzer(str(mock_project_structure))
        results = analyzer.analyze_unused_imports()
        
        assert isinstance(results, list)
        # Should find unused_module as unused
        unused_imports = [r for r in results if not r.is_used]
        assert any('unused_module' in r.import_name for r in unused_imports)
    
    def test_analyze_unused_imports_with_fix(self, mock_project_structure):
        """Test unused import analysis with fix mode."""
        test_file = mock_project_structure / "test_fix.py"
        test_content = """
import os
import unused_module

def main():
    print(os.getcwd())
"""
        test_file.write_text(test_content)
        
        with patch('builtins.open', mock_open(read_data=test_content)) as mock_file:
            analyzer = code_analyzer.CodeAnalyzer(str(mock_project_structure))
            results = analyzer.analyze_unused_imports(fix_mode=True)
            
            # Should attempt to write fixed content
            assert mock_file.called
            
            # Check that unused imports are detected
            unused_imports = [r for r in results if not r.is_used]
            assert len(unused_imports) > 0
    
    def test_analyze_coupling(self, mock_project_structure):
        """Test coupling analysis."""
        analyzer = code_analyzer.CodeAnalyzer(str(mock_project_structure))
        
        with patch.object(analyzer, '_calculate_module_coupling') as mock_coupling:
            mock_coupling.return_value = [
                code_analyzer.CouplingMetric(
                    module_a="main.py",
                    module_b="utils.py", 
                    coupling_strength=0.5,
                    coupling_type="afferent",
                    shared_dependencies=["json"],
                    refactoring_opportunity="Consider interface extraction"
                )
            ]
            
            results = analyzer.analyze_coupling()
            
            assert len(results) == 1
            assert results[0].module_a == "main.py"
            assert results[0].coupling_strength == 0.5
    
    def test_detect_architecture_violations(self, mock_project_structure):
        """Test architecture violation detection."""
        analyzer = code_analyzer.CodeAnalyzer(str(mock_project_structure))
        
        with patch.object(analyzer, '_check_layered_architecture') as mock_check, \
             patch.object(analyzer, '_detect_circular_imports') as mock_circular:
            
            mock_check.return_value = []
            mock_circular.return_value = [
                code_analyzer.ArchitectureViolation(
                    file_path="main.py",
                    violation_type="circular_dependency",
                    severity="HIGH",
                    description="Circular import detected",
                    suggestion="Refactor to break cycle",
                    pattern_violated="layered_architecture"
                )
            ]
            
            results = analyzer.detect_architecture_violations()
            
            assert len(results) == 1
            assert results[0].violation_type == "circular_dependency"
            assert results[0].severity == "HIGH"
    
    def test_calculate_technical_debt(self, mock_project_structure):
        """Test technical debt calculation."""
        analyzer = code_analyzer.CodeAnalyzer(str(mock_project_structure))
        
        with patch.object(analyzer, '_analyze_file_complexity') as mock_complexity:
            mock_complexity.return_value = code_analyzer.TechnicalDebt(
                file_path="main.py",
                debt_score=7.5,
                complexity_metrics={
                    "cyclomatic_complexity": 5,
                    "cognitive_complexity": 8,
                    "lines_of_code": 150
                },
                debt_indicators=["Long function", "High complexity"],
                refactoring_priority="MEDIUM",
                estimated_effort="4-8 hours"
            )
            
            results = analyzer.calculate_technical_debt()
            
            assert len(results) > 0
            assert results[0].debt_score == 7.5
            assert "Long function" in results[0].debt_indicators
    
    def test_analyze_ai_context_windows(self, mock_project_structure):
        """Test AI context window analysis."""
        # Create a large test file
        large_content = "import os\n" + "# " + "x" * 10000 + "\nprint('test')\n"
        test_file = mock_project_structure / "large_file.py"
        test_file.write_text(large_content)
        
        analyzer = code_analyzer.CodeAnalyzer(str(mock_project_structure))
        results = analyzer.analyze_ai_context_windows()
        
        assert len(results) > 0
        # Large file should have WARNING or CRITICAL health
        large_file_result = next((r for r in results if "large_file.py" in r.file_path), None)
        assert large_file_result is not None
        assert large_file_result.context_health in ["WARNING", "CRITICAL"]
    
    def test_extract_imports_from_ast(self):
        """Test import extraction from AST."""
        test_code = """
import os
import sys as system
from pathlib import Path, PurePath
from typing import Dict, List, Optional
import json as j
"""
        
        analyzer = code_analyzer.CodeAnalyzer(".")
        imports = analyzer._extract_imports_from_ast(test_code)
        
        expected_imports = ['os', 'sys', 'pathlib', 'typing', 'json']
        for expected in expected_imports:
            assert any(expected in imp for imp in imports)
    
    def test_extract_imports_from_invalid_ast(self):
        """Test import extraction from invalid Python code."""
        invalid_code = "import os\nif True\n    print('invalid')"
        
        analyzer = code_analyzer.CodeAnalyzer(".")
        imports = analyzer._extract_imports_from_ast(invalid_code)
        
        # Should return empty list for invalid syntax
        assert imports == []
    
    def test_is_import_used(self):
        """Test import usage detection."""
        code = """
import os
import unused_import
from pathlib import Path

def main():
    current_dir = os.getcwd()
    my_path = Path('/test')
    return current_dir
"""
        
        analyzer = code_analyzer.CodeAnalyzer(".")
        
        assert analyzer._is_import_used("os", code) is True
        assert analyzer._is_import_used("Path", code) is True
        assert analyzer._is_import_used("unused_import", code) is False
    
    def test_calculate_cyclomatic_complexity(self):
        """Test cyclomatic complexity calculation."""
        complex_code = """
def complex_function(x):
    if x > 0:
        if x > 10:
            return "high"
        elif x > 5:
            return "medium"
        else:
            return "low"
    else:
        return "negative"
"""
        
        analyzer = code_analyzer.CodeAnalyzer(".")
        complexity = analyzer._calculate_cyclomatic_complexity(complex_code)
        
        # Should be > 1 due to multiple branches
        assert complexity > 1
    
    def test_calculate_cognitive_complexity(self):
        """Test cognitive complexity calculation."""
        code = """
def cognitive_function(items):
    total = 0
    for item in items:
        if item > 0:
            if item % 2 == 0:
                total += item
            else:
                total += item * 2
    return total
"""
        
        analyzer = code_analyzer.CodeAnalyzer(".")
        complexity = analyzer._calculate_cognitive_complexity(code)
        
        # Should be > 0 due to nested conditions
        assert complexity > 0
    
    def test_estimate_refactoring_effort(self):
        """Test refactoring effort estimation."""
        analyzer = code_analyzer.CodeAnalyzer(".")
        
        # Low complexity
        effort_low = analyzer._estimate_refactoring_effort(3.0, 50, 2)
        assert "1-2 hours" in effort_low
        
        # High complexity  
        effort_high = analyzer._estimate_refactoring_effort(9.0, 500, 10)
        assert "days" in effort_high.lower()
    
    def test_get_refactoring_priority(self):
        """Test refactoring priority assessment."""
        analyzer = code_analyzer.CodeAnalyzer(".")
        
        assert analyzer._get_refactoring_priority(2.0) == "LOW"
        assert analyzer._get_refactoring_priority(6.0) == "MEDIUM"
        assert analyzer._get_refactoring_priority(9.0) == "HIGH"


class TestImportAnalysis:
    """Test ImportAnalysis dataclass."""
    
    def test_import_analysis_creation(self):
        """Test ImportAnalysis dataclass creation."""
        analysis = code_analyzer.ImportAnalysis(
            file_path="test.py",
            import_name="os",
            import_type="import",
            is_used=True,
            usage_count=3,
            line_number=1,
            suggestions=[]
        )
        
        assert analysis.file_path == "test.py"
        assert analysis.import_name == "os"
        assert analysis.is_used is True
        assert analysis.usage_count == 3
    
    def test_import_analysis_to_dict(self):
        """Test ImportAnalysis conversion to dictionary.""" 
        analysis = code_analyzer.ImportAnalysis(
            file_path="test.py",
            import_name="os",
            import_type="import",
            is_used=True,
            usage_count=3,
            line_number=1,
            suggestions=["Add type hints"]
        )
        
        analysis_dict = asdict(analysis)
        
        assert analysis_dict["file_path"] == "test.py"
        assert analysis_dict["import_name"] == "os"
        assert analysis_dict["suggestions"] == ["Add type hints"]


class TestCouplingMetric:
    """Test CouplingMetric dataclass."""
    
    def test_coupling_metric_creation(self):
        """Test CouplingMetric dataclass creation."""
        metric = code_analyzer.CouplingMetric(
            module_a="main.py",
            module_b="utils.py",
            coupling_strength=0.7,
            coupling_type="bidirectional",
            shared_dependencies=["json", "os"],
            refactoring_opportunity="Extract interface"
        )
        
        assert metric.module_a == "main.py"
        assert metric.coupling_strength == 0.7
        assert "json" in metric.shared_dependencies


class TestArchitectureViolation:
    """Test ArchitectureViolation dataclass."""
    
    def test_architecture_violation_creation(self):
        """Test ArchitectureViolation dataclass creation."""
        violation = code_analyzer.ArchitectureViolation(
            file_path="main.py",
            violation_type="layer_violation",
            severity="MEDIUM",
            description="UI layer accessing data layer directly",
            suggestion="Add service layer",
            pattern_violated="layered_architecture"
        )
        
        assert violation.file_path == "main.py"
        assert violation.severity == "MEDIUM"
        assert violation.pattern_violated == "layered_architecture"


class TestTechnicalDebt:
    """Test TechnicalDebt dataclass."""
    
    def test_technical_debt_creation(self):
        """Test TechnicalDebt dataclass creation."""
        debt = code_analyzer.TechnicalDebt(
            file_path="complex.py",
            debt_score=8.5,
            complexity_metrics={
                "cyclomatic_complexity": 15,
                "cognitive_complexity": 20,
                "lines_of_code": 300
            },
            debt_indicators=["Long function", "High complexity", "Code duplication"],
            refactoring_priority="HIGH",
            estimated_effort="1-2 days"
        )
        
        assert debt.debt_score == 8.5
        assert debt.refactoring_priority == "HIGH"
        assert len(debt.debt_indicators) == 3


class TestAIContextAnalysis:
    """Test AIContextAnalysis dataclass."""
    
    def test_ai_context_analysis_creation(self):
        """Test AIContextAnalysis dataclass creation."""
        analysis = code_analyzer.AIContextAnalysis(
            file_path="large_file.py",
            token_count=5000,
            context_health="WARNING",
            estimated_split_points=[100, 200, 300],
            refactoring_suggestions=["Split into classes", "Extract functions"],
            ai_friendliness_score=6.5
        )
        
        assert analysis.token_count == 5000
        assert analysis.context_health == "WARNING"
        assert len(analysis.estimated_split_points) == 3
        assert analysis.ai_friendliness_score == 6.5


class TestCodeAnalyzerIntegration:
    """Integration tests for CodeAnalyzer."""
    
    def test_full_analysis_workflow(self, mock_project_structure):
        """Test complete analysis workflow."""
        analyzer = code_analyzer.CodeAnalyzer(str(mock_project_structure))
        
        # Run all analysis types
        import_results = analyzer.analyze_unused_imports()
        coupling_results = analyzer.analyze_coupling()
        violation_results = analyzer.detect_architecture_violations()
        debt_results = analyzer.calculate_technical_debt()
        context_results = analyzer.analyze_ai_context_windows()
        
        # All should return lists
        assert isinstance(import_results, list)
        assert isinstance(coupling_results, list)
        assert isinstance(violation_results, list)
        assert isinstance(debt_results, list)
        assert isinstance(context_results, list)


class TestCommandLineInterface:
    """Test the command line interface."""
    
    @patch('code_analyzer.CodeAnalyzer')
    def test_main_function(self, mock_analyzer_class):
        """Test main function execution."""
        mock_analyzer = MagicMock()
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.analyze_unused_imports.return_value = []
        
        with patch('sys.argv', ['code_analyzer.py', '/test/path']), \
             patch('code_analyzer.argparse.ArgumentParser.parse_args') as mock_parse:
            
            mock_args = MagicMock()
            mock_args.project_path = '/test/path'
            mock_args.fix_imports = False
            mock_args.analyze_coupling = False
            mock_parse.return_value = mock_args
            
            # Import and call main function if it exists
            if hasattr(code_analyzer, 'main'):
                code_analyzer.main()
                
                mock_analyzer_class.assert_called_once_with('/test/path')


@pytest.mark.parametrize("fix_mode", [True, False])
def test_fix_mode_parameter(mock_project_structure, fix_mode):
    """Test fix mode parameter handling."""
    analyzer = code_analyzer.CodeAnalyzer(str(mock_project_structure))
    
    with patch('builtins.open', mock_open(read_data="import os\nimport unused\n")):
        results = analyzer.analyze_unused_imports(fix_mode=fix_mode)
        
        # Should always return results regardless of fix_mode
        assert isinstance(results, list)