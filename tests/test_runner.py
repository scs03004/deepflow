"""
Test runner for the deepflow test suite.
Provides utilities for running tests with different configurations and reporting.
"""

import pytest
import sys
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
import argparse


class DeepflowTestRunner:
    """Test runner for deepflow package tests."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize test runner."""
        if project_root is None:
            project_root = Path(__file__).parent.parent
        
        self.project_root = project_root
        self.tests_dir = project_root / "tests"
        
        # Ensure project paths are in sys.path
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        if str(project_root / "tools") not in sys.path:
            sys.path.insert(0, str(project_root / "tools"))
    
    def run_unit_tests(self, verbose: bool = False) -> int:
        """Run unit tests only."""
        args = [str(self.tests_dir / "unit")]
        if verbose:
            args.extend(["-v", "--tb=short"])
        args.extend(["-m", "unit"])
        
        return pytest.main(args)
    
    def run_integration_tests(self, verbose: bool = False) -> int:
        """Run integration tests only."""
        args = [str(self.tests_dir / "integration")]
        if verbose:
            args.extend(["-v", "--tb=short"])
        args.extend(["-m", "integration"])
        
        return pytest.main(args)
    
    def run_mcp_tests(self, verbose: bool = False) -> int:
        """Run MCP tests only."""
        args = [str(self.tests_dir / "mcp")]
        if verbose:
            args.extend(["-v", "--tb=short"])
        args.extend(["-m", "mcp"])
        
        return pytest.main(args)
    
    def run_all_tests(self, verbose: bool = False, coverage: bool = True) -> int:
        """Run all tests with optional coverage."""
        args = [str(self.tests_dir)]
        
        if verbose:
            args.extend(["-v", "--tb=short"])
        
        if coverage:
            args.extend(["--cov=tools", "--cov=deepflow", "--cov-report=term-missing"])
        
        return pytest.main(args)
    
    def run_fast_tests(self, verbose: bool = False) -> int:
        """Run fast tests only (exclude slow marked tests)."""
        args = [str(self.tests_dir), "-m", "not slow"]
        if verbose:
            args.extend(["-v", "--tb=short"])
        
        return pytest.main(args)
    
    def run_specific_test(self, test_pattern: str, verbose: bool = False) -> int:
        """Run tests matching a specific pattern."""
        args = [str(self.tests_dir), "-k", test_pattern]
        if verbose:
            args.extend(["-v", "--tb=short"])
        
        return pytest.main(args)
    
    def run_with_markers(self, markers: List[str], verbose: bool = False) -> int:
        """Run tests with specific markers."""
        marker_expr = " and ".join(markers)
        args = [str(self.tests_dir), "-m", marker_expr]
        if verbose:
            args.extend(["-v", "--tb=short"])
        
        return pytest.main(args)
    
    def run_parallel_tests(self, num_workers: int = 4, verbose: bool = False) -> int:
        """Run tests in parallel using pytest-xdist."""
        args = [str(self.tests_dir), "-n", str(num_workers)]
        if verbose:
            args.extend(["-v", "--tb=short"])
        
        return pytest.main(args)
    
    def generate_coverage_report(self, format_type: str = "html") -> int:
        """Generate coverage report in specified format."""
        args = [str(self.tests_dir), "--cov=tools", "--cov=deepflow"]
        
        if format_type == "html":
            args.append("--cov-report=html")
        elif format_type == "xml":
            args.append("--cov-report=xml")
        elif format_type == "json":
            args.append("--cov-report=json")
        else:
            args.append("--cov-report=term-missing")
        
        return pytest.main(args)
    
    def validate_test_environment(self) -> Dict[str, Any]:
        """Validate test environment and return status."""
        status = {
            "project_root_exists": self.project_root.exists(),
            "tests_dir_exists": self.tests_dir.exists(),
            "tools_importable": False,
            "deepflow_importable": False,
            "mcp_available": False,
            "test_dependencies": {}
        }
        
        # Check if tools are importable
        try:
            import dependency_visualizer
            status["tools_importable"] = True
        except ImportError:
            pass
        
        # Check if deepflow is importable
        try:
            import deepflow
            status["deepflow_importable"] = True
            status["mcp_available"] = getattr(deepflow, 'MCP_AVAILABLE', False)
        except ImportError:
            pass
        
        # Check test dependencies
        test_deps = ['pytest', 'pytest-cov', 'pytest-asyncio', 'pytest-mock']
        for dep in test_deps:
            try:
                __import__(dep.replace('-', '_'))
                status["test_dependencies"][dep] = True
            except ImportError:
                status["test_dependencies"][dep] = False
        
        return status
    
    def list_available_tests(self) -> Dict[str, List[str]]:
        """List all available tests by category."""
        tests = {
            "unit": [],
            "integration": [],
            "mcp": []
        }
        
        for category in tests.keys():
            test_dir = self.tests_dir / category
            if test_dir.exists():
                for test_file in test_dir.glob("test_*.py"):
                    tests[category].append(test_file.name)
        
        return tests


def main():
    """Main entry point for test runner CLI."""
    parser = argparse.ArgumentParser(description="Deepflow Test Runner")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage", "-c", action="store_true", help="Generate coverage report")
    parser.add_argument("--parallel", "-p", type=int, metavar="N", help="Run tests in parallel with N workers")
    parser.add_argument("--fast", action="store_true", help="Run fast tests only")
    parser.add_argument("--pattern", "-k", help="Run tests matching pattern")
    parser.add_argument("--markers", "-m", nargs="+", help="Run tests with specific markers")
    parser.add_argument("--validate", action="store_true", help="Validate test environment")
    parser.add_argument("--list", action="store_true", help="List available tests")
    
    subparsers = parser.add_subparsers(dest="command", help="Test commands")
    
    # Unit tests
    subparsers.add_parser("unit", help="Run unit tests")
    
    # Integration tests
    subparsers.add_parser("integration", help="Run integration tests")
    
    # MCP tests
    subparsers.add_parser("mcp", help="Run MCP tests")
    
    # Coverage report
    coverage_parser = subparsers.add_parser("coverage", help="Generate coverage report")
    coverage_parser.add_argument("--format", choices=["html", "xml", "json", "term"], 
                                default="html", help="Coverage report format")
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = DeepflowTestRunner()
    
    # Handle special commands
    if args.validate:
        status = runner.validate_test_environment()
        print("Test Environment Status:")
        for key, value in status.items():
            print(f"  {key}: {value}")
        return 0
    
    if args.list:
        tests = runner.list_available_tests()
        print("Available Tests:")
        for category, test_files in tests.items():
            print(f"  {category.upper()}:")
            for test_file in test_files:
                print(f"    {test_file}")
        return 0
    
    # Run tests based on command
    exit_code = 0
    
    if args.command == "unit":
        exit_code = runner.run_unit_tests(verbose=args.verbose)
    elif args.command == "integration":
        exit_code = runner.run_integration_tests(verbose=args.verbose)
    elif args.command == "mcp":
        exit_code = runner.run_mcp_tests(verbose=args.verbose)
    elif args.command == "coverage":
        exit_code = runner.generate_coverage_report(format_type=args.format)
    elif args.parallel:
        exit_code = runner.run_parallel_tests(num_workers=args.parallel, verbose=args.verbose)
    elif args.fast:
        exit_code = runner.run_fast_tests(verbose=args.verbose)
    elif args.pattern:
        exit_code = runner.run_specific_test(args.pattern, verbose=args.verbose)
    elif args.markers:
        exit_code = runner.run_with_markers(args.markers, verbose=args.verbose)
    else:
        # Run all tests
        exit_code = runner.run_all_tests(verbose=args.verbose, coverage=args.coverage)
    
    return exit_code


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)