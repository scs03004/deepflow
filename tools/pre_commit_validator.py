#!/usr/bin/env python3
"""
Pre-commit Dependency Validator
==============================

Git hooks that validate dependencies before commits:
- Import validation - Check for broken imports before commit
- Dependency impact analysis - Warn about high-risk changes
- Auto-test triggering - Run tests based on changed dependencies
- Documentation sync - Ensure docs match code structure

Usage:
    # Install hooks
    python pre_commit_validator.py --install /path/to/project

    # Manual validation
    python pre_commit_validator.py --validate /path/to/project

    # Check specific files
    python pre_commit_validator.py --check-files file1.py file2.py
"""

import ast
import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from typing import Dict, List, Set, Optional
from dataclasses import dataclass
import importlib.util
import tempfile

try:
    import git
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import track
    import yaml
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Install with: pip install GitPython rich PyYAML")
    sys.exit(1)


@dataclass
class ValidationResult:
    """Results of dependency validation."""

    file_path: str
    issues: List[str]
    warnings: List[str]
    risk_level: str
    requires_testing: bool
    affected_components: List[str]


@dataclass
class ChangeImpact:
    """Impact analysis of file changes."""

    changed_files: List[str]
    affected_modules: Set[str]
    risk_assessment: str
    required_tests: List[str]
    documentation_updates: List[str]
    deployment_impact: bool


class DependencyValidator:
    """Core dependency validation engine."""

    def __init__(self, project_path: str):
        self.project_path = Path(project_path).resolve()
        self.console = Console()
        self.git_repo = None

        try:
            self.git_repo = git.Repo(self.project_path)
        except git.InvalidGitRepositoryError:
            self.console.print("[yellow]Warning: Not a Git repository[/yellow]")

    def validate_imports(self, files: List[str]) -> List[ValidationResult]:
        """Validate imports in the specified files."""
        results = []

        for file_path in track(files, description="Validating imports..."):
            result = self._validate_file_imports(file_path)
            results.append(result)

        return results

    def _validate_file_imports(self, file_path: str) -> ValidationResult:
        """Validate imports in a single file."""
        file_path = Path(file_path)
        issues = []
        warnings = []
        risk_level = "LOW"
        requires_testing = False
        affected_components = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse AST
            tree = ast.parse(content, filename=str(file_path))

            # Extract and validate imports
            imports = self._extract_imports(tree)

            for imp in imports:
                validation = self._validate_import(imp, file_path)
                if validation["status"] == "error":
                    issues.append(validation["message"])
                    risk_level = "HIGH"
                    requires_testing = True
                elif validation["status"] == "warning":
                    warnings.append(validation["message"])
                    if risk_level == "LOW":
                        risk_level = "MEDIUM"

            # Check for circular dependencies
            circular_deps = self._check_circular_dependencies(file_path, imports)
            if circular_deps:
                issues.extend(circular_deps)
                risk_level = "HIGH"

            # Determine affected components
            affected_components = self._get_affected_components(file_path)

        except SyntaxError as e:
            issues.append(f"Syntax error: {e}")
            risk_level = "HIGH"
            requires_testing = True
        except Exception as e:
            issues.append(f"Analysis error: {e}")
            risk_level = "MEDIUM"

        return ValidationResult(
            file_path=str(file_path),
            issues=issues,
            warnings=warnings,
            risk_level=risk_level,
            requires_testing=requires_testing,
            affected_components=affected_components,
        )

    def _extract_imports(self, tree: ast.AST) -> List[Dict]:
        """Extract all imports from AST."""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(
                        {
                            "type": "import",
                            "module": alias.name,
                            "name": alias.asname or alias.name,
                            "line": node.lineno,
                        }
                    )

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(
                        {
                            "type": "from_import",
                            "module": module,
                            "name": alias.name,
                            "alias": alias.asname,
                            "line": node.lineno,
                            "level": node.level,  # For relative imports
                        }
                    )

        return imports

    def _validate_import(self, imp: Dict, file_path: Path) -> Dict:
        """Validate a single import statement."""
        module_name = imp["module"]

        # Skip validation for relative imports and common stdlib modules
        if imp.get("level", 0) > 0:  # Relative import
            return {"status": "ok", "message": ""}

        # Check if it's a standard library module
        if self._is_stdlib_module(module_name):
            return {"status": "ok", "message": ""}

        # Check if module exists
        try:
            # Try to find the module
            spec = importlib.util.find_spec(module_name)
            if spec is None:
                # Check if it's an internal module
                if self._is_internal_module(module_name, file_path):
                    return {"status": "ok", "message": ""}
                else:
                    return {
                        "status": "error",
                        "message": f"Module '{module_name}' not found (line {imp['line']})",
                    }

            # Check if module is deprecated or has known issues
            deprecation_warning = self._check_deprecation(module_name)
            if deprecation_warning:
                return {
                    "status": "warning",
                    "message": f"Module '{module_name}' {deprecation_warning} (line {imp['line']})",
                }

        except Exception as e:
            return {
                "status": "warning",
                "message": f"Could not validate module '{module_name}': {e} (line {imp['line']})",
            }

        return {"status": "ok", "message": ""}

    def _is_stdlib_module(self, module_name: str) -> bool:
        """Check if module is part of Python standard library."""
        stdlib_modules = {
            "os",
            "sys",
            "json",
            "datetime",
            "time",
            "random",
            "math",
            "collections",
            "itertools",
            "functools",
            "pathlib",
            "re",
            "urllib",
            "http",
            "socket",
            "threading",
            "multiprocessing",
            "subprocess",
            "argparse",
            "logging",
            "unittest",
            "typing",
            "dataclasses",
            "enum",
            "abc",
            "io",
            "tempfile",
            "shutil",
            "csv",
            "sqlite3",
            "pickle",
            "base64",
            "hashlib",
            "hmac",
        }

        return module_name.split(".")[0] in stdlib_modules

    def _is_internal_module(self, module_name: str, file_path: Path) -> bool:
        """Check if module is internal to the project."""
        # Convert module name to file path
        parts = module_name.split(".")

        # Check relative to project root
        for i in range(len(parts)):
            potential_path = self.project_path / "/".join(parts[: i + 1])
            if (
                potential_path.with_suffix(".py").exists()
                or (potential_path / "__init__.py").exists()
            ):
                return True

        # Check relative to current file directory
        current_dir = file_path.parent
        for i in range(len(parts)):
            potential_path = current_dir / "/".join(parts[: i + 1])
            if (
                potential_path.with_suffix(".py").exists()
                or (potential_path / "__init__.py").exists()
            ):
                return True

        return False

    def _check_deprecation(self, module_name: str) -> Optional[str]:
        """Check if module has deprecation warnings."""
        deprecated_modules = {
            "imp": "is deprecated, use importlib instead",
            "formatter": "is deprecated",
            "optparse": "is deprecated, use argparse instead",
        }

        return deprecated_modules.get(module_name)

    def _check_circular_dependencies(self, file_path: Path, imports: List[Dict]) -> List[str]:
        """Check for potential circular dependencies."""
        issues = []

        # Get current module name
        rel_path = file_path.relative_to(self.project_path)
        current_module = str(rel_path.with_suffix("")).replace(os.sep, ".")

        for imp in imports:
            module_name = imp["module"]
            if self._is_internal_module(module_name, file_path):
                # Check if the imported module imports back to current
                if self._imports_back_to(module_name, current_module):
                    issues.append(
                        f"Potential circular dependency: {current_module} ↔ {module_name}"
                    )

        return issues

    def _imports_back_to(self, module_name: str, target_module: str) -> bool:
        """Check if module_name imports target_module (simplified check)."""
        # This is a simplified implementation
        # A full implementation would build a complete dependency graph
        try:
            module_path = self._module_to_path(module_name)
            if module_path and module_path.exists():
                with open(module_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Simple string search (could be improved with AST)
                    return target_module in content
        except Exception:
            pass

        return False

    def _module_to_path(self, module_name: str) -> Optional[Path]:
        """Convert module name to file path."""
        parts = module_name.split(".")
        potential_path = self.project_path / "/".join(parts)

        if potential_path.with_suffix(".py").exists():
            return potential_path.with_suffix(".py")
        elif (potential_path / "__init__.py").exists():
            return potential_path / "__init__.py"

        return None

    def _get_affected_components(self, file_path: Path) -> List[str]:
        """Determine which components might be affected by changes to this file."""
        components = []

        # Analyze file path to determine component type
        rel_path = file_path.relative_to(self.project_path)
        path_parts = rel_path.parts

        if "api" in path_parts or "routes" in path_parts:
            components.append("API")
        if "models" in path_parts or "database" in path_parts:
            components.append("Database")
        if "web" in path_parts or "static" in path_parts:
            components.append("Frontend")
        if "tests" in path_parts:
            components.append("Testing")
        if "config" in str(file_path).lower():
            components.append("Configuration")
        if "main.py" in str(file_path) or "app.py" in str(file_path):
            components.append("Core")

        return components

    def analyze_change_impact(self, changed_files: List[str]) -> ChangeImpact:
        """Analyze the impact of changed files."""
        affected_modules = set()
        risk_assessment = "LOW"
        required_tests = []
        documentation_updates = []
        deployment_impact = False

        for file_path in changed_files:
            path = Path(file_path)

            # Determine affected modules
            if self._is_high_impact_file(path):
                risk_assessment = "HIGH"
                deployment_impact = True
                required_tests.extend(["integration", "full"])
            elif self._is_medium_impact_file(path):
                if risk_assessment == "LOW":
                    risk_assessment = "MEDIUM"
                required_tests.extend(["unit", "component"])

            # Check for documentation updates needed
            if self._requires_doc_update(path):
                documentation_updates.append(str(path))

            # Find modules that depend on this file
            dependents = self._find_dependents(path)
            affected_modules.update(dependents)

        return ChangeImpact(
            changed_files=changed_files,
            affected_modules=affected_modules,
            risk_assessment=risk_assessment,
            required_tests=list(set(required_tests)),
            documentation_updates=documentation_updates,
            deployment_impact=deployment_impact,
        )

    def _is_high_impact_file(self, file_path: Path) -> bool:
        """Check if file is high-impact."""
        high_impact_patterns = [
            "main.py",
            "app.py",
            "config.py",
            "settings.py",
            "database.py",
            "models.py",
            "__init__.py",
        ]

        return any(pattern in str(file_path) for pattern in high_impact_patterns)

    def _is_medium_impact_file(self, file_path: Path) -> bool:
        """Check if file is medium-impact."""
        medium_impact_dirs = ["api", "routes", "models", "services", "utils"]

        return any(dir_name in file_path.parts for dir_name in medium_impact_dirs)

    def _requires_doc_update(self, file_path: Path) -> bool:
        """Check if file changes require documentation updates."""
        # Check if it's an API file or configuration file
        return any(
            keyword in str(file_path).lower() for keyword in ["api", "config", "main", "routes"]
        )

    def _find_dependents(self, file_path: Path) -> List[str]:
        """Find files that depend on the given file."""
        dependents = []

        # Get module name
        try:
            rel_path = file_path.relative_to(self.project_path)
            module_name = str(rel_path.with_suffix("")).replace(os.sep, ".")

            # Search for imports of this module
            for root, dirs, files in os.walk(self.project_path):
                # Skip non-source directories
                dirs[:] = [
                    d
                    for d in dirs
                    if d not in {".git", "__pycache__", ".pytest_cache", "node_modules"}
                ]

                for file in files:
                    if file.endswith(".py"):
                        check_file = Path(root) / file
                        if self._file_imports_module(check_file, module_name):
                            dependents.append(str(check_file.relative_to(self.project_path)))

        except Exception:
            pass

        return dependents

    def _file_imports_module(self, file_path: Path, module_name: str) -> bool:
        """Check if file imports the specified module."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                # Simple string search (could be improved)
                return module_name in content
        except Exception:
            return False


class PreCommitHookInstaller:
    """Install and manage pre-commit hooks."""

    def __init__(self, project_path: str):
        self.project_path = Path(project_path).resolve()
        self.console = Console()

    def install_hooks(self):
        """Install pre-commit hooks."""
        self.console.print("[bold blue]Installing pre-commit hooks...[/bold blue]")

        # Create hooks directory
        hooks_dir = self.project_path / ".git" / "hooks"
        hooks_dir.mkdir(exist_ok=True)

        # Install pre-commit hook
        pre_commit_hook = hooks_dir / "pre-commit"
        self._write_pre_commit_hook(pre_commit_hook)

        # Make executable
        pre_commit_hook.chmod(0o755)

        # Create configuration file
        config_file = self.project_path / ".deepflow-validator.yml"
        self._write_config_file(config_file)

        self.console.print("[green]✅ Pre-commit hooks installed successfully![/green]")
        self.console.print(f"Configuration file created: {config_file}")

    def _write_pre_commit_hook(self, hook_path: Path):
        """Write the pre-commit hook script."""
        hook_content = f'''#!/usr/bin/env python3
"""Pre-commit hook for dependency validation."""

import sys
import subprocess
from pathlib import Path

# Path to this script
SCRIPT_PATH = Path(__file__).parent.parent.parent / "deepflow" / "tools" / "pre_commit_validator.py"

def main():
    """Run pre-commit validation."""
    # Get staged files
    result = subprocess.run(['git', 'diff', '--cached', '--name-only', '--diff-filter=ACM'], 
                          capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Error getting staged files")
        return 1
    
    staged_files = [f for f in result.stdout.strip().split('\\n') if f.endswith('.py')]
    
    if not staged_files:
        return 0  # No Python files to check
    
    # Run validation
    cmd = [sys.executable, str(SCRIPT_PATH), '--check-files'] + staged_files
    result = subprocess.run(cmd)
    
    return result.returncode

if __name__ == "__main__":
    sys.exit(main())
'''

        with open(hook_path, "w") as f:
            f.write(hook_content)

    def _write_config_file(self, config_path: Path):
        """Write configuration file."""
        config = {
            "validation": {
                "check_imports": True,
                "check_circular_deps": True,
                "require_tests": True,
                "check_documentation": True,
            },
            "risk_levels": {
                "high_impact_files": ["main.py", "app.py", "config.py", "settings.py"],
                "require_full_tests": ["config.py", "main.py"],
                "skip_validation": ["migrations/"],
            },
            "testing": {
                "test_command": "pytest",
                "test_paths": ["tests/"],
                "required_coverage": 80,
            },
        }

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Pre-commit dependency validation")
    parser.add_argument(
        "--install", metavar="PROJECT_PATH", help="Install pre-commit hooks in project"
    )
    parser.add_argument(
        "--validate", metavar="PROJECT_PATH", help="Validate all Python files in project"
    )
    parser.add_argument("--check-files", nargs="+", metavar="FILE", help="Check specific files")
    parser.add_argument(
        "--impact-analysis", metavar="PROJECT_PATH", help="Analyze impact of current changes"
    )

    args = parser.parse_args()

    if args.install:
        installer = PreCommitHookInstaller(args.install)
        installer.install_hooks()
        return

    if args.validate:
        validator = DependencyValidator(args.validate)
        # Find all Python files
        python_files = []
        for root, dirs, files in os.walk(args.validate):
            dirs[:] = [d for d in dirs if d not in {".git", "__pycache__"}]
            python_files.extend([os.path.join(root, f) for f in files if f.endswith(".py")])

        results = validator.validate_imports(python_files)
        _display_results(results)
        return

    if args.check_files:
        # Use current directory as project path
        validator = DependencyValidator(".")
        results = validator.validate_imports(args.check_files)
        exit_code = _display_results(results)
        sys.exit(exit_code)

    if args.impact_analysis:
        validator = DependencyValidator(args.impact_analysis)

        # Get changed files from git
        try:
            repo = git.Repo(args.impact_analysis)
            changed_files = [item.a_path for item in repo.index.diff("HEAD")]

            if changed_files:
                impact = validator.analyze_change_impact(changed_files)
                _display_impact_analysis(impact)
            else:
                print("No changes detected")
        except Exception as e:
            print(f"Error analyzing changes: {e}")
        return

    parser.print_help()


def _display_results(results: List[ValidationResult]) -> int:
    """Display validation results."""
    console = Console()

    # Summary table
    table = Table(title="Dependency Validation Results")
    table.add_column("File")
    table.add_column("Risk Level")
    table.add_column("Issues")
    table.add_column("Warnings")
    table.add_column("Testing Required")

    exit_code = 0

    for result in results:
        risk_color = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "green"}.get(
            result.risk_level, "white"
        )

        if result.issues:
            exit_code = 1

        table.add_row(
            result.file_path,
            f"[{risk_color}]{result.risk_level}[/{risk_color}]",
            str(len(result.issues)),
            str(len(result.warnings)),
            "Yes" if result.requires_testing else "No",
        )

    console.print(table)

    # Detailed issues
    for result in results:
        if result.issues or result.warnings:
            console.print(f"\n[bold]{result.file_path}[/bold]")

            for issue in result.issues:
                console.print(f"  [red]❌ {issue}[/red]")

            for warning in result.warnings:
                console.print(f"  [yellow]⚠️  {warning}[/yellow]")

    return exit_code


def _display_impact_analysis(impact: ChangeImpact):
    """Display change impact analysis."""
    console = Console()

    # Impact summary
    console.print(
        Panel.fit(
            f"[bold]Change Impact Analysis[/bold]\n\n"
            f"Risk Level: [{'red' if impact.risk_assessment == 'HIGH' else 'yellow' if impact.risk_assessment == 'MEDIUM' else 'green'}]{impact.risk_assessment}[/]\n"
            f"Affected Modules: {len(impact.affected_modules)}\n"
            f"Required Tests: {', '.join(impact.required_tests) if impact.required_tests else 'None'}\n"
            f"Deployment Impact: {'Yes' if impact.deployment_impact else 'No'}",
            title="Impact Summary",
        )
    )

    # Changed files
    if impact.changed_files:
        console.print("\n[bold]Changed Files:[/bold]")
        for file in impact.changed_files:
            console.print(f"  • {file}")

    # Documentation updates
    if impact.documentation_updates:
        console.print("\n[bold]Documentation Updates Required:[/bold]")
        for doc in impact.documentation_updates:
            console.print(f"  • {doc}")


if __name__ == "__main__":
    main()
