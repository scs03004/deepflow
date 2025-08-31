#!/usr/bin/env python3
"""
Smart Code Analysis Tools
========================

Advanced code analysis for dependency optimization:
- Unused import detection and cleanup suggestions
- Coupling analysis for refactoring opportunities
- Architecture violation detection against established patterns
- Technical debt scoring based on dependency complexity

Usage:
    python code_analyzer.py /path/to/project
    python code_analyzer.py /path/to/project --fix-imports
    python code_analyzer.py /path/to/project --analyze-coupling
"""

import ast
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import re

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import track
    import networkx as nx
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Install with: pip install rich networkx pandas")
    sys.exit(1)


@dataclass
class ImportAnalysis:
    """Import usage analysis result."""

    file_path: str
    import_name: str
    import_type: str  # 'import', 'from_import'
    is_used: bool
    usage_count: int
    line_number: int
    suggestions: List[str]


@dataclass
class CouplingMetric:
    """Coupling metric between modules."""

    module_a: str
    module_b: str
    coupling_strength: float
    coupling_type: str  # 'afferent', 'efferent', 'bidirectional'
    shared_dependencies: List[str]
    refactoring_opportunity: str


@dataclass
class ArchitectureViolation:
    """Architecture pattern violation."""

    file_path: str
    violation_type: str
    severity: str  # 'HIGH', 'MEDIUM', 'LOW'
    description: str
    suggestion: str
    pattern_violated: str


@dataclass
class TechnicalDebt:
    """Technical debt assessment."""

    file_path: str
    debt_score: float
    complexity_metrics: Dict[str, float]
    debt_indicators: List[str]
    refactoring_priority: str
    estimated_effort: str


@dataclass
class AIContextAnalysis:
    """AI context window analysis result."""
    
    file_path: str
    token_count: int
    context_health: str  # 'GOOD', 'WARNING', 'CRITICAL'
    estimated_split_points: List[int]
    refactoring_suggestions: List[str]
    ai_friendliness_score: float


@dataclass
class PatternConsistency:
    """Pattern consistency analysis result."""
    
    pattern_type: str  # 'error_handling', 'logging', 'imports', 'naming'
    consistency_score: float  # 0.0 - 1.0
    total_instances: int
    consistent_instances: int
    violations: List[Dict[str, str]]  # file_path, line, description
    recommended_standard: str


@dataclass
class AICodeMetrics:
    """AI-specific code quality metrics."""
    
    file_path: str
    likely_ai_generated: bool
    confidence_score: float
    pattern_consistency_scores: Dict[str, float]
    context_window_health: str
    ai_optimization_suggestions: List[str]


# AI-specific utility functions
def estimate_tokens(content: str) -> int:
    """Estimate token count for AI context analysis."""
    # Rough estimation: ~4 characters per token for code
    return len(content) // 4


def get_context_health(token_count: int) -> str:
    """Determine AI context health based on token count."""
    if token_count < 2000:
        return "GOOD"
    elif token_count < 4000:
        return "WARNING"
    else:
        return "CRITICAL"


def detect_ai_patterns(content: str) -> Tuple[bool, float]:
    """Detect if code is likely AI-generated based on patterns."""
    ai_indicators = [
        len(re.findall(r'# .*implementation.*', content, re.IGNORECASE)),
        len(re.findall(r'# TODO:.*', content)),
        len(re.findall(r'# NOTE:.*', content)),
        len(re.findall(r'# Helper function', content, re.IGNORECASE)),
        len(re.findall(r'def .*_helper\(', content)),
        len(re.findall(r'# Main.*function', content, re.IGNORECASE)),
    ]
    
    total_indicators = sum(ai_indicators)
    lines = len(content.split('\n'))
    
    if lines == 0:
        return False, 0.0
    
    indicator_ratio = total_indicators / lines
    confidence = min(indicator_ratio * 10, 1.0)  # Scale to 0-1
    likely_ai = confidence > 0.3
    
    return likely_ai, confidence


class ImportAnalyzer:
    """Helper class for import-related analysis."""
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
    
    def extract_imports(self, tree: ast.AST) -> List[Dict]:
        """Extract detailed import information."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(self._process_import_node(node))
            elif isinstance(node, ast.ImportFrom):
                imports.extend(self._process_from_import_node(node))
        
        return imports
    
    def _process_import_node(self, node: ast.Import) -> List[Dict]:
        """Process a regular import node."""
        return [
            {
                "type": "import",
                "module": alias.name,
                "name": alias.asname or alias.name.split(".")[-1],
                "full_name": alias.name,
                "line": node.lineno,
            }
            for alias in node.names
        ]
    
    def _process_from_import_node(self, node: ast.ImportFrom) -> List[Dict]:
        """Process a from...import node."""
        module = node.module or ""
        imports = []
        
        for alias in node.names:
            if alias.name == "*":  # Skip star imports
                continue
            
            imports.append({
                "type": "from_import",
                "module": module,
                "name": alias.asname or alias.name,
                "full_name": f"{module}.{alias.name}" if module else alias.name,
                "line": node.lineno,
            })
        
        return imports
    
    def count_usage(self, content: str, imp: Dict) -> int:
        """Count how many times an import is used."""
        import_name = imp["name"]
        
        # Different patterns for different import types
        if imp["type"] == "import":
            pattern = r'\b' + re.escape(import_name) + r'\.'
        else:  # from_import
            pattern = r'\b' + re.escape(import_name) + r'\b'
        
        # Exclude the import line itself
        lines = content.splitlines()
        search_content = "\n".join(
            line for i, line in enumerate(lines, 1) if i != imp["line"]
        )
        
        return len(re.findall(pattern, search_content))


class ComplexityAnalyzer:
    """Helper class for complexity analysis."""
    
    @staticmethod
    def calculate_cyclomatic_complexity(tree: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, (ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(node, (ast.BoolOp, ast.Compare)):
                complexity += len(node.values) - 1 if hasattr(node, 'values') else 0
        
        return complexity
    
    @staticmethod
    def calculate_cognitive_complexity(tree: ast.AST) -> int:
        """Calculate cognitive complexity (simpler version)."""
        class CognitiveVisitor(ast.NodeVisitor):
            def __init__(self):
                self.cognitive = 0
                self.nesting = 0
            
            def visit_If(self, node):
                self.cognitive += 1 + self.nesting
                self.nesting += 1
                self.generic_visit(node)
                self.nesting -= 1
            
            def visit_For(self, node):
                self.cognitive += 1 + self.nesting
                self.nesting += 1
                self.generic_visit(node)
                self.nesting -= 1
            
            def visit_While(self, node):
                self.cognitive += 1 + self.nesting
                self.nesting += 1
                self.generic_visit(node)
                self.nesting -= 1
        
        visitor = CognitiveVisitor()
        visitor.visit(tree)
        return visitor.cognitive


class TechnicalDebtCalculator:
    """Helper class for calculating technical debt scores."""
    
    def __init__(self, complexity_analyzer: ComplexityAnalyzer):
        self.complexity_analyzer = complexity_analyzer
    
    def calculate_debt_score(self, tree: ast.AST, content: str, file_path: str) -> TechnicalDebt:
        """Calculate meaningful technical debt for a file."""
        complexity_metrics = self._calculate_complexity_metrics(tree, content)
        
        # Calculate base debt score based on actual problems
        base_score = (
            complexity_metrics["code_smells"] * 3.0 +  # Real problems weight heavily
            max(0, complexity_metrics["cyclomatic_complexity"] - 5) * 0.5 +  # Only penalize high complexity
            max(0, complexity_metrics["cognitive_complexity"] - 10) * 0.3  # Only penalize very high cognitive load
        )
        
        # Add specific issue scores
        indicators = self._identify_debt_indicators(tree, complexity_metrics)
        performance_penalty = len([i for i in indicators if 'performance' in i.lower() or 'sync database' in i.lower() or 'n+1' in i.lower()]) * 5
        security_penalty = len([i for i in indicators if 'security' in i.lower() or 'injection' in i.lower() or 'secret' in i.lower()]) * 10
        architecture_penalty = len([i for i in indicators if 'architecture' in i.lower() or 'concerns' in i.lower()]) * 3
        
        base_score += performance_penalty + security_penalty + architecture_penalty
        
        # Determine priority based on issue types, not score
        has_security_issues = any('security' in i.lower() or 'injection' in i.lower() or 'secret' in i.lower() for i in indicators)
        has_performance_issues = any('performance' in i.lower() or 'sync database' in i.lower() or 'n+1' in i.lower() for i in indicators)
        
        if has_security_issues:
            priority = "HIGH"
            effort = "1-2 days"
        elif has_performance_issues or base_score > 15:
            priority = "MEDIUM"
            effort = "4-8 hours"
        elif base_score > 5:
            priority = "LOW"
            effort = "1-2 hours"
        else:
            priority = "NONE"
            effort = "No action needed"
        
        return TechnicalDebt(
            file_path=file_path,
            debt_score=base_score,
            complexity_metrics=complexity_metrics,
            debt_indicators=indicators,
            refactoring_priority=priority,
            estimated_effort=effort
        )
    
    def _calculate_complexity_metrics(self, tree: ast.AST, content: str) -> Dict[str, float]:
        """Calculate various complexity metrics."""
        metrics = {
            "lines_of_code": len(content.splitlines()),
            "coupling_factor": 0,
            "code_smells": 0,
        }
        
        # Use helper class for complexity calculations
        metrics["cyclomatic_complexity"] = self.complexity_analyzer.calculate_cyclomatic_complexity(tree)
        metrics["cognitive_complexity"] = self.complexity_analyzer.calculate_cognitive_complexity(tree)
        
        # Coupling factor (number of imports)
        import_count = sum(1 for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)))
        metrics["coupling_factor"] = import_count
        
        # Code smells detection
        metrics["code_smells"] = self._count_code_smells(tree)
        
        return metrics
    
    def _detect_performance_issues(self, tree: ast.AST) -> List[str]:
        """Detect actual performance problems."""
        issues = []
        
        for node in ast.walk(tree):
            # Sync database calls in async functions
            if isinstance(node, ast.AsyncFunctionDef):
                for child in ast.walk(node):
                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                        if any(db_method in str(child.func.attr) for db_method in ['execute', 'query', 'commit']):
                            # Check if it's not awaited
                            parent = getattr(child, 'parent', None)
                            if not isinstance(parent, ast.Await):
                                issues.append("Sync database call in async function")
            
            # N+1 query patterns (loops with queries)
            elif isinstance(node, (ast.For, ast.While)):
                for child in ast.walk(node):
                    if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                        if any(query_method in str(child.func.attr) for query_method in ['query', 'filter', 'get']):
                            issues.append("Potential N+1 query pattern in loop")
        
        return issues
    
    def _detect_security_issues(self, tree: ast.AST) -> List[str]:
        """Detect security vulnerabilities."""
        issues = []
        
        for node in ast.walk(tree):
            # SQL injection via string concatenation
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)):
                if isinstance(node.left, ast.Str) and any(sql in node.left.s.upper() for sql in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']):
                    issues.append("Potential SQL injection via string concatenation")
            
            # Missing authentication on endpoints
            elif isinstance(node, ast.FunctionDef):
                decorators = [d.id if hasattr(d, 'id') else str(d) for d in node.decorator_list]
                if any(route in decorators for route in ['route', 'get', 'post', 'put', 'delete']):
                    if not any(auth in decorators for auth in ['login_required', 'requires_auth', 'authenticated']):
                        # Check if function name suggests it handles sensitive data
                        if any(sensitive in node.name.lower() for sensitive in ['admin', 'delete', 'update', 'create', 'private']):
                            issues.append(f"Endpoint '{node.name}' may need authentication")
            
            # Secrets in code
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if any(secret_word in target.id.lower() for secret_word in ['password', 'token', 'key', 'secret']):
                            if isinstance(node.value, ast.Str):
                                issues.append(f"Hardcoded secret in variable '{target.id}'")
        
        return issues
    
    def _detect_architecture_smells(self, tree: ast.AST) -> List[str]:
        """Detect architecture violations."""
        issues = []
        
        # Database models with business logic
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                
                has_db_methods = any(db_method in methods for db_method in ['save', 'delete', 'query'])
                has_business_logic = any(biz_method in methods for biz_method in ['calculate', 'validate', 'process', 'transform'])
                
                if has_db_methods and has_business_logic:
                    issues.append(f"Class '{node.name}' mixes database and business concerns")
        
        return issues
    
    def _count_function_responsibilities(self, node: ast.FunctionDef) -> int:
        """Count distinct responsibilities in a function."""
        responsibilities = set()
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                method = child.func.attr
                if any(validation in method.lower() for validation in ['validate', 'check', 'verify']):
                    responsibilities.add('validation')
                elif any(data in method.lower() for data in ['save', 'load', 'store', 'fetch']):
                    responsibilities.add('data_access')
                elif any(calc in method.lower() for calc in ['calculate', 'compute', 'process']):
                    responsibilities.add('computation')
                elif any(format_word in method.lower() for format_word in ['format', 'render', 'display']):
                    responsibilities.add('presentation')
        
        return len(responsibilities)
    
    def _analyze_class_jobs(self, node: ast.ClassDef) -> set:
        """Analyze what jobs a class is doing."""
        jobs = set()
        methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
        
        if any(data_method in methods for data_method in ['save', 'load', 'query', 'delete']):
            jobs.add('data_persistence')
        if any(validation_method in methods for validation_method in ['validate', 'check', 'verify']):
            jobs.add('validation')
        if any(calculation_method in methods for calculation_method in ['calculate', 'compute', 'process']):
            jobs.add('business_logic')
        if any(ui_method in methods for ui_method in ['render', 'display', 'format']):
            jobs.add('presentation')
        if any(network_method in methods for network_method in ['send', 'receive', 'request', 'post']):
            jobs.add('networking')
        
        return jobs
    
    def _detect_error_handling_issues(self, tree: ast.AST) -> List[str]:
        """Detect poor error handling patterns."""
        issues = []
        
        for node in ast.walk(tree):
            # Bare except clauses (catch all exceptions)
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    issues.append("Bare except clause - catching all exceptions")
                elif hasattr(node.type, 'id') and node.type.id == 'Exception':
                    issues.append("Catching generic Exception - too broad")
            
            # Empty except blocks (swallowing exceptions)
            elif isinstance(node, ast.Try):
                for handler in node.handlers:
                    if len(handler.body) == 1 and isinstance(handler.body[0], ast.Pass):
                        issues.append("Empty except block - silently swallowing exceptions")
        
        return issues
    
    def _find_circular_dependencies(self) -> List[List[str]]:
        """Find circular dependencies in the import graph."""
        try:
            cycles = list(nx.simple_cycles(self.import_graph))
            return [cycle for cycle in cycles if len(cycle) > 1]  # Ignore self-loops
        except Exception:
            return []
    
    def _find_tight_coupling(self) -> List[CouplingMetric]:
        """Find modules with excessive shared dependencies."""
        coupling_metrics = []
        
        nodes = list(self.import_graph.nodes())
        for i, module_a in enumerate(nodes):
            for module_b in nodes[i+1:]:
                deps_a = set(self.import_graph.successors(module_a))
                deps_b = set(self.import_graph.successors(module_b))
                shared_deps = deps_a.intersection(deps_b)
                
                # Only flag if there are many shared dependencies (indicates tight coupling)
                if len(shared_deps) >= 5:
                    coupling_metrics.append(
                        CouplingMetric(
                            module_a=module_a,
                            module_b=module_b,
                            coupling_strength=len(shared_deps) / 10.0,  # Normalize
                            coupling_type="tight_coupling",
                            shared_dependencies=list(shared_deps),
                            refactoring_opportunity=f"Consider extracting {len(shared_deps)} shared dependencies to common module",
                        )
                    )
        
        return coupling_metrics
    
    def _count_code_smells(self, tree: ast.AST) -> int:
        """Count meaningful code smells."""
        smells = 0
        
        for node in ast.walk(tree):
            # Functions with too many responsibilities
            if isinstance(node, ast.FunctionDef):
                responsibilities = self._count_function_responsibilities(node)
                if responsibilities > 3:
                    smells += 1
            
            # Classes doing multiple jobs (god classes)
            elif isinstance(node, ast.ClassDef):
                job_types = self._analyze_class_jobs(node)
                if len(job_types) > 2:
                    smells += 1
        
        return smells
    
    def _identify_debt_indicators(self, tree: ast.AST, metrics: Dict[str, float]) -> List[str]:
        """Identify meaningful technical debt indicators."""
        indicators = []
        
        # Function-level complexity (not file-level)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_complexity = self.complexity_analyzer.calculate_cyclomatic_complexity(node)
                if func_complexity > 10:
                    indicators.append(f"Function '{node.name}' has high complexity ({func_complexity})")
        
        # Performance problems
        performance_issues = self._detect_performance_issues(tree)
        indicators.extend(performance_issues)
        
        # Security vulnerabilities
        security_issues = self._detect_security_issues(tree)
        indicators.extend(security_issues)
        
        # Business logic scattered across layers
        architecture_issues = self._detect_architecture_smells(tree)
        indicators.extend(architecture_issues)
        
        # Error handling issues
        error_issues = self._detect_error_handling_issues(tree)
        indicators.extend(error_issues)
        
        return indicators


class CodeAnalyzer:
    """Advanced code analysis engine."""

    def __init__(self, project_path: str):
        self.project_path = Path(project_path).resolve()
        self.console = Console()
        self.import_graph = nx.DiGraph()
        self.module_metrics = {}
        
        # Initialize helper classes
        self.import_analyzer = ImportAnalyzer(self.project_path)
        self.complexity_analyzer = ComplexityAnalyzer()
        self.debt_calculator = TechnicalDebtCalculator(self.complexity_analyzer)

    def analyze_unused_imports(self, fix_mode: bool = False) -> List[ImportAnalysis]:
        """Analyze and optionally fix unused imports."""
        self.console.print("[bold blue]Analyzing unused imports...[/bold blue]")

        results = []
        python_files = list(self.project_path.rglob("*.py"))

        for file_path in track(python_files, description="Analyzing files..."):
            if self._should_skip_file(file_path):
                continue

            file_results = self._analyze_file_imports(file_path)
            results.extend(file_results)

            if fix_mode:
                self._fix_unused_imports(file_path, file_results)

        return results

    def _analyze_file_imports(self, file_path: Path) -> List[ImportAnalysis]:
        """Analyze imports in a single file."""
        results = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))

            # Extract imports using helper class
            imports = self.import_analyzer.extract_imports(tree)

            # Check usage for each import
            for imp in imports:
                usage_count = self.import_analyzer.count_usage(content, imp)
                is_used = usage_count > 0

                suggestions = []
                if not is_used:
                    suggestions.append("Remove unused import")
                elif usage_count == 1:
                    suggestions.append("Consider inline usage if simple")

                results.append(
                    ImportAnalysis(
                        file_path=str(file_path.relative_to(self.project_path)),
                        import_name=imp["name"],
                        import_type=imp["type"],
                        is_used=is_used,
                        usage_count=usage_count,
                        line_number=imp["line"],
                        suggestions=suggestions,
                    )
                )

        except Exception as e:
            self.console.print(f"[red]Error analyzing {file_path}: {e}[/red]")

        return results


    def _fix_unused_imports(self, file_path: Path, analyses: List[ImportAnalysis]):
        """Fix unused imports in a file."""
        unused_imports = [a for a in analyses if not a.is_used]

        if not unused_imports:
            return

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Remove unused import lines (in reverse order to preserve line numbers)
            lines_to_remove = sorted([a.line_number - 1 for a in unused_imports], reverse=True)

            for line_idx in lines_to_remove:
                if 0 <= line_idx < len(lines):
                    lines.pop(line_idx)

            # Write back the cleaned file
            with open(file_path, "w", encoding="utf-8") as f:
                f.writelines(lines)

            self.console.print(
                f"[green][OK] Fixed {len(unused_imports)} unused imports in {file_path.name}[/green]"
            )

        except Exception as e:
            self.console.print(f"[red]Error fixing imports in {file_path}: {e}[/red]")

    def analyze_coupling(self) -> List[CouplingMetric]:
        """Analyze meaningful coupling issues between modules."""
        self.console.print("[bold blue]Analyzing module coupling...[/bold blue]")

        # Build dependency graph
        self._build_dependency_graph()

        # Find actual coupling problems
        coupling_metrics = []

        # Check for circular dependencies (major issue)
        circular_deps = self._find_circular_dependencies()
        for cycle in circular_deps:
            for i in range(len(cycle)):
                module_a = cycle[i]
                module_b = cycle[(i + 1) % len(cycle)]
                coupling_metrics.append(
                    CouplingMetric(
                        module_a=module_a,
                        module_b=module_b,
                        coupling_strength=1.0,  # Maximum strength for circular deps
                        coupling_type="circular",
                        shared_dependencies=[],
                        refactoring_opportunity="CRITICAL: Break circular dependency - consider dependency inversion or interface extraction",
                    )
                )

        # Check for tight coupling (many shared dependencies)
        tight_coupling = self._find_tight_coupling()
        coupling_metrics.extend(tight_coupling)

        return coupling_metrics

    def _find_circular_dependencies(self) -> List[List[str]]:
        """Find circular dependencies in the import graph."""
        try:
            cycles = list(nx.simple_cycles(self.import_graph))
            return [cycle for cycle in cycles if len(cycle) > 1]  # Ignore self-loops
        except Exception:
            return []
    
    def _find_tight_coupling(self) -> List[CouplingMetric]:
        """Find modules with excessive shared dependencies."""
        coupling_metrics = []
        
        nodes = list(self.import_graph.nodes())
        for i, module_a in enumerate(nodes):
            for module_b in nodes[i+1:]:
                deps_a = set(self.import_graph.successors(module_a))
                deps_b = set(self.import_graph.successors(module_b))
                shared_deps = deps_a.intersection(deps_b)
                
                # Only flag if there are many shared dependencies (indicates tight coupling)
                if len(shared_deps) >= 3:  # Threshold for "tight coupling"
                    coupling_strength = len(shared_deps) / max(len(deps_a), len(deps_b), 1)
                    
                    if coupling_strength > 0.5:  # More than 50% shared dependencies
                        coupling_metrics.append(
                            CouplingMetric(
                                module_a=module_a,
                                module_b=module_b,
                                coupling_strength=coupling_strength,
                                coupling_type="tight",
                                shared_dependencies=list(shared_deps),
                                refactoring_opportunity=f"Consider extracting shared logic into common module or interface - {len(shared_deps)} shared dependencies"
                            )
                        )
        
        return coupling_metrics

    def _build_dependency_graph(self):
        """Build the dependency graph."""
        python_files = list(self.project_path.rglob("*.py"))

        for file_path in python_files:
            if self._should_skip_file(file_path):
                continue

            module_name = self._path_to_module_name(file_path)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content, filename=str(file_path))
                imports = self.import_analyzer.extract_imports(tree)

                # Add node
                self.import_graph.add_node(module_name, file_path=str(file_path))

                # Add edges for internal dependencies
                for imp in imports:
                    if self._is_internal_import(imp):
                        target_module = self._resolve_import_to_module(imp)
                        if target_module:
                            self.import_graph.add_edge(module_name, target_module)

            except Exception as e:
                self.console.print(f"[yellow]Warning: Could not analyze {file_path}: {e}[/yellow]")

    def _path_to_module_name(self, file_path: Path) -> str:
        """Convert file path to module name."""
        rel_path = file_path.relative_to(self.project_path)
        return str(rel_path.with_suffix("")).replace(os.sep, ".")

    def _is_internal_import(self, imp: Dict) -> bool:
        """Check if import is internal to the project."""
        module = imp.get("module", "")

        # Check for relative imports
        if module.startswith("."):
            return True

        # Check if module exists in project
        parts = module.split(".")
        for i in range(len(parts)):
            potential_path = self.project_path / "/".join(parts[: i + 1])
            if (
                potential_path.with_suffix(".py").exists()
                or (potential_path / "__init__.py").exists()
            ):
                return True

        return False

    def _resolve_import_to_module(self, imp: Dict) -> Optional[str]:
        """Resolve import to internal module name."""
        module = imp.get("module", "")

        if module.startswith("."):
            # Relative import - would need more context to resolve
            return None

        # Check if it maps to a file in our project
        parts = module.split(".")
        for i in range(len(parts)):
            potential_path = self.project_path / "/".join(parts[: i + 1])
            if potential_path.with_suffix(".py").exists():
                return str(potential_path.relative_to(self.project_path).with_suffix("")).replace(
                    os.sep, "."
                )
            elif (potential_path / "__init__.py").exists():
                return str(potential_path.relative_to(self.project_path)).replace(os.sep, ".")

        return None

    def _calculate_coupling(self, module_a: str, module_b: str) -> Dict:
        """Calculate coupling metrics between two modules."""
        # Afferent coupling (how many modules depend on this one)
        afferent_a = len(list(self.import_graph.predecessors(module_a)))
        afferent_b = len(list(self.import_graph.predecessors(module_b)))

        # Efferent coupling (how many modules this one depends on)
        efferent_a = len(list(self.import_graph.successors(module_a)))
        efferent_b = len(list(self.import_graph.successors(module_b)))

        # Direct coupling
        direct_coupling = 0
        if self.import_graph.has_edge(module_a, module_b):
            direct_coupling += 1
        if self.import_graph.has_edge(module_b, module_a):
            direct_coupling += 1

        # Shared dependencies
        deps_a = set(self.import_graph.successors(module_a))
        deps_b = set(self.import_graph.successors(module_b))
        shared_deps = list(deps_a.intersection(deps_b))

        # Calculate overall coupling strength
        coupling_strength = (
            direct_coupling * 0.5
            + len(shared_deps) * 0.2
            + min(afferent_a, afferent_b) * 0.15
            + min(efferent_a, efferent_b) * 0.15
        )

        # Determine coupling type
        coupling_type = "independent"
        if self.import_graph.has_edge(module_a, module_b) and self.import_graph.has_edge(
            module_b, module_a
        ):
            coupling_type = "bidirectional"
        elif self.import_graph.has_edge(module_a, module_b):
            coupling_type = "efferent"
        elif self.import_graph.has_edge(module_b, module_a):
            coupling_type = "afferent"

        # Generate refactoring suggestion
        refactor_suggestion = "No immediate action needed"
        if coupling_type == "bidirectional":
            refactor_suggestion = "Consider breaking circular dependency"
        elif len(shared_deps) > 3:
            refactor_suggestion = "Consider extracting shared dependencies to common module"
        elif coupling_strength > 0.7:
            refactor_suggestion = "High coupling detected - consider refactoring"

        return {
            "strength": coupling_strength,
            "type": coupling_type,
            "shared_deps": shared_deps,
            "refactor_suggestion": refactor_suggestion,
        }

    def detect_architecture_violations(self) -> List[ArchitectureViolation]:
        """Detect violations of common architecture patterns."""
        self.console.print("[bold blue]Detecting architecture violations...[/bold blue]")

        violations = []
        python_files = list(self.project_path.rglob("*.py"))

        for file_path in track(python_files, description="Checking patterns..."):
            if self._should_skip_file(file_path):
                continue

            file_violations = self._check_file_patterns(file_path)
            violations.extend(file_violations)

        return violations

    def _check_file_patterns(self, file_path: Path) -> List[ArchitectureViolation]:
        """Check architectural patterns in a file."""
        violations = []

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))
            rel_path = str(file_path.relative_to(self.project_path))

            # Check various patterns
            violations.extend(self._check_layered_architecture(tree, rel_path))
            violations.extend(self._check_dependency_inversion(tree, rel_path))
            violations.extend(self._check_single_responsibility(tree, rel_path))
            violations.extend(self._check_interface_segregation(tree, rel_path))

        except Exception as e:
            self.console.print(
                f"[yellow]Warning: Could not check patterns in {file_path}: {e}[/yellow]"
            )

        return violations

    def _check_layered_architecture(
        self, tree: ast.AST, file_path: str
    ) -> List[ArchitectureViolation]:
        """Check for layered architecture violations."""
        violations = []

        # Define typical layers
        layers = {
            "presentation": ["web", "ui", "views", "controllers"],
            "business": ["services", "business", "logic", "core"],
            "data": ["models", "database", "repositories", "dao"],
        }

        # Determine current layer
        current_layer = None
        for layer, keywords in layers.items():
            if any(keyword in file_path.lower() for keyword in keywords):
                current_layer = layer
                break

        if not current_layer:
            return violations

        # Check imports for layer violations
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_path = self._get_import_path(node)

                if import_path:
                    # Check if importing from a higher layer
                    target_layer = None
                    for layer, keywords in layers.items():
                        if any(keyword in import_path.lower() for keyword in keywords):
                            target_layer = layer
                            break

                    if target_layer and self._is_layer_violation(current_layer, target_layer):
                        violations.append(
                            ArchitectureViolation(
                                file_path=file_path,
                                violation_type="layer_violation",
                                severity="MEDIUM",
                                description=f"{current_layer} layer importing from {target_layer} layer",
                                suggestion="Consider dependency inversion or refactoring",
                                pattern_violated="Layered Architecture",
                            )
                        )

        return violations

    def _is_layer_violation(self, current_layer: str, target_layer: str) -> bool:
        """Check if importing from target layer violates layered architecture."""
        layer_order = ["data", "business", "presentation"]

        try:
            current_idx = layer_order.index(current_layer)
            target_idx = layer_order.index(target_layer)

            # Violation if importing from a higher layer
            return target_idx > current_idx
        except ValueError:
            return False

    def _check_dependency_inversion(
        self, tree: ast.AST, file_path: str
    ) -> List[ArchitectureViolation]:
        """Check for dependency inversion principle violations."""
        violations = []

        # Look for direct instantiation of concrete classes in high-level modules
        high_level_indicators = ["service", "controller", "manager", "facade"]

        if any(indicator in file_path.lower() for indicator in high_level_indicators):
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        # Direct instantiation - check if it's a concrete class
                        class_name = node.func.id
                        if self._is_likely_concrete_class(class_name):
                            violations.append(
                                ArchitectureViolation(
                                    file_path=file_path,
                                    violation_type="dependency_inversion",
                                    severity="LOW",
                                    description=f"Direct instantiation of {class_name}",
                                    suggestion="Consider dependency injection",
                                    pattern_violated="Dependency Inversion Principle",
                                )
                            )

        return violations

    def _is_likely_concrete_class(self, class_name: str) -> bool:
        """Check if name is likely a concrete class."""
        concrete_indicators = ["Service", "Repository", "Manager", "Handler", "Provider"]
        return any(indicator in class_name for indicator in concrete_indicators)

    def _check_single_responsibility(
        self, tree: ast.AST, file_path: str
    ) -> List[ArchitectureViolation]:
        """Check for single responsibility principle violations."""
        violations = []

        class_methods = defaultdict(list)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        class_methods[node.name].append(item.name)

        for class_name, methods in class_methods.items():
            # Check for classes with too many methods
            if len(methods) > 15:
                violations.append(
                    ArchitectureViolation(
                        file_path=file_path,
                        violation_type="single_responsibility",
                        severity="MEDIUM",
                        description=f"Class {class_name} has {len(methods)} methods",
                        suggestion="Consider splitting into smaller, focused classes",
                        pattern_violated="Single Responsibility Principle",
                    )
                )

            # Check for mixed responsibilities (heuristic)
            responsibility_keywords = {
                "data": ["get", "set", "load", "save", "store"],
                "validation": ["validate", "check", "verify"],
                "formatting": ["format", "render", "display"],
                "calculation": ["calculate", "compute", "process"],
            }

            detected_responsibilities = set()
            for method in methods:
                method_lower = method.lower()
                for responsibility, keywords in responsibility_keywords.items():
                    if any(keyword in method_lower for keyword in keywords):
                        detected_responsibilities.add(responsibility)

            if len(detected_responsibilities) > 2:
                violations.append(
                    ArchitectureViolation(
                        file_path=file_path,
                        violation_type="mixed_responsibilities",
                        severity="LOW",
                        description=f"Class {class_name} has mixed responsibilities: {', '.join(detected_responsibilities)}",
                        suggestion="Consider separating concerns into different classes",
                        pattern_violated="Single Responsibility Principle",
                    )
                )

        return violations

    def _check_interface_segregation(
        self, tree: ast.AST, file_path: str
    ) -> List[ArchitectureViolation]:
        """Check for interface segregation principle violations."""
        violations = []

        # Look for large interfaces (classes with many abstract methods)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                abstract_methods = []

                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        # Check for abstract method indicators
                        has_decorator = any(
                            isinstance(d, ast.Name)
                            and d.id in ["abstractmethod", "abc.abstractmethod"]
                            for d in item.decorator_list
                        )

                        if has_decorator or (
                            len(item.body) == 1 and isinstance(item.body[0], ast.Raise)
                        ):
                            abstract_methods.append(item.name)

                if len(abstract_methods) > 8:
                    violations.append(
                        ArchitectureViolation(
                            file_path=file_path,
                            violation_type="interface_segregation",
                            severity="MEDIUM",
                            description=f"Interface {node.name} has {len(abstract_methods)} abstract methods",
                            suggestion="Consider splitting into smaller, more focused interfaces",
                            pattern_violated="Interface Segregation Principle",
                        )
                    )

        return violations

    def calculate_technical_debt(self) -> List[TechnicalDebt]:
        """Calculate technical debt metrics."""
        self.console.print("[bold blue]Calculating technical debt...[/bold blue]")

        debt_results = []
        python_files = list(self.project_path.rglob("*.py"))

        for file_path in track(python_files, description="Analyzing debt..."):
            if self._should_skip_file(file_path):
                continue

            debt = self._calculate_file_debt(file_path)
            if debt:
                debt_results.append(debt)

        return debt_results

    def _calculate_file_debt(self, file_path: Path) -> Optional[TechnicalDebt]:
        """Calculate technical debt for a single file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content, filename=str(file_path))
            
            # Use helper class for debt calculation
            relative_path = str(file_path.relative_to(self.project_path))
            return self.debt_calculator.calculate_debt_score(tree, content, relative_path)

        except Exception as e:
            self.console.print(
                f"[yellow]Warning: Could not calculate debt for {file_path}: {e}[/yellow]"
            )
            return None


    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped in analysis."""
        skip_patterns = {
            "__pycache__",
            ".git",
            ".pytest_cache",
            "node_modules",
            "venv",
            ".venv",
            "env",
            ".env",
            "build",
            "dist",
        }

        return any(part in skip_patterns for part in file_path.parts) or file_path.name.startswith(
            "."
        )

    def _get_import_path(self, node: ast.AST) -> Optional[str]:
        """Get import path from AST node."""
        if isinstance(node, ast.Import):
            return node.names[0].name if node.names else None
        elif isinstance(node, ast.ImportFrom):
            return node.module
        return None

    def analyze_ai_context_windows(self) -> List[AIContextAnalysis]:
        """Analyze files for AI context window efficiency."""
        self.console.print("[bold blue]Analyzing AI context windows...[/bold blue]")
        
        results = []
        python_files = [f for f in self.project_path.rglob("*.py") if not self._should_skip_file(f)]
        
        for file_path in track(python_files, description="Analyzing context windows..."):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                token_count = estimate_tokens(content)
                context_health = get_context_health(token_count)
                
                # Analyze potential split points (class/function boundaries)
                tree = ast.parse(content)
                split_points = []
                suggestions = []
                
                for node in ast.walk(tree):
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and hasattr(node, 'lineno'):
                        split_points.append(node.lineno)
                
                # Generate suggestions based on context health
                if context_health == "CRITICAL":
                    suggestions.append("Consider splitting this file into smaller modules")
                    suggestions.append("Extract classes/functions into separate files")
                elif context_health == "WARNING":
                    suggestions.append("Monitor file size growth")
                    suggestions.append("Consider refactoring if adding more functionality")
                
                # Calculate AI-friendliness score
                lines = len(content.split('\n'))
                ai_friendliness = max(0.0, min(1.0, (4000 - token_count) / 4000))
                
                results.append(AIContextAnalysis(
                    file_path=str(file_path.relative_to(self.project_path)),
                    token_count=token_count,
                    context_health=context_health,
                    estimated_split_points=split_points[:5],  # Top 5 split points
                    refactoring_suggestions=suggestions,
                    ai_friendliness_score=ai_friendliness
                ))
                
            except Exception as e:
                self.console.print(f"[red]Error analyzing {file_path}: {e}[/red]")
        
        return results

    def analyze_pattern_consistency(self) -> List[PatternConsistency]:
        """Analyze pattern consistency across the codebase."""
        self.console.print("[bold blue]Analyzing pattern consistency...[/bold blue]")
        
        patterns = {
            'error_handling': self._analyze_error_handling_patterns(),
            'logging': self._analyze_logging_patterns(),
            'imports': self._analyze_import_patterns(),
            'naming': self._analyze_naming_patterns(),
        }
        
        results = []
        for pattern_type, analysis in patterns.items():
            results.append(analysis)
        
        return results

    def _analyze_error_handling_patterns(self) -> PatternConsistency:
        """Analyze error handling pattern consistency."""
        python_files = [f for f in self.project_path.rglob("*.py") if not self._should_skip_file(f)]
        
        try_except_patterns = []
        raise_patterns = []
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Try):
                        # Analyze exception handling patterns
                        if node.handlers:
                            handler = node.handlers[0]
                            pattern = f"except {handler.type.id if handler.type and hasattr(handler.type, 'id') else 'Exception'}"
                            try_except_patterns.append({
                                'file': str(file_path.relative_to(self.project_path)),
                                'line': node.lineno,
                                'pattern': pattern
                            })
                    
                    elif isinstance(node, ast.Raise):
                        if node.exc and hasattr(node.exc, 'id'):
                            pattern = f"raise {node.exc.id}"
                            raise_patterns.append({
                                'file': str(file_path.relative_to(self.project_path)),
                                'line': node.lineno,
                                'pattern': pattern
                            })
            except:
                continue
        
        # Calculate consistency
        all_patterns = try_except_patterns + raise_patterns
        if not all_patterns:
            return PatternConsistency(
                pattern_type='error_handling',
                consistency_score=1.0,
                total_instances=0,
                consistent_instances=0,
                violations=[],
                recommended_standard="No error handling patterns found"
            )
        
        pattern_counts = Counter(p['pattern'] for p in all_patterns)
        most_common = pattern_counts.most_common(1)[0] if pattern_counts else ('', 0)
        consistent_count = most_common[1]
        consistency_score = consistent_count / len(all_patterns) if all_patterns else 0.0
        
        violations = [p for p in all_patterns if p['pattern'] != most_common[0]]
        
        return PatternConsistency(
            pattern_type='error_handling',
            consistency_score=consistency_score,
            total_instances=len(all_patterns),
            consistent_instances=consistent_count,
            violations=violations,
            recommended_standard=most_common[0]
        )

    def _analyze_logging_patterns(self) -> PatternConsistency:
        """Analyze logging pattern consistency."""
        python_files = [f for f in self.project_path.rglob("*.py") if not self._should_skip_file(f)]
        
        logging_patterns = []
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Look for logging patterns
                log_calls = re.findall(r'(log\w*\.(debug|info|warning|error|critical))', content, re.IGNORECASE)
                print_calls = re.findall(r'print\s*\(', content)
                
                for match in log_calls:
                    logging_patterns.append({
                        'file': str(file_path.relative_to(self.project_path)),
                        'pattern': f"logging.{match[1]}"
                    })
                
                for _ in print_calls:
                    logging_patterns.append({
                        'file': str(file_path.relative_to(self.project_path)),
                        'pattern': "print()"
                    })
            except:
                continue
        
        if not logging_patterns:
            return PatternConsistency(
                pattern_type='logging',
                consistency_score=1.0,
                total_instances=0,
                consistent_instances=0,
                violations=[],
                recommended_standard="No logging patterns found"
            )
        
        pattern_counts = Counter(p['pattern'] for p in logging_patterns)
        most_common = pattern_counts.most_common(1)[0]
        consistent_count = most_common[1]
        consistency_score = consistent_count / len(logging_patterns)
        
        violations = [p for p in logging_patterns if p['pattern'] != most_common[0]]
        
        return PatternConsistency(
            pattern_type='logging',
            consistency_score=consistency_score,
            total_instances=len(logging_patterns),
            consistent_instances=consistent_count,
            violations=violations,
            recommended_standard=most_common[0]
        )

    def _analyze_import_patterns(self) -> PatternConsistency:
        """Analyze import organization patterns."""
        python_files = [f for f in self.project_path.rglob("*.py") if not self._should_skip_file(f)]
        
        import_styles = []
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                # Check import organization
                imports_section = []
                for i, line in enumerate(lines[:50]):  # Check first 50 lines
                    if line.strip().startswith(('import ', 'from ')):
                        imports_section.append((i, line.strip()))
                
                if imports_section:
                    # Determine style based on organization
                    stdlib_first = any('from typing' in line or 'import os' in line for _, line in imports_section[:3])
                    style = "stdlib_first" if stdlib_first else "mixed"
                    
                    import_styles.append({
                        'file': str(file_path.relative_to(self.project_path)),
                        'pattern': style
                    })
            except:
                continue
        
        if not import_styles:
            return PatternConsistency(
                pattern_type='imports',
                consistency_score=1.0,
                total_instances=0,
                consistent_instances=0,
                violations=[],
                recommended_standard="No import patterns found"
            )
        
        pattern_counts = Counter(p['pattern'] for p in import_styles)
        most_common = pattern_counts.most_common(1)[0]
        consistent_count = most_common[1]
        consistency_score = consistent_count / len(import_styles)
        
        violations = [p for p in import_styles if p['pattern'] != most_common[0]]
        
        return PatternConsistency(
            pattern_type='imports',
            consistency_score=consistency_score,
            total_instances=len(import_styles),
            consistent_instances=consistent_count,
            violations=violations,
            recommended_standard=most_common[0]
        )

    def _analyze_naming_patterns(self) -> PatternConsistency:
        """Analyze naming convention consistency."""
        python_files = [f for f in self.project_path.rglob("*.py") if not self._should_skip_file(f)]
        
        naming_patterns = []
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if '_' in node.name:
                            pattern = "snake_case"
                        elif node.name[0].isupper():
                            pattern = "PascalCase"
                        elif node.name[0].islower() and not '_' in node.name:
                            pattern = "camelCase"
                        else:
                            pattern = "mixed"
                        
                        naming_patterns.append({
                            'file': str(file_path.relative_to(self.project_path)),
                            'line': node.lineno,
                            'pattern': pattern,
                            'name': node.name
                        })
            except:
                continue
        
        if not naming_patterns:
            return PatternConsistency(
                pattern_type='naming',
                consistency_score=1.0,
                total_instances=0,
                consistent_instances=0,
                violations=[],
                recommended_standard="No naming patterns found"
            )
        
        pattern_counts = Counter(p['pattern'] for p in naming_patterns)
        most_common = pattern_counts.most_common(1)[0]
        consistent_count = most_common[1]
        consistency_score = consistent_count / len(naming_patterns)
        
        violations = [p for p in naming_patterns if p['pattern'] != most_common[0]]
        
        return PatternConsistency(
            pattern_type='naming',
            consistency_score=consistency_score,
            total_instances=len(naming_patterns),
            consistent_instances=consistent_count,
            violations=violations,
            recommended_standard=most_common[0]
        )

    def analyze_ai_code_metrics(self) -> List[AICodeMetrics]:
        """Analyze AI-specific code quality metrics."""
        self.console.print("[bold blue]Analyzing AI code metrics...[/bold blue]")
        
        results = []
        python_files = [f for f in self.project_path.rglob("*.py") if not self._should_skip_file(f)]
        
        for file_path in track(python_files, description="Analyzing AI metrics..."):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Detect if likely AI-generated
                likely_ai, confidence = detect_ai_patterns(content)
                
                # Get context window health
                token_count = estimate_tokens(content)
                context_health = get_context_health(token_count)
                
                # Analyze pattern consistency (simplified for individual file)
                pattern_scores = {
                    'error_handling': 0.8,  # Would implement actual analysis
                    'logging': 0.9,
                    'naming': 0.85,
                    'imports': 0.75
                }
                
                # Generate AI optimization suggestions
                suggestions = []
                if context_health == "CRITICAL":
                    suggestions.append("File too large for optimal AI processing - consider splitting")
                if likely_ai and confidence > 0.5:
                    suggestions.append("Review AI-generated code for consistency with project patterns")
                if token_count > 3000:
                    suggestions.append("Consider extracting classes or functions to improve AI context efficiency")
                
                results.append(AICodeMetrics(
                    file_path=str(file_path.relative_to(self.project_path)),
                    likely_ai_generated=likely_ai,
                    confidence_score=confidence,
                    pattern_consistency_scores=pattern_scores,
                    context_window_health=context_health,
                    ai_optimization_suggestions=suggestions
                ))
                
            except Exception as e:
                self.console.print(f"[red]Error analyzing {file_path}: {e}[/red]")
        
        return results


class CLIHandler:
    """Handles command-line interface and result processing."""
    
    def __init__(self):
        self.console = Console()
    
    def create_parser(self):
        """Create and configure argument parser."""
        parser = argparse.ArgumentParser(description="Smart code analysis for dependency optimization")
        parser.add_argument("project_path", help="Path to the project to analyze")
        parser.add_argument(
            "--fix-imports", action="store_true", help="Automatically fix unused imports"
        )
        parser.add_argument("--analyze-coupling", action="store_true", help="Analyze module coupling")
        parser.add_argument(
            "--check-architecture", action="store_true", help="Check for architecture violations"
        )
        parser.add_argument("--calculate-debt", action="store_true", help="Calculate technical debt")
        parser.add_argument("--output", help="Output file for results (JSON format)")
        parser.add_argument("--all", action="store_true", help="Run all analyses")
        
        # AI-specific analysis options
        parser.add_argument("--ai-metrics", action="store_true", help="Enable AI-specific code quality metrics")
        parser.add_argument("--pattern-consistency", action="store_true", help="Analyze pattern consistency across codebase")
        parser.add_argument("--context-analysis", action="store_true", help="Analyze files for AI context window efficiency")
        
        return parser
    
    def run_analysis(self, args):
        """Run the complete analysis based on arguments."""
        analyzer = CodeAnalyzer(args.project_path)
        results = {}
        
        # Run traditional analyses
        if (
            args.all
            or args.fix_imports
            or not any([args.analyze_coupling, args.check_architecture, args.calculate_debt])
        ):
            import_results = self._run_import_analysis(analyzer, args)
            results.update(import_results)
        
        if args.all or args.analyze_coupling:
            coupling_results = self._run_coupling_analysis(analyzer)
            results.update(coupling_results)
        
        if args.all or args.check_architecture:
            arch_results = self._run_architecture_analysis(analyzer)
            results.update(arch_results)
        
        if args.all or args.calculate_debt:
            debt_results = self._run_debt_analysis(analyzer)
            results.update(debt_results)
        
        # Run AI-specific analyses  
        ai_issues = 0
        if args.all or args.ai_metrics or args.pattern_consistency or args.context_analysis:
            ai_results, ai_issues = self._run_ai_analyses(analyzer, args)
            results.update(ai_results)
        
        # Save results and display summary
        self._save_results(results, args.output)
        self._display_summary(results, ai_issues)
        
        return results
    
    def _run_import_analysis(self, analyzer, args):
        """Run import analysis and return results."""
        import_results = analyzer.analyze_unused_imports(fix_mode=args.fix_imports)
        results = {"imports": [asdict(r) for r in import_results]}
        
        # Display results
        unused_count = sum(1 for r in import_results if not r.is_used)
        self.console.print(
            f"\n[bold]Import Analysis:[/bold] Found {unused_count} unused imports out of {len(import_results)} total"
        )
        
        if unused_count > 0:
            self._display_import_table(import_results)
        
        return results
    
    def _display_import_table(self, import_results):
        """Display unused imports in a table."""
        table = Table(title="Unused Imports")
        table.add_column("File")
        table.add_column("Import")
        table.add_column("Line")
        
        for result in import_results:
            if not result.is_used:
                table.add_row(
                    str(Path(result.file_path).name),
                    result.import_name,
                    str(result.line_number)
                )
        
        self.console.print(table)
    
    def _run_coupling_analysis(self, analyzer):
        """Run coupling analysis and return results."""
        coupling_results = analyzer.analyze_coupling()
        results = {"coupling": [asdict(c) for c in coupling_results]}
        
        # Display results  
        self.console.print(f"\n[bold]Coupling Analysis:[/bold] Found {len(coupling_results)} coupling relationships")
        
        if coupling_results:
            table = Table(title="Module Coupling")
            table.add_column("Module A")
            table.add_column("Module B")
            table.add_column("Strength")
            table.add_column("Type")
            
            for coupling in coupling_results[:10]:  # Show top 10
                table.add_row(
                    coupling.module_a,
                    coupling.module_b,
                    f"{coupling.coupling_strength:.2f}",
                    coupling.coupling_type
                )
            
            self.console.print(table)
        
        return results
    
    def _run_architecture_analysis(self, analyzer):
        """Run architecture analysis and return results."""
        arch_results = analyzer.detect_architecture_violations()
        results = {"architecture": [asdict(v) for v in arch_results]}
        
        # Display results
        high_severity = [v for v in arch_results if v.severity == "HIGH"]
        self.console.print(f"\n[bold]Architecture Analysis:[/bold] Found {len(arch_results)} violations ({len(high_severity)} high priority)")
        
        if arch_results:
            table = Table(title="Architecture Violations")
            table.add_column("File")
            table.add_column("Type")
            table.add_column("Severity")
            table.add_column("Description")
            
            for violation in arch_results[:10]:  # Show top 10
                table.add_row(
                    str(Path(violation.file_path).name),
                    violation.violation_type,
                    violation.severity,
                    violation.description[:60] + "..." if len(violation.description) > 60 else violation.description
                )
            
            self.console.print(table)
        
        return results
    
    def _run_debt_analysis(self, analyzer):
        """Run technical debt analysis and return results."""
        debt_results = analyzer.calculate_technical_debt()
        results = {"debt": [asdict(d) for d in debt_results]}
        
        # Display results
        self.console.print(f"\n[bold]Technical Debt Analysis:[/bold] Analyzed {len(debt_results)} files")
        
        if debt_results:
            # Sort by debt score
            debt_results.sort(key=lambda x: x.debt_score, reverse=True)
            
            table = Table(title="Technical Debt")
            table.add_column("File")
            table.add_column("Score")
            table.add_column("Priority")
            table.add_column("Effort")
            
            for debt in debt_results[:15]:  # Show top 15
                table.add_row(
                    str(Path(debt.file_path).name),
                    f"{debt.debt_score:.1f}",
                    debt.refactoring_priority,
                    debt.estimated_effort
                )
            
            self.console.print(table)
        
        return results
    
    def _run_ai_analyses(self, analyzer, args):
        """Run AI-specific analyses and return results."""
        results = {}
        ai_issues = 0
        
        if args.pattern_consistency or args.all:
            pattern_results, pattern_issues = self._run_pattern_analysis(analyzer)
            results["patterns"] = pattern_results
            ai_issues += pattern_issues
        
        if args.context_analysis or args.all:
            context_results, context_issues = self._run_context_analysis(analyzer)
            results["ai_context"] = context_results
            ai_issues += context_issues
        
        if args.ai_metrics or args.all:
            metrics_results, metrics_issues = self._run_ai_metrics_analysis(analyzer)
            results["ai_metrics"] = metrics_results
            ai_issues += metrics_issues
        
        return results, ai_issues
    
    def _run_pattern_analysis(self, analyzer):
        """Run pattern consistency analysis."""
        pattern_results = analyzer.analyze_pattern_consistency()
        results = [asdict(p) for p in pattern_results]
        
        self.console.print(f"\n[bold]Pattern Analysis:[/bold] Analyzed {len(pattern_results)} patterns")
        
        inconsistent_patterns = [p for p in pattern_results if p.consistency_score < 0.8]
        if inconsistent_patterns:
            self._display_pattern_table(inconsistent_patterns)
        
        return results, len(inconsistent_patterns)
    
    def _display_pattern_table(self, inconsistent_patterns):
        """Display pattern inconsistencies table."""
        table = Table(title="Pattern Inconsistencies")
        table.add_column("Pattern Type")
        table.add_column("Consistency Score")
        table.add_column("Files Affected")
        table.add_column("Primary Variant")
        
        for pattern in inconsistent_patterns[:10]:  # Show top 10
            table.add_row(
                pattern.pattern_type,
                f"{pattern.consistency_score:.2f}",
                str(len(pattern.file_examples)),
                pattern.primary_variant[:50] + "..." if len(pattern.primary_variant) > 50 else pattern.primary_variant
            )
        
        self.console.print(table)
    
    def _run_context_analysis(self, analyzer):
        """Run AI context window analysis."""
        context_results = analyzer.analyze_ai_context_windows()
        results = [asdict(c) for c in context_results]
        
        self.console.print(f"\n[bold]AI Context Analysis:[/bold] Analyzed {len(context_results)} files")
        
        problematic_files = [f for f in context_results if f.context_health in ["WARNING", "CRITICAL"]]
        if problematic_files:
            self._display_context_table(problematic_files)
        
        return results, len(problematic_files)
    
    def _display_context_table(self, problematic_files):
        """Display context window issues table."""
        table = Table(title="Context Window Issues")
        table.add_column("File")
        table.add_column("Token Count")
        table.add_column("Health")
        table.add_column("AI Friendliness")
        
        for file_analysis in problematic_files[:10]:  # Show top 10
            table.add_row(
                str(Path(file_analysis.file_path).name),
                f"{file_analysis.token_count:,}",
                file_analysis.context_health,
                f"{file_analysis.ai_friendliness_score:.2f}"
            )
        
        self.console.print(table)
    
    def _run_ai_metrics_analysis(self, analyzer):
        """Run AI code metrics analysis."""
        ai_metrics = analyzer.analyze_ai_code_metrics()
        results = [asdict(m) for m in ai_metrics]
        
        self.console.print(f"\n[bold]AI Code Metrics:[/bold] Analyzed {len(ai_metrics)} files")
        
        low_readability = [m for m in ai_metrics if m.ai_readability_score < 0.7]
        if low_readability:
            self._display_ai_metrics_table(low_readability)
        
        return results, len(low_readability)
    
    def _display_ai_metrics_table(self, low_readability):
        """Display AI readability issues table."""
        table = Table(title="AI Readability Issues")
        table.add_column("File")
        table.add_column("Readability")
        table.add_column("Maintainability")
        table.add_column("Context Efficiency")
        
        for metrics in low_readability[:10]:  # Show top 10
            table.add_row(
                str(Path(metrics.file_path).name),
                f"{metrics.ai_readability_score:.2f}",
                f"{metrics.ai_maintainability_score:.2f}",
                f"{metrics.context_efficiency_score:.2f}"
            )
        
        self.console.print(table)
    
    def _save_results(self, results, output_file):
        """Save results to JSON file."""
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            self.console.print(f"\n[green]Results saved to {output_file}[/green]")
    
    def _display_summary(self, results, ai_issues=0):
        """Display analysis summary."""
        total_issues = 0
        traditional_issues = 0
        
        # Count issues from different analyses
        if "imports" in results:
            unused_imports = sum(1 for r in results["imports"] if not r["is_used"])
            total_issues += unused_imports
            traditional_issues += unused_imports
        
        if "coupling" in results:
            high_coupling = len([c for c in results["coupling"] if c["coupling_strength"] > 0.7])
            total_issues += high_coupling
            traditional_issues += high_coupling
        
        if "architecture" in results:
            arch_violations = len([v for v in results["architecture"] if v["severity"] in ["HIGH", "MEDIUM"]])
            total_issues += arch_violations
            traditional_issues += arch_violations
        
        if "debt" in results:
            high_debt = len([d for d in results["debt"] if d["refactoring_priority"] in ["HIGH", "MEDIUM"]])
            total_issues += high_debt
            traditional_issues += high_debt
        
        total_issues += ai_issues
        
        self.console.print(f"\n[bold]Summary:[/bold] Found {total_issues} total issues requiring attention")
        if ai_issues > 0:
            self.console.print(f"  Traditional issues: {traditional_issues}")
            self.console.print(f"  AI-specific issues: {ai_issues}")
            self.console.print(f"\n[blue]Tip:[/blue] Use --ai-metrics to focus on AI development hygiene")


def main():
    """Main CLI interface."""
    cli_handler = CLIHandler()
    parser = cli_handler.create_parser()
    args = parser.parse_args()
    
    try:
        cli_handler.run_analysis(args)
    except Exception as e:
        cli_handler.console.print(f"[red]Error during analysis: {e}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()
