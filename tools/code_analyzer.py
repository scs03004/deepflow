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
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import importlib.util
import re

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import track
    from rich.tree import Tree
    import networkx as nx
    import pandas as pd
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


class CodeAnalyzer:
    """Advanced code analysis engine."""
    
    def __init__(self, project_path: str):
        self.project_path = Path(project_path).resolve()
        self.console = Console()
        self.import_graph = nx.DiGraph()
        self.module_metrics = {}
        
    def analyze_unused_imports(self, fix_mode: bool = False) -> List[ImportAnalysis]:
        """Analyze and optionally fix unused imports."""
        self.console.print("[bold blue]Analyzing unused imports...[/bold blue]")
        
        results = []
        python_files = list(self.project_path.rglob('*.py'))
        
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
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(file_path))
            
            # Extract imports
            imports = self._extract_detailed_imports(tree)
            
            # Check usage for each import
            for imp in imports:
                usage_count = self._count_import_usage(content, imp)
                is_used = usage_count > 0
                
                suggestions = []
                if not is_used:
                    suggestions.append("Remove unused import")
                elif usage_count == 1:
                    suggestions.append("Consider inline usage if simple")
                
                results.append(ImportAnalysis(
                    file_path=str(file_path.relative_to(self.project_path)),
                    import_name=imp['name'],
                    import_type=imp['type'],
                    is_used=is_used,
                    usage_count=usage_count,
                    line_number=imp['line'],
                    suggestions=suggestions
                ))
        
        except Exception as e:
            self.console.print(f"[red]Error analyzing {file_path}: {e}[/red]")
        
        return results
    
    def _extract_detailed_imports(self, tree: ast.AST) -> List[Dict]:
        """Extract detailed import information."""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'type': 'import',
                        'module': alias.name,
                        'name': alias.asname or alias.name.split('.')[-1],
                        'full_name': alias.name,
                        'line': node.lineno
                    })
            
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    if alias.name == '*':
                        # Skip star imports for now
                        continue
                    
                    imports.append({
                        'type': 'from_import',
                        'module': module,
                        'name': alias.asname or alias.name,
                        'full_name': f"{module}.{alias.name}" if module else alias.name,
                        'line': node.lineno
                    })
        
        return imports
    
    def _count_import_usage(self, content: str, imp: Dict) -> int:
        """Count how many times an import is used."""
        import_name = imp['name']
        
        # Remove the import line itself
        lines = content.split('\n')
        code_without_imports = '\n'.join(
            line for i, line in enumerate(lines, 1)
            if i != imp['line']
        )
        
        # Count occurrences
        # This is a simplified approach - a full implementation would use AST
        pattern = r'\b' + re.escape(import_name) + r'\b'
        matches = re.findall(pattern, code_without_imports)
        
        # Filter out matches in comments and strings
        filtered_count = 0
        for line in code_without_imports.split('\n'):
            # Skip comments
            if line.strip().startswith('#'):
                continue
            
            # Count matches in this line
            line_matches = len(re.findall(pattern, line))
            filtered_count += line_matches
        
        return filtered_count
    
    def _fix_unused_imports(self, file_path: Path, analyses: List[ImportAnalysis]):
        """Fix unused imports in a file."""
        unused_imports = [a for a in analyses if not a.is_used]
        
        if not unused_imports:
            return
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Remove unused import lines (in reverse order to preserve line numbers)
            lines_to_remove = sorted([a.line_number - 1 for a in unused_imports], reverse=True)
            
            for line_idx in lines_to_remove:
                if 0 <= line_idx < len(lines):
                    lines.pop(line_idx)
            
            # Write back the cleaned file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            
            self.console.print(f"[green]✅ Fixed {len(unused_imports)} unused imports in {file_path.name}[/green]")
        
        except Exception as e:
            self.console.print(f"[red]Error fixing imports in {file_path}: {e}[/red]")
    
    def analyze_coupling(self) -> List[CouplingMetric]:
        """Analyze coupling between modules."""
        self.console.print("[bold blue]Analyzing module coupling...[/bold blue]")
        
        # Build dependency graph
        self._build_dependency_graph()
        
        # Calculate coupling metrics
        coupling_metrics = []
        
        for module_a in self.import_graph.nodes():
            for module_b in self.import_graph.nodes():
                if module_a != module_b:
                    coupling = self._calculate_coupling(module_a, module_b)
                    if coupling['strength'] > 0.1:  # Only significant couplings
                        coupling_metrics.append(CouplingMetric(
                            module_a=module_a,
                            module_b=module_b,
                            coupling_strength=coupling['strength'],
                            coupling_type=coupling['type'],
                            shared_dependencies=coupling['shared_deps'],
                            refactoring_opportunity=coupling['refactor_suggestion']
                        ))
        
        return coupling_metrics
    
    def _build_dependency_graph(self):
        """Build the dependency graph."""
        python_files = list(self.project_path.rglob('*.py'))
        
        for file_path in python_files:
            if self._should_skip_file(file_path):
                continue
            
            module_name = self._path_to_module_name(file_path)
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content, filename=str(file_path))
                imports = self._extract_detailed_imports(tree)
                
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
        return str(rel_path.with_suffix('')).replace(os.sep, '.')
    
    def _is_internal_import(self, imp: Dict) -> bool:
        """Check if import is internal to the project."""
        module = imp.get('module', '')
        
        # Check for relative imports
        if module.startswith('.'):
            return True
        
        # Check if module exists in project
        parts = module.split('.')
        for i in range(len(parts)):
            potential_path = self.project_path / '/'.join(parts[:i+1])
            if potential_path.with_suffix('.py').exists() or \
               (potential_path / '__init__.py').exists():
                return True
        
        return False
    
    def _resolve_import_to_module(self, imp: Dict) -> Optional[str]:
        """Resolve import to internal module name."""
        module = imp.get('module', '')
        
        if module.startswith('.'):
            # Relative import - would need more context to resolve
            return None
        
        # Check if it maps to a file in our project
        parts = module.split('.')
        for i in range(len(parts)):
            potential_path = self.project_path / '/'.join(parts[:i+1])
            if potential_path.with_suffix('.py').exists():
                return str(potential_path.relative_to(self.project_path).with_suffix('')).replace(os.sep, '.')
            elif (potential_path / '__init__.py').exists():
                return str(potential_path.relative_to(self.project_path)).replace(os.sep, '.')
        
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
            direct_coupling * 0.5 +
            len(shared_deps) * 0.2 +
            min(afferent_a, afferent_b) * 0.15 +
            min(efferent_a, efferent_b) * 0.15
        )
        
        # Determine coupling type
        coupling_type = 'independent'
        if self.import_graph.has_edge(module_a, module_b) and self.import_graph.has_edge(module_b, module_a):
            coupling_type = 'bidirectional'
        elif self.import_graph.has_edge(module_a, module_b):
            coupling_type = 'efferent'
        elif self.import_graph.has_edge(module_b, module_a):
            coupling_type = 'afferent'
        
        # Generate refactoring suggestion
        refactor_suggestion = "No immediate action needed"
        if coupling_type == 'bidirectional':
            refactor_suggestion = "Consider breaking circular dependency"
        elif len(shared_deps) > 3:
            refactor_suggestion = "Consider extracting shared dependencies to common module"
        elif coupling_strength > 0.7:
            refactor_suggestion = "High coupling detected - consider refactoring"
        
        return {
            'strength': coupling_strength,
            'type': coupling_type,
            'shared_deps': shared_deps,
            'refactor_suggestion': refactor_suggestion
        }
    
    def detect_architecture_violations(self) -> List[ArchitectureViolation]:
        """Detect violations of common architecture patterns."""
        self.console.print("[bold blue]Detecting architecture violations...[/bold blue]")
        
        violations = []
        python_files = list(self.project_path.rglob('*.py'))
        
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
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(file_path))
            rel_path = str(file_path.relative_to(self.project_path))
            
            # Check various patterns
            violations.extend(self._check_layered_architecture(tree, rel_path))
            violations.extend(self._check_dependency_inversion(tree, rel_path))
            violations.extend(self._check_single_responsibility(tree, rel_path))
            violations.extend(self._check_interface_segregation(tree, rel_path))
        
        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not check patterns in {file_path}: {e}[/yellow]")
        
        return violations
    
    def _check_layered_architecture(self, tree: ast.AST, file_path: str) -> List[ArchitectureViolation]:
        """Check for layered architecture violations."""
        violations = []
        
        # Define typical layers
        layers = {
            'presentation': ['web', 'ui', 'views', 'controllers'],
            'business': ['services', 'business', 'logic', 'core'],
            'data': ['models', 'database', 'repositories', 'dao']
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
                        violations.append(ArchitectureViolation(
                            file_path=file_path,
                            violation_type='layer_violation',
                            severity='MEDIUM',
                            description=f"{current_layer} layer importing from {target_layer} layer",
                            suggestion="Consider dependency inversion or refactoring",
                            pattern_violated='Layered Architecture'
                        ))
        
        return violations
    
    def _is_layer_violation(self, current_layer: str, target_layer: str) -> bool:
        """Check if importing from target layer violates layered architecture."""
        layer_order = ['data', 'business', 'presentation']
        
        try:
            current_idx = layer_order.index(current_layer)
            target_idx = layer_order.index(target_layer)
            
            # Violation if importing from a higher layer
            return target_idx > current_idx
        except ValueError:
            return False
    
    def _check_dependency_inversion(self, tree: ast.AST, file_path: str) -> List[ArchitectureViolation]:
        """Check for dependency inversion principle violations."""
        violations = []
        
        # Look for direct instantiation of concrete classes in high-level modules
        high_level_indicators = ['service', 'controller', 'manager', 'facade']
        
        if any(indicator in file_path.lower() for indicator in high_level_indicators):
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        # Direct instantiation - check if it's a concrete class
                        class_name = node.func.id
                        if self._is_likely_concrete_class(class_name):
                            violations.append(ArchitectureViolation(
                                file_path=file_path,
                                violation_type='dependency_inversion',
                                severity='LOW',
                                description=f"Direct instantiation of {class_name}",
                                suggestion="Consider dependency injection",
                                pattern_violated='Dependency Inversion Principle'
                            ))
        
        return violations
    
    def _is_likely_concrete_class(self, class_name: str) -> bool:
        """Check if name is likely a concrete class."""
        concrete_indicators = ['Service', 'Repository', 'Manager', 'Handler', 'Provider']
        return any(indicator in class_name for indicator in concrete_indicators)
    
    def _check_single_responsibility(self, tree: ast.AST, file_path: str) -> List[ArchitectureViolation]:
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
                violations.append(ArchitectureViolation(
                    file_path=file_path,
                    violation_type='single_responsibility',
                    severity='MEDIUM',
                    description=f"Class {class_name} has {len(methods)} methods",
                    suggestion="Consider splitting into smaller, focused classes",
                    pattern_violated='Single Responsibility Principle'
                ))
            
            # Check for mixed responsibilities (heuristic)
            responsibility_keywords = {
                'data': ['get', 'set', 'load', 'save', 'store'],
                'validation': ['validate', 'check', 'verify'],
                'formatting': ['format', 'render', 'display'],
                'calculation': ['calculate', 'compute', 'process']
            }
            
            detected_responsibilities = set()
            for method in methods:
                method_lower = method.lower()
                for responsibility, keywords in responsibility_keywords.items():
                    if any(keyword in method_lower for keyword in keywords):
                        detected_responsibilities.add(responsibility)
            
            if len(detected_responsibilities) > 2:
                violations.append(ArchitectureViolation(
                    file_path=file_path,
                    violation_type='mixed_responsibilities',
                    severity='LOW',
                    description=f"Class {class_name} has mixed responsibilities: {', '.join(detected_responsibilities)}",
                    suggestion="Consider separating concerns into different classes",
                    pattern_violated='Single Responsibility Principle'
                ))
        
        return violations
    
    def _check_interface_segregation(self, tree: ast.AST, file_path: str) -> List[ArchitectureViolation]:
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
                            isinstance(d, ast.Name) and d.id in ['abstractmethod', 'abc.abstractmethod']
                            for d in item.decorator_list
                        )
                        
                        if has_decorator or (len(item.body) == 1 and isinstance(item.body[0], ast.Raise)):
                            abstract_methods.append(item.name)
                
                if len(abstract_methods) > 8:
                    violations.append(ArchitectureViolation(
                        file_path=file_path,
                        violation_type='interface_segregation',
                        severity='MEDIUM',
                        description=f"Interface {node.name} has {len(abstract_methods)} abstract methods",
                        suggestion="Consider splitting into smaller, more focused interfaces",
                        pattern_violated='Interface Segregation Principle'
                    ))
        
        return violations
    
    def calculate_technical_debt(self) -> List[TechnicalDebt]:
        """Calculate technical debt metrics."""
        self.console.print("[bold blue]Calculating technical debt...[/bold blue]")
        
        debt_results = []
        python_files = list(self.project_path.rglob('*.py'))
        
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
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(file_path))
            
            # Calculate various complexity metrics
            metrics = self._calculate_complexity_metrics(tree, content)
            
            # Calculate debt score
            debt_score = (
                metrics['cyclomatic_complexity'] * 0.3 +
                metrics['cognitive_complexity'] * 0.3 +
                metrics['coupling_factor'] * 0.2 +
                metrics['code_smells'] * 0.2
            )
            
            # Identify debt indicators
            debt_indicators = []
            if metrics['cyclomatic_complexity'] > 10:
                debt_indicators.append("High cyclomatic complexity")
            if metrics['cognitive_complexity'] > 15:
                debt_indicators.append("High cognitive complexity")
            if metrics['code_smells'] > 5:
                debt_indicators.append("Multiple code smells detected")
            if metrics['lines_of_code'] > 500:
                debt_indicators.append("Large file size")
            
            # Determine priority and effort
            if debt_score > 15:
                priority = "HIGH"
                effort = "3-5 days"
            elif debt_score > 8:
                priority = "MEDIUM"
                effort = "1-2 days"
            else:
                priority = "LOW"
                effort = "2-4 hours"
            
            return TechnicalDebt(
                file_path=str(file_path.relative_to(self.project_path)),
                debt_score=debt_score,
                complexity_metrics=metrics,
                debt_indicators=debt_indicators,
                refactoring_priority=priority,
                estimated_effort=effort
            )
        
        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not calculate debt for {file_path}: {e}[/yellow]")
            return None
    
    def _calculate_complexity_metrics(self, tree: ast.AST, content: str) -> Dict[str, float]:
        """Calculate various complexity metrics."""
        metrics = {
            'lines_of_code': len(content.splitlines()),
            'cyclomatic_complexity': 0,
            'cognitive_complexity': 0,
            'coupling_factor': 0,
            'code_smells': 0
        }
        
        # Cyclomatic complexity
        complexity_nodes = (ast.If, ast.While, ast.For, ast.With, ast.Try, 
                          ast.ExceptHandler, ast.Assert, ast.comprehension)
        
        for node in ast.walk(tree):
            if isinstance(node, complexity_nodes):
                metrics['cyclomatic_complexity'] += 1
            elif isinstance(node, ast.BoolOp):
                metrics['cyclomatic_complexity'] += len(node.values) - 1
        
        # Cognitive complexity (simplified)
        metrics['cognitive_complexity'] = metrics['cyclomatic_complexity'] * 1.2
        
        # Coupling factor (number of imports)
        import_count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_count += 1
        metrics['coupling_factor'] = import_count
        
        # Code smells (simplified detection)
        smells = 0
        
        # Long parameter lists
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if len(node.args.args) > 5:
                    smells += 1
        
        # Long classes
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                method_count = sum(1 for item in node.body if isinstance(item, ast.FunctionDef))
                if method_count > 20:
                    smells += 1
        
        # Magic numbers
        for node in ast.walk(tree):
            if isinstance(node, ast.Num) and isinstance(node.n, (int, float)):
                if node.n not in [0, 1, -1] and abs(node.n) > 10:
                    smells += 0.5
        
        metrics['code_smells'] = smells
        
        return metrics
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped in analysis."""
        skip_patterns = {
            '__pycache__', '.git', '.pytest_cache', 'node_modules',
            'venv', '.venv', 'env', '.env', 'build', 'dist'
        }
        
        return any(part in skip_patterns for part in file_path.parts) or \
               file_path.name.startswith('.')
    
    def _get_import_path(self, node: ast.AST) -> Optional[str]:
        """Get import path from AST node."""
        if isinstance(node, ast.Import):
            return node.names[0].name if node.names else None
        elif isinstance(node, ast.ImportFrom):
            return node.module
        return None


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description='Smart code analysis for dependency optimization')
    parser.add_argument('project_path', help='Path to the project to analyze')
    parser.add_argument('--fix-imports', action='store_true',
                       help='Automatically fix unused imports')
    parser.add_argument('--analyze-coupling', action='store_true',
                       help='Analyze module coupling')
    parser.add_argument('--check-architecture', action='store_true',
                       help='Check for architecture violations')
    parser.add_argument('--calculate-debt', action='store_true',
                       help='Calculate technical debt')
    parser.add_argument('--output', help='Output file for results (JSON format)')
    parser.add_argument('--all', action='store_true',
                       help='Run all analyses')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = CodeAnalyzer(args.project_path)
    console = Console()
    
    results = {}
    
    try:
        if args.all or args.fix_imports or not any([args.analyze_coupling, args.check_architecture, args.calculate_debt]):
            # Analyze unused imports
            import_results = analyzer.analyze_unused_imports(fix_mode=args.fix_imports)
            results['imports'] = [asdict(r) for r in import_results]
            
            # Display results
            unused_count = sum(1 for r in import_results if not r.is_used)
            console.print(f"\n[bold]Import Analysis:[/bold] Found {unused_count} unused imports out of {len(import_results)} total")
            
            if unused_count > 0:
                table = Table(title="Unused Imports")
                table.add_column("File")
                table.add_column("Import")
                table.add_column("Line")
                
                for result in import_results:
                    if not result.is_used:
                        table.add_row(result.file_path, result.import_name, str(result.line_number))
                
                console.print(table)
        
        if args.all or args.analyze_coupling:
            # Analyze coupling
            coupling_results = analyzer.analyze_coupling()
            results['coupling'] = [asdict(r) for r in coupling_results]
            
            console.print(f"\n[bold]Coupling Analysis:[/bold] Found {len(coupling_results)} significant couplings")
            
            if coupling_results:
                high_coupling = [r for r in coupling_results if r.coupling_strength > 0.5]
                if high_coupling:
                    table = Table(title="High Coupling Detected")
                    table.add_column("Module A")
                    table.add_column("Module B")
                    table.add_column("Strength")
                    table.add_column("Type")
                    table.add_column("Suggestion")
                    
                    for result in high_coupling[:10]:  # Top 10
                        table.add_row(
                            result.module_a,
                            result.module_b,
                            f"{result.coupling_strength:.2f}",
                            result.coupling_type,
                            result.refactoring_opportunity
                        )
                    
                    console.print(table)
        
        if args.all or args.check_architecture:
            # Check architecture violations
            violations = analyzer.detect_architecture_violations()
            results['violations'] = [asdict(v) for v in violations]
            
            console.print(f"\n[bold]Architecture Analysis:[/bold] Found {len(violations)} violations")
            
            if violations:
                high_severity = [v for v in violations if v.severity == 'HIGH']
                if high_severity:
                    table = Table(title="High Severity Violations")
                    table.add_column("File")
                    table.add_column("Type")
                    table.add_column("Description")
                    table.add_column("Pattern")
                    
                    for violation in high_severity:
                        table.add_row(
                            violation.file_path,
                            violation.violation_type,
                            violation.description,
                            violation.pattern_violated
                        )
                    
                    console.print(table)
        
        if args.all or args.calculate_debt:
            # Calculate technical debt
            debt_results = analyzer.calculate_technical_debt()
            results['debt'] = [asdict(d) for d in debt_results]
            
            console.print(f"\n[bold]Technical Debt Analysis:[/bold] Analyzed {len(debt_results)} files")
            
            if debt_results:
                high_debt = sorted([d for d in debt_results if d.refactoring_priority == 'HIGH'],
                                 key=lambda x: x.debt_score, reverse=True)
                
                if high_debt:
                    table = Table(title="High Priority Technical Debt")
                    table.add_column("File")
                    table.add_column("Debt Score")
                    table.add_column("Indicators")
                    table.add_column("Effort")
                    
                    for debt in high_debt[:10]:  # Top 10
                        table.add_row(
                            debt.file_path,
                            f"{debt.debt_score:.1f}",
                            ", ".join(debt.debt_indicators),
                            debt.estimated_effort
                        )
                    
                    console.print(table)
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            console.print(f"\n[green]✅ Results saved to:[/green] {args.output}")
        
        # Summary
        total_issues = sum([
            sum(1 for r in results.get('imports', []) if not r['is_used']),
            len(results.get('coupling', [])),
            len(results.get('violations', [])),
            sum(1 for d in results.get('debt', []) if d['refactoring_priority'] in ['HIGH', 'MEDIUM'])
        ])
        
        console.print(f"\n[bold]Summary:[/bold] Found {total_issues} total issues requiring attention")
    
    except Exception as e:
        console.print(f"[red]Error during analysis: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()