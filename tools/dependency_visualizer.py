#!/usr/bin/env python3
"""
Dependency Graph Visualizer
==========================

Automated dependency graph visualization tool that generates:
- Text-based dependency trees for quick analysis
- Interactive HTML graphs with clickable nodes
- Risk heat maps highlighting critical dependencies
- Circular dependency detection with visual warnings
- Multiple export formats (PNG, SVG, HTML, JSON)

Usage:
    python dependency_visualizer.py /path/to/project
    python dependency_visualizer.py /path/to/project --format html
    python dependency_visualizer.py /path/to/project --output dependency_graph.html
"""

import ast
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import importlib.util

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from rich.console import Console
    from rich.tree import Tree
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import track

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

    # Fallback console
    class Console:
        def print(self, *args, **kwargs):
            print(*args)

    def track(items, description="Processing..."):
        print(f"{description}")
        return items


@dataclass
class DependencyNode:
    """Represents a single node in the dependency graph."""

    name: str
    file_path: str
    module_type: str  # 'internal', 'external', 'stdlib'
    imports: List[str]
    imported_by: List[str]
    risk_level: str  # 'HIGH', 'MEDIUM', 'LOW'
    lines_of_code: int
    last_modified: Optional[str] = None


@dataclass
class DependencyGraph:
    """Complete dependency graph representation."""

    nodes: Dict[str, DependencyNode]
    edges: List[Tuple[str, str, str]]  # (from, to, relationship_type)
    circular_dependencies: List[List[str]]
    external_dependencies: Dict[str, List[str]]
    metrics: Dict[str, int]


class DependencyAnalyzer:
    """Core dependency analysis engine."""

    def __init__(self, project_path: str):
        self.project_path = Path(project_path).resolve()
        self.console = Console()
        if NETWORKX_AVAILABLE:
            self.graph = nx.DiGraph()
        else:
            # Fallback simple graph representation
            self.graph = defaultdict(list)
        self.nodes = {}
        self.external_deps = defaultdict(list)

    def analyze_project(self) -> DependencyGraph:
        """Analyze the entire project and build dependency graph."""
        self.console.print(f"[bold blue]Analyzing project:[/bold blue] {self.project_path}")

        # Find all Python files
        python_files = self._find_python_files()
        self.console.print(f"Found {len(python_files)} Python files")

        # Analyze each file
        for file_path in track(python_files, description="Analyzing files..."):
            self._analyze_file(file_path)

        # Detect circular dependencies
        circular_deps = self._detect_circular_dependencies()

        # Calculate risk levels
        self._calculate_risk_levels()

        # Build final graph object
        return DependencyGraph(
            nodes=self.nodes,
            edges=[(u, v, self.graph[u][v].get("type", "import")) for u, v in self.graph.edges()],
            circular_dependencies=circular_deps,
            external_dependencies=dict(self.external_deps),
            metrics=self._calculate_metrics(),
        )

    def _find_python_files(self) -> List[Path]:
        """Find all Python files in the project."""
        python_files = []
        for root, dirs, files in os.walk(self.project_path):
            # Skip common non-source directories
            dirs[:] = [
                d
                for d in dirs
                if d
                not in {
                    ".git",
                    "__pycache__",
                    ".pytest_cache",
                    "node_modules",
                    "venv",
                    ".venv",
                    "env",
                    ".env",
                    "build",
                    "dist",
                }
            ]

            for file in files:
                if file.endswith(".py") and not file.startswith("."):
                    python_files.append(Path(root) / file)

        return python_files

    def _analyze_file(self, file_path: Path):
        """Analyze a single Python file for imports and dependencies."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Parse the AST
            tree = ast.parse(content, filename=str(file_path))

            # Extract module name
            rel_path = file_path.relative_to(self.project_path)
            module_name = str(rel_path.with_suffix("")).replace(os.sep, ".")

            # Analyze imports
            imports = self._extract_imports(tree, file_path)

            # Create node
            node = DependencyNode(
                name=module_name,
                file_path=str(rel_path),
                module_type="internal",
                imports=imports["internal"],
                imported_by=[],
                risk_level="LOW",  # Will be calculated later
                lines_of_code=len(content.splitlines()),
                last_modified=None,  # Could add file stat info
            )

            self.nodes[module_name] = node

            # Add to graph
            self.graph.add_node(module_name, **asdict(node))

            # Add edges for internal imports
            for imported_module in imports["internal"]:
                self.graph.add_edge(module_name, imported_module, type="import")

            # Track external dependencies
            for ext_dep in imports["external"]:
                self.external_deps[ext_dep].append(module_name)

        except Exception as e:
            self.console.print(f"[red]Error analyzing {file_path}: {e}[/red]")

    def _extract_imports(self, tree: ast.AST, file_path: Path) -> Dict[str, List[str]]:
        """Extract imports from AST."""
        imports = {"internal": [], "external": [], "stdlib": []}

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports["external"].append(alias.name.split(".")[0])

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module
                    if module.startswith("."):
                        # Relative import - convert to absolute
                        imports["internal"].append(self._resolve_relative_import(module, file_path))
                    elif self._is_internal_module(module):
                        imports["internal"].append(module)
                    else:
                        imports["external"].append(module.split(".")[0])

        return imports

    def _resolve_relative_import(self, module: str, file_path: Path) -> str:
        """Resolve relative imports to absolute module names."""
        # Simple implementation - could be more sophisticated
        rel_path = file_path.relative_to(self.project_path)
        current_module = str(rel_path.parent).replace(os.sep, ".")

        if module.startswith(".."):
            # Go up one level for each additional dot
            levels = len(module) - len(module.lstrip("."))
            parts = current_module.split(".")
            if levels <= len(parts):
                base = ".".join(parts[:-levels]) if levels < len(parts) else ""
                return f"{base}.{module.lstrip('.')}" if base else module.lstrip(".")

        return f"{current_module}.{module.lstrip('.')}"

    def _is_internal_module(self, module: str) -> bool:
        """Check if a module is internal to the project."""
        # Check if the module path exists in the project
        parts = module.split(".")
        for i in range(len(parts)):
            partial_path = Path(self.project_path) / "/".join(parts[: i + 1])
            if partial_path.with_suffix(".py").exists() or (partial_path / "__init__.py").exists():
                return True
        return False

    def _detect_circular_dependencies(self) -> List[List[str]]:
        """Detect circular dependencies in the graph."""
        cycles = []
        try:
            # Find all strongly connected components with more than one node
            sccs = list(nx.strongly_connected_components(self.graph))
            for scc in sccs:
                if len(scc) > 1:
                    cycles.append(list(scc))

            # Also check for simple cycles
            try:
                simple_cycles = list(nx.simple_cycles(self.graph))
                for cycle in simple_cycles:
                    if cycle not in cycles:
                        cycles.append(cycle)
            except nx.NetworkXError:
                pass  # Graph might be too large for cycle detection

        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not detect all cycles: {e}[/yellow]")

        return cycles

    def _calculate_risk_levels(self):
        """Calculate risk levels for each node based on connectivity."""
        for module_name, node in self.nodes.items():
            # Calculate in-degree (how many modules depend on this)
            in_degree = self.graph.in_degree(module_name)
            # Calculate out-degree (how many modules this depends on)
            out_degree = self.graph.out_degree(module_name)

            # Risk calculation based on connectivity and file size
            risk_score = in_degree * 2 + out_degree + (node.lines_of_code / 100)

            if risk_score > 10:
                node.risk_level = "HIGH"
            elif risk_score > 5:
                node.risk_level = "MEDIUM"
            else:
                node.risk_level = "LOW"

            # Update imported_by relationships
            node.imported_by = list(self.graph.predecessors(module_name))

    def _calculate_metrics(self) -> Dict[str, int]:
        """Calculate various project metrics."""
        return {
            "total_files": len(self.nodes),
            "total_imports": sum(len(node.imports) for node in self.nodes.values()),
            "external_dependencies": len(self.external_deps),
            "high_risk_files": sum(1 for node in self.nodes.values() if node.risk_level == "HIGH"),
            "circular_dependencies": len(self._detect_circular_dependencies()),
            "total_lines_of_code": sum(node.lines_of_code for node in self.nodes.values()),
        }


class DependencyVisualizer:
    """Dependency graph visualization engine."""

    def __init__(self, graph: DependencyGraph):
        self.graph = graph
        self.console = Console()

    def generate_text_tree(self) -> str:
        """Generate a text-based dependency tree."""
        from rich.console import Console
        from io import StringIO

        output = []

        # Find root nodes (nodes with no dependencies)
        root_nodes = [name for name, node in self.graph.nodes.items() if not node.imports]

        if not root_nodes:
            # If no true roots, pick nodes with fewest dependencies
            root_nodes = sorted(
                self.graph.nodes.keys(), key=lambda x: len(self.graph.nodes[x].imports)
            )[:3]

        for root in root_nodes:
            tree = Tree(f"[bold blue]{root}[/bold blue]")
            self._build_tree_recursive(tree, root, visited=set())

            # Render tree to string
            console = Console(file=StringIO(), width=80)
            console.print(tree)
            tree_str = console.file.getvalue()
            output.append(tree_str)

        return "\n\n".join(output)

    def _build_tree_recursive(
        self, tree: Tree, node_name: str, visited: Set[str], max_depth: int = 5
    ):
        """Recursively build tree structure."""
        if node_name in visited or max_depth <= 0:
            return

        visited.add(node_name)
        node = self.graph.nodes[node_name]

        for imported in node.imports:
            if imported in self.graph.nodes:
                risk_color = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "green"}.get(
                    self.graph.nodes[imported].risk_level, "white"
                )

                subtree = tree.add(f"[{risk_color}]{imported}[/{risk_color}]")
                self._build_tree_recursive(subtree, imported, visited.copy(), max_depth - 1)

    def generate_mermaid_graph(self) -> str:
        """Generate Mermaid syntax for dependency graph."""
        lines = ["graph TD"]

        # Add nodes with risk-based styling
        for node_name, node in self.graph.nodes.items():
            # Sanitize node names for Mermaid (replace dots and special chars)
            safe_name = node_name.replace(".", "_").replace("-", "_").replace("/", "_")
            display_name = node_name.split(".")[-1]  # Show just the module name

            lines.append(f'    {safe_name}["{display_name}"]')

            # Add edges
            for imported in node.imports:
                if imported in self.graph.nodes:
                    safe_imported = imported.replace(".", "_").replace("-", "_").replace("/", "_")
                    lines.append(f"    {safe_name} --> {safe_imported}")

        # Add risk-based styling classes
        lines.extend(
            [
                "",
                "    %% Risk-based styling",
                "    classDef high fill:#ff6b6b,stroke:#d63031,stroke-width:3px,color:#fff",
                "    classDef medium fill:#ffd93d,stroke:#f39c12,stroke-width:2px,color:#2d3436",
                "    classDef low fill:#6bcf7f,stroke:#00b894,stroke-width:1px,color:#2d3436",
                "",
            ]
        )

        # Apply classes to nodes based on risk level
        for node_name, node in self.graph.nodes.items():
            safe_name = node_name.replace(".", "_").replace("-", "_").replace("/", "_")
            risk_class = node.risk_level.lower()
            lines.append(f"    class {safe_name} {risk_class}")

        return "\n".join(lines)

    def save_mermaid_syntax(self, output_path: str):
        """Save just the Mermaid syntax to a .mmd file for use in GitHub, etc."""
        mermaid_syntax = self.generate_mermaid_graph()

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(mermaid_syntax)

        self.console.print(f"[green]Mermaid syntax saved to:[/green] {output_path}")
        self.console.print("[blue]💡 Tip:[/blue] You can include this in GitHub README with:")
        self.console.print(f"```mermaid\\n{mermaid_syntax[:100]}...\\n```")

    def generate_mermaid_html(self, output_path: str):
        """Generate HTML file with Mermaid diagram."""
        mermaid_graph = self.generate_mermaid_graph()

        # Create comprehensive HTML with Mermaid
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dependency Graph - {len(self.graph.nodes)} modules</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10.6.1/dist/mermaid.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f8f9fa;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 20px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #e9ecef;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .metric {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #495057;
        }}
        .metric-label {{
            font-size: 14px;
            color: #6c757d;
            margin-top: 5px;
        }}
        .legend {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 3px;
        }}
        .high {{ background: #ff6b6b; }}
        .medium {{ background: #ffd93d; }}
        .low {{ background: #6bcf7f; }}
        .diagram-container {{
            text-align: center;
            margin: 20px 0;
            min-height: 400px;
        }}
        .warning {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 20px;
        }}
        @media (max-width: 768px) {{
            .metrics {{ grid-template-columns: 1fr; }}
            .legend {{ flex-direction: column; align-items: center; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🗺️ Dependency Graph</h1>
            <p>Visual representation of project dependencies and relationships</p>
        </div>
        
        <div class="metrics">
            <div class="metric">
                <div class="metric-value">{self.graph.metrics['total_files']}</div>
                <div class="metric-label">Total Files</div>
            </div>
            <div class="metric">
                <div class="metric-value">{self.graph.metrics['total_imports']}</div>
                <div class="metric-label">Total Imports</div>
            </div>
            <div class="metric">
                <div class="metric-value">{self.graph.metrics['external_dependencies']}</div>
                <div class="metric-label">External Dependencies</div>
            </div>
            <div class="metric">
                <div class="metric-value">{self.graph.metrics['high_risk_files']}</div>
                <div class="metric-label">High Risk Files</div>
            </div>
            <div class="metric">
                <div class="metric-value">{self.graph.metrics['circular_dependencies']}</div>
                <div class="metric-label">Circular Dependencies</div>
            </div>
            <div class="metric">
                <div class="metric-value">{self.graph.metrics.get('total_lines_of_code', 'N/A')}</div>
                <div class="metric-label">Lines of Code</div>
            </div>
        </div>
        
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color high"></div>
                <span>High Risk</span>
            </div>
            <div class="legend-item">
                <div class="legend-color medium"></div>
                <span>Medium Risk</span>
            </div>
            <div class="legend-item">
                <div class="legend-color low"></div>
                <span>Low Risk</span>
            </div>
        </div>
        
        {f'<div class="warning">⚠️ {len(self.graph.circular_dependencies)} Circular Dependencies Detected</div>' if self.graph.circular_dependencies else ''}
        
        <div class="diagram-container">
            <div class="mermaid">
{mermaid_graph}
            </div>
        </div>
        
        <div style="margin-top: 30px; padding-top: 20px; border-top: 1px solid #e9ecef; text-align: center; color: #6c757d; font-size: 12px;">
            Generated by <a href="https://github.com/scs03004/dependency-toolkit" target="_blank">Dependency Toolkit</a>
        </div>
    </div>
    
    <script>
        mermaid.initialize({{ 
            startOnLoad: true,
            theme: 'base',
            themeVariables: {{
                primaryColor: '#f8f9fa',
                primaryTextColor: '#495057',
                primaryBorderColor: '#dee2e6',
                lineColor: '#adb5bd',
                secondaryColor: '#e9ecef',
                tertiaryColor: '#f8f9fa'
            }},
            flowchart: {{
                useMaxWidth: true,
                htmlLabels: true,
                curve: 'basis'
            }}
        }});
    </script>
</body>
</html>"""

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        self.console.print(f"[green]Mermaid HTML graph saved to:[/green] {output_path}")

    def generate_html_interactive(self, output_path: str):
        """Generate interactive HTML visualization - now uses Mermaid by default."""
        # Use Mermaid as the primary approach
        self.generate_mermaid_html(output_path)
        return

        # Fallback to Plotly if specifically requested
        if not PLOTLY_AVAILABLE or not NETWORKX_AVAILABLE:
            self.console.print(
                "[yellow]Warning: Plotly or NetworkX not available. Using Mermaid instead.[/yellow]"
            )
            self.generate_mermaid_html(output_path)
            return

        # Prepare data for Plotly
        node_names = list(self.graph.nodes.keys())
        node_colors = {"HIGH": "red", "MEDIUM": "orange", "LOW": "green"}

        # Create network layout
        G = nx.DiGraph()
        for name, node in self.graph.nodes.items():
            G.add_node(name, **asdict(node))

        for edge in self.graph.edges:
            G.add_edge(edge[0], edge[1])

        # Generate layout
        try:
            pos = nx.spring_layout(G, k=1, iterations=50)
        except:
            # Fallback for large graphs
            pos = nx.random_layout(G)

        # Create traces for nodes
        node_trace = go.Scatter(
            x=[pos[node][0] for node in node_names],
            y=[pos[node][1] for node in node_names],
            mode="markers+text",
            text=node_names,
            textposition="middle center",
            hovertemplate="<b>%{text}</b><br>"
            + "Risk Level: %{customdata[0]}<br>"
            + "Lines of Code: %{customdata[1]}<br>"
            + "Imports: %{customdata[2]}<br>"
            + "Imported By: %{customdata[3]}",
            customdata=[
                [
                    self.graph.nodes[name].risk_level,
                    self.graph.nodes[name].lines_of_code,
                    len(self.graph.nodes[name].imports),
                    len(self.graph.nodes[name].imported_by),
                ]
                for name in node_names
            ],
            marker=dict(
                size=[
                    max(10, min(50, self.graph.nodes[name].lines_of_code / 10))
                    for name in node_names
                ],
                color=[node_colors[self.graph.nodes[name].risk_level] for name in node_names],
                line=dict(width=1, color="black"),
            ),
            name="Modules",
        )

        # Create traces for edges
        edge_x = []
        edge_y = []
        for edge in self.graph.edges:
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=1, color="lightgray"),
            hoverinfo="none",
            mode="lines",
            name="Dependencies",
        )

        # Create the figure
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                title=dict(
                    text=f'Dependency Graph - {self.graph.metrics["total_files"]} modules',
                    font=dict(size=16),
                ),
                showlegend=True,
                hovermode="closest",
                margin=dict(b=20, l=5, r=5, t=40),
                annotations=[
                    dict(
                        text="Risk Levels: Red=HIGH, Orange=MEDIUM, Green=LOW<br>"
                        + f"Total Files: {self.graph.metrics['total_files']}, "
                        + f"External Deps: {self.graph.metrics['external_dependencies']}, "
                        + f"Circular Deps: {self.graph.metrics['circular_dependencies']}",
                        showarrow=False,
                        xref="paper",
                        yref="paper",
                        x=0.005,
                        y=-0.002,
                        xanchor="left",
                        yanchor="bottom",
                        font=dict(size=12),
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            ),
        )

        # Add circular dependency annotations
        if self.graph.circular_dependencies:
            fig.add_annotation(
                text=f"⚠️ {len(self.graph.circular_dependencies)} Circular Dependencies Detected",
                x=0.5,
                y=1.1,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=14, color="red"),
                bgcolor="yellow",
                bordercolor="red",
                borderwidth=2,
            )

        # Save the HTML file
        fig.write_html(output_path)
        self.console.print(f"[green]Interactive HTML graph saved to:[/green] {output_path}")

    def generate_risk_heatmap(self, output_path: str):
        """Generate risk level heatmap."""
        # Prepare data
        nodes_data = []
        for name, node in self.graph.nodes.items():
            nodes_data.append(
                {
                    "Module": name,
                    "Risk Level": node.risk_level,
                    "Lines of Code": node.lines_of_code,
                    "Imports": len(node.imports),
                    "Imported By": len(node.imported_by),
                    "Risk Score": len(node.imported_by) * 2
                    + len(node.imports)
                    + (node.lines_of_code / 100),
                }
            )

        df = pd.DataFrame(nodes_data)

        # Create heatmap
        fig = px.scatter(
            df,
            x="Imports",
            y="Imported By",
            size="Lines of Code",
            color="Risk Level",
            hover_name="Module",
            title="Module Risk Assessment Heatmap",
            color_discrete_map={"HIGH": "red", "MEDIUM": "orange", "LOW": "green"},
        )

        fig.update_layout(
            xaxis_title="Number of Dependencies",
            yaxis_title="Number of Dependents",
            title_font_size=16,
        )

        # Save the HTML file
        fig.write_html(output_path)
        self.console.print(f"[green]Risk heatmap saved to:[/green] {output_path}")

    def generate_summary_report(self) -> str:
        """Generate a text summary report."""
        report = []

        # Header
        report.append("# Dependency Analysis Report")
        report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Metrics
        report.append("## 📊 Project Metrics")
        for key, value in self.graph.metrics.items():
            report.append(f"- **{key.replace('_', ' ').title()}**: {value}")
        report.append("")

        # High-risk files
        high_risk = [name for name, node in self.graph.nodes.items() if node.risk_level == "HIGH"]

        if high_risk:
            report.append("## 🚨 High-Risk Files")
            report.append("These files have high impact on the codebase:")
            for file in high_risk[:10]:  # Top 10
                node = self.graph.nodes[file]
                report.append(
                    f"- **{file}** ({node.lines_of_code} LOC, "
                    f"{len(node.imported_by)} dependents)"
                )
            report.append("")

        # Circular dependencies
        if self.graph.circular_dependencies:
            report.append("## ⚠️ Circular Dependencies")
            for i, cycle in enumerate(self.graph.circular_dependencies):
                report.append(f"**Cycle {i+1}**: {' → '.join(cycle + [cycle[0]])}")
            report.append("")

        # External dependencies
        if self.graph.external_dependencies:
            report.append("## 📦 External Dependencies")
            sorted_deps = sorted(
                self.graph.external_dependencies.items(), key=lambda x: len(x[1]), reverse=True
            )
            for dep, users in sorted_deps[:15]:  # Top 15
                report.append(f"- **{dep}**: Used by {len(users)} modules")
            report.append("")

        return "\n".join(report)


def main():
    """Main CLI interface."""
    # Set up proper encoding for Windows console
    import sys

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except AttributeError:
            # Fallback for older Python versions
            import codecs

            sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer)
            sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer)

    parser = argparse.ArgumentParser(description="Analyze and visualize project dependencies")
    parser.add_argument("project_path", help="Path to the project to analyze")
    parser.add_argument(
        "--format",
        choices=["text", "html", "mermaid", "syntax", "heatmap", "all"],
        default="mermaid",
        help="Output format (mermaid=HTML+Mermaid, syntax=raw .mmd file)",
    )
    parser.add_argument("--output", help="Output file path (default: auto-generated)")
    parser.add_argument("--no-interactive", action="store_true", help="Skip interactive features")

    args = parser.parse_args()

    # Initialize analyzer
    analyzer = DependencyAnalyzer(args.project_path)

    # Analyze project
    try:
        dependency_graph = analyzer.analyze_project()
    except Exception as e:
        print(f"Error analyzing project: {e}")
        sys.exit(1)

    # Initialize visualizer
    visualizer = DependencyVisualizer(dependency_graph)

    # Generate outputs
    base_name = Path(args.project_path).name

    if args.format in ["text", "all"]:
        # Text output
        text_tree = visualizer.generate_text_tree()
        print("\n" + "=" * 50)
        print("DEPENDENCY TREE")
        print("=" * 50)
        print(text_tree)

        # Summary report
        summary = visualizer.generate_summary_report()
        print("\n" + "=" * 50)
        print("SUMMARY REPORT")
        print("=" * 50)
        print(summary)

    if args.format in ["mermaid", "all"]:
        # Mermaid HTML graph (web-native)
        output_path = args.output or f"{base_name}_dependency_graph.html"
        visualizer.generate_mermaid_html(output_path)

    if args.format in ["syntax"]:
        # Raw Mermaid syntax for GitHub/docs
        output_path = args.output or f"{base_name}_dependency_graph.mmd"
        visualizer.save_mermaid_syntax(output_path)

    if args.format in ["html"]:
        # Legacy Plotly HTML graph
        output_path = args.output or f"{base_name}_dependency_graph_plotly.html"
        visualizer.generate_html_interactive(output_path)

    if args.format in ["heatmap", "all"]:
        # Risk heatmap
        heatmap_path = args.output or f"{base_name}_risk_heatmap.html"
        if args.format == "heatmap":
            heatmap_path = args.output or f"{base_name}_risk_heatmap.html"
        else:
            heatmap_path = f"{base_name}_risk_heatmap.html"
        visualizer.generate_risk_heatmap(heatmap_path)

    print(f"\n✅ Analysis complete for {dependency_graph.metrics['total_files']} files")
    if dependency_graph.circular_dependencies:
        print(f"⚠️  Found {len(dependency_graph.circular_dependencies)} circular dependencies")


if __name__ == "__main__":
    main()
