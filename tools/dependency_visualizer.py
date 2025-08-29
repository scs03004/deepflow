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
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

PLOTLY_AVAILABLE = False
PANDAS_AVAILABLE = False

try:
    from rich.console import Console
    from rich.tree import Tree
    from rich.progress import track

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

    # Fallback console and tree
    class Console:
        def print(self, *args, **kwargs):
            print(*args)
    
    class Tree:
        def __init__(self, label):
            self.label = label
            self.children = []
        
        def add(self, label):
            child = Tree(label)
            self.children.append(child)
            return child

    def track(items, description="Processing..."):
        print(f"{description}")
        return items


def estimate_token_count(file_path: str) -> int:
    """Estimate token count for AI context window analysis."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        # Rough estimation: ~4 characters per token for code
        return len(content) // 4
    except:
        return 0


def get_ai_context_health(token_count: int) -> str:
    """Determine AI context health based on token count."""
    if token_count < 2000:
        return "GOOD"
    elif token_count < 4000:
        return "WARNING"
    else:
        return "CRITICAL"


class MermaidRenderer:
    """Helper class for generating Mermaid diagrams."""
    
    def __init__(self, graph):
        self.graph = graph
    
    def generate_graph_definition(self, ai_awareness: bool = False) -> str:
        """Generate the core Mermaid graph definition."""
        node_count = len(self.graph.nodes)
        
        # Choose layout based on graph size - force vertical layouts
        if node_count > 50:
            layout = "graph TD"  # Top-down for large graphs (force vertical)
        elif node_count > 20:
            layout = "graph TD"  # Top-down for medium graphs (force vertical)  
        else:
            layout = "graph TD"  # Top-down for small graphs
            
        lines = [layout]
        
        # Add nodes with styling
        for name, node in self.graph.nodes.items():
            node_id = self._sanitize_node_id(name)
            display_name = self._get_display_name(name, node, ai_awareness)
            
            # Choose node style based on risk level and graph size
            if hasattr(node, 'risk_level'):
                shape = self._get_node_shape(node.risk_level, node_count > 30)
            else:
                shape = self._get_default_shape(node_count > 30)
            
            # Create the node definition with the display name
            if "name" in shape:
                # Replace "name" placeholder with the actual display name
                node_definition = shape.replace("name", display_name)
                lines.append(f"    {node_id}{node_definition}")
            else:
                # Fallback for shapes that don't use "name" placeholder
                lines.append(f"    {node_id}{display_name}")
        
        # Add edges with better spacing for large graphs
        edge_count = 0
        max_edges_per_node = 10 if node_count > 50 else 20
        
        for name, node in self.graph.nodes.items():
            node_id = self._sanitize_node_id(name)
            edges_added = 0
            
            for dependency in node.imports:
                if edges_added >= max_edges_per_node:
                    # Add a summary node for remaining dependencies
                    summary_id = f"{node_id}_more"
                    remaining = len(node.imports) - edges_added
                    lines.append(f"    {summary_id}[\"...+{remaining} more\"]")
                    lines.append(f"    {node_id} -.-> {summary_id}")
                    break
                    
                dep_id = self._sanitize_node_id(dependency)
                if dep_id in [self._sanitize_node_id(n) for n in self.graph.nodes]:
                    lines.append(f"    {node_id} --> {dep_id}")
                    edges_added += 1
                    edge_count += 1
        
        # Add styling with better colors for readability
        lines.extend(self._get_styling_definitions(node_count))
        
        return "\n".join(lines)
    
    def _sanitize_node_id(self, name: str) -> str:
        """Sanitize node name for Mermaid compatibility."""
        return name.replace(".", "_").replace("/", "_").replace("-", "_")
    
    def _get_display_name(self, name: str, node, ai_awareness: bool) -> str:
        """Get the display name for a node (just the label text, no shape)."""
        if ai_awareness and hasattr(node, 'token_count'):
            health = get_ai_context_health(node.token_count)
            health_emoji = {"GOOD": "✅", "WARNING": "⚠️", "CRITICAL": "🚨"}.get(health, "")
            return f"{name}\\n{node.token_count} tokens {health_emoji}"
        else:
            return name
    
    def _get_node_shape(self, risk_level: str, compact: bool = False) -> str:
        """Get Mermaid node shape based on risk level."""
        if compact:
            shapes = {
                "HIGH": '["name"]',
                "MEDIUM": '["name"]', 
                "LOW": '["name"]'
            }
        else:
            shapes = {
                "HIGH": '(("name"))',
                "MEDIUM": '["name"]', 
                "LOW": '("name")'
            }
        return shapes.get(risk_level, '["name"]')
    
    def _get_default_shape(self, compact: bool = False) -> str:
        """Get default node shape."""
        return '["name"]' if compact else '("name")'
    
    def _get_styling_definitions(self, node_count: int = 0) -> List[str]:
        """Get CSS styling definitions for the graph."""
        # Adjust font size based on graph size
        if node_count > 50:
            font_size = "12px"
        elif node_count > 20:
            font_size = "14px"
        else:
            font_size = "16px"
            
        return [
            f"    classDef high fill:#ff6b6b,stroke:#d63031,stroke-width:2px,color:#fff,font-size:{font_size}",
            f"    classDef medium fill:#fdcb6e,stroke:#e17055,stroke-width:2px,color:#2d3436,font-size:{font_size}",
            f"    classDef low fill:#00b894,stroke:#00a085,stroke-width:2px,color:#fff,font-size:{font_size}",
            f"    classDef external fill:#74b9ff,stroke:#0984e3,stroke-width:1px,color:#fff,font-size:{font_size}",
            f"    classDef summary fill:#ddd,stroke:#999,stroke-width:1px,color:#666,font-size:12px",
        ]


class HTMLTemplateGenerator:
    """Helper class for generating HTML templates."""
    
    def __init__(self, graph):
        self.graph = graph
    
    def generate_mermaid_html(self, mermaid_graph: str, ai_awareness: bool = False, title_suffix: str = "") -> str:
        """Generate complete HTML page with Mermaid diagram."""
        metrics = self._calculate_metrics()
        styles = self._get_html_styles()
        scripts = self._get_html_scripts()
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dependency Graph{title_suffix} - {len(self.graph.nodes)} modules</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10.6.1/dist/mermaid.min.js"></script>
    {styles}
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔗 Dependency Graph Analysis{title_suffix}</h1>
            <p class="subtitle">Interactive visualization of project dependencies</p>
        </div>
        
        {self._generate_metrics_section(metrics, ai_awareness)}
        
        <div class="diagram-info">
            <h3>📊 Interactive Dependency Diagram</h3>
            <p><strong>Zoom:</strong> Use buttons, mouse wheel, or keyboard (Ctrl/Cmd + +/- to zoom, Ctrl/Cmd + 0 to reset)</p>
            <p><strong>Pan:</strong> Click and drag to move around, or use arrow keys for precise movement</p>
            <p><strong>Mobile:</strong> Touch and drag to pan, pinch to zoom</p>
        </div>
        
        <div class="graph-container" style="position: relative;">
            <div class="zoom-controls">
                <button class="zoom-btn" onclick="zoomIn()" title="Zoom In (Ctrl/Cmd + +)">+</button>
                <button class="zoom-btn" onclick="zoomOut()" title="Zoom Out (Ctrl/Cmd + -)">-</button>
                <button class="zoom-btn" onclick="resetZoom()" title="Reset Zoom (Ctrl/Cmd + 0)">100%</button>
                <span class="zoom-btn" id="zoom-level" style="background: #28a745; cursor: default;">100%</span>
            </div>
            <div class="mermaid" id="diagram">
{mermaid_graph}
            </div>
        </div>
        
        {self._generate_legend_section(ai_awareness)}
        {self._generate_details_section()}
        
        <footer>
            <p>Generated by Dependency Visualizer • <a href="https://github.com/scs03004/deepflow">Deepflow</a></p>
        </footer>
    </div>
    
    {scripts}
</body>
</html>"""
    
    def _calculate_metrics(self) -> Dict[str, int]:
        """Calculate basic metrics for the graph."""
        metrics = {
            "total_modules": len(self.graph.nodes),
            "total_dependencies": sum(len(node.imports) for node in self.graph.nodes.values()),
            "external_deps": len([n for n in self.graph.nodes.values() if getattr(n, 'module_type', 'internal') == "external"]),
            "internal_deps": len([n for n in self.graph.nodes.values() if getattr(n, 'module_type', 'internal') == "internal"])
        }
        return metrics
    
    def _get_html_styles(self) -> str:
        """Get CSS styles for the HTML page."""
        return """<style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f8f9fa;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 20px;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #e9ecef;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        .metric {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }
        .graph-container {
            text-align: center;
            margin: 20px 0;
            overflow: auto;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            background: #fafafa;
            padding: 20px;
            min-height: 600px;
        }
        .mermaid {
            min-width: 100%;
            min-height: 600px;
            background: white;
            border-radius: 4px;
            padding: 20px;
            cursor: grab;
            overflow: hidden;
            position: relative;
        }
        .mermaid:active {
            cursor: grabbing;
        }
        .mermaid svg {
            transition: transform 0.1s ease-out;
        }
        .zoom-controls {
            position: absolute;
            top: 10px;
            right: 10px;
            display: flex;
            gap: 5px;
            z-index: 1000;
        }
        .zoom-btn {
            background: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 8px 12px;
            cursor: pointer;
            font-size: 14px;
        }
        .zoom-btn:hover {
            background: #0056b3;
        }
        .diagram-info {
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }
        </style>"""
    
    def _get_html_scripts(self) -> str:
        """Get JavaScript for the HTML page."""
        return """<script>
        let zoomLevel = 1.0;
        const zoomStep = 0.2;
        let panX = 0;
        let panY = 0;
        let isDragging = false;
        let dragStart = { x: 0, y: 0 };
        
        mermaid.initialize({ 
            startOnLoad: true,
            theme: 'neutral',
            flowchart: { 
                useMaxWidth: true, 
                htmlLabels: true,
                curve: 'basis',
                padding: 20,
                rankdir: 'TD'  // Force top-down direction
            },
            themeVariables: {
                fontSize: '16px',
                fontFamily: 'Arial, sans-serif'
            }
        });
        
        function zoomIn() {
            zoomLevel += zoomStep;
            updateTransform();
        }
        
        function zoomOut() {
            zoomLevel = Math.max(0.2, zoomLevel - zoomStep);
            updateTransform();
        }
        
        function resetZoom() {
            zoomLevel = 1.0;
            panX = 0;
            panY = 0;
            updateTransform();
        }
        
        function updateTransform() {
            const diagram = document.querySelector('.mermaid svg') || document.querySelector('.mermaid');
            if (diagram) {
                diagram.style.transform = `translate(${panX}px, ${panY}px) scale(${zoomLevel})`;
                diagram.style.transformOrigin = '0 0';
                const zoomDisplay = document.getElementById('zoom-level');
                if (zoomDisplay) {
                    zoomDisplay.textContent = Math.round(zoomLevel * 100) + '%';
                }
            }
        }
        
        // Initialize drag functionality when mermaid is ready
        setTimeout(function() {
            const diagramContainer = document.querySelector('.mermaid');
            if (diagramContainer) {
                setupDragFunctionality(diagramContainer);
            }
        }, 1000);
        
        function setupDragFunctionality(container) {
            // Mouse events for desktop
            container.addEventListener('mousedown', startDrag);
            document.addEventListener('mousemove', drag);
            document.addEventListener('mouseup', endDrag);
            
            // Touch events for mobile
            container.addEventListener('touchstart', startTouchDrag, { passive: false });
            document.addEventListener('touchmove', touchDrag, { passive: false });
            document.addEventListener('touchend', endTouchDrag);
            
            // Prevent context menu on right-click
            container.addEventListener('contextmenu', function(e) {
                e.preventDefault();
            });
            
            // Mouse wheel zoom
            container.addEventListener('wheel', function(e) {
                e.preventDefault();
                const delta = e.deltaY > 0 ? -zoomStep : zoomStep;
                const rect = container.getBoundingClientRect();
                const mouseX = e.clientX - rect.left;
                const mouseY = e.clientY - rect.top;
                
                // Zoom towards mouse cursor
                const oldZoom = zoomLevel;
                zoomLevel = Math.max(0.2, Math.min(5.0, zoomLevel + delta));
                
                if (zoomLevel !== oldZoom) {
                    const zoomChange = zoomLevel - oldZoom;
                    panX -= (mouseX - panX) * (zoomChange / oldZoom);
                    panY -= (mouseY - panY) * (zoomChange / oldZoom);
                    updateTransform();
                }
            });
        }
        
        function startDrag(e) {
            isDragging = true;
            dragStart.x = e.clientX - panX;
            dragStart.y = e.clientY - panY;
            e.preventDefault();
        }
        
        function drag(e) {
            if (!isDragging) return;
            e.preventDefault();
            panX = e.clientX - dragStart.x;
            panY = e.clientY - dragStart.y;
            updateTransform();
        }
        
        function endDrag(e) {
            isDragging = false;
        }
        
        function startTouchDrag(e) {
            if (e.touches.length === 1) {
                isDragging = true;
                const touch = e.touches[0];
                dragStart.x = touch.clientX - panX;
                dragStart.y = touch.clientY - panY;
                e.preventDefault();
            }
        }
        
        function touchDrag(e) {
            if (!isDragging || e.touches.length !== 1) return;
            e.preventDefault();
            const touch = e.touches[0];
            panX = touch.clientX - dragStart.x;
            panY = touch.clientY - dragStart.y;
            updateTransform();
        }
        
        function endTouchDrag(e) {
            isDragging = false;
        }
        
        // Add keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            if (e.ctrlKey || e.metaKey) {
                if (e.key === '=' || e.key === '+') {
                    e.preventDefault();
                    zoomIn();
                } else if (e.key === '-') {
                    e.preventDefault();
                    zoomOut();
                } else if (e.key === '0') {
                    e.preventDefault();
                    resetZoom();
                }
            }
            // Arrow key panning
            const panSpeed = 50;
            if (e.key === 'ArrowLeft') {
                e.preventDefault();
                panX += panSpeed;
                updateTransform();
            } else if (e.key === 'ArrowRight') {
                e.preventDefault();
                panX -= panSpeed;
                updateTransform();
            } else if (e.key === 'ArrowUp') {
                e.preventDefault();
                panY += panSpeed;
                updateTransform();
            } else if (e.key === 'ArrowDown') {
                e.preventDefault();
                panY -= panSpeed;
                updateTransform();
            }
        });
        </script>"""
    
    def _generate_metrics_section(self, metrics: Dict[str, int], ai_awareness: bool) -> str:
        """Generate the metrics section of the HTML."""
        ai_section = ""
        if ai_awareness:
            ai_section = f"""
            <div class="metric">
                <div class="metric-value">🤖</div>
                <div class="metric-label">AI Analysis</div>
            </div>"""
        
        return f"""<div class="metrics">
            <div class="metric">
                <div class="metric-value">{metrics['total_modules']}</div>
                <div class="metric-label">Total Modules</div>
            </div>
            <div class="metric">
                <div class="metric-value">{metrics['total_dependencies']}</div>
                <div class="metric-label">Dependencies</div>
            </div>
            <div class="metric">
                <div class="metric-value">{metrics['external_deps']}</div>
                <div class="metric-label">External</div>
            </div>
            <div class="metric">
                <div class="metric-value">{metrics['internal_deps']}</div>
                <div class="metric-label">Internal</div>
            </div>{ai_section}
        </div>"""
    
    def _generate_legend_section(self, ai_awareness: bool) -> str:
        """Generate the legend section."""
        ai_legend = ""
        if ai_awareness:
            ai_legend = """
            <h3>AI Context Health</h3>
            <ul>
                <li>✅ Good (&lt; 2K tokens)</li>
                <li>⚠️ Warning (2K-4K tokens)</li>
                <li>🚨 Critical (&gt; 4K tokens)</li>
            </ul>"""
        
        return f"""<div class="legend">
            <h3>Legend</h3>
            <ul>
                <li>🟢 Low Risk Dependencies</li>
                <li>🟡 Medium Risk Dependencies</li>
                <li>🔴 High Risk Dependencies</li>
            </ul>{ai_legend}
        </div>"""
    
    def _generate_details_section(self) -> str:
        """Generate the details section."""
        return """<div class="details">
            <h3>Analysis Details</h3>
            <p>This graph shows the dependency relationships in your project. Click on nodes to explore connections.</p>
        </div>"""


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
    # AI-specific fields
    ai_context_tokens: Optional[int] = None
    ai_context_health: Optional[str] = None  # 'GOOD', 'WARNING', 'CRITICAL'
    pattern_consistency_score: Optional[float] = None


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

    def __init__(self, project_path: str, ai_awareness: bool = False):
        self.project_path = Path(project_path).resolve()
        self.ai_awareness = ai_awareness
        self.console = Console()
        if NETWORKX_AVAILABLE:
            self.graph = nx.DiGraph()
        else:
            # Fallback simple graph representation
            self.graph = defaultdict(list)
        self.nodes = {}
        self.external_deps = defaultdict(list)

    def analyze_project(self, exclude_tests: bool = False, include_tests_only: bool = False) -> DependencyGraph:
        """Analyze the entire project and build dependency graph."""
        filter_desc = ""
        if exclude_tests:
            filter_desc = " (excluding tests)"
        elif include_tests_only:
            filter_desc = " (tests only)"
            
        self.console.print(f"[bold blue]Analyzing project:[/bold blue] {self.project_path}{filter_desc}")

        # Find all Python files
        python_files = self._find_python_files(exclude_tests=exclude_tests, include_tests_only=include_tests_only)
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

    def _find_python_files(self, exclude_tests: bool = False, include_tests_only: bool = False) -> List[Path]:
        """Find all Python files in the project with optional test filtering."""
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
            
            # Apply test directory filtering
            if exclude_tests:
                dirs[:] = [d for d in dirs if not self._is_test_directory(d)]
            elif include_tests_only:
                dirs[:] = [d for d in dirs if self._is_test_directory(d) or any(self._is_test_file(f) for f in files)]

            for file in files:
                if file.endswith(".py") and not file.startswith("."):
                    file_path = Path(root) / file
                    
                    # Apply file-level test filtering
                    if exclude_tests and self._is_test_file(file):
                        continue
                    elif include_tests_only and not self._is_test_file(file):
                        continue
                        
                    python_files.append(file_path)

        return python_files
    
    def _is_test_directory(self, dirname: str) -> bool:
        """Check if a directory name indicates it contains tests."""
        test_patterns = {'test', 'tests', 'testing', '__tests__', 'spec', 'specs'}
        return dirname.lower() in test_patterns
    
    def _is_test_file(self, filename: str) -> bool:
        """Check if a filename indicates it's a test file."""
        filename_lower = filename.lower()
        return (
            filename_lower.startswith('test_') or
            filename_lower.endswith('_test.py') or
            filename_lower.startswith('spec_') or
            filename_lower.endswith('_spec.py') or
            'test' in filename_lower and (
                'conftest.py' in filename_lower or
                'test_' in filename_lower or
                '_test' in filename_lower
            )
        )

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

            # AI-specific analysis
            ai_context_tokens = None
            ai_context_health = None
            pattern_consistency_score = None
            
            if self.ai_awareness:
                ai_context_tokens = estimate_token_count(file_path)
                ai_context_health = get_ai_context_health(ai_context_tokens)
                # Placeholder for pattern consistency analysis
                pattern_consistency_score = 0.85  # Would implement actual analysis
            
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
                ai_context_tokens=ai_context_tokens,
                ai_context_health=ai_context_health,
                pattern_consistency_score=pattern_consistency_score,
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
        
        # Initialize helper classes
        self.mermaid_renderer = MermaidRenderer(graph)
        self.html_generator = HTMLTemplateGenerator(graph)

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

    def generate_mermaid_graph(self, ai_awareness: bool = False) -> str:
        """Generate Mermaid syntax for dependency graph."""
        return self.mermaid_renderer.generate_graph_definition(ai_awareness)

    def save_mermaid_syntax(self, output_path: str, ai_awareness: bool = False):
        """Save just the Mermaid syntax to a .mmd file for use in GitHub, etc."""
        mermaid_syntax = self.generate_mermaid_graph(ai_awareness)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(mermaid_syntax)

        self.console.print(f"[green]Mermaid syntax saved to:[/green] {output_path}")
        self.console.print("[blue]💡 Tip:[/blue] You can include this in GitHub README with:")
        self.console.print(f"```mermaid\\n{mermaid_syntax[:100]}...\\n```")

    def generate_mermaid_html(self, output_path: str, ai_awareness: bool = False, title_suffix: str = ""):
        """Generate HTML file with Mermaid diagram."""
        mermaid_graph = self.generate_mermaid_graph(ai_awareness)
        html_content = self.html_generator.generate_mermaid_html(mermaid_graph, ai_awareness, title_suffix)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        self.console.print(f"[green]Mermaid HTML graph saved to:[/green] {output_path}")

    def generate_html_interactive(self, output_path: str):
        """Generate interactive HTML visualization - now uses Mermaid by default."""
        # Use Mermaid as the primary approach
        self.generate_mermaid_html(output_path)

    def generate_risk_heatmap(self, output_path: str):
        """Generate risk level heatmap."""
        # Prepare data
        nodes_data = []
        for name, node in self.graph.nodes.items():
            nodes_data.append({
                "name": name,
                "risk_level": getattr(node, 'risk_level', 'MEDIUM'),
                "imports_count": len(node.imports)
            })
        
        self.console.print(f"[green]Risk heatmap would be generated with {len(nodes_data)} nodes[/green]")
        # Note: This method would need additional implementation for actual heatmap generation

    def generate_summary_report(self) -> str:
        """Generate a summary report of the dependency analysis."""
        report_lines = []
        
        # Basic statistics
        total_modules = len(self.graph.nodes)
        
        report_lines.append(f"📊 Dependency Analysis Summary")
        report_lines.append(f"Total modules analyzed: {total_modules}")
        
        return "\n".join(report_lines)


def main():
    """Main CLI interface."""
    # Set up proper encoding for Windows console
    import sys
    import codecs
    if sys.platform == "win32":
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer)
    
    parser = argparse.ArgumentParser(description="Generate dependency visualization")
    parser.add_argument("project_path", help="Path to the project to analyze")
    parser.add_argument("--format", choices=["text", "html", "mermaid", "syntax"], 
                       default="html", help="Output format")
    parser.add_argument("--output", help="Output file path (default: output/visualizations/)")
    parser.add_argument("--ai-awareness", action="store_true", 
                       help="Enable AI context window analysis")
    parser.add_argument("--exclude-tests", action="store_true",
                       help="Exclude test files from the main dependency graph")
    parser.add_argument("--tests-only", action="store_true",
                       help="Generate a separate dependency graph for tests only")
    parser.add_argument("--include-external", action="store_true",
                       help="Include external dependencies in the visualization")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    import os
    output_dir = Path(args.project_path) / "output" / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create analyzer and generate graph(s)
    analyzer = DependencyAnalyzer(args.project_path, ai_awareness=args.ai_awareness)
    
    if args.tests_only:
        # Generate tests-only dependency graph
        graph = analyzer.analyze_project(include_tests_only=True)
        suffix = "_tests_only"
        title_suffix = " (Tests Only)"
    elif args.exclude_tests:
        # Generate main dependency graph without tests
        graph = analyzer.analyze_project(exclude_tests=True)
        suffix = "_no_tests"
        title_suffix = " (Excluding Tests)"
    else:
        # Generate complete dependency graph
        graph = analyzer.analyze_project()
        suffix = ""
        title_suffix = ""
    
    visualizer = DependencyVisualizer(graph)
    
    # Generate output based on format
    if args.format == "text":
        output = visualizer.generate_text_tree()
        if args.output:
            output_path = args.output
        else:
            output_path = output_dir / f"dependency_graph{suffix}.txt"
            
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Text dependency tree saved to: {output_path}")
    
    elif args.format in ["html", "mermaid"]:
        if args.output:
            output_path = args.output
        else:
            output_path = output_dir / f"dependency_graph{suffix}.html"
            
        visualizer.generate_mermaid_html(str(output_path), args.ai_awareness, title_suffix)
        print(f"Mermaid HTML graph saved to: {output_path}")
        
    elif args.format == "syntax":
        if args.output:
            output_path = args.output
        else:
            output_path = output_dir / f"dependency_graph{suffix}.mmd"
            
        visualizer.save_mermaid_syntax(str(output_path), args.ai_awareness)
        print(f"Mermaid syntax saved to: {output_path}")
    
    # If exclude-tests is used, also offer to generate tests-only graph
    if args.exclude_tests and not args.tests_only:
        print(f"\nTip: Run with --tests-only to generate a separate tests dependency graph")
        print(f"Command: {sys.argv[0]} {args.project_path} --tests-only")


if __name__ == "__main__":
    main()
