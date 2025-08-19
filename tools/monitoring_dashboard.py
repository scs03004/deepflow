#!/usr/bin/env python3
"""
Real-time Monitoring Dashboard
=============================

Live dependency health monitoring:
- Real-time dependency dashboard showing system health
- Performance impact tracking of dependency changes
- Usage analytics for dependency optimization
- Automated alerts for circular dependencies

Usage:
    python monitoring_dashboard.py --start /path/to/project
    python monitoring_dashboard.py --analyze /path/to/project
    python monitoring_dashboard.py --server --port 8080
"""

import os
import sys
import json
import time
import threading
import argparse
from pathlib import Path
from typing import List
from dataclasses import dataclass, asdict
from datetime import datetime
import psutil

try:
    from flask import Flask, render_template, jsonify
    from flask_socketio import SocketIO, emit
    from rich.console import Console
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Install with: pip install flask flask-socketio pandas plotly psutil")
    sys.exit(1)


@dataclass
class DependencyMetric:
    """Real-time dependency metric."""

    timestamp: datetime
    module_name: str
    import_count: int
    memory_usage: float
    cpu_usage: float
    error_count: int
    circular_deps: int


@dataclass
class SystemHealth:
    """System health metrics."""

    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    active_modules: int
    total_imports: int
    error_rate: float


class DependencyMonitor:
    """Real-time dependency monitoring system."""

    def __init__(self, project_path: str):
        self.project_path = Path(project_path).resolve()
        self.console = Console()
        self.metrics_history = []
        self.health_history = []
        self.is_monitoring = False
        self.monitoring_thread = None

        # Initialize Flask app for web dashboard
        self.app = Flask(__name__)
        # Use environment variable for secret key, fall back to generated key
        import secrets

        self.app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        self._setup_routes()

    def start_monitoring(self, interval: int = 10):
        """Start real-time monitoring."""
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, args=(interval,))
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()

        self.console.print(f"[green]✅ Monitoring started for:[/green] {self.project_path}")
        self.console.print(f"[blue]📊 Dashboard available at:[/blue] http://localhost:5000")

    def stop_monitoring(self):
        """Stop monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        self.console.print("[yellow]⏹️  Monitoring stopped[/yellow]")

    def _monitoring_loop(self, interval: int):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                health = self._collect_system_health()

                # Store metrics
                self.metrics_history.append(metrics)
                self.health_history.append(health)

                # Keep only last 100 entries
                if len(self.metrics_history) > 100:
                    self.metrics_history.pop(0)
                if len(self.health_history) > 100:
                    self.health_history.pop(0)

                # Emit to web dashboard
                self.socketio.emit(
                    "metrics_update",
                    {
                        "metrics": [asdict(m) for m in self.metrics_history[-10:]],
                        "health": [asdict(h) for h in self.health_history[-10:]],
                    },
                )

                # Check for alerts
                self._check_alerts(metrics, health)

                time.sleep(interval)

            except Exception as e:
                self.console.print(f"[red]Monitoring error: {e}[/red]")
                time.sleep(interval)

    def _collect_metrics(self) -> List[DependencyMetric]:
        """Collect dependency metrics."""
        metrics = []

        try:
            # Run dependency analysis
            from .dependency_visualizer import DependencyAnalyzer

            analyzer = DependencyAnalyzer(str(self.project_path))
            dep_graph = analyzer.analyze_project()

            # Create metrics for each module
            for name, node in dep_graph.nodes.items():
                metric = DependencyMetric(
                    timestamp=datetime.now(),
                    module_name=name,
                    import_count=len(node.imports),
                    memory_usage=0.0,  # Would need runtime analysis
                    cpu_usage=0.0,  # Would need runtime analysis
                    error_count=0,  # Would need error tracking
                    circular_deps=len([c for c in dep_graph.circular_dependencies if name in c]),
                )
                metrics.append(metric)

        except Exception as e:
            self.console.print(f"[yellow]Warning: Could not collect metrics: {e}[/yellow]")

        return metrics

    def _collect_system_health(self) -> SystemHealth:
        """Collect system health metrics."""
        return SystemHealth(
            timestamp=datetime.now(),
            cpu_percent=psutil.cpu_percent(interval=1),
            memory_percent=psutil.virtual_memory().percent,
            disk_usage=psutil.disk_usage(str(self.project_path)).percent,
            active_modules=len([f for f in self.project_path.rglob("*.py")]),
            total_imports=0,  # Would calculate from analysis
            error_rate=0.0,  # Would track from logs
        )

    def _check_alerts(self, metrics: List[DependencyMetric], health: SystemHealth):
        """Check for alert conditions."""
        alerts = []

        # High CPU usage
        if health.cpu_percent > 80:
            alerts.append(f"High CPU usage: {health.cpu_percent:.1f}%")

        # High memory usage
        if health.memory_percent > 85:
            alerts.append(f"High memory usage: {health.memory_percent:.1f}%")

        # Circular dependencies
        circular_count = sum(m.circular_deps for m in metrics)
        if circular_count > 0:
            alerts.append(f"Circular dependencies detected: {circular_count}")

        # Emit alerts
        if alerts:
            self.socketio.emit("alerts", {"alerts": alerts})
            for alert in alerts:
                self.console.print(f"[red]🚨 ALERT: {alert}[/red]")

    def _setup_routes(self):
        """Set up Flask routes for web dashboard."""

        @self.app.route("/")
        def dashboard():
            return render_template("dashboard.html")

        @self.app.route("/api/metrics")
        def get_metrics():
            return jsonify(
                {
                    "metrics": [asdict(m) for m in self.metrics_history],
                    "health": [asdict(h) for h in self.health_history],
                }
            )

        @self.app.route("/api/health")
        def get_health():
            if self.health_history:
                latest = self.health_history[-1]
                return jsonify(asdict(latest))
            return jsonify({"error": "No health data available"})

        @self.app.route("/api/dependency-graph")
        def get_dependency_graph():
            try:
                from .dependency_visualizer import DependencyAnalyzer

                analyzer = DependencyAnalyzer(str(self.project_path))
                dep_graph = analyzer.analyze_project()

                return jsonify(
                    {
                        "nodes": [asdict(node) for node in dep_graph.nodes.values()],
                        "edges": dep_graph.edges,
                        "metrics": dep_graph.metrics,
                        "circular_dependencies": dep_graph.circular_dependencies,
                    }
                )
            except Exception as e:
                return jsonify({"error": str(e)})

        @self.socketio.on("connect")
        def handle_connect():
            emit("status", {"msg": "Connected to dependency monitor"})

        @self.socketio.on("request_update")
        def handle_request_update():
            if self.metrics_history and self.health_history:
                emit(
                    "metrics_update",
                    {
                        "metrics": [asdict(m) for m in self.metrics_history[-10:]],
                        "health": [asdict(h) for h in self.health_history[-10:]],
                    },
                )

    def run_dashboard_server(self, host="127.0.0.1", port=5000, debug=False):
        """Run the web dashboard server."""
        # Create templates directory and template
        self._create_dashboard_template()

        self.console.print(f"[blue]🌐 Starting dashboard server at http://{host}:{port}[/blue]")
        self.socketio.run(self.app, host=host, port=port, debug=debug)

    def _create_dashboard_template(self):
        """Create the dashboard HTML template."""
        templates_dir = Path(__file__).parent / "templates"
        templates_dir.mkdir(exist_ok=True)

        template_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dependency Monitor Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }
        .widget {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .widget h3 {
            margin-top: 0;
            color: #333;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        .alert {
            background: #ff4757;
            color: white;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
        }
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        .status-good { background-color: #2ed573; }
        .status-warning { background-color: #ffa502; }
        .status-error { background-color: #ff4757; }
        #dependency-graph {
            height: 500px;
        }
        #performance-chart {
            height: 300px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Dependency Monitor Dashboard</h1>
        <p>Real-time monitoring of project dependencies and system health</p>
        <span id="connection-status">
            <span class="status-indicator status-good"></span>
            Connected
        </span>
    </div>

    <div class="dashboard-grid">
        <div class="widget">
            <h3>System Health</h3>
            <div>CPU: <span id="cpu-usage" class="metric-value">0%</span></div>
            <div>Memory: <span id="memory-usage" class="metric-value">0%</span></div>
            <div>Disk: <span id="disk-usage" class="metric-value">0%</span></div>
        </div>

        <div class="widget">
            <h3>Dependency Status</h3>
            <div>Active Modules: <span id="active-modules" class="metric-value">0</span></div>
            <div>Total Imports: <span id="total-imports" class="metric-value">0</span></div>
            <div>Circular Deps: <span id="circular-deps" class="metric-value">0</span></div>
        </div>
    </div>

    <div class="widget">
        <h3>Performance Trends</h3>
        <div id="performance-chart"></div>
    </div>

    <div class="widget">
        <h3>Dependency Graph</h3>
        <div id="dependency-graph"></div>
    </div>

    <div class="widget">
        <h3>Alerts</h3>
        <div id="alerts-container">
            <p>No alerts at this time.</p>
        </div>
    </div>

    <script>
        const socket = io();
        let performanceData = [];
        let dependencyData = null;

        socket.on('connect', function() {
            console.log('Connected to dependency monitor');
            socket.emit('request_update');
        });

        socket.on('metrics_update', function(data) {
            updateDashboard(data);
        });

        socket.on('alerts', function(data) {
            updateAlerts(data.alerts);
        });

        function updateDashboard(data) {
            const health = data.health;
            const metrics = data.metrics;

            if (health && health.length > 0) {
                const latest = health[health.length - 1];
                document.getElementById('cpu-usage').textContent = latest.cpu_percent.toFixed(1) + '%';
                document.getElementById('memory-usage').textContent = latest.memory_percent.toFixed(1) + '%';
                document.getElementById('disk-usage').textContent = latest.disk_usage.toFixed(1) + '%';
                document.getElementById('active-modules').textContent = latest.active_modules;
                document.getElementById('total-imports').textContent = latest.total_imports;
            }

            if (metrics && metrics.length > 0) {
                const circularDeps = metrics.reduce((sum, m) => sum + m.circular_deps, 0);
                document.getElementById('circular-deps').textContent = circularDeps;
                
                // Update performance chart
                updatePerformanceChart(health);
            }
        }

        function updatePerformanceChart(healthData) {
            if (!healthData || healthData.length === 0) return;

            const timestamps = healthData.map(h => h.timestamp);
            const cpuData = healthData.map(h => h.cpu_percent);
            const memoryData = healthData.map(h => h.memory_percent);

            const trace1 = {
                x: timestamps,
                y: cpuData,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'CPU %',
                line: { color: '#667eea' }
            };

            const trace2 = {
                x: timestamps,
                y: memoryData,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Memory %',
                line: { color: '#764ba2' }
            };

            const layout = {
                title: 'System Performance Over Time',
                xaxis: { title: 'Time' },
                yaxis: { title: 'Usage %', range: [0, 100] },
                height: 300
            };

            Plotly.newPlot('performance-chart', [trace1, trace2], layout, {responsive: true});
        }

        function updateAlerts(alerts) {
            const container = document.getElementById('alerts-container');
            
            if (alerts && alerts.length > 0) {
                container.innerHTML = alerts.map(alert => 
                    `<div class="alert">🚨 ${alert}</div>`
                ).join('');
            } else {
                container.innerHTML = '<p>No alerts at this time.</p>';
            }
        }

        // Load dependency graph
        fetch('/api/dependency-graph')
            .then(response => response.json())
            .then(data => {
                if (data.nodes) {
                    updateDependencyGraph(data);
                }
            })
            .catch(error => console.error('Error loading dependency graph:', error));

        function updateDependencyGraph(data) {
            // Simple network visualization
            const nodes = data.nodes.map((node, index) => ({
                x: Math.cos(index / data.nodes.length * 2 * Math.PI),
                y: Math.sin(index / data.nodes.length * 2 * Math.PI),
                text: node.name,
                mode: 'markers+text',
                marker: {
                    size: Math.max(5, node.lines_of_code / 20),
                    color: node.risk_level === 'HIGH' ? 'red' : 
                           node.risk_level === 'MEDIUM' ? 'orange' : 'green'
                }
            }));

            const layout = {
                title: 'Dependency Network',
                showlegend: false,
                xaxis: { showgrid: false, zeroline: false, showticklabels: false },
                yaxis: { showgrid: false, zeroline: false, showticklabels: false },
                height: 500
            };

            Plotly.newPlot('dependency-graph', nodes, layout, {responsive: true});
        }

        // Request updates every 10 seconds
        setInterval(() => {
            socket.emit('request_update');
        }, 10000);
    </script>
</body>
</html>"""

        template_file = templates_dir / "dashboard.html"
        with open(template_file, "w") as f:
            f.write(template_content)

    def generate_analysis_report(self) -> str:
        """Generate a comprehensive analysis report."""
        if not self.metrics_history or not self.health_history:
            return "No monitoring data available. Start monitoring first."

        report = []
        report.append("# Dependency Monitoring Analysis Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Project: {self.project_path.name}")
        report.append("")

        # Summary statistics
        latest_health = self.health_history[-1]
        avg_cpu = sum(h.cpu_percent for h in self.health_history) / len(self.health_history)
        avg_memory = sum(h.memory_percent for h in self.health_history) / len(self.health_history)

        report.append("## System Health Summary")
        report.append(f"- Current CPU Usage: {latest_health.cpu_percent:.1f}%")
        report.append(f"- Average CPU Usage: {avg_cpu:.1f}%")
        report.append(f"- Current Memory Usage: {latest_health.memory_percent:.1f}%")
        report.append(f"- Average Memory Usage: {avg_memory:.1f}%")
        report.append(f"- Disk Usage: {latest_health.disk_usage:.1f}%")
        report.append("")

        # Dependency analysis
        if self.metrics_history:
            total_circular = sum(m.circular_deps for m in self.metrics_history[-1])
            report.append("## Dependency Analysis")
            report.append(f"- Active Modules: {latest_health.active_modules}")
            report.append(f"- Circular Dependencies: {total_circular}")
            report.append("")

        # Recommendations
        report.append("## Recommendations")

        if avg_cpu > 70:
            report.append("- ⚠️ High average CPU usage detected. Consider optimizing performance.")

        if avg_memory > 80:
            report.append("- ⚠️ High memory usage detected. Review memory-intensive operations.")

        if total_circular > 0:
            report.append(
                "- 🔄 Circular dependencies found. Consider refactoring to remove cycles."
            )

        if avg_cpu < 50 and avg_memory < 70:
            report.append("- ✅ System performance looks good.")

        return "\n".join(report)


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Real-time dependency monitoring")
    parser.add_argument("--start", metavar="PROJECT_PATH", help="Start monitoring for project")
    parser.add_argument("--server", action="store_true", help="Start web dashboard server")
    parser.add_argument(
        "--port", type=int, default=5000, help="Dashboard server port (default: 5000)"
    )
    parser.add_argument(
        "--interval", type=int, default=10, help="Monitoring interval in seconds (default: 10)"
    )
    parser.add_argument("--analyze", metavar="PROJECT_PATH", help="Generate analysis report")

    args = parser.parse_args()

    if args.start:
        monitor = DependencyMonitor(args.start)

        try:
            if args.server:
                # Start monitoring in background
                monitor.start_monitoring(args.interval)
                # Run dashboard server
                monitor.run_dashboard_server(port=args.port)
            else:
                # Console monitoring only
                monitor.start_monitoring(args.interval)

                console = Console()

                with Live(console=console, refresh_per_second=1) as live:
                    try:
                        while monitor.is_monitoring:
                            # Create live dashboard
                            layout = Layout()

                            # System health table
                            health_table = Table(title="System Health")
                            health_table.add_column("Metric")
                            health_table.add_column("Value")

                            if monitor.health_history:
                                latest = monitor.health_history[-1]
                                health_table.add_row("CPU Usage", f"{latest.cpu_percent:.1f}%")
                                health_table.add_row(
                                    "Memory Usage", f"{latest.memory_percent:.1f}%"
                                )
                                health_table.add_row("Disk Usage", f"{latest.disk_usage:.1f}%")
                                health_table.add_row("Active Modules", str(latest.active_modules))

                            live.update(Panel(health_table, title="Dependency Monitor"))
                            time.sleep(1)

                    except KeyboardInterrupt:
                        monitor.stop_monitoring()
                        console.print("\n[yellow]Monitoring stopped by user[/yellow]")

        except KeyboardInterrupt:
            monitor.stop_monitoring()

        return

    if args.analyze:
        monitor = DependencyMonitor(args.analyze)
        # Would need to load historical data
        report = monitor.generate_analysis_report()
        print(report)
        return

    if args.server:
        # Server-only mode
        monitor = DependencyMonitor(".")
        monitor.run_dashboard_server(port=args.port)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
