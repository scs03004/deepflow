#!/usr/bin/env python3
"""
Documentation Automation Tool
============================

Auto-generated dependency documentation:
- Auto-generated dependency maps from code analysis
- Living architecture diagrams that update with changes
- Impact analysis reports for stakeholders
- Dependency changelogs showing evolution over time

Usage:
    python doc_generator.py /path/to/project
    python doc_generator.py /path/to/project --format markdown
    python doc_generator.py /path/to/project --output docs/
"""

import ast
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import importlib.util
import subprocess

try:
    import networkx as nx
    import pandas as pd
    from jinja2 import Template, Environment, FileSystemLoader
    from rich.console import Console
    from rich.progress import track
    import yaml
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Install with: pip install -r requirements.txt")
    sys.exit(1)

# Import our dependency visualizer
try:
    from .dependency_visualizer import DependencyAnalyzer, DependencyGraph
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.dirname(__file__))
    from dependency_visualizer import DependencyAnalyzer, DependencyGraph


@dataclass
class ProjectMetadata:
    """Project metadata for documentation."""

    name: str
    path: str
    language: str
    framework: Optional[str]
    version: Optional[str]
    description: Optional[str]
    total_files: int
    total_lines: int
    last_updated: str


@dataclass
class APIEndpoint:
    """API endpoint information."""

    path: str
    method: str
    function: str
    file_path: str
    dependencies: List[str]
    parameters: List[str]
    returns: Optional[str]


@dataclass
class DatabaseModel:
    """Database model information."""

    name: str
    file_path: str
    fields: List[str]
    relationships: List[str]
    dependencies: List[str]


class DocumentationGenerator:
    """Core documentation generation engine."""

    def __init__(self, project_path: str):
        self.project_path = Path(project_path).resolve()
        self.console = Console()
        self.project_metadata = None
        self.dependency_graph = None

        # Set up Jinja2 environment
        template_dir = Path(__file__).parent.parent / "templates"
        self.jinja_env = Environment(
            loader=FileSystemLoader([str(template_dir), str(Path(__file__).parent)]),
            autoescape=True,  # Fix Jinja2 XSS vulnerability
        )

    def generate_all_documentation(self, output_dir: str = None) -> Dict[str, str]:
        """Generate all documentation formats."""
        if output_dir is None:
            output_dir = self.project_path / "docs"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(exist_ok=True)

        self.console.print(
            f"[bold blue]Generating documentation for:[/bold blue] {self.project_path}"
        )

        # Analyze project
        self._analyze_project()

        generated_files = {}

        # Generate dependency map
        dep_map_path = output_dir / "DEPENDENCY_MAP.md"
        generated_files["dependency_map"] = str(dep_map_path)
        self._generate_dependency_map(dep_map_path)

        # Generate API documentation
        api_doc_path = output_dir / "API_DOCUMENTATION.md"
        generated_files["api_docs"] = str(api_doc_path)
        self._generate_api_documentation(api_doc_path)

        # Generate architecture overview
        arch_path = output_dir / "ARCHITECTURE.md"
        generated_files["architecture"] = str(arch_path)
        self._generate_architecture_overview(arch_path)

        # Generate change impact checklist
        checklist_path = output_dir / "CHANGE_IMPACT_CHECKLIST.md"
        generated_files["change_checklist"] = str(checklist_path)
        self._generate_change_checklist(checklist_path)

        # Generate metrics dashboard
        metrics_path = output_dir / "project_metrics.json"
        generated_files["metrics"] = str(metrics_path)
        self._generate_metrics_dashboard(metrics_path)

        self.console.print(f"[green]✅ Documentation generated in:[/green] {output_dir}")
        return generated_files

    def _analyze_project(self):
        """Analyze the project structure and dependencies."""
        self.console.print("Analyzing project structure...")

        # Get project metadata
        self.project_metadata = self._extract_project_metadata()

        # Analyze dependencies
        analyzer = DependencyAnalyzer(str(self.project_path))
        self.dependency_graph = analyzer.analyze_project()

    def _extract_project_metadata(self) -> ProjectMetadata:
        """Extract project metadata."""
        # Try to get project name from various sources
        project_name = self.project_path.name

        # Check for setup.py, pyproject.toml, package.json, etc.
        description = None
        version = None
        framework = None

        # Check setup.py
        setup_py = self.project_path / "setup.py"
        if setup_py.exists():
            try:
                with open(setup_py, "r") as f:
                    content = f.read()
                    # Simple regex-based extraction (could be improved)
                    if "name=" in content:
                        import re

                        match = re.search(r'name=[\'"]([^\'"]+)[\'"]', content)
                        if match:
                            project_name = match.group(1)
            except Exception:
                pass

        # Check pyproject.toml
        pyproject = self.project_path / "pyproject.toml"
        if pyproject.exists():
            try:
                import toml

                with open(pyproject, "r") as f:
                    data = toml.load(f)
                    if "tool" in data and "poetry" in data["tool"]:
                        poetry = data["tool"]["poetry"]
                        project_name = poetry.get("name", project_name)
                        description = poetry.get("description")
                        version = poetry.get("version")
            except Exception:
                pass

        # Check for common frameworks
        if (self.project_path / "main.py").exists():
            # Check for FastAPI, Flask, Django, etc.
            framework = self._detect_framework()

        # Count files and lines
        total_files = 0
        total_lines = 0

        for file_path in self.project_path.rglob("*.py"):
            if not any(part.startswith(".") for part in file_path.parts):
                total_files += 1
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        total_lines += len(f.readlines())
                except Exception:
                    pass

        return ProjectMetadata(
            name=project_name,
            path=str(self.project_path),
            language="Python",
            framework=framework,
            version=version,
            description=description,
            total_files=total_files,
            total_lines=total_lines,
            last_updated=datetime.now().isoformat(),
        )

    def _detect_framework(self) -> Optional[str]:
        """Detect the web framework being used."""
        frameworks = {
            "fastapi": "FastAPI",
            "flask": "Flask",
            "django": "Django",
            "tornado": "Tornado",
            "aiohttp": "aiohttp",
            "starlette": "Starlette",
        }

        # Check requirements files
        req_files = ["requirements.txt", "pyproject.toml", "setup.py"]

        for req_file in req_files:
            req_path = self.project_path / req_file
            if req_path.exists():
                try:
                    with open(req_path, "r") as f:
                        content = f.read().lower()
                        for fw_name, fw_display in frameworks.items():
                            if fw_name in content:
                                return fw_display
                except Exception:
                    pass

        # Check imports in main files
        main_files = ["main.py", "app.py", "server.py"]

        for main_file in main_files:
            main_path = self.project_path / main_file
            if main_path.exists():
                try:
                    with open(main_path, "r") as f:
                        content = f.read().lower()
                        for fw_name, fw_display in frameworks.items():
                            if f"import {fw_name}" in content or f"from {fw_name}" in content:
                                return fw_display
                except Exception:
                    pass

        return None

    def _generate_dependency_map(self, output_path: Path):
        """Generate comprehensive dependency map."""
        template = self.jinja_env.get_template("dependency_map_template.md")

        # Prepare template data
        template_data = {
            "project_name": self.project_metadata.name,
            "generated_date": datetime.now().strftime("%Y-%m-%d"),
            "project_metadata": self.project_metadata,
            "dependency_graph": self.dependency_graph,
            "high_risk_files": [
                name
                for name, node in self.dependency_graph.nodes.items()
                if node.risk_level == "HIGH"
            ],
            "circular_dependencies": self.dependency_graph.circular_dependencies,
            "external_dependencies": self.dependency_graph.external_dependencies,
            "metrics": self.dependency_graph.metrics,
        }

        # Render template
        rendered = template.render(**template_data)

        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(rendered)

        self.console.print(f"[green]✅ Dependency map generated:[/green] {output_path}")

    def _generate_api_documentation(self, output_path: Path):
        """Generate API documentation."""
        api_endpoints = self._extract_api_endpoints()

        template_content = """# API Documentation

**Project**: {{ project_name }}
**Generated**: {{ generated_date }}
**Framework**: {{ framework or "Unknown" }}

## Overview

This document provides comprehensive API documentation for {{ project_name }}.

### Base URL
- **Development**: `http://localhost:8000`
- **Production**: `https://your-domain.com`

## Endpoints

{% for endpoint in endpoints %}
### {{ endpoint.method }} {{ endpoint.path }}

**Handler**: `{{ endpoint.function }}` in `{{ endpoint.file_path }}`

{% if endpoint.parameters %}
**Parameters**:
{% for param in endpoint.parameters %}
- `{{ param }}`
{% endfor %}
{% endif %}

**Dependencies**:
{% for dep in endpoint.dependencies %}
- {{ dep }}
{% endfor %}

{% if endpoint.returns %}
**Returns**: {{ endpoint.returns }}
{% endif %}

---

{% endfor %}

## Error Handling

Common HTTP status codes used:
- `200 OK` - Successful request
- `400 Bad Request` - Invalid request parameters
- `401 Unauthorized` - Authentication required
- `404 Not Found` - Resource not found
- `500 Internal Server Error` - Server error

## Rate Limiting

API requests may be rate-limited. Check response headers:
- `X-RateLimit-Limit` - Maximum requests per window
- `X-RateLimit-Remaining` - Remaining requests in window
- `X-RateLimit-Reset` - Window reset time

## Authentication

{% if 'auth' in project_name.lower() or 'api_key' in ' '.join(dependencies) %}
This API uses authentication. Include your API key in requests:

```bash
curl -H "Authorization: Bearer YOUR_API_KEY" \\
     https://api.example.com/endpoint
```
{% else %}
Authentication details not detected in code analysis.
{% endif %}

---

*Generated automatically by Dependency Toolkit*
"""

        template = Template(template_content)
        rendered = template.render(
            project_name=self.project_metadata.name,
            generated_date=datetime.now().strftime("%Y-%m-%d"),
            framework=self.project_metadata.framework,
            endpoints=api_endpoints,
            dependencies=list(self.dependency_graph.external_dependencies.keys()),
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(rendered)

        self.console.print(f"[green]✅ API documentation generated:[/green] {output_path}")

    def _extract_api_endpoints(self) -> List[APIEndpoint]:
        """Extract API endpoints from code."""
        endpoints = []

        # Look for API route files
        api_files = []
        for file_path in self.project_path.rglob("*.py"):
            if any(
                keyword in str(file_path).lower()
                for keyword in ["route", "api", "endpoint", "view"]
            ):
                api_files.append(file_path)

        for file_path in api_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Parse AST to find route decorators
                tree = ast.parse(content, filename=str(file_path))

                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Check for route decorators
                        route_info = self._extract_route_info(node)
                        if route_info:
                            endpoints.append(
                                APIEndpoint(
                                    path=route_info["path"],
                                    method=route_info["method"],
                                    function=node.name,
                                    file_path=str(file_path.relative_to(self.project_path)),
                                    dependencies=self._get_function_dependencies(node),
                                    parameters=self._get_function_parameters(node),
                                    returns=self._get_return_type(node),
                                )
                            )

            except Exception as e:
                self.console.print(f"[yellow]Warning: Could not parse {file_path}: {e}[/yellow]")

        return endpoints

    def _extract_route_info(self, func_node: ast.FunctionDef) -> Optional[Dict]:
        """Extract route information from function decorators."""
        for decorator in func_node.decorator_list:
            if isinstance(decorator, ast.Call):
                if isinstance(decorator.func, ast.Attribute):
                    # @app.get(), @router.post(), etc.
                    method = decorator.func.attr.upper()
                    if decorator.args and isinstance(decorator.args[0], ast.Str):
                        path = decorator.args[0].s
                        return {"path": path, "method": method}
                elif isinstance(decorator.func, ast.Name):
                    # @get(), @post(), etc.
                    method = decorator.func.id.upper()
                    if decorator.args and isinstance(decorator.args[0], ast.Str):
                        path = decorator.args[0].s
                        return {"path": path, "method": method}

        return None

    def _get_function_dependencies(self, func_node: ast.FunctionDef) -> List[str]:
        """Get dependencies used within a function."""
        dependencies = []

        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    dependencies.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    dependencies.append(node.func.attr)

        return list(set(dependencies))

    def _get_function_parameters(self, func_node: ast.FunctionDef) -> List[str]:
        """Get function parameters."""
        params = []
        for arg in func_node.args.args:
            if arg.arg != "self":
                params.append(arg.arg)
        return params

    def _get_return_type(self, func_node: ast.FunctionDef) -> Optional[str]:
        """Get function return type annotation."""
        if func_node.returns:
            if isinstance(func_node.returns, ast.Name):
                return func_node.returns.id
            elif isinstance(func_node.returns, ast.Constant):
                return str(func_node.returns.value)
        return None

    def _generate_architecture_overview(self, output_path: Path):
        """Generate architecture overview documentation."""
        template_content = """# Architecture Overview

**Project**: {{ project_name }}
**Generated**: {{ generated_date }}
**Language**: {{ language }}
**Framework**: {{ framework or "Not detected" }}

## Project Structure

```
{{ project_name }}/
{% for component in components %}
├── {{ component.path }}{% if component.description %} - {{ component.description }}{% endif %}
{% endfor %}
```

## Component Overview

### Core Components

{% for component in core_components %}
#### {{ component.name }}
- **Path**: `{{ component.path }}`
- **Purpose**: {{ component.description }}
- **Dependencies**: {{ component.dependencies|join(', ') if component.dependencies else 'None' }}
- **Risk Level**: {{ component.risk_level }}

{% endfor %}

## Dependency Flow

### High-Level Architecture

```
{{ architecture_diagram }}
```

### Data Flow

1. **User Request** → Web Interface
2. **Web Interface** → API Layer  
3. **API Layer** → Business Logic
4. **Business Logic** → Database Layer
5. **Database Layer** → Response

## Technology Stack

### Backend
{% for tech in backend_tech %}
- **{{ tech.name }}**: {{ tech.purpose }}
{% endfor %}

### Frontend
{% for tech in frontend_tech %}
- **{{ tech.name }}**: {{ tech.purpose }}
{% endfor %}

### Database
{% for tech in database_tech %}
- **{{ tech.name }}**: {{ tech.purpose }}
{% endfor %}

## Security Considerations

{% if security_features %}
### Implemented Security Features
{% for feature in security_features %}
- {{ feature }}
{% endfor %}
{% else %}
Security features analysis not available from code inspection.
{% endif %}

## Performance Considerations

- **Total Files**: {{ total_files }}
- **Total Lines of Code**: {{ total_lines }}
- **External Dependencies**: {{ external_deps_count }}
- **Circular Dependencies**: {{ circular_deps_count }}

## Deployment Architecture

{% if deployment_files %}
### Deployment Configuration
{% for file in deployment_files %}
- `{{ file.name }}`: {{ file.purpose }}
{% endfor %}
{% endif %}

## Monitoring and Logging

{% if monitoring_detected %}
- Monitoring capabilities detected in codebase
- Health check endpoints available
{% else %}
- No explicit monitoring configuration detected
{% endif %}

---

*Generated automatically by Dependency Toolkit*
"""

        # Analyze components
        components = self._analyze_components()

        template = Template(template_content)
        rendered = template.render(
            project_name=self.project_metadata.name,
            generated_date=datetime.now().strftime("%Y-%m-%d"),
            language=self.project_metadata.language,
            framework=self.project_metadata.framework,
            components=components["all"],
            core_components=components["core"],
            architecture_diagram=self._generate_ascii_architecture(),
            backend_tech=self._detect_backend_tech(),
            frontend_tech=self._detect_frontend_tech(),
            database_tech=self._detect_database_tech(),
            security_features=self._detect_security_features(),
            total_files=self.project_metadata.total_files,
            total_lines=self.project_metadata.total_lines,
            external_deps_count=len(self.dependency_graph.external_dependencies),
            circular_deps_count=len(self.dependency_graph.circular_dependencies),
            deployment_files=self._detect_deployment_files(),
            monitoring_detected=self._detect_monitoring(),
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(rendered)

        self.console.print(f"[green]✅ Architecture overview generated:[/green] {output_path}")

    def _analyze_components(self) -> Dict[str, List]:
        """Analyze project components."""
        all_components = []
        core_components = []

        # Analyze directory structure
        for item in self.project_path.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                component = {
                    "name": item.name,
                    "path": item.name + "/",
                    "description": self._get_component_description(item.name),
                    "dependencies": [],
                    "risk_level": "LOW",
                }

                all_components.append(component)

                # Identify core components
                if item.name in ["api", "models", "web", "core", "src"]:
                    component["risk_level"] = "HIGH"
                    core_components.append(component)

        # Add important files
        important_files = ["main.py", "app.py", "config.py", "requirements.txt"]
        for file_name in important_files:
            file_path = self.project_path / file_name
            if file_path.exists():
                all_components.append(
                    {
                        "name": file_name,
                        "path": file_name,
                        "description": self._get_file_description(file_name),
                        "dependencies": [],
                        "risk_level": "HIGH" if file_name in ["main.py", "config.py"] else "MEDIUM",
                    }
                )

        return {"all": all_components, "core": core_components}

    def _get_component_description(self, component_name: str) -> str:
        """Get description for a component directory."""
        descriptions = {
            "api": "API endpoints and route handlers",
            "models": "Database models and schemas",
            "web": "Web interface and static files",
            "static": "Static assets (CSS, JS, images)",
            "templates": "HTML templates",
            "tests": "Test suite and fixtures",
            "docs": "Project documentation",
            "scripts": "Utility and deployment scripts",
            "config": "Configuration files",
            "utils": "Utility functions and helpers",
            "core": "Core business logic",
            "services": "Service layer components",
            "database": "Database configuration and migrations",
        }

        return descriptions.get(component_name, "Project component")

    def _get_file_description(self, file_name: str) -> str:
        """Get description for important files."""
        descriptions = {
            "main.py": "Application entry point",
            "app.py": "Application factory/configuration",
            "config.py": "Configuration settings",
            "requirements.txt": "Python dependencies",
            "setup.py": "Package setup configuration",
            "Dockerfile": "Docker container configuration",
            "docker-compose.yml": "Docker compose configuration",
        }

        return descriptions.get(file_name, "Project file")

    def _generate_ascii_architecture(self) -> str:
        """Generate simple ASCII architecture diagram."""
        return """
User/Browser
     ↓
┌─────────────┐
│ Web Interface│
└─────────────┘
     ↓
┌─────────────┐
│  API Layer  │
└─────────────┘
     ↓
┌─────────────┐
│   Models    │
└─────────────┘
     ↓
┌─────────────┐
│  Database   │
└─────────────┘
"""

    def _detect_backend_tech(self) -> List[Dict]:
        """Detect backend technologies."""
        tech_stack = []

        external_deps = self.dependency_graph.external_dependencies

        tech_mapping = {
            "fastapi": {"name": "FastAPI", "purpose": "Modern web framework"},
            "flask": {"name": "Flask", "purpose": "Lightweight web framework"},
            "django": {"name": "Django", "purpose": "Full-featured web framework"},
            "sqlalchemy": {"name": "SQLAlchemy", "purpose": "Database ORM"},
            "pydantic": {"name": "Pydantic", "purpose": "Data validation"},
            "uvicorn": {"name": "Uvicorn", "purpose": "ASGI web server"},
            "gunicorn": {"name": "Gunicorn", "purpose": "WSGI web server"},
        }

        for dep in external_deps:
            if dep.lower() in tech_mapping:
                tech_stack.append(tech_mapping[dep.lower()])

        return tech_stack

    def _detect_frontend_tech(self) -> List[Dict]:
        """Detect frontend technologies."""
        tech_stack = []

        # Check for static files
        web_dir = self.project_path / "web"
        static_dir = self.project_path / "static"

        if web_dir.exists() or static_dir.exists():
            tech_stack.append({"name": "HTML/CSS/JavaScript", "purpose": "Web interface"})

        # Check for specific frontend frameworks in package files
        if (self.project_path / "package.json").exists():
            tech_stack.append({"name": "Node.js", "purpose": "JavaScript runtime"})

        return tech_stack

    def _detect_database_tech(self) -> List[Dict]:
        """Detect database technologies."""
        tech_stack = []

        external_deps = self.dependency_graph.external_dependencies

        db_mapping = {
            "sqlite3": {"name": "SQLite", "purpose": "Embedded database"},
            "aiosqlite": {"name": "SQLite (Async)", "purpose": "Async SQLite driver"},
            "psycopg2": {"name": "PostgreSQL", "purpose": "PostgreSQL database"},
            "asyncpg": {"name": "PostgreSQL (Async)", "purpose": "Async PostgreSQL driver"},
            "pymongo": {"name": "MongoDB", "purpose": "Document database"},
            "redis": {"name": "Redis", "purpose": "In-memory data store"},
        }

        for dep in external_deps:
            if dep.lower() in db_mapping:
                tech_stack.append(db_mapping[dep.lower()])

        return tech_stack

    def _detect_security_features(self) -> List[str]:
        """Detect security features."""
        features = []

        external_deps = self.dependency_graph.external_dependencies

        if "slowapi" in external_deps:
            features.append("Rate limiting (slowapi)")

        if "passlib" in external_deps or "bcrypt" in external_deps:
            features.append("Password hashing")

        if "jwt" in external_deps or "pyjwt" in external_deps:
            features.append("JWT authentication")

        if "cryptography" in external_deps:
            features.append("Cryptographic operations")

        return features

    def _detect_deployment_files(self) -> List[Dict]:
        """Detect deployment configuration files."""
        deployment_files = []

        files = {
            "Dockerfile": "Docker container configuration",
            "docker-compose.yml": "Docker compose orchestration",
            "requirements.txt": "Python dependencies",
            "pyproject.toml": "Python project configuration",
            ".env": "Environment variables",
            "nginx.conf": "Nginx web server configuration",
        }

        for file_name, purpose in files.items():
            if (self.project_path / file_name).exists():
                deployment_files.append({"name": file_name, "purpose": purpose})

        return deployment_files

    def _detect_monitoring(self) -> bool:
        """Detect monitoring capabilities."""
        # Check for health endpoints or monitoring code
        for file_path in self.project_path.rglob("*.py"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().lower()
                    if "health" in content or "monitor" in content or "metrics" in content:
                        return True
            except Exception:
                pass

        return False

    def _generate_change_checklist(self, output_path: Path):
        """Generate customized change impact checklist."""
        # Read the template and customize it for this project
        template_path = Path(__file__).parent.parent / "templates" / "CHANGE_IMPACT_CHECKLIST.md"

        if template_path.exists():
            with open(template_path, "r", encoding="utf-8") as f:
                template_content = f.read()

            # Customize template for this project
            customized_content = template_content.replace(
                "Project Change Impact Checklist Template",
                f"{self.project_metadata.name} Change Impact Checklist",
            )

            # Add project-specific test commands if detected
            if (self.project_path / "pytest.ini").exists() or any(
                "pytest" in dep for dep in self.dependency_graph.external_dependencies
            ):
                customized_content = customized_content.replace(
                    "pytest tests/test_basic.py", "pytest"
                )

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(customized_content)
        else:
            # Fallback: create basic checklist
            basic_checklist = f"""# {self.project_metadata.name} Change Impact Checklist

Generated automatically for {self.project_metadata.name}.

## Before Making Changes

1. **Identify change type**
2. **Search for references**
3. **Check dependency impact**

## Testing Requirements

- Run test suite: `pytest`
- Check all components work
- Verify documentation is up to date

## High-Risk Files

{chr(10).join(f"- {name}" for name, node in self.dependency_graph.nodes.items() if node.risk_level == 'HIGH')}

---

*Generated automatically by Dependency Toolkit*
"""

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(basic_checklist)

        self.console.print(f"[green]✅ Change checklist generated:[/green] {output_path}")

    def _generate_metrics_dashboard(self, output_path: Path):
        """Generate project metrics in JSON format."""
        metrics = {
            "project": {
                "name": self.project_metadata.name,
                "language": self.project_metadata.language,
                "framework": self.project_metadata.framework,
                "total_files": self.project_metadata.total_files,
                "total_lines": self.project_metadata.total_lines,
                "last_updated": self.project_metadata.last_updated,
            },
            "dependencies": {
                "total_imports": self.dependency_graph.metrics.get("total_imports", 0),
                "external_dependencies": len(self.dependency_graph.external_dependencies),
                "circular_dependencies": len(self.dependency_graph.circular_dependencies),
                "high_risk_files": self.dependency_graph.metrics.get("high_risk_files", 0),
            },
            "external_packages": list(self.dependency_graph.external_dependencies.keys()),
            "risk_assessment": {
                "high_risk_files": [
                    name
                    for name, node in self.dependency_graph.nodes.items()
                    if node.risk_level == "HIGH"
                ],
                "circular_dependencies": self.dependency_graph.circular_dependencies,
            },
            "generated_timestamp": datetime.now().isoformat(),
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)

        self.console.print(f"[green]✅ Metrics dashboard generated:[/green] {output_path}")


# Create Jinja2 template for dependency map
DEPENDENCY_MAP_TEMPLATE = """# {{ project_name }} Dependency Map

**Generated**: {{ generated_date }}
**Last Updated**: {{ generated_date }}

## 🏗️ Architecture Overview

### Project Metadata
- **Name**: {{ project_metadata.name }}
- **Language**: {{ project_metadata.language }}
- **Framework**: {{ project_metadata.framework or "Not detected" }}
- **Total Files**: {{ project_metadata.total_files }}
- **Total Lines**: {{ project_metadata.total_lines }}

## 📁 File Dependencies

### High-Risk Files (Changes Affect Many Components)
| File | Risk Level | Lines of Code | Dependencies | Dependents |
|------|------------|---------------|--------------|------------|
{% for file_name in high_risk_files %}
{%- set node = dependency_graph.nodes[file_name] %}
| {{ node.file_path }} | {{ node.risk_level }} | {{ node.lines_of_code }} | {{ node.imports|length }} | {{ node.imported_by|length }} |
{% endfor %}

## 🔗 Import Dependency Graph

### Internal Dependencies
{% for name, node in dependency_graph.nodes.items() %}
**{{ name }}** (`{{ node.file_path }}`)
{% for imported in node.imports %}
  ↳ {{ imported }}
{% endfor %}

{% endfor %}

### External Dependencies
{% for dep, users in external_dependencies.items() %}
- **{{ dep }}**: Used by {{ users|length }} modules
{% endfor %}

{% if circular_dependencies %}
## ⚠️ Circular Dependencies

{% for cycle in circular_dependencies %}
**Cycle {{ loop.index }}**: {{ cycle|join(' → ') }} → {{ cycle[0] }}
{% endfor %}
{% endif %}

## 📈 Metrics

- **Total Python Files**: {{ metrics.total_files }}
- **Total Import Statements**: {{ metrics.total_imports }}
- **External Dependencies**: {{ metrics.external_dependencies }}
- **High-Risk Files**: {{ metrics.high_risk_files }}
- **Circular Dependencies**: {{ metrics.circular_dependencies }}
- **Total Lines of Code**: {{ metrics.total_lines_of_code }}

---

*Generated automatically by Dependency Toolkit on {{ generated_date }}*
"""


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

    parser = argparse.ArgumentParser(description="Generate project documentation")
    parser.add_argument("project_path", help="Path to the project to document")
    parser.add_argument(
        "--format", choices=["markdown", "html", "json"], default="markdown", help="Output format"
    )
    parser.add_argument("--output", help="Output directory (default: project_path/docs)")
    parser.add_argument(
        "--type",
        choices=["all", "dependency-map", "api", "architecture"],
        default="all",
        help="Documentation type to generate",
    )

    args = parser.parse_args()

    # Initialize generator
    generator = DocumentationGenerator(args.project_path)

    try:
        # Generate documentation
        if args.type == "all":
            generated_files = generator.generate_all_documentation(args.output)
            print(f"\n✅ Generated {len(generated_files)} documentation files:")
            for doc_type, file_path in generated_files.items():
                print(f"  • {doc_type}: {file_path}")
        else:
            # Generate specific documentation type
            output_dir = Path(args.output) if args.output else Path(args.project_path) / "docs"
            output_dir.mkdir(exist_ok=True)

            if args.type == "dependency-map":
                generator._analyze_project()
                generator._generate_dependency_map(output_dir / "DEPENDENCY_MAP.md")
            elif args.type == "api":
                generator._analyze_project()
                generator._generate_api_documentation(output_dir / "API_DOCUMENTATION.md")
            elif args.type == "architecture":
                generator._analyze_project()
                generator._generate_architecture_overview(output_dir / "ARCHITECTURE.md")

    except Exception as e:
        print(f"Error generating documentation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Create the dependency map template file if running directly
    template_dir = Path(__file__).parent.parent / "templates"
    template_dir.mkdir(exist_ok=True)

    template_file = template_dir / "dependency_map_template.md"
    if not template_file.exists():
        with open(template_file, "w") as f:
            f.write(DEPENDENCY_MAP_TEMPLATE)

    main()
