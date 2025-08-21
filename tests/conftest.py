"""
Pytest configuration and shared fixtures for deepflow tests.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any
from unittest.mock import MagicMock, patch
import pytest

# Add tools and deepflow to Python path for testing
project_root = Path(__file__).parent.parent
tools_dir = project_root / "tools"
deepflow_dir = project_root / "deepflow"

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(tools_dir))


@pytest.fixture
def mock_project_structure():
    """Create a temporary project structure for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        project_path = Path(tmpdir)
        
        # Create Python files with dependencies
        (project_path / "main.py").write_text("""
import os
import sys
import json
from pathlib import Path
from datetime import datetime
import unused_import

def main():
    print("Hello, world!")
    return json.dumps({"status": "ok"})

if __name__ == "__main__":
    main()
""")
        
        (project_path / "utils.py").write_text("""
import json
import requests
from typing import Dict, List

def process_data(data: Dict) -> List:
    return list(data.values())

def fetch_data(url: str) -> Dict:
    response = requests.get(url)
    return response.json()
""")
        
        (project_path / "models").mkdir(exist_ok=True)
        (project_path / "models" / "__init__.py").write_text("")
        
        (project_path / "models" / "user.py").write_text("""
from dataclasses import dataclass
from typing import Optional

@dataclass
class User:
    id: int
    name: str
    email: Optional[str] = None
""")
        
        # Create requirements.txt
        (project_path / "requirements.txt").write_text("""
requests>=2.25.0
numpy>=1.20.0
pandas>=1.3.0
""")
        
        # Create pyproject.toml
        (project_path / "pyproject.toml").write_text("""
[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "test-project"
version = "1.0.0"
""")
        
        yield project_path


@pytest.fixture
def sample_dependency_graph():
    """Sample dependency graph data for testing."""
    return {
        "nodes": [
            {"id": "main.py", "label": "main", "type": "module", "token_count": 150},
            {"id": "utils.py", "label": "utils", "type": "module", "token_count": 200},
            {"id": "models/user.py", "label": "user", "type": "module", "token_count": 100},
            {"id": "json", "label": "json", "type": "stdlib", "token_count": 0},
            {"id": "requests", "label": "requests", "type": "external", "token_count": 0},
        ],
        "edges": [
            {"source": "main.py", "target": "json", "type": "import"},
            {"source": "utils.py", "target": "json", "type": "import"},
            {"source": "utils.py", "target": "requests", "type": "import"},
            {"source": "models/user.py", "target": "dataclasses", "type": "import"},
        ]
    }


@pytest.fixture
def mock_import_analysis():
    """Mock import analysis results."""
    return [
        {
            "file_path": "main.py",
            "import_name": "unused_import",
            "import_type": "import",
            "is_used": False,
            "usage_count": 0,
            "line_number": 6,
            "suggestions": ["Remove unused import"]
        },
        {
            "file_path": "utils.py", 
            "import_name": "requests",
            "import_type": "import",
            "is_used": True,
            "usage_count": 1,
            "line_number": 2,
            "suggestions": []
        }
    ]


@pytest.fixture
def mock_coupling_metrics():
    """Mock coupling analysis results."""
    return [
        {
            "module_a": "main.py",
            "module_b": "utils.py",
            "coupling_strength": 0.3,
            "coupling_type": "afferent",
            "shared_dependencies": ["json"],
            "refactoring_opportunity": "Consider extracting shared utilities"
        }
    ]


@pytest.fixture
def mock_architecture_violations():
    """Mock architecture violation results."""
    return [
        {
            "file_path": "main.py",
            "violation_type": "circular_dependency",
            "severity": "HIGH",
            "description": "Circular import detected between main.py and utils.py",
            "suggestion": "Refactor to break circular dependency",
            "pattern_violated": "layered_architecture"
        }
    ]


@pytest.fixture
def mock_technical_debt():
    """Mock technical debt assessment results."""
    return [
        {
            "file_path": "main.py",
            "debt_score": 7.5,
            "complexity_metrics": {
                "cyclomatic_complexity": 3,
                "cognitive_complexity": 5,
                "lines_of_code": 25
            },
            "debt_indicators": ["Long function", "Too many imports"],
            "refactoring_priority": "MEDIUM",
            "estimated_effort": "2-4 hours"
        }
    ]


@pytest.fixture
def mock_ai_context_analysis():
    """Mock AI context analysis results.""" 
    return [
        {
            "file_path": "main.py",
            "token_count": 150,
            "context_health": "GOOD",
            "estimated_split_points": [],
            "refactoring_suggestions": [],
            "ai_friendliness_score": 8.5
        },
        {
            "file_path": "utils.py",
            "token_count": 4500,
            "context_health": "WARNING", 
            "estimated_split_points": [50, 100],
            "refactoring_suggestions": ["Split into smaller functions", "Extract utility classes"],
            "ai_friendliness_score": 6.0
        }
    ]


@pytest.fixture
def mock_validation_result():
    """Mock validation result for pre-commit validation."""
    return {
        "is_valid": True,
        "errors": [],
        "warnings": ["Consider updating dependency versions"],
        "suggestions": ["Add type hints to improve code quality"]
    }


@pytest.fixture
def mock_networkx():
    """Mock NetworkX for testing without actual graph operations."""
    with patch('networkx.DiGraph') as mock_digraph:
        mock_graph = MagicMock()
        mock_digraph.return_value = mock_graph
        
        # Mock graph methods
        mock_graph.add_node = MagicMock()
        mock_graph.add_edge = MagicMock()
        mock_graph.nodes.return_value = ["main.py", "utils.py"]
        mock_graph.edges.return_value = [("main.py", "utils.py")]
        mock_graph.number_of_nodes.return_value = 2
        mock_graph.number_of_edges.return_value = 1
        
        yield mock_graph


@pytest.fixture
def mock_file_operations():
    """Mock file system operations for testing."""
    mock_data = {}
    
    def mock_open(filename, mode='r', *args, **kwargs):
        if 'w' in mode:
            # Writing mode - store data
            class MockFile:
                def __init__(self, filename):
                    self.filename = filename
                    self.content = ""
                
                def write(self, data):
                    self.content += data
                    mock_data[self.filename] = self.content
                
                def __enter__(self):
                    return self
                
                def __exit__(self, *args):
                    pass
            
            return MockFile(filename)
        else:
            # Reading mode - return stored data
            import io
            content = mock_data.get(filename, "")
            return io.StringIO(content)
    
    with patch('builtins.open', side_effect=mock_open):
        yield mock_data


@pytest.fixture
def mock_subprocess():
    """Mock subprocess operations for testing CLI commands."""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Mock command output",
            stderr=""
        )
        yield mock_run


@pytest.fixture
def mock_mcp_server():
    """Mock MCP server for testing MCP integration."""
    mock_server = MagicMock()
    
    # Mock server methods
    mock_server.call_tool = MagicMock()
    mock_server.run = MagicMock()
    mock_server.create_initialization_options = MagicMock()
    
    return mock_server


@pytest.fixture
def sample_mcp_tool_request():
    """Sample MCP tool request for testing."""
    return {
        "method": "tools/call",
        "params": {
            "name": "analyze_dependencies",
            "arguments": {
                "project_path": ".",
                "format": "json",
                "ai_awareness": True
            }
        }
    }


@pytest.fixture
def mock_rich_console():
    """Mock Rich console for testing output formatting."""
    with patch('rich.console.Console') as mock_console_class:
        mock_console = MagicMock()
        mock_console_class.return_value = mock_console
        yield mock_console


@pytest.fixture
def environment_variables():
    """Fixture to set and clean up environment variables."""
    original_env = os.environ.copy()
    
    def set_env(**kwargs):
        for key, value in kwargs.items():
            os.environ[key] = str(value)
    
    yield set_env
    
    # Cleanup
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(autouse=True)
def clean_imports():
    """Automatically clean up imported modules between tests."""
    modules_before = set(sys.modules.keys())
    yield
    
    # Remove any modules that were imported during the test
    modules_after = set(sys.modules.keys())
    for module in modules_after - modules_before:
        if module.startswith(('tools.', 'deepflow.')):
            sys.modules.pop(module, None)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "mcp: mark test as an MCP protocol test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically mark tests based on their location."""
    for item in items:
        # Mark tests based on directory structure
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "mcp" in str(item.fspath):
            item.add_marker(pytest.mark.mcp)