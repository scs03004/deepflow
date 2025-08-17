# Contributing to Dependency Toolkit

Thank you for your interest in contributing to the Dependency Toolkit! This document provides guidelines and information for contributors.

## 🚀 Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/dependency-toolkit.git
   cd dependency-toolkit
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```
5. **Run tests** to ensure everything works:
   ```bash
   pytest
   ```

## 📋 Development Guidelines

### Code Style

We use several tools to maintain code quality:

- **Black** for code formatting
- **MyPy** for type checking
- **Flake8** for linting
- **Pytest** for testing

Run all checks before submitting:
```bash
# Format code
black .

# Type checking
mypy tools/

# Linting
flake8 tools/

# Tests
pytest --cov=tools
```

### Pre-commit Hooks

We recommend using pre-commit hooks:
```bash
pip install pre-commit
pre-commit install
```

This will automatically run checks before each commit.

### Code Structure

```
dependency-toolkit/
├── tools/                  # Core tools and engines
│   ├── dependency_visualizer.py
│   ├── pre_commit_validator.py
│   ├── doc_generator.py
│   ├── ci_cd_integrator.py
│   ├── monitoring_dashboard.py
│   └── code_analyzer.py
├── templates/              # Jinja2 templates
│   ├── DEPENDENCY_MAP_TEMPLATE.md
│   └── CHANGE_IMPACT_CHECKLIST.md
├── tests/                  # Test suite
├── docs/                   # Documentation
├── examples/               # Usage examples
└── .github/workflows/      # CI/CD workflows
```

## 🛠️ Types of Contributions

### 1. Bug Reports

When reporting bugs, please include:

- **Environment details**: OS, Python version, dependency versions
- **Reproduction steps**: Clear steps to reproduce the issue
- **Expected vs actual behavior**: What should happen vs what actually happens
- **Error messages**: Full error traces and logs
- **Minimal example**: Smallest possible code that demonstrates the issue

Use the bug report template in GitHub Issues.

### 2. Feature Requests

For new features:

- **Use case description**: Why is this feature needed?
- **Proposed solution**: How should it work?
- **Alternatives considered**: What other approaches were considered?
- **Implementation ideas**: Technical approach (if you have ideas)

Use the feature request template in GitHub Issues.

### 3. Code Contributions

#### Adding New Tools

To add a new analysis tool:

1. **Create the tool file** in `tools/` directory
2. **Follow the existing pattern**:
   ```python
   #!/usr/bin/env python3
   """
   Tool Name
   =========
   
   Description of what the tool does.
   
   Usage:
       python tool_name.py /path/to/project
   """
   
   import argparse
   from pathlib import Path
   from rich.console import Console
   
   class ToolEngine:
       def __init__(self, project_path: str):
           self.project_path = Path(project_path).resolve()
           self.console = Console()
       
       def analyze(self):
           # Implementation here
           pass
   
   def main():
       parser = argparse.ArgumentParser(description='Tool description')
       parser.add_argument('project_path', help='Path to project')
       args = parser.parse_args()
       
       tool = ToolEngine(args.project_path)
       tool.analyze()
   
   if __name__ == "__main__":
       main()
   ```

3. **Add tests** in `tests/test_tool_name.py`
4. **Update documentation** in README.md
5. **Add entry point** in setup.py

#### Adding Language Support

Currently supports Python. To add support for other languages:

1. **Create language parser** in `tools/parsers/`
2. **Implement common interface**:
   ```python
   class LanguageParser:
       def extract_imports(self, file_path: Path) -> List[Import]:
           pass
       
       def extract_dependencies(self, file_path: Path) -> List[Dependency]:
           pass
       
       def calculate_metrics(self, file_path: Path) -> Dict[str, float]:
           pass
   ```

3. **Register parser** in tool engines
4. **Add tests** for the new language
5. **Update documentation**

#### Improving Visualizations

For visualization enhancements:

1. **Use Plotly** for interactive visualizations
2. **Maintain responsive design** for web outputs
3. **Support dark/light themes**
4. **Add accessibility features**
5. **Test on different screen sizes**

### 4. Documentation

Documentation improvements are always welcome:

- **Fix typos and grammar**
- **Add usage examples**
- **Improve explanations**
- **Add screenshots/diagrams**
- **Translate to other languages**

Documentation is written in Markdown and uses:
- **GitHub Pages** for hosting
- **MkDocs** for generation (planned)
- **Mermaid** for diagrams (where applicable)

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=tools --cov-report=html

# Run specific test file
pytest tests/test_dependency_visualizer.py

# Run tests matching pattern
pytest -k "test_import_analysis"
```

### Writing Tests

- **Use pytest fixtures** for common setup
- **Test both success and failure cases**
- **Mock external dependencies** (file system, network)
- **Use temporary directories** for file operations
- **Follow AAA pattern**: Arrange, Act, Assert

Example test:
```python
import pytest
from pathlib import Path
from tools.dependency_visualizer import DependencyAnalyzer

@pytest.fixture
def sample_project(tmp_path):
    """Create a sample project for testing."""
    # Create sample Python files
    (tmp_path / "main.py").write_text("import os\nfrom utils import helper")
    (tmp_path / "utils.py").write_text("def helper(): pass")
    return tmp_path

def test_dependency_analysis(sample_project):
    """Test basic dependency analysis."""
    analyzer = DependencyAnalyzer(str(sample_project))
    graph = analyzer.analyze_project()
    
    assert len(graph.nodes) == 2
    assert "main" in graph.nodes
    assert "utils" in graph.nodes
```

## 📚 Documentation Standards

### Docstrings

Use Google-style docstrings:

```python
def analyze_dependencies(project_path: str, include_external: bool = True) -> DependencyGraph:
    """Analyze project dependencies.
    
    Args:
        project_path: Path to the project root directory.
        include_external: Whether to include external dependencies in analysis.
        
    Returns:
        DependencyGraph object containing analysis results.
        
    Raises:
        ValueError: If project_path does not exist.
        AnalysisError: If analysis cannot be completed.
        
    Example:
        >>> graph = analyze_dependencies("/path/to/project")
        >>> print(f"Found {len(graph.nodes)} modules")
    """
```

### README Updates

When adding features, update the main README.md:

1. **Add to feature list**
2. **Update usage examples**
3. **Add to table of contents**
4. **Update installation instructions** (if needed)

## 🚦 Pull Request Process

### Before Submitting

- [ ] **Tests pass** locally
- [ ] **Code is formatted** with Black
- [ ] **Type checking passes** with MyPy
- [ ] **Linting passes** with Flake8
- [ ] **Documentation updated** (if applicable)
- [ ] **CHANGELOG.md updated** (if applicable)

### PR Description

Include:

- **Summary** of changes
- **Issue reference** (if applicable): "Fixes #123"
- **Type of change**: Bug fix, feature, documentation, etc.
- **Testing done**: How you tested the changes
- **Screenshots** (for UI changes)

### Review Process

1. **Automated checks** must pass (GitHub Actions)
2. **Code review** by maintainer
3. **Discussion** and feedback incorporation
4. **Final approval** and merge

### Merge Requirements

- At least one approving review
- All checks passing
- No merge conflicts
- Branch up to date with main

## 🏗️ Architecture Guidelines

### Separation of Concerns

- **Analysis engines**: Core logic for dependency analysis
- **Visualizers**: Handle output generation and formatting
- **CLI interfaces**: User-facing command-line tools
- **Web interfaces**: Dashboard and real-time features

### Error Handling

- **Use specific exceptions** for different error types
- **Provide helpful error messages** with context
- **Log errors appropriately** for debugging
- **Graceful degradation** when possible

### Performance

- **Use generators** for large datasets
- **Implement progress bars** for long operations
- **Cache expensive computations** when appropriate
- **Support parallel processing** where beneficial

## 🌍 Community Guidelines

### Code of Conduct

This project follows the [Contributor Covenant](https://www.contributor-covenant.org/) Code of Conduct. Please read and follow it.

### Communication

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and community discussion
- **Pull Requests**: Code contributions and reviews

### Recognition

Contributors are recognized in:
- **CONTRIBUTORS.md** file
- **GitHub contributors** page
- **Release notes** for significant contributions

## 🆘 Getting Help

If you need help:

1. **Check existing issues** for similar problems
2. **Read the documentation** thoroughly
3. **Create a GitHub issue** with detailed information
4. **Join GitHub Discussions** for community support

## 📝 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the Dependency Toolkit! Your efforts help make dependency management better for everyone. 🙏