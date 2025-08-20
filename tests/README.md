# Deepflow Test Suite

Comprehensive test suite for the deepflow PyPI package including unit tests, integration tests, and MCP protocol tests.

## Test Structure

```
tests/
├── README.md                      # This file
├── conftest.py                    # Pytest configuration and shared fixtures
├── test_runner.py                 # Custom test runner with advanced features
├── unit/                          # Unit tests for core functionality
│   ├── __init__.py
│   ├── test_dependency_visualizer.py
│   ├── test_code_analyzer.py
│   ├── test_doc_generator.py
│   └── test_tools_import.py
├── integration/                   # Integration tests
│   ├── __init__.py
│   ├── test_cli_commands.py
│   ├── test_package_imports.py
│   └── test_optional_dependencies.py
└── mcp/                          # MCP protocol tests
    ├── __init__.py
    ├── test_mcp_server.py
    ├── test_mcp_tools.py
    ├── test_mcp_fallbacks.py
    └── test_mcp_entry_points.py
```

## Running Tests

### Quick Start

```bash
# Run all tests
pytest tests/

# Run all tests with coverage
pytest tests/ --cov=tools --cov=deepflow --cov-report=html

# Run specific test categories
pytest tests/unit/        # Unit tests only
pytest tests/integration/ # Integration tests only
pytest tests/mcp/         # MCP tests only
```

### Using the Custom Test Runner

```bash
# Run all tests
python tests/test_runner.py

# Run specific test types
python tests/test_runner.py unit
python tests/test_runner.py integration
python tests/test_runner.py mcp

# Run with options
python tests/test_runner.py --verbose --coverage
python tests/test_runner.py --fast  # Skip slow tests
python tests/test_runner.py --parallel 4  # Run in parallel

# Generate coverage report
python tests/test_runner.py coverage --format html

# Validate test environment
python tests/test_runner.py --validate

# List available tests
python tests/test_runner.py --list
```

### Test Markers

Tests are organized with pytest markers:

- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests  
- `@pytest.mark.mcp` - MCP protocol tests
- `@pytest.mark.slow` - Slow-running tests

Run tests by marker:
```bash
pytest -m unit           # Unit tests only
pytest -m integration    # Integration tests only
pytest -m "not slow"     # Skip slow tests
pytest -m "unit and not slow"  # Fast unit tests
```

## Test Categories

### Unit Tests (`tests/unit/`)

Test individual components in isolation with extensive mocking:

- **test_dependency_visualizer.py**: Tests for dependency analysis and visualization
- **test_code_analyzer.py**: Tests for code quality analysis and metrics
- **test_doc_generator.py**: Tests for documentation generation
- **test_tools_import.py**: Tests for graceful import handling and fallbacks

### Integration Tests (`tests/integration/`)

Test component interactions and end-to-end workflows:

- **test_cli_commands.py**: CLI entry points and command execution
- **test_package_imports.py**: Package import scenarios and module loading
- **test_optional_dependencies.py**: Optional dependency handling and graceful degradation

### MCP Tests (`tests/mcp/`)

Test Model Context Protocol integration:

- **test_mcp_server.py**: MCP server functionality and async operations
- **test_mcp_tools.py**: MCP tool exposure and protocol compliance
- **test_mcp_fallbacks.py**: Graceful fallbacks when MCP is unavailable
- **test_mcp_entry_points.py**: MCP CLI entry points and command interface

## Test Configuration

### pytest.ini Options

The test suite is configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
minversion = "6.0"
addopts = "-ra -q --strict-markers --cov=tools --cov=deepflow --cov-report=term-missing"
testpaths = ["tests"]
markers = [
    "unit: unit tests",
    "integration: integration tests", 
    "mcp: MCP protocol tests",
    "slow: slow running tests",
]
asyncio_mode = "auto"
```

### Coverage Configuration

Coverage is configured to track both `tools/` and `deepflow/` packages:

```toml
[tool.coverage.run]
source = ["tools", "deepflow"]
branch = true
omit = [
    "*/tests/*",
    "*/test_*", 
    "*/venv/*",
    "*/build/*",
    "*/dist/*",
]

[tool.coverage.report]
fail_under = 85
show_missing = true
exclude_lines = [
    "pragma: no cover",
    "except ImportError:",
    "if.*AVAILABLE.*:",
]
```

## Test Fixtures

### Shared Fixtures (`conftest.py`)

- **mock_project_structure**: Creates temporary project with realistic Python files
- **sample_dependency_graph**: Sample dependency data for testing visualizations
- **mock_import_analysis**: Mock results for import analysis
- **mock_coupling_metrics**: Mock coupling analysis data
- **mock_networkx**: Mocked NetworkX for testing without graph operations
- **mock_file_operations**: Mock file system operations
- **mock_subprocess**: Mock subprocess calls for CLI testing

### Usage Example

```python
def test_dependency_analysis(mock_project_structure, sample_dependency_graph):
    """Test dependency analysis with mock data."""
    analyzer = DependencyAnalyzer(str(mock_project_structure))
    result = analyzer.analyze()
    assert result is not None
```

## Test Dependencies

### Required Dependencies

Install test dependencies:

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Or install test dependencies directly
pip install pytest pytest-cov pytest-asyncio pytest-mock pytest-xdist
```

### Optional Dependencies

The test suite handles missing optional dependencies gracefully:

- **NetworkX**: Required for dependency graph analysis
- **Plotly**: Required for interactive HTML visualizations  
- **Rich**: Required for enhanced console output
- **Jinja2**: Required for template rendering
- **MCP**: Required for MCP protocol functionality

## Mocking Strategy

### External Dependencies

All external dependencies are mocked to ensure:
- Tests run without installing heavy dependencies
- Consistent behavior across environments
- Fast test execution
- Isolation of component logic

### File System Operations

File operations are mocked using:
- `unittest.mock.mock_open` for file I/O
- `tempfile.TemporaryDirectory` for real file operations when needed
- Custom fixtures for project structure simulation

### Network and Subprocess

- **Subprocess calls**: Mocked to avoid external command execution
- **Network requests**: Mocked to prevent external API calls
- **Async operations**: Properly handled with `pytest-asyncio`

## Error Handling Tests

### Graceful Fallbacks

Tests verify graceful handling of:
- Missing optional dependencies
- Invalid file paths
- Malformed configuration files
- Network connectivity issues
- Permission errors

### Error Messages

Tests ensure error messages are:
- Helpful and actionable
- Include installation instructions
- Provide debugging context
- Follow consistent formatting

## Performance Testing

### Test Execution Speed

- Unit tests: < 0.1s per test
- Integration tests: < 1s per test
- Full suite: < 30s on modern hardware

### Memory Usage

Tests monitor and limit:
- Memory usage during large project analysis
- File handle leaks
- Module import overhead

## Continuous Integration

### GitHub Actions Configuration

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, 3.10, 3.11, 3.12]
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      - name: Run tests
        run: |
          pytest tests/ --cov=tools --cov=deepflow --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

### Pre-commit Hooks

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: tests
        name: Run tests
        entry: pytest
        args: [tests/, --fast]
        language: system
        pass_filenames: false
```

## Debugging Tests

### Running Individual Tests

```bash
# Run specific test file
pytest tests/unit/test_dependency_visualizer.py -v

# Run specific test function
pytest tests/unit/test_dependency_visualizer.py::test_init -v

# Run with debugging
pytest tests/unit/test_dependency_visualizer.py::test_init -v -s --pdb
```

### Common Issues

1. **Import Errors**: Ensure project root is in `PYTHONPATH`
2. **Missing Dependencies**: Use mocks or install optional dependencies
3. **Path Issues**: Use absolute paths in test fixtures
4. **Async Issues**: Mark async tests with `@pytest.mark.asyncio`

### Test Output

```bash
# Verbose output with full tracebacks
pytest tests/ -v --tb=long

# Show local variables in tracebacks
pytest tests/ --tb=auto --showlocals

# Stop on first failure
pytest tests/ -x

# Run last failed tests only
pytest tests/ --lf
```

## Contributing

### Adding New Tests

1. **Choose appropriate category**: unit, integration, or mcp
2. **Use descriptive names**: `test_function_behavior_condition`
3. **Include docstrings**: Describe what the test validates
4. **Use appropriate fixtures**: Leverage existing fixtures when possible
5. **Mock external dependencies**: Keep tests isolated and fast

### Test Structure Template

```python
"""
Tests for new_module.py
"""

import pytest
from unittest.mock import patch, MagicMock


class TestNewModule:
    """Test cases for NewModule class."""
    
    def test_init(self):
        """Test NewModule initialization."""
        module = NewModule()
        assert module is not None
    
    def test_method_success_case(self, mock_project_structure):
        """Test method under normal conditions."""
        module = NewModule(str(mock_project_structure))
        result = module.method()
        assert result == expected_value
    
    def test_method_error_handling(self):
        """Test method error handling."""
        module = NewModule()
        with pytest.raises(ValueError, match="expected error"):
            module.method(invalid_input)
    
    @patch('new_module.external_dependency')
    def test_method_with_mocked_dependency(self, mock_dep):
        """Test method with mocked external dependency."""
        mock_dep.return_value = "mocked_result"
        module = NewModule()
        result = module.method()
        assert result == "expected_with_mock"
        mock_dep.assert_called_once()


@pytest.mark.parametrize("input_value,expected", [
    ("input1", "output1"),
    ("input2", "output2"),
])
def test_parameterized_behavior(input_value, expected):
    """Test behavior with different inputs."""
    module = NewModule()
    result = module.process(input_value)
    assert result == expected
```

### Best Practices

1. **One assertion per test**: Keep tests focused and specific
2. **Arrange-Act-Assert**: Structure tests clearly
3. **Mock at boundaries**: Mock at the edge of your system
4. **Test edge cases**: Include boundary conditions and error cases
5. **Use meaningful assertions**: Assert on specific values, not just existence

## Maintenance

### Regular Tasks

1. **Update test dependencies**: Keep pytest and plugins current
2. **Review coverage reports**: Identify gaps in test coverage
3. **Clean up obsolete tests**: Remove tests for deprecated functionality
4. **Update fixtures**: Keep test data realistic and current

### Monitoring

- **Coverage trends**: Track coverage percentage over time
- **Test execution time**: Monitor for performance regressions
- **Flaky tests**: Identify and fix unstable tests
- **Dependency updates**: Update mocks when dependencies change

## Troubleshooting

### Common Test Failures

1. **Import errors**: Check `sys.path` setup in conftest.py
2. **Mock issues**: Verify mock targets and return values
3. **Async test failures**: Ensure proper async/await usage
4. **Fixture conflicts**: Check fixture scope and dependencies

### Performance Issues

1. **Slow tests**: Profile and optimize or mark as slow
2. **Memory leaks**: Check for unclosed resources
3. **Import overhead**: Consider lazy imports in fixtures

### Environment Issues

1. **Different Python versions**: Test across supported versions
2. **Operating system differences**: Use cross-platform path handling
3. **Dependency versions**: Test with minimum and latest versions

For additional help, see the [contributing guide](../CONTRIBUTING.md) and [integration guide](../INTEGRATION_GUIDE.md).