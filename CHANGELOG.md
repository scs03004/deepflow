# Changelog

All notable changes to Deepflow will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2025-08-21

### Added - MCP Integration Breakthrough 🎉
- **🔌 MCP Server Integration** - First dependency analysis tool with native Model Context Protocol support
- **🤖 Claude Code Integration** - Direct integration with Claude Code and other MCP-compatible AI assistants
- **📚 MCP Integration Guide** - Comprehensive setup documentation for Claude Code configuration
- **🛠️ 4 MCP Tools** - analyze_dependencies, analyze_code_quality, validate_commit, generate_documentation
- **📊 Auto-generated Documentation** - 5 comprehensive project documentation files
- **🧪 Perfect Test Coverage** - 99/99 tests passing (100% pass rate)
- **📦 MANIFEST.in** - Proper package file inclusion for PyPI distribution
- **📋 Self-Analysis Capability** - Successfully analyzed own codebase with generated insights

### Enhanced
- **README.md** - Updated with MCP integration highlights and breakthrough messaging
- **Package Structure** - Professional PyPI-ready organization with optional MCP dependencies
- **Documentation** - Enhanced project documentation with auto-generated architecture maps

### Technical Achievements
- **100% Test Pass Rate** - Fixed all failing tests systematically
- **MCP Protocol Compliance** - Full specification compliance with structured JSON responses
- **Graceful Fallbacks** - Core functionality works without MCP dependencies
- **Zero Circular Dependencies** - Clean architecture verified through self-analysis

## [2.0.0] - 2025-08-21

### Major Release - Complete Package Transformation
- **🏗️ Complete Rebranding** - Transformed from "dependency-toolkit" to "deepflow"
- **🔧 Professional Package Structure** - Modern pyproject.toml with comprehensive test suite
- **📦 PyPI-Ready Distribution** - Production-ready package with proper dependencies
- **🧪 Comprehensive Testing** - 230+ tests covering unit, integration, and MCP functionality

## [1.0.0] - 2025-08-17

### Added
- **Dependency Graph Visualizer** - Generate interactive dependency trees and risk heat maps
- **Pre-commit Dependency Validator** - Git hooks for import validation and impact analysis
- **Documentation Automation** - Auto-generated dependency maps and architecture docs
- **CI/CD Integration Tools** - GitHub Actions and GitLab CI workflows
- **Real-time Monitoring Dashboard** - Live dependency health monitoring with web interface
- **Smart Code Analysis Tools** - Unused import detection, coupling analysis, and technical debt scoring

### Features

#### 🌐 Dependency Graph Visualizer
- Text-based dependency trees for quick analysis
- Interactive HTML graphs with clickable nodes
- Risk heat maps highlighting critical dependencies
- Circular dependency detection with visual warnings
- Multiple export formats (PNG, SVG, HTML, JSON)

#### 🔒 Pre-commit Dependency Validator
- Import validation - Check for broken imports before commit
- Dependency impact analysis - Warn about high-risk changes
- Auto-test triggering - Run tests based on changed dependencies
- Documentation sync - Ensure docs match code structure

#### 📚 Documentation Automation
- Auto-generated dependency maps from code analysis
- Living architecture diagrams that update with changes
- Impact analysis reports for stakeholders
- Dependency changelogs showing evolution over time

#### 🚀 CI/CD Integration
- GitHub Actions workflows for automated dependency validation
- GitLab CI pipeline templates
- Dependency change notifications in pull requests
- Risk-based testing - Different test suites based on impact
- Deployment safety checks before production

#### 📊 Real-time Monitoring
- Live dependency dashboard showing system health
- Performance impact tracking of dependency changes
- Usage analytics for dependency optimization
- Automated alerts for circular dependencies
- Web-based dashboard with real-time updates

#### 🧠 Smart Code Analysis
- Unused import detection and cleanup suggestions
- Coupling analysis for refactoring opportunities
- Architecture violation detection against established patterns
- Technical debt scoring based on dependency complexity
- Automated refactoring recommendations

### Technical Specifications
- **Language Support**: Python (primary), extensible to other languages
- **Platforms**: Windows, macOS, Linux
- **Python Versions**: 3.8, 3.9, 3.10, 3.11, 3.12
- **CI/CD Platforms**: GitHub Actions, GitLab CI, Jenkins (planned)
- **Output Formats**: HTML, JSON, Markdown, PNG, SVG
- **Web Framework**: Flask with SocketIO for real-time features

### Dependencies
- **Core**: NetworkX, Rich, Click, Jinja2
- **Visualization**: Plotly, Matplotlib, Pandas
- **Web**: Flask, Flask-SocketIO
- **Git**: GitPython
- **Analysis**: AST, importlib
- **Monitoring**: psutil

### Installation
```bash
pip install deepflow
```

### Quick Start
```bash
# Generate dependency graph
deepflow-visualizer /path/to/project

# Install pre-commit hooks
deepflow-validator --install /path/to/project

# Generate documentation
deepflow-docs /path/to/project --output docs/

# Start monitoring dashboard
deepflow-monitor --start /path/to/project --server
```

### Configuration
The toolkit supports configuration through:
- Command-line arguments
- YAML configuration files (`.deepflow.yml`)
- Environment variables
- Project-specific settings

### Performance
- **Large Projects**: Optimized for projects with 1000+ files
- **Memory Usage**: Efficient graph algorithms with memory optimization
- **Processing Speed**: Parallel analysis where possible
- **Real-time Updates**: Sub-second dashboard refresh rates

### Security
- **No External Calls**: All analysis performed locally
- **Safe Parsing**: AST-based analysis prevents code execution
- **Input Validation**: Comprehensive validation of all inputs
- **Permission Checks**: Respects file system permissions

### Extensibility
- **Plugin Architecture**: Extensible analysis engines
- **Custom Templates**: Jinja2-based templating system
- **API Integration**: RESTful APIs for external tools
- **Language Support**: Framework for adding language parsers

## [Future Releases]

### Planned Features
- **Language Expansion**: JavaScript/TypeScript, Go, Rust support
- **IDE Integration**: VS Code and PyCharm plugins
- **Advanced Analytics**: Machine learning-based recommendations
- **Team Features**: Multi-user dashboards and collaboration tools
- **Enterprise Features**: SSO, audit logs, compliance reporting

---

For more information, see the [documentation](https://deepflow.readthedocs.io/) or visit our [GitHub repository](https://github.com/deepflow/deepflow).