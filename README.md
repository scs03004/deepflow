# 🗺️ Dependency Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![GitHub Issues](https://img.shields.io/github/issues/scs03004/dependency-toolkit)](https://github.com/scs03004/dependency-toolkit/issues)
[![GitHub Stars](https://img.shields.io/github/stars/scs03004/dependency-toolkit)](https://github.com/scs03004/dependency-toolkit)

A **production-ready** suite of tools for managing, visualizing, and validating project dependencies across any codebase. Built for developers who need **enterprise-grade dependency management** without the complexity.

## 🎯 **Overview**

The Dependency Toolkit provides **six powerful tools** for comprehensive dependency management:

1. **🌐 Automated Dependency Graph Visualization** - Interactive dependency trees with risk analysis
2. **🔒 Pre-commit Dependency Validation** - Git hooks preventing broken dependencies
3. **🚀 CI/CD Integration Tools** - Automated workflows for GitHub Actions & GitLab CI
4. **📊 Real-time Monitoring Dashboard** - Live dependency health monitoring with web interface
5. **🧠 Smart Code Analysis Tools** - Unused import detection and technical debt scoring
6. **📚 Documentation Automation** - Auto-generated dependency maps and architecture docs

## 🚀 **Quick Start**

### Installation
```bash
# Install directly from GitHub
pip install git+https://github.com/scs03004/dependency-toolkit.git

# Or clone and install locally
git clone https://github.com/scs03004/dependency-toolkit.git
cd dependency-toolkit
pip install -e .
```

### Basic Usage
```bash
# Generate dependency graph for a project
dependency-visualizer /path/to/project --format html

# Set up pre-commit validation
dependency-validator --install /path/to/project

# Analyze code quality and unused imports
dependency-analyzer /path/to/project --all

# Generate documentation
dependency-docs /path/to/project --output docs/

# Start real-time monitoring dashboard
dependency-monitor --start /path/to/project --server
```

## 📊 **Example Output**

### Real Project Analysis (NPCGPT)
```
✅ NPCGPT Dependency Analysis Results:
📁 Total Python files analyzed: 43
🔗 Total import relationships: 62
📦 External dependencies: 55
⚠️  Circular dependencies: 0
🔴 High-risk files: 12

📊 Project Metrics:
  • Total Files: 43
  • Total Imports: 63
  • External Dependencies: 55
  • High Risk Files: 12
  • Circular Dependencies: 0
  • Total Lines Of Code: 13,667

✅ No circular dependencies detected!

🔴 High-Risk Files (require careful change management):
  • config.py (138 LOC, 7 dependents)
  • main.py (152 LOC, 3 dependents)
  • api\routes.py (1489 LOC, 1 dependents)
  • models\database.py (37 LOC, 11 dependents)
  • llm\personality_engine.py (487 LOC, 5 dependents)
```

### Code Quality Analysis
```
Import Analysis: Found 37 unused imports out of 157 total

Technical Debt Analysis: Analyzed 7 files
High Priority Technical Debt:
  • tools\code_analyzer.py - Debt Score: 121.1 (3-5 days effort)
  • tools\doc_generator.py - Debt Score: 82.3 (3-5 days effort)
  • Large file size, High cyclomatic complexity detected

Architecture Analysis: Found 4 violations
  • Single responsibility violations
  • Interface segregation issues

Summary: Found 47 total issues requiring attention
```
python tools/dependency_visualizer.py /path/to/project

# Validate dependencies
python tools/pre_commit_validator.py /path/to/project

# Generate documentation
python tools/doc_generator.py /path/to/project
```

## 🛠️ **Tools**

### 1. Dependency Graph Visualizer
- **Text-based dependency trees** for quick analysis
- **Interactive HTML graphs** with clickable nodes
- **Risk heat maps** highlighting critical dependencies
- **Circular dependency detection** with visual warnings
- **Export formats**: PNG, SVG, HTML, JSON

### 2. Pre-commit Validator
- **Import validation** - Check for broken imports before commit
- **Dependency impact analysis** - Warn about high-risk changes
- **Auto-test triggering** - Run tests based on changed dependencies
- **Documentation sync** - Ensure docs match code structure

### 3. CI/CD Integration
- **GitHub Actions workflows** for automated dependency checking
- **Dependency change notifications** in pull requests
- **Risk-based testing** - Different test suites based on impact
- **Deployment safety checks** before production

### 4. Real-time Monitoring
- **Live dependency dashboard** showing system health
- **Performance impact tracking** of dependency changes
- **Usage analytics** for dependency optimization
- **Automated alerts** for circular dependencies

### 5. Smart Code Analysis
- **Unused import detection** and cleanup suggestions
- **Coupling analysis** for refactoring opportunities
- **Architecture violation detection** against established patterns
- **Technical debt scoring** based on dependency complexity

### 6. Documentation Automation
- **Auto-generated dependency maps** from code analysis
- **Living architecture diagrams** that update with changes
- **Impact analysis reports** for stakeholders
- **Dependency changelogs** showing evolution over time

## 📁 **Project Structure**

```
dependency-toolkit/
├── tools/                          # Core dependency management tools
│   ├── dependency_visualizer.py    # Graph generation and visualization
│   ├── pre_commit_validator.py     # Git hook validation
│   ├── ci_cd_integrator.py         # CI/CD pipeline tools
│   ├── monitoring_dashboard.py     # Real-time monitoring
│   ├── code_analyzer.py            # Smart analysis tools
│   └── doc_generator.py            # Documentation automation
├── templates/                      # Reusable templates
│   ├── DEPENDENCY_MAP_TEMPLATE.md  # Standard dependency map format
│   ├── CHANGE_IMPACT_CHECKLIST.md  # Change management checklist
│   └── dependency_report.html      # HTML report template
├── .github/workflows/              # CI/CD workflows
│   ├── dependency_check.yml        # Automated dependency validation
│   └── documentation_update.yml    # Auto-update documentation
├── tests/                          # Comprehensive test suite
├── examples/                       # Example usage and integrations
├── docs/                          # Detailed documentation
└── config/                        # Configuration files
```

## 🎯 **Use Cases**

### For Individual Developers
- **Understand codebase structure** before making changes
- **Avoid breaking dependencies** with pre-commit validation
- **Track technical debt** and refactoring opportunities

### For Teams
- **Coordinate changes** across complex codebases
- **Standardize dependency management** across projects
- **Reduce integration issues** with automated validation

### For DevOps
- **Automate dependency checking** in CI/CD pipelines
- **Monitor system health** in real-time
- **Generate compliance reports** for stakeholders

## 📚 **Documentation**

- **[Getting Started Guide](docs/getting-started.md)** - Installation and basic usage
- **[Tool Reference](docs/tool-reference.md)** - Detailed documentation for each tool
- **[Integration Guide](docs/integration.md)** - CI/CD and workflow integration
- **[API Reference](docs/api-reference.md)** - Programmatic usage
- **[Best Practices](docs/best-practices.md)** - Recommended workflows

## 🤝 **Contributing**

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 **License**

MIT License - see [LICENSE](LICENSE) for details.

## 🌟 **Features**

- ✅ **Language Agnostic** - Works with Python, JavaScript, TypeScript, and more
- ✅ **CI/CD Ready** - GitHub Actions, GitLab CI, Jenkins integration
- ✅ **Cross-Platform** - Windows, macOS, Linux support
- ✅ **Extensible** - Plugin architecture for custom analysis
- ✅ **Fast** - Optimized for large codebases
- ✅ **Open Source** - Free to use and modify

---

**Built with ❤️ for better dependency management**