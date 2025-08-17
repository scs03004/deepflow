# 🗺️ Dependency Toolkit

A comprehensive suite of tools for managing, visualizing, and validating project dependencies across any codebase.

## 🎯 **Overview**

The Dependency Toolkit provides six powerful tools for dependency management:

1. **🌐 Automated Dependency Graph Visualization** - Generate interactive dependency trees and risk heat maps
2. **🔒 Pre-commit Dependency Validation** - Git hooks that validate dependencies before commits
3. **🚀 CI/CD Integration Tools** - Automated dependency checking in pipelines
4. **📊 Real-time Monitoring Dashboard** - Live dependency health monitoring
5. **🧠 Smart Code Analysis Tools** - Import usage analysis and refactoring suggestions
6. **📚 Documentation Automation** - Auto-generated dependency documentation

## 🚀 **Quick Start**

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd dependency-toolkit

# Install dependencies
pip install -r requirements.txt

# Run setup
python setup.py
```

### Basic Usage
```bash
# Generate dependency graph for a project
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