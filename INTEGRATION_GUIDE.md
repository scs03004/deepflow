# 🔗 Integration Guide: Using Dependency Toolkit Across Projects

This guide shows how to integrate the Dependency Toolkit into your existing projects and establish it as a standard workflow.

## 🎯 **Integration Strategies**

### **Strategy 1: Package Installation (Recommended)**

**Best for**: All projects, especially when you want consistent tooling across teams.

1. **Publish to GitHub** (or PyPI):
   ```bash
   # Create GitHub repository
   git remote add origin https://github.com/your-username/dependency-toolkit.git
   git push -u origin main
   ```

2. **Install in target projects**:
   ```bash
   # In NPCGPT or any project
   pip install git+https://github.com/your-username/dependency-toolkit.git
   
   # Or add to requirements-dev.txt
   echo "dependency-toolkit @ git+https://github.com/your-username/dependency-toolkit.git" >> requirements-dev.txt
   ```

3. **Use immediately**:
   ```bash
   dependency-visualizer .
   dependency-validator --install .
   dependency-docs . --output docs/
   ```

### **Strategy 2: Git Submodule**

**Best for**: When you want the source code available in each project.

```bash
# In NPCGPT project
git submodule add https://github.com/your-username/dependency-toolkit.git tools/dependency-toolkit

# Use tools
python tools/dependency-toolkit/tools/dependency_visualizer.py .
```

### **Strategy 3: Shared Development Environment**

**Best for**: Team environments with shared tooling.

```bash
# Install globally or in shared environment
pip install dependency-toolkit

# Available across all projects
cd any-project && dependency-visualizer .
```

## 🛠️ **NPCGPT Integration Example**

Here's how to integrate with the NPCGPT project:

### **Step 1: Install Dependency Toolkit**

```bash
cd NPCGPT
pip install git+https://github.com/your-username/dependency-toolkit.git
```

### **Step 2: Update Development Dependencies**

Add to `requirements-dev.txt`:
```txt
# Dependency management
dependency-toolkit @ git+https://github.com/your-username/dependency-toolkit.git

# Existing dev dependencies
pytest>=7.0.0
black>=23.0.0
# ... other deps
```

### **Step 3: Generate Initial Analysis**

```bash
# Generate comprehensive dependency analysis
dependency-visualizer . --format all

# This creates:
# - NPCGPT_dependency_graph.html (interactive visualization)
# - NPCGPT_risk_heatmap.html (risk assessment)
# - Text output in terminal
```

### **Step 4: Set Up Pre-commit Hooks**

```bash
# Install git hooks for automatic validation
dependency-validator --install .

# This creates:
# - .git/hooks/pre-commit (validation script)
# - .dependency-validator.yml (configuration)
```

### **Step 5: Generate Documentation**

```bash
# Generate project documentation
dependency-docs . --output docs/

# This creates:
# - docs/DEPENDENCY_MAP.md
# - docs/API_DOCUMENTATION.md  
# - docs/ARCHITECTURE.md
# - docs/CHANGE_IMPACT_CHECKLIST.md
```

### **Step 6: Set Up CI/CD Integration**

```bash
# Generate GitHub Actions workflows
dependency-ci --setup-github .

# This creates workflows in .github/workflows/:
# - dependency-check.yml
# - documentation-update.yml
# - risk-based-testing.yml
# - deployment-safety.yml
```

## 📋 **Daily Workflow Integration**

### **For Developers Working on NPCGPT:**

1. **Before making changes**:
   ```bash
   # Check current dependency status
   dependency-validator --validate .
   
   # Analyze impact of planned changes
   dependency-validator --impact-analysis .
   ```

2. **During development**:
   ```bash
   # Fix code quality issues
   dependency-analyzer . --fix-imports --analyze-coupling
   
   # Monitor real-time dependency health
   dependency-monitor --start . --server
   # Opens dashboard at http://localhost:5000
   ```

3. **Before committing**:
   ```bash
   # Pre-commit hooks automatically run
   git commit -m "Your changes"
   
   # Manual validation if needed
   dependency-validator --check-files file1.py file2.py
   ```

4. **After major changes**:
   ```bash
   # Update documentation
   dependency-docs . --output docs/
   
   # Generate new dependency graph
   dependency-visualizer . --format html
   ```

### **For Project Maintenance:**

1. **Weekly dependency review**:
   ```bash
   # Generate comprehensive analysis
   dependency-analyzer . --all --output weekly_analysis.json
   
   # Check for technical debt
   dependency-analyzer . --calculate-debt
   ```

2. **Before releases**:
   ```bash
   # Full dependency validation
   dependency-validator --validate .
   
   # Generate deployment safety report
   dependency-ci --validate-changes .
   ```

## 🌍 **Multi-Project Setup**

### **Standardizing Across Multiple Projects:**

1. **Create organization standards**:
   ```bash
   # Fork dependency-toolkit for your organization
   git clone https://github.com/your-username/dependency-toolkit.git
   cd dependency-toolkit
   
   # Customize for your standards
   # Edit templates/CHANGE_IMPACT_CHECKLIST.md
   # Update .github/workflows templates
   ```

2. **Install in all projects**:
   ```bash
   # Script to install in all projects
   for project in project1 project2 npcgpt; do
     cd $project
     pip install git+https://github.com/your-org/dependency-toolkit.git
     dependency-validator --install .
     dependency-ci --setup-github .
     cd ..
   done
   ```

3. **Centralized monitoring**:
   ```bash
   # Set up central monitoring dashboard
   dependency-monitor --server --port 8080
   # Monitor multiple projects from one dashboard
   ```

## 🔧 **Configuration Management**

### **Project-Specific Configuration**

Create `.dependency-toolkit.yml` in each project:

```yaml
# NPCGPT-specific configuration
validation:
  check_imports: true
  check_circular_deps: true
  require_tests: true
  
risk_levels:
  high_impact_files:
    - "main.py"
    - "config.py" 
    - "api/routes.py"
    - "models/database.py"
  
testing:
  test_command: "pytest"
  coverage_threshold: 80
  
documentation:
  auto_update: true
  include_architecture: true
  
monitoring:
  alert_thresholds:
    cpu_percent: 80
    memory_percent: 85
    circular_deps: 0
```

### **Team Standards Configuration**

Create organization templates:

```yaml
# Organization defaults
validation:
  enforce_pre_commit: true
  block_circular_deps: true
  
ci_cd:
  platform: "github"
  auto_deploy_docs: true
  risk_based_testing: true
  
code_quality:
  fix_unused_imports: true
  enforce_type_hints: true
  max_complexity: 10
```

## 📊 **Dashboard Integration**

### **Real-time Project Monitoring**

```bash
# Start monitoring for NPCGPT
cd NPCGPT
dependency-monitor --start . --server --port 5000

# View dashboard at http://localhost:5000
# Features:
# - Live dependency graph
# - Performance metrics
# - Risk alerts
# - Change impact visualization
```

### **Multi-Project Dashboard**

```bash
# Monitor multiple projects
dependency-monitor --server --port 8080 \
  --projects "NPCGPT:./NPCGPT,Project2:./Project2"
```

## 🚀 **Automation Examples**

### **GitHub Actions Integration**

The toolkit automatically creates workflows that:

1. **Validate dependencies** on every PR
2. **Update documentation** on main branch changes
3. **Run risk-based testing** based on files changed
4. **Check deployment safety** before releases
5. **Generate dependency reports** as artifacts

### **Pre-commit Hook Integration**

Automatically runs on every commit:

1. **Import validation** - No broken imports
2. **Circular dependency check** - No new cycles
3. **Impact analysis** - Warn about high-risk changes
4. **Test triggering** - Run appropriate tests

## 💡 **Best Practices**

### **For Individual Projects**

1. **Start small**: Begin with visualization and validation
2. **Gradual adoption**: Add monitoring and automation over time
3. **Team training**: Ensure team understands the tools
4. **Regular reviews**: Weekly dependency health checks

### **For Organizations**

1. **Standardize configurations** across projects
2. **Central monitoring** for enterprise visibility
3. **Automated reporting** for stakeholders
4. **Integration with existing tools** (JIRA, Slack, etc.)

### **For Large Codebases**

1. **Performance optimization**: Use parallel analysis
2. **Incremental analysis**: Focus on changed components
3. **Historical tracking**: Monitor trends over time
4. **Alerting systems**: Proactive issue detection

---

This integration approach provides **scalable dependency management** that grows with your projects and organization. The toolkit becomes a natural part of the development workflow, providing continuous insights and automated quality assurance.

## 🎯 **Quick Start Commands**

```bash
# Install and get started in any project
pip install dependency-toolkit
dependency-visualizer .                    # See your dependencies
dependency-validator --install .           # Set up validation
dependency-docs . --output docs/          # Generate docs
dependency-monitor --start . --server     # Start monitoring
```

The toolkit is designed to be **immediately useful** while providing **enterprise-scale** capabilities as your needs grow.