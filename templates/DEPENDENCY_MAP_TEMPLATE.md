# 🗺️ Project Dependency Map Template

Use this template to create a comprehensive dependency map for any project.

## 📋 **How to Create a Dependency Map**

### Step 1: Analyze Project Structure
```bash
# Get overall project structure
find . -type f -name "*.py" | head -20
find . -type f -name "*.js" | head -10
find . -type f -name "*.html" | head -10
find . -type f -name "*.md" | head -10
```

### Step 2: Map Import Dependencies
```bash
# Find all Python imports
grep -r "^from\|^import" . --include="*.py" | sort | uniq

# Find relative imports
grep -r "from \." . --include="*.py"

# Find external dependencies
grep -r "import " . --include="*.py" | grep -v "from \."
```

### Step 3: Find Configuration References
```bash
# Find config file usage
grep -r "config\." . --include="*.py"
grep -r "Config" . --include="*.py"
grep -r "settings" . --include="*.py"

# Find environment variables
grep -r "os\.environ\|getenv" . --include="*.py"
```

### Step 4: Map API Dependencies
```bash
# Find API endpoint definitions
grep -r "@app\.\|@router\." . --include="*.py"
grep -r "def.*(" . --include="routes*.py"

# Find frontend API calls
grep -r "fetch\|axios\|ajax" . --include="*.js"
grep -r "/api/" . --include="*.js" --include="*.html"
```

### Step 5: Database Dependencies
```bash
# Find model definitions
grep -r "class.*Model\|class.*Base" . --include="*.py"
grep -r "db\.\|database\." . --include="*.py"

# Find migration references
find . -name "*migration*" -o -name "*migrate*"
```

## 📊 **Dependency Map Template**

```markdown
# [PROJECT NAME] Dependency Map

**Generated**: [DATE]
**Last Updated**: [DATE]

## 🏗️ Architecture Overview

### Core Components
- **Entry Point**: [main.py, app.py, etc.]
- **Configuration**: [config files]
- **Database**: [database files/models]
- **API Layer**: [API files]
- **Frontend**: [web interface files]
- **External Services**: [third-party integrations]

## 📁 File Dependencies

### High-Risk Files (Changes Affect Many Components)
| File | Depends On | Dependents | Risk Level | Update Impact |
|------|------------|------------|------------|---------------|
| config.py | environment | all modules | HIGH | Full retest needed |
| main.py | config, models, api | deployment | HIGH | Integration testing |

### Medium-Risk Files
| File | Depends On | Dependents | Risk Level | Update Impact |
|------|------------|------------|------------|---------------|
| [file] | [dependencies] | [dependents] | MEDIUM | Targeted testing |

### Low-Risk Files
| File | Depends On | Dependents | Risk Level | Update Impact |
|------|------------|------------|------------|---------------|
| [file] | [dependencies] | [dependents] | LOW | Minimal testing |

## 🔗 Import Dependency Graph

### Internal Dependencies
```
main.py
├── config.py
├── api/
│   ├── routes.py
│   └── models.py
├── models/
│   ├── database.py
│   └── [model_files].py
└── [other_modules]/
```

### External Dependencies
- **[package_name]**: Used by [files] for [purpose]
- **[package_name]**: Used by [files] for [purpose]

## 🌐 API Dependencies

### Endpoint Mapping
| Endpoint | Handler | Frontend Usage | Dependencies |
|----------|---------|----------------|--------------|
| /api/[endpoint] | [function] | [js_file:line] | [models, services] |

### Frontend → Backend Calls
| Frontend File | API Calls | Purpose |
|---------------|-----------|---------|
| [file.js] | [endpoints] | [functionality] |

## 💾 Database Dependencies

### Model Relationships
```
[Model1] ←→ [Model2] (relationship_type)
[Model2] ←→ [Model3] (relationship_type)
```

### Migration Dependencies
- **[migration_file]**: Depends on [previous_migration]
- **Data Initialization**: [init_scripts] → [models]

## 📚 Documentation Dependencies

### File References in Documentation
| Doc File | References | Update Trigger |
|----------|------------|----------------|
| README.md | [file_paths] | When files move |
| [guide].md | [code_examples] | When API changes |

## 🧪 Test Dependencies

### Test Coverage Map
| Source File | Test File | Coverage Type |
|-------------|-----------|---------------|
| [source] | [test] | Unit/Integration |

### Test Data Dependencies
- **Fixtures**: [fixture_files] used by [test_files]
- **Mock Data**: [mock_files] simulate [external_services]

## 🔧 Tool Dependencies

### Build/Deployment Tools
- **[tool]**: Requires [files/config]
- **Scripts**: [script_files] depend on [requirements]

### Development Tools
- **[tool]**: Configuration in [config_file]

## ⚠️ Circular Dependencies

### Detected Circular Dependencies
- **[file1] ↔ [file2]**: [description of relationship]
- **[module1] ↔ [module2]**: [description of relationship]

### Recommendations
- [How to resolve circular dependencies]

## 🚨 Critical Update Paths

### When Changing [Critical Component]:
1. **Files to Update**: [list]
2. **Tests to Run**: [test_categories]
3. **Documentation to Update**: [doc_files]
4. **Verification Steps**: [checklist]

## 📈 Metrics

- **Total Python Files**: [count]
- **Total Import Statements**: [count]
- **External Dependencies**: [count]
- **API Endpoints**: [count]
- **Database Models**: [count]
- **Test Files**: [count]

## 🔄 Update History

| Date | Change | Impact | Files Updated |
|------|--------|--------|---------------|
| [date] | [description] | [scope] | [file_list] |
```

## 🛠️ **Agent Prompt for Dependency Analysis**

Use this prompt to generate a dependency map with an agent:

```
Analyze this project and create a comprehensive dependency map following the template above. 

Focus on:
1. File import relationships and dependencies
2. Configuration usage across modules  
3. API endpoint mappings and frontend calls
4. Database model relationships
5. Test coverage and dependencies
6. Documentation references to code files
7. Build/deployment tool requirements
8. Circular dependencies and risks

For each dependency, indicate:
- Type of dependency (import, reference, data, etc.)
- Risk level if dependency breaks
- Required updates when source changes
- Testing requirements

Create the map as [PROJECT_NAME]_DEPENDENCY_MAP.md
```

## 💡 **Best Practices**

1. **Generate Early**: Create dependency map in first project session
2. **Update Regularly**: Regenerate after major architectural changes
3. **Use for Planning**: Reference before making any significant changes
4. **Document in Logs**: Record dependency changes in session logs
5. **Cross-Reference**: Link dependency map with change impact checklist
6. **Automate When Possible**: Use scripts to detect some dependencies
7. **Review Regularly**: Check for new circular dependencies

This template ensures consistent dependency tracking across all projects and helps maintain code quality and system integrity.