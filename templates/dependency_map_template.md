# {{ project_name }} Dependency Map

**Generated**: {{ generated_date }}
**Project Path**: {{ project_metadata.path }}
**Language**: {{ project_metadata.language }}
**Framework**: {{ project_metadata.framework or "Not detected" }}

## 📊 Project Metrics

- **Total Files**: {{ dependency_graph.metrics.total_files }}
- **Total Imports**: {{ dependency_graph.metrics.total_imports }}
- **External Dependencies**: {{ dependency_graph.metrics.external_dependencies }}
- **High Risk Files**: {{ dependency_graph.metrics.high_risk_files }}
- **Circular Dependencies**: {{ dependency_graph.metrics.circular_dependencies }}
- **Lines of Code**: {{ dependency_graph.metrics.total_lines_of_code }}

## 🚨 High-Risk Files (Critical Components)

{% for file in high_risk_files %}
### {{ file.name }}
- **Path**: `{{ file.file_path }}`
- **Risk Level**: {{ file.risk_level }}
- **Lines of Code**: {{ file.lines_of_code }}
- **Imports**: {{ file.imports|length }} modules
- **Imported By**: {{ file.imported_by|length }} modules
- **Dependencies**: {{ file.imports|join(', ') if file.imports else 'None' }}
- **Dependents**: {{ file.imported_by|join(', ') if file.imported_by else 'None' }}

{% endfor %}

## 🔗 All Internal Dependencies

{% for name, node in dependency_graph.nodes.items() %}
### {{ name }}
- **File**: `{{ node.file_path }}`
- **Risk Level**: {{ node.risk_level }}
- **LOC**: {{ node.lines_of_code }}
- **Imports**: {% if node.imports %}{{ node.imports|join(', ') }}{% else %}None{% endif %}
- **Imported By**: {% if node.imported_by %}{{ node.imported_by|join(', ') }}{% else %}None{% endif %}

{% endfor %}

## 📦 External Dependencies

{% for dep, users in dependency_graph.external_dependencies.items() %}
### {{ dep }}
- **Used By**: {{ users|join(', ') }}
- **Module Count**: {{ users|length }}

{% endfor %}

## ⚠️ Circular Dependencies

{% if dependency_graph.circular_dependencies %}
{% for cycle in dependency_graph.circular_dependencies %}
**Cycle {{ loop.index }}**: {{ cycle|join(' → ') }} → {{ cycle[0] }}

{% endfor %}

### Recommendations
- Review these circular imports for potential refactoring
- Consider dependency injection or event-driven patterns
- Move shared code to common modules
{% else %}
✅ **No circular dependencies detected!**
{% endif %}

## 🧪 Testing Strategy

### Priority Testing Areas
1. **High-Risk Files**: Focus on {{ dependency_graph.metrics.high_risk_files }} critical components
2. **External Integrations**: Test all {{ dependency_graph.metrics.external_dependencies }} external dependencies
3. **Circular Dependencies**: {% if dependency_graph.circular_dependencies %}{{ dependency_graph.circular_dependencies|length }} cycles need integration tests{% else %}None detected{% endif %}

### Recommended Test Coverage
- **Unit Tests**: All high-risk files ({{ dependency_graph.metrics.high_risk_files }} files)
- **Integration Tests**: Cross-module dependency chains
- **External Tests**: Mock all {{ dependency_graph.metrics.external_dependencies }} external services

## 🚨 Change Impact Analysis

### When Updating High-Risk Files:

{% for file in high_risk_files %}
**{{ file.name }}**:
1. **Direct Impact**: {{ file.imported_by|length }} modules need review
2. **Test Coverage**: Run integration tests for dependent modules  
3. **Documentation**: Update if public interfaces change
4. **Dependent Files**: {% for dep in file.imported_by %}{{ dep }}{% if not loop.last %}, {% endif %}{% endfor %}

{% endfor %}

## 📈 Quality Metrics

- **Dependency Complexity**: {{ dependency_graph.metrics.total_imports }} total import relationships
- **Modularity Score**: {% if dependency_graph.metrics.total_files > 0 %}{{ "%.1f"|format(dependency_graph.metrics.total_imports / dependency_graph.metrics.total_files) }}{% else %}0{% endif %} imports per file
- **External Coupling**: {{ dependency_graph.metrics.external_dependencies }} external dependencies
- **Risk Distribution**: {{ dependency_graph.metrics.high_risk_files }} high-risk files

## 💡 Maintenance Recommendations

### Immediate Actions
{% if dependency_graph.circular_dependencies %}
- 🔴 **Resolve {{ dependency_graph.circular_dependencies|length }} circular dependencies**
{% endif %}
{% if dependency_graph.metrics.high_risk_files > 5 %}
- 🟡 **Consider refactoring {{ dependency_graph.metrics.high_risk_files }} high-risk files**
{% endif %}
- 🟢 **Add dependency tracking to CI/CD pipeline**

### Regular Monitoring
- Review dependency graph after major changes
- Monitor for new circular dependencies
- Track external dependency updates and security advisories
- Validate test coverage for high-risk modules

---

*Generated automatically by [Deepflow](https://github.com/scs03004/deepflow) on {{ generated_date }}*