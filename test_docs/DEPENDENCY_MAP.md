# dependency-toolkit Dependency Map

**Generated**: 2025-08-17
**Project Path**: C:\Users\Sebastian\NPCGPT\NPCGPT\dependency-toolkit
**Language**: Python
**Framework**: Not detected

## 📊 Project Metrics

- **Total Files**: 7
- **Total Imports**: 0
- **External Dependencies**: 33
- **High Risk Files**: 1
- **Circular Dependencies**: 0
- **Lines of Code**: 5154

## 🚨 High-Risk Files (Critical Components)


### 
- **Path**: ``
- **Risk Level**: 
- **Lines of Code**: 
- **Imports**: 0 modules
- **Imported By**: 0 modules
- **Dependencies**: None
- **Dependents**: None



## 🔗 All Internal Dependencies


### setup
- **File**: `setup.py`
- **Risk Level**: LOW
- **LOC**: 65
- **Imports**: None
- **Imported By**: None


### tools.ci_cd_integrator
- **File**: `tools\ci_cd_integrator.py`
- **Risk Level**: MEDIUM
- **LOC**: 751
- **Imports**: None
- **Imported By**: None


### tools.code_analyzer
- **File**: `tools\code_analyzer.py`
- **Risk Level**: MEDIUM
- **LOC**: 986
- **Imports**: None
- **Imported By**: None


### tools.dependency_visualizer
- **File**: `tools\dependency_visualizer.py`
- **Risk Level**: MEDIUM
- **LOC**: 926
- **Imports**: None
- **Imported By**: None


### tools.doc_generator
- **File**: `tools\doc_generator.py`
- **Risk Level**: HIGH
- **LOC**: 1060
- **Imports**: None
- **Imported By**: None


### tools.monitoring_dashboard
- **File**: `tools\monitoring_dashboard.py`
- **Risk Level**: MEDIUM
- **LOC**: 659
- **Imports**: None
- **Imported By**: None


### tools.pre_commit_validator
- **File**: `tools\pre_commit_validator.py`
- **Risk Level**: MEDIUM
- **LOC**: 707
- **Imports**: None
- **Imported By**: None



## 📦 External Dependencies


### setuptools
- **Used By**: setup
- **Module Count**: 1


### pathlib
- **Used By**: setup, tools.ci_cd_integrator, tools.code_analyzer, tools.dependency_visualizer, tools.doc_generator, tools.monitoring_dashboard, tools.pre_commit_validator
- **Module Count**: 7


### os
- **Used By**: tools.ci_cd_integrator, tools.code_analyzer, tools.dependency_visualizer, tools.doc_generator, tools.monitoring_dashboard, tools.pre_commit_validator
- **Module Count**: 6


### sys
- **Used By**: tools.ci_cd_integrator, tools.code_analyzer, tools.dependency_visualizer, tools.dependency_visualizer, tools.doc_generator, tools.doc_generator, tools.monitoring_dashboard, tools.pre_commit_validator
- **Module Count**: 8


### json
- **Used By**: tools.ci_cd_integrator, tools.code_analyzer, tools.dependency_visualizer, tools.doc_generator, tools.monitoring_dashboard, tools.pre_commit_validator
- **Module Count**: 6


### argparse
- **Used By**: tools.ci_cd_integrator, tools.code_analyzer, tools.dependency_visualizer, tools.doc_generator, tools.monitoring_dashboard, tools.pre_commit_validator
- **Module Count**: 6


### subprocess
- **Used By**: tools.ci_cd_integrator, tools.doc_generator, tools.monitoring_dashboard, tools.pre_commit_validator
- **Module Count**: 4


### typing
- **Used By**: tools.ci_cd_integrator, tools.code_analyzer, tools.dependency_visualizer, tools.doc_generator, tools.monitoring_dashboard, tools.pre_commit_validator
- **Module Count**: 6


### dataclasses
- **Used By**: tools.ci_cd_integrator, tools.code_analyzer, tools.dependency_visualizer, tools.doc_generator, tools.monitoring_dashboard, tools.pre_commit_validator
- **Module Count**: 6


### datetime
- **Used By**: tools.ci_cd_integrator, tools.doc_generator, tools.monitoring_dashboard
- **Module Count**: 3


### yaml
- **Used By**: tools.ci_cd_integrator, tools.doc_generator, tools.pre_commit_validator
- **Module Count**: 3


### rich
- **Used By**: tools.ci_cd_integrator, tools.ci_cd_integrator, tools.code_analyzer, tools.code_analyzer, tools.code_analyzer, tools.code_analyzer, tools.code_analyzer, tools.dependency_visualizer, tools.dependency_visualizer, tools.dependency_visualizer, tools.dependency_visualizer, tools.dependency_visualizer, tools.dependency_visualizer, tools.doc_generator, tools.doc_generator, tools.monitoring_dashboard, tools.monitoring_dashboard, tools.monitoring_dashboard, tools.monitoring_dashboard, tools.monitoring_dashboard, tools.pre_commit_validator, tools.pre_commit_validator, tools.pre_commit_validator, tools.pre_commit_validator
- **Module Count**: 24


### git
- **Used By**: tools.ci_cd_integrator, tools.pre_commit_validator
- **Module Count**: 2


### ast
- **Used By**: tools.code_analyzer, tools.dependency_visualizer, tools.doc_generator, tools.pre_commit_validator
- **Module Count**: 4


### collections
- **Used By**: tools.code_analyzer, tools.dependency_visualizer
- **Module Count**: 2


### importlib
- **Used By**: tools.code_analyzer, tools.dependency_visualizer, tools.doc_generator, tools.pre_commit_validator
- **Module Count**: 4


### re
- **Used By**: tools.code_analyzer, tools.doc_generator
- **Module Count**: 2


### networkx
- **Used By**: tools.code_analyzer, tools.dependency_visualizer, tools.doc_generator
- **Module Count**: 3


### pandas
- **Used By**: tools.code_analyzer, tools.dependency_visualizer, tools.doc_generator, tools.monitoring_dashboard
- **Module Count**: 4


### matplotlib
- **Used By**: tools.dependency_visualizer
- **Module Count**: 1


### plotly
- **Used By**: tools.dependency_visualizer, tools.dependency_visualizer, tools.dependency_visualizer, tools.monitoring_dashboard, tools.monitoring_dashboard, tools.monitoring_dashboard
- **Module Count**: 6


### io
- **Used By**: tools.dependency_visualizer
- **Module Count**: 1


### codecs
- **Used By**: tools.dependency_visualizer, tools.doc_generator
- **Module Count**: 2


### jinja2
- **Used By**: tools.doc_generator
- **Module Count**: 1


### dependency_visualizer
- **Used By**: tools.doc_generator, tools.doc_generator, tools.monitoring_dashboard, tools.monitoring_dashboard
- **Module Count**: 4


### toml
- **Used By**: tools.doc_generator
- **Module Count**: 1


### time
- **Used By**: tools.monitoring_dashboard
- **Module Count**: 1


### threading
- **Used By**: tools.monitoring_dashboard
- **Module Count**: 1


### psutil
- **Used By**: tools.monitoring_dashboard
- **Module Count**: 1


### flask
- **Used By**: tools.monitoring_dashboard
- **Module Count**: 1


### flask_socketio
- **Used By**: tools.monitoring_dashboard
- **Module Count**: 1


### secrets
- **Used By**: tools.monitoring_dashboard
- **Module Count**: 1


### tempfile
- **Used By**: tools.pre_commit_validator
- **Module Count**: 1



## ⚠️ Circular Dependencies


✅ **No circular dependencies detected!**


## 🧪 Testing Strategy

### Priority Testing Areas
1. **High-Risk Files**: Focus on 1 critical components
2. **External Integrations**: Test all 33 external dependencies
3. **Circular Dependencies**: None detected

### Recommended Test Coverage
- **Unit Tests**: All high-risk files (1 files)
- **Integration Tests**: Cross-module dependency chains
- **External Tests**: Mock all 33 external services

## 🚨 Change Impact Analysis

### When Updating High-Risk Files:


****:
1. **Direct Impact**: 0 modules need review
2. **Test Coverage**: Run integration tests for dependent modules  
3. **Documentation**: Update if public interfaces change
4. **Dependent Files**: 



## 📈 Quality Metrics

- **Dependency Complexity**: 0 total import relationships
- **Modularity Score**: 0.0 imports per file
- **External Coupling**: 33 external dependencies
- **Risk Distribution**: 1 high-risk files

## 💡 Maintenance Recommendations

### Immediate Actions


- 🟢 **Add dependency tracking to CI/CD pipeline**

### Regular Monitoring
- Review dependency graph after major changes
- Monitor for new circular dependencies
- Track external dependency updates and security advisories
- Validate test coverage for high-risk modules

---

*Generated automatically by [Dependency Toolkit](https://github.com/scs03004/dependency-toolkit) on 2025-08-17*