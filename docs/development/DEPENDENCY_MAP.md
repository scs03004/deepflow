# deepflow Dependency Map

**Generated**: 2025-08-23
**Project Path**: C:\Users\Sebastian\PycharmProjects\npcgpt-dependency\dependency-toolkit
**Language**: Python
**Framework**: Not detected

## 📊 Project Metrics

- **Total Files**: 31
- **Total Imports**: 43
- **External Dependencies**: 45
- **High Risk Files**: 3
- **Circular Dependencies**: 0
- **Lines of Code**: 12822

## 🚨 High-Risk Files (Critical Components)


### 
- **Path**: ``
- **Risk Level**: 
- **Lines of Code**: 
- **Imports**: 0 modules
- **Imported By**: 0 modules
- **Dependencies**: None
- **Dependents**: None


### 
- **Path**: ``
- **Risk Level**: 
- **Lines of Code**: 
- **Imports**: 0 modules
- **Imported By**: 0 modules
- **Dependencies**: None
- **Dependents**: None


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
- **LOC**: 90
- **Imports**: None
- **Imported By**: None


### deepflow.tools
- **File**: `deepflow\tools.py`
- **Risk Level**: LOW
- **LOC**: 157
- **Imports**: None
- **Imported By**: None


### deepflow.__init__
- **File**: `deepflow\__init__.py`
- **Risk Level**: LOW
- **LOC**: 65
- **Imports**: None
- **Imported By**: None


### deepflow.mcp.server
- **File**: `deepflow\mcp\server.py`
- **Risk Level**: HIGH
- **LOC**: 510
- **Imports**: None
- **Imported By**: tests.test_fallbacks_comprehensive, tests.mcp.test_mcp_entry_points, tests.mcp.test_mcp_fallbacks, tests.mcp.test_mcp_tools


### deepflow.mcp.__init__
- **File**: `deepflow\mcp\__init__.py`
- **Risk Level**: LOW
- **LOC**: 39
- **Imports**: None
- **Imported By**: None


### deepflow.mcp.__main__
- **File**: `deepflow\mcp\__main__.py`
- **Risk Level**: LOW
- **LOC**: 12
- **Imports**: None
- **Imported By**: None


### tests.conftest
- **File**: `tests\conftest.py`
- **Risk Level**: LOW
- **LOC**: 392
- **Imports**: None
- **Imported By**: None


### tests.test_fallbacks_comprehensive
- **File**: `tests\test_fallbacks_comprehensive.py`
- **Risk Level**: MEDIUM
- **LOC**: 473
- **Imports**: deepflow.mcp.server, deepflow.mcp
- **Imported By**: None


### tests.test_runner
- **File**: `tests\test_runner.py`
- **Risk Level**: LOW
- **LOC**: 249
- **Imports**: None
- **Imported By**: None


### tests.__init__
- **File**: `tests\__init__.py`
- **Risk Level**: LOW
- **LOC**: 12
- **Imports**: None
- **Imported By**: None


### tests.integration.test_cli_commands
- **File**: `tests\integration\test_cli_commands.py`
- **Risk Level**: MEDIUM
- **LOC**: 621
- **Imports**: deepflow.mcp, deepflow.mcp
- **Imported By**: None


### tests.integration.test_optional_dependencies
- **File**: `tests\integration\test_optional_dependencies.py`
- **Risk Level**: MEDIUM
- **LOC**: 475
- **Imports**: deepflow
- **Imported By**: None


### tests.integration.test_package_imports
- **File**: `tests\integration\test_package_imports.py`
- **Risk Level**: MEDIUM
- **LOC**: 481
- **Imports**: deepflow, deepflow, deepflow
- **Imported By**: None


### tests.integration.__init__
- **File**: `tests\integration\__init__.py`
- **Risk Level**: LOW
- **LOC**: 1
- **Imports**: None
- **Imported By**: None


### tests.mcp.test_mcp_entry_points
- **File**: `tests\mcp\test_mcp_entry_points.py`
- **Risk Level**: MEDIUM
- **LOC**: 387
- **Imports**: deepflow.mcp.server, deepflow.mcp, deepflow.mcp, deepflow.mcp.server, deepflow.mcp.server, deepflow.mcp.server, deepflow.mcp, deepflow.mcp, deepflow.mcp.server, deepflow.mcp.server, deepflow.mcp.server, deepflow.mcp.server
- **Imported By**: None


### tests.mcp.test_mcp_fallbacks
- **File**: `tests\mcp\test_mcp_fallbacks.py`
- **Risk Level**: MEDIUM
- **LOC**: 358
- **Imports**: deepflow.mcp.server, deepflow.mcp.server, deepflow.mcp.server, deepflow.mcp.server, deepflow.mcp.server, deepflow.mcp.server, deepflow.mcp.server, deepflow.mcp, deepflow.mcp
- **Imported By**: None


### tests.mcp.test_mcp_server
- **File**: `tests\mcp\test_mcp_server.py`
- **Risk Level**: MEDIUM
- **LOC**: 472
- **Imports**: deepflow.mcp
- **Imported By**: None


### tests.mcp.test_mcp_tools
- **File**: `tests\mcp\test_mcp_tools.py`
- **Risk Level**: MEDIUM
- **LOC**: 513
- **Imports**: deepflow.mcp.server, deepflow.mcp.server, deepflow.mcp.server, deepflow.mcp.server, deepflow.mcp.server, deepflow.mcp.server, deepflow.mcp.server, deepflow.mcp.server, deepflow.mcp.server, deepflow.mcp.server, deepflow.mcp.server, deepflow.mcp.server, deepflow.mcp.server
- **Imported By**: None


### tests.mcp.__init__
- **File**: `tests\mcp\__init__.py`
- **Risk Level**: LOW
- **LOC**: 1
- **Imports**: None
- **Imported By**: None


### tests.unit.test_code_analyzer
- **File**: `tests\unit\test_code_analyzer.py`
- **Risk Level**: LOW
- **LOC**: 428
- **Imports**: None
- **Imported By**: None


### tests.unit.test_dependency_visualizer
- **File**: `tests\unit\test_dependency_visualizer.py`
- **Risk Level**: LOW
- **LOC**: 296
- **Imports**: None
- **Imported By**: None


### tests.unit.test_doc_generator
- **File**: `tests\unit\test_doc_generator.py`
- **Risk Level**: LOW
- **LOC**: 321
- **Imports**: None
- **Imported By**: None


### tests.unit.test_tools_import
- **File**: `tests\unit\test_tools_import.py`
- **Risk Level**: LOW
- **LOC**: 320
- **Imports**: None
- **Imported By**: None


### tests.unit.__init__
- **File**: `tests\unit\__init__.py`
- **Risk Level**: LOW
- **LOC**: 1
- **Imports**: None
- **Imported By**: None


### tools.ai_session_tracker
- **File**: `tools\ai_session_tracker.py`
- **Risk Level**: MEDIUM
- **LOC**: 537
- **Imports**: None
- **Imported By**: None


### tools.ci_cd_integrator
- **File**: `tools\ci_cd_integrator.py`
- **Risk Level**: MEDIUM
- **LOC**: 749
- **Imports**: None
- **Imported By**: None


### tools.code_analyzer
- **File**: `tools\code_analyzer.py`
- **Risk Level**: HIGH
- **LOC**: 1655
- **Imports**: None
- **Imported By**: None


### tools.dependency_visualizer
- **File**: `tools\dependency_visualizer.py`
- **Risk Level**: MEDIUM
- **LOC**: 754
- **Imports**: None
- **Imported By**: None


### tools.doc_generator
- **File**: `tools\doc_generator.py`
- **Risk Level**: HIGH
- **LOC**: 1092
- **Imports**: None
- **Imported By**: None


### tools.monitoring_dashboard
- **File**: `tools\monitoring_dashboard.py`
- **Risk Level**: MEDIUM
- **LOC**: 654
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
- **Used By**: setup, deepflow.tools, deepflow.mcp.server, tests.conftest, tests.test_fallbacks_comprehensive, tests.test_runner, tests.integration.test_cli_commands, tests.integration.test_optional_dependencies, tests.integration.test_package_imports, tests.integration.test_package_imports, tests.mcp.test_mcp_entry_points, tests.mcp.test_mcp_fallbacks, tests.mcp.test_mcp_server, tests.unit.test_code_analyzer, tests.unit.test_dependency_visualizer, tests.unit.test_doc_generator, tests.unit.test_tools_import, tools.ai_session_tracker, tools.ci_cd_integrator, tools.code_analyzer, tools.dependency_visualizer, tools.doc_generator, tools.monitoring_dashboard, tools.pre_commit_validator
- **Module Count**: 24


### sys
- **Used By**: deepflow.tools, deepflow.mcp.server, deepflow.mcp.server, tests.conftest, tests.test_fallbacks_comprehensive, tests.test_runner, tests.integration.test_cli_commands, tests.integration.test_optional_dependencies, tests.integration.test_package_imports, tests.integration.test_package_imports, tests.mcp.test_mcp_entry_points, tests.mcp.test_mcp_fallbacks, tests.mcp.test_mcp_server, tests.unit.test_code_analyzer, tests.unit.test_dependency_visualizer, tests.unit.test_doc_generator, tests.unit.test_tools_import, tools.ai_session_tracker, tools.ci_cd_integrator, tools.code_analyzer, tools.dependency_visualizer, tools.dependency_visualizer, tools.doc_generator, tools.doc_generator, tools.monitoring_dashboard, tools.pre_commit_validator
- **Module Count**: 26


### warnings
- **Used By**: deepflow.tools
- **Module Count**: 1


### typing
- **Used By**: deepflow.tools, deepflow.mcp.server, tests.conftest, tests.test_runner, tools.ai_session_tracker, tools.ci_cd_integrator, tools.code_analyzer, tools.dependency_visualizer, tools.doc_generator, tools.monitoring_dashboard, tools.pre_commit_validator
- **Module Count**: 11


### dependency_visualizer
- **Used By**: deepflow.tools, deepflow.mcp.server, tests.test_fallbacks_comprehensive, tests.test_fallbacks_comprehensive, tests.test_fallbacks_comprehensive, tests.test_fallbacks_comprehensive, tests.test_runner, tests.integration.test_cli_commands, tests.integration.test_cli_commands, tests.integration.test_cli_commands, tests.integration.test_cli_commands, tests.integration.test_cli_commands, tests.integration.test_cli_commands, tests.integration.test_cli_commands, tests.integration.test_optional_dependencies, tests.integration.test_optional_dependencies, tests.integration.test_optional_dependencies, tests.integration.test_optional_dependencies, tests.integration.test_optional_dependencies, tests.integration.test_optional_dependencies, tests.integration.test_optional_dependencies, tests.integration.test_optional_dependencies, tests.integration.test_optional_dependencies, tests.integration.test_optional_dependencies, tests.integration.test_optional_dependencies, tests.integration.test_package_imports, tests.integration.test_package_imports, tests.integration.test_package_imports, tests.mcp.test_mcp_fallbacks, tests.unit.test_dependency_visualizer, tests.unit.test_tools_import, tests.unit.test_tools_import, tests.unit.test_tools_import, tests.unit.test_tools_import, tests.unit.test_tools_import, tools.doc_generator, tools.doc_generator, tools.monitoring_dashboard, tools.monitoring_dashboard
- **Module Count**: 39


### code_analyzer
- **Used By**: deepflow.tools, deepflow.mcp.server, tests.integration.test_cli_commands, tests.integration.test_cli_commands, tests.unit.test_code_analyzer, tests.unit.test_tools_import, tests.unit.test_tools_import
- **Module Count**: 7


### monitoring_dashboard
- **Used By**: deepflow.tools
- **Module Count**: 1


### doc_generator
- **Used By**: deepflow.tools, deepflow.mcp.server, tests.integration.test_cli_commands, tests.integration.test_cli_commands, tests.integration.test_cli_commands, tests.integration.test_optional_dependencies, tests.integration.test_optional_dependencies, tests.unit.test_doc_generator, tests.unit.test_tools_import
- **Module Count**: 9


### asyncio
- **Used By**: deepflow.mcp.server, tests.test_fallbacks_comprehensive, tests.mcp.test_mcp_entry_points, tests.mcp.test_mcp_fallbacks, tests.mcp.test_mcp_fallbacks, tests.mcp.test_mcp_server
- **Module Count**: 6


### json
- **Used By**: deepflow.mcp.server, tests.integration.test_package_imports, tests.mcp.test_mcp_tools, tests.unit.test_dependency_visualizer, tools.ai_session_tracker, tools.ci_cd_integrator, tools.code_analyzer, tools.doc_generator, tools.monitoring_dashboard, tools.pre_commit_validator
- **Module Count**: 10


### logging
- **Used By**: deepflow.mcp.server
- **Module Count**: 1


### mcp
- **Used By**: deepflow.mcp.server, deepflow.mcp.server, deepflow.mcp.server, deepflow.mcp.__init__, deepflow.mcp.__init__
- **Module Count**: 5


### pre_commit_validator
- **Used By**: deepflow.mcp.server
- **Module Count**: 1


### server
- **Used By**: deepflow.mcp.__main__
- **Module Count**: 1


### os
- **Used By**: tests.conftest, tests.test_fallbacks_comprehensive, tests.integration.test_package_imports, tools.ci_cd_integrator, tools.code_analyzer, tools.dependency_visualizer, tools.doc_generator, tools.monitoring_dashboard, tools.pre_commit_validator
- **Module Count**: 9


### tempfile
- **Used By**: tests.conftest, tools.pre_commit_validator
- **Module Count**: 2


### unittest
- **Used By**: tests.conftest, tests.test_fallbacks_comprehensive, tests.integration.test_cli_commands, tests.integration.test_optional_dependencies, tests.integration.test_package_imports, tests.mcp.test_mcp_entry_points, tests.mcp.test_mcp_fallbacks, tests.mcp.test_mcp_server, tests.mcp.test_mcp_tools, tests.unit.test_code_analyzer, tests.unit.test_dependency_visualizer, tests.unit.test_doc_generator, tests.unit.test_tools_import
- **Module Count**: 13


### pytest
- **Used By**: tests.conftest, tests.test_fallbacks_comprehensive, tests.test_runner, tests.integration.test_cli_commands, tests.integration.test_optional_dependencies, tests.integration.test_package_imports, tests.mcp.test_mcp_entry_points, tests.mcp.test_mcp_fallbacks, tests.mcp.test_mcp_server, tests.mcp.test_mcp_tools, tests.unit.test_code_analyzer, tests.unit.test_dependency_visualizer, tests.unit.test_doc_generator, tests.unit.test_tools_import
- **Module Count**: 14


### io
- **Used By**: tests.conftest, tools.dependency_visualizer
- **Module Count**: 2


### subprocess
- **Used By**: tests.test_fallbacks_comprehensive, tests.mcp.test_mcp_entry_points, tools.pre_commit_validator
- **Module Count**: 3


### importlib
- **Used By**: tests.test_fallbacks_comprehensive, tests.integration.test_package_imports, tools.pre_commit_validator
- **Module Count**: 3


### deepflow
- **Used By**: tests.test_fallbacks_comprehensive, tests.test_fallbacks_comprehensive, tests.test_fallbacks_comprehensive, tests.test_fallbacks_comprehensive, tests.test_fallbacks_comprehensive, tests.test_fallbacks_comprehensive, tests.test_runner, tests.integration.test_optional_dependencies, tests.integration.test_optional_dependencies, tests.integration.test_optional_dependencies, tests.integration.test_package_imports, tests.integration.test_package_imports, tests.integration.test_package_imports, tests.integration.test_package_imports, tests.integration.test_package_imports, tests.integration.test_package_imports, tests.integration.test_package_imports, tests.integration.test_package_imports, tests.integration.test_package_imports, tests.integration.test_package_imports, tests.integration.test_package_imports, tests.mcp.test_mcp_entry_points, tests.mcp.test_mcp_fallbacks, tests.mcp.test_mcp_fallbacks, tests.mcp.test_mcp_fallbacks, tests.mcp.test_mcp_fallbacks, tests.mcp.test_mcp_fallbacks, tests.unit.test_tools_import, tests.unit.test_tools_import, tests.unit.test_tools_import
- **Module Count**: 30


### threading
- **Used By**: tests.test_fallbacks_comprehensive, tools.monitoring_dashboard
- **Module Count**: 2


### argparse
- **Used By**: tests.test_runner, tools.ai_session_tracker, tools.ci_cd_integrator, tools.code_analyzer, tools.dependency_visualizer, tools.doc_generator, tools.monitoring_dashboard, tools.pre_commit_validator
- **Module Count**: 8


### networkx
- **Used By**: tests.integration.test_optional_dependencies, tools.code_analyzer, tools.dependency_visualizer
- **Module Count**: 3


### time
- **Used By**: tests.integration.test_package_imports, tools.monitoring_dashboard
- **Module Count**: 2


### types
- **Used By**: tests.mcp.test_mcp_entry_points
- **Module Count**: 1


### re
- **Used By**: tests.mcp.test_mcp_entry_points, tools.code_analyzer, tools.doc_generator
- **Module Count**: 3


### builtins
- **Used By**: tests.mcp.test_mcp_fallbacks
- **Module Count**: 1


### dataclasses
- **Used By**: tests.unit.test_code_analyzer, tools.ai_session_tracker, tools.ci_cd_integrator, tools.code_analyzer, tools.dependency_visualizer, tools.doc_generator, tools.monitoring_dashboard, tools.pre_commit_validator
- **Module Count**: 8


### ast
- **Used By**: tests.unit.test_code_analyzer, tests.unit.test_code_analyzer, tests.unit.test_doc_generator, tests.unit.test_doc_generator, tests.unit.test_doc_generator, tools.code_analyzer, tools.dependency_visualizer, tools.doc_generator, tools.pre_commit_validator
- **Module Count**: 9


### datetime
- **Used By**: tools.ai_session_tracker, tools.doc_generator, tools.monitoring_dashboard
- **Module Count**: 3


### hashlib
- **Used By**: tools.ai_session_tracker
- **Module Count**: 1


### rich
- **Used By**: tools.ai_session_tracker, tools.ai_session_tracker, tools.ai_session_tracker, tools.ci_cd_integrator, tools.ci_cd_integrator, tools.code_analyzer, tools.code_analyzer, tools.code_analyzer, tools.dependency_visualizer, tools.dependency_visualizer, tools.dependency_visualizer, tools.dependency_visualizer, tools.doc_generator, tools.monitoring_dashboard, tools.monitoring_dashboard, tools.monitoring_dashboard, tools.monitoring_dashboard, tools.monitoring_dashboard, tools.pre_commit_validator, tools.pre_commit_validator, tools.pre_commit_validator, tools.pre_commit_validator
- **Module Count**: 22


### git
- **Used By**: tools.ai_session_tracker, tools.ci_cd_integrator, tools.pre_commit_validator
- **Module Count**: 3


### collections
- **Used By**: tools.ai_session_tracker, tools.code_analyzer, tools.dependency_visualizer
- **Module Count**: 3


### yaml
- **Used By**: tools.ci_cd_integrator, tools.pre_commit_validator
- **Module Count**: 2


### codecs
- **Used By**: tools.dependency_visualizer, tools.doc_generator
- **Module Count**: 2


### jinja2
- **Used By**: tools.doc_generator
- **Module Count**: 1


### toml
- **Used By**: tools.doc_generator
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



## ⚠️ Circular Dependencies


✅ **No circular dependencies detected!**


## 🧪 Testing Strategy

### Priority Testing Areas
1. **High-Risk Files**: Focus on 3 critical components
2. **External Integrations**: Test all 45 external dependencies
3. **Circular Dependencies**: None detected

### Recommended Test Coverage
- **Unit Tests**: All high-risk files (3 files)
- **Integration Tests**: Cross-module dependency chains
- **External Tests**: Mock all 45 external services

## 🚨 Change Impact Analysis

### When Updating High-Risk Files:


****:
1. **Direct Impact**: 0 modules need review
2. **Test Coverage**: Run integration tests for dependent modules  
3. **Documentation**: Update if public interfaces change
4. **Dependent Files**: 


****:
1. **Direct Impact**: 0 modules need review
2. **Test Coverage**: Run integration tests for dependent modules  
3. **Documentation**: Update if public interfaces change
4. **Dependent Files**: 


****:
1. **Direct Impact**: 0 modules need review
2. **Test Coverage**: Run integration tests for dependent modules  
3. **Documentation**: Update if public interfaces change
4. **Dependent Files**: 



## 📈 Quality Metrics

- **Dependency Complexity**: 43 total import relationships
- **Modularity Score**: 1.4 imports per file
- **External Coupling**: 45 external dependencies
- **Risk Distribution**: 3 high-risk files

## 💡 Maintenance Recommendations

### Immediate Actions


- 🟢 **Add dependency tracking to CI/CD pipeline**

### Regular Monitoring
- Review dependency graph after major changes
- Monitor for new circular dependencies
- Track external dependency updates and security advisories
- Validate test coverage for high-risk modules

---

*Generated automatically by [Deepflow](https://github.com/scs03004/deepflow) on 2025-08-23*