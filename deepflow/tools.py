#!/usr/bin/env python3
"""
Deepflow Tools Module
====================

Provides access to deepflow tools with graceful fallbacks.
This module ensures tools can be imported even if some dependencies are missing.
"""

import sys
import warnings
from pathlib import Path
from typing import Any

# Add tools directory to path for imports
_tools_dir = Path(__file__).parent.parent / "tools"
if str(_tools_dir) not in sys.path:
    sys.path.insert(0, str(_tools_dir))

# Tool availability flags
VISUALIZER_AVAILABLE = False
ANALYZER_AVAILABLE = False
VALIDATOR_AVAILABLE = False
MONITOR_AVAILABLE = False
DOC_GENERATOR_AVAILABLE = False

# Graceful imports with availability tracking
try:
    from dependency_visualizer import DependencyVisualizer
    VISUALIZER_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Dependency visualizer not available: {e}")
    DependencyVisualizer = None

try:
    from code_analyzer import CodeAnalyzer
    ANALYZER_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Code analyzer not available: {e}")
    CodeAnalyzer = None

# Note: pre_commit_validator has hard sys.exit() calls, so we skip it for now
VALIDATOR_AVAILABLE = False
PreCommitValidator = None

# Uncomment when the tool is fixed to not call sys.exit()
# try:
#     from pre_commit_validator import PreCommitValidator
#     VALIDATOR_AVAILABLE = True
# except ImportError as e:
#     warnings.warn(f"Pre-commit validator not available: {e}")
#     PreCommitValidator = None

try:
    from monitoring_dashboard import DependencyMonitor
    MONITOR_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Monitoring dashboard not available: {e}")
    DependencyMonitor = None

try:
    from doc_generator import DocumentationGenerator
    DOC_GENERATOR_AVAILABLE = True
except ImportError as e:
    warnings.warn(f"Documentation generator not available: {e}")
    DocumentationGenerator = None

# Additional tools with known dependency issues - import with extra caution
CI_CD_INTEGRATOR_AVAILABLE = False
AI_SESSION_TRACKER_AVAILABLE = False

# Note: ci_cd_integrator and ai_session_tracker have hard sys.exit() calls
# on import failure, so we skip them for now to ensure graceful operation
CICDIntegrator = None
AISessionTracker = None

# Uncomment these when the tools are fixed to not call sys.exit()
# try:
#     from ci_cd_integrator import CICDIntegrator
#     CI_CD_INTEGRATOR_AVAILABLE = True
# except ImportError as e:
#     warnings.warn(f"CI/CD integrator not available: {e}")
#     CICDIntegrator = None
#
# try:
#     from ai_session_tracker import AISessionTracker  
#     AI_SESSION_TRACKER_AVAILABLE = True
# except ImportError as e:
#     warnings.warn(f"AI session tracker not available: {e}")
#     AISessionTracker = None

# Availability summary
TOOLS_AVAILABLE = {
    "dependency_visualizer": VISUALIZER_AVAILABLE,
    "code_analyzer": ANALYZER_AVAILABLE,
    "pre_commit_validator": VALIDATOR_AVAILABLE,
    "monitoring_dashboard": MONITOR_AVAILABLE,
    "doc_generator": DOC_GENERATOR_AVAILABLE,
    "ci_cd_integrator": CI_CD_INTEGRATOR_AVAILABLE,
    "ai_session_tracker": AI_SESSION_TRACKER_AVAILABLE,
}

def get_available_tools():
    """Get list of available tools."""
    return [tool for tool, available in TOOLS_AVAILABLE.items() if available]

def get_unavailable_tools():
    """Get list of unavailable tools."""
    return [tool for tool, available in TOOLS_AVAILABLE.items() if not available]

def check_tool_availability(tool_name: str) -> bool:
    """Check if a specific tool is available."""
    return TOOLS_AVAILABLE.get(tool_name, False)

def require_tool(tool_name: str) -> Any:
    """
    Require a tool to be available, raising ImportError if not.
    
    Args:
        tool_name: Name of the tool to require
        
    Returns:
        The tool class if available
        
    Raises:
        ImportError: If the tool is not available
    """
    tool_map = {
        "dependency_visualizer": DependencyVisualizer,
        "code_analyzer": CodeAnalyzer,
        "pre_commit_validator": DependencyValidator,
        "monitoring_dashboard": DependencyMonitor,
        "doc_generator": DocumentationGenerator,
        "ci_cd_integrator": CICDIntegrator,
        "ai_session_tracker": AISessionTracker,
    }
    
    if not check_tool_availability(tool_name):
        raise ImportError(f"Tool '{tool_name}' is not available. Check dependencies.")
    
    return tool_map[tool_name]

# Export available tools
__all__ = [
    "DependencyVisualizer",
    "CodeAnalyzer", 
    "DependencyValidator",
    "DependencyMonitor",
    "DocumentationGenerator",
    "CICDIntegrator",
    "AISessionTracker",
    "TOOLS_AVAILABLE",
    "get_available_tools",
    "get_unavailable_tools", 
    "check_tool_availability",
    "require_tool",
]