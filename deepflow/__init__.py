#!/usr/bin/env python3
"""
Deepflow - AI-Assisted Codebase Management Tools
================================================

Keep AI-assisted codebases clean, consistent, and maintainable.
Specialized tools for Claude Code, Cursor, GitHub Copilot users.

This package provides:
- Dependency visualization and analysis
- Code quality validation and monitoring
- Documentation generation
- CI/CD integration tools
- AI session tracking
- MCP (Model Context Protocol) integration

Usage:
    import deepflow
    
    # Import specific tools
    from deepflow.tools import dependency_visualizer
    from deepflow.tools import code_analyzer
    
    # MCP integration (optional)
    try:
        from deepflow.mcp import server
    except ImportError:
        # MCP not installed - graceful fallback
        pass
"""

__version__ = "2.0.0"
__author__ = "Deepflow Team"
__email__ = "team@deepflow.dev"

# Core package metadata
__all__ = [
    "__version__",
    "__author__", 
    "__email__",
]

# Graceful imports - tools should work even if some dependencies are missing
try:
    from . import tools
    TOOLS_AVAILABLE = True
except ImportError:
    # Tools directory might not be importable as a package yet
    TOOLS_AVAILABLE = False

try:
    from . import mcp
    MCP_AVAILABLE = getattr(mcp, 'MCP_AVAILABLE', False)
except ImportError:
    # MCP integration is optional
    MCP_AVAILABLE = False

# Update __all__ to include availability info
__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "TOOLS_AVAILABLE",
    "MCP_AVAILABLE",
]