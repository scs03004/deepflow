#!/usr/bin/env python3
"""
Deepflow MCP Integration
=======================

Model Context Protocol (MCP) integration for Deepflow tools.
Exposes deepflow functionality through the MCP protocol for use with
AI assistants and language models.

This module provides:
- MCP server implementation
- Tool exposure through MCP protocol
- Graceful fallback if MCP dependencies not installed

Usage:
    # Start MCP server
    python -m deepflow.mcp.server
    
    # Or use entry point
    deepflow-mcp-server

Requirements:
    Install with MCP support: pip install deepflow[mcp]
"""

# Graceful imports for MCP dependencies
try:
    import mcp
    from mcp.server import Server
    from mcp.types import Tool, TextContent
    
    MCP_AVAILABLE = True
    __all__ = ["server", "MCP_AVAILABLE"]
    
except ImportError:
    MCP_AVAILABLE = False
    __all__ = ["MCP_AVAILABLE"]

# Version info
__version__ = "2.0.0"