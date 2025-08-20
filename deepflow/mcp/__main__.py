#!/usr/bin/env python3
"""
Entry point for running the MCP server as a module.

Usage:
    python -m deepflow.mcp.server
"""

from .server import main

if __name__ == "__main__":
    main()