# Deepflow MCP Integration

This document describes the Model Context Protocol (MCP) integration for the Deepflow package.

## Overview

Deepflow now supports MCP integration, allowing AI assistants and language models to access deepflow tools through the standardized MCP protocol. This enables seamless integration with Claude Code, Cursor, and other MCP-compatible AI development tools.

## Installation

### Basic Installation
```bash
pip install deepflow
```

### With MCP Support
```bash
pip install deepflow[mcp]
```

### All Features
```bash
pip install deepflow[all]
```

## Usage

### Starting the MCP Server

#### Using the entry point:
```bash
deepflow-mcp-server
```

#### Using Python module:
```bash
python -m deepflow.mcp.server
```

### Available MCP Tools

The MCP server exposes the following tools:

1. **analyze_dependencies**
   - Analyzes project dependencies and creates visualizations
   - Supports text, HTML, and JSON output formats
   - AI-aware dependency analysis

2. **analyze_code_quality**
   - Analyzes code quality and detects issues
   - Unused import detection
   - Coupling analysis
   - Architecture violation detection
   - Technical debt assessment
   - AI context window analysis

3. **validate_commit**
   - Validates code changes before commit
   - Pre-commit hook integration
   - Dependency and pattern checking

4. **generate_documentation**
   - Generates project documentation
   - Dependency maps
   - Architecture overviews
   - API documentation

### Tool Parameters

Each tool accepts various parameters to customize its behavior. Refer to the MCP tool schema for detailed parameter information.

## Integration with AI Assistants

### Claude Code
When running with Claude Code, the MCP server can be configured as a development tool to provide real-time code analysis and dependency management.

### Cursor
Similar integration capabilities with Cursor IDE for enhanced AI-assisted development.

### GitHub Copilot
Can be used alongside Copilot to provide additional code quality insights.

## Architecture

### Package Structure
```
deepflow/
├── __init__.py          # Main package initialization
├── tools.py            # Tool imports with graceful fallbacks
└── mcp/                # MCP integration subpackage
    ├── __init__.py     # MCP package initialization
    ├── server.py       # Main MCP server implementation
    └── __main__.py     # Module entry point
```

### Graceful Fallbacks

The implementation includes comprehensive graceful fallbacks:

1. **Missing MCP Dependencies**: If `mcp>=1.2.0` is not installed, the package continues to work without MCP functionality
2. **Missing Tool Dependencies**: Individual tools that can't be imported are disabled gracefully
3. **Import Safety**: Tools with hard `sys.exit()` calls are excluded to prevent crashes

## Configuration

### Entry Points

The following entry points are configured:

- `deepflow-mcp-server`: Main MCP server entry point
- Traditional tool entry points continue to work as before

### Optional Dependencies

- `mcp`: For MCP protocol support (`mcp>=1.2.0`)
- `dev`: Development dependencies
- `docs`: Documentation dependencies
- `all`: All optional dependencies

## Troubleshooting

### Common Issues

1. **MCP Dependencies Not Found**
   ```
   ERROR: MCP dependencies not found. Install with: pip install deepflow[mcp]
   ```
   **Solution**: Install with MCP support: `pip install deepflow[mcp]`

2. **Tool Import Failures**
   Some tools may fail to import due to missing dependencies. The package will continue to work with available tools.

3. **Windows Console Encoding**
   Some output may have encoding issues on Windows. This is handled gracefully in the implementation.

### Testing

Run the integration test to verify everything is working:

```bash
python test_mcp_integration.py
```

## Development

### Adding New MCP Tools

To add new MCP tools:

1. Add the tool to `deepflow/mcp/server.py`
2. Define the tool schema and handler
3. Update the `get_tools()` method
4. Test the integration

### Tool Safety

When adding new tools, ensure they:
- Don't call `sys.exit()` on import failure
- Handle missing dependencies gracefully
- Return appropriate error messages for MCP clients

## Compatibility

- Python 3.8+
- MCP SDK 1.2.0+
- Compatible with existing deepflow CLI tools
- Backwards compatible with non-MCP usage

## License

Same as the main Deepflow package (MIT License).