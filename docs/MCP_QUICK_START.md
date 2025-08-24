# Deepflow MCP Integration - Quick Start Guide

**Get started with Deepflow's Model Context Protocol integration in under 5 minutes!**

## What is MCP Integration?

Deepflow's MCP integration allows AI assistants like Claude Code to directly use dependency analysis tools during development sessions. This enables real-time code quality feedback, dependency visualization, and architectural insights.

## 🚀 Quick Setup

### 1. Install Deepflow with MCP Support

```bash
# Install with MCP dependencies
pip install deepflow[mcp]

# Verify installation
deepflow-mcp-server --help
```

### 2. Start the MCP Server

```bash
# Start the server (runs in background)
deepflow-mcp-server
```

The server will start and wait for connections from MCP clients.

### 3. Connect from Claude Code

In Claude Code, the MCP tools are automatically available when the server is running:

- `analyze_dependencies` - Analyze project dependencies
- `analyze_code_quality` - Code quality and technical debt analysis
- `validate_commit` - Pre-commit validation
- `generate_documentation` - Auto-generate project docs

## 🎯 First Analysis

Try this in Claude Code once connected:

```
Analyze the dependencies in this project and show me any issues.
```

Claude Code will automatically use the `analyze_dependencies` tool to provide insights.

## 📊 Available Tools

### 1. Dependency Analysis
```json
{
  "tool": "analyze_dependencies",
  "parameters": {
    "project_path": ".",
    "format": "text",
    "ai_awareness": true
  }
}
```

**What it does**: Maps your project's dependency structure, identifies circular dependencies, and highlights high-risk areas.

### 2. Code Quality Analysis
```json
{
  "tool": "analyze_code_quality", 
  "parameters": {
    "project_path": ".",
    "analysis_type": "all",
    "fix_imports": false
  }
}
```

**What it does**: Finds unused imports, measures technical debt, analyzes coupling, and provides refactoring suggestions.

### 3. Commit Validation
```json
{
  "tool": "validate_commit",
  "parameters": {
    "project_path": ".",
    "check_dependencies": true,
    "check_patterns": true
  }
}
```

**What it does**: Validates staged changes, checks for breaking changes, and provides impact analysis.

### 4. Documentation Generation
```json
{
  "tool": "generate_documentation",
  "parameters": {
    "project_path": ".",
    "doc_type": "dependency_map",
    "output_path": "docs/dependencies.md"
  }
}
```

**What it does**: Auto-generates dependency maps, architecture overviews, and API documentation.

## 🎭 Example Session

Here's what a typical Claude Code session looks like with Deepflow MCP:

**You**: "Help me clean up this codebase and identify any architectural issues"

**Claude Code**: *Uses analyze_dependencies and analyze_code_quality tools*

"I've analyzed your codebase and found several opportunities for improvement:

1. **Circular Dependencies**: There's a circular dependency between `auth/models.py` and `users/services.py`
2. **Unused Imports**: 23 unused imports across 8 files
3. **Technical Debt**: High coupling in the `core` module suggests refactoring opportunities

Would you like me to fix the unused imports automatically?"

**You**: "Yes, fix the imports and show me the refactoring suggestions"

**Claude Code**: *Uses analyze_code_quality with fix_imports=true*

"I've cleaned up the unused imports and here are the refactoring suggestions..."

## 🛠️ Configuration

### Server Options

The MCP server can be configured with environment variables:

```bash
# Set cache timeout (default: 300 seconds)
export DEEPFLOW_CACHE_TIMEOUT=600

# Set log level
export DEEPFLOW_LOG_LEVEL=DEBUG

# Set maximum project size for analysis
export DEEPFLOW_MAX_PROJECT_SIZE=100000

# Start server with custom settings
deepflow-mcp-server
```

### Claude Code Configuration

Add to your Claude Code MCP configuration:

```json
{
  "mcpServers": {
    "deepflow": {
      "command": "deepflow-mcp-server",
      "args": []
    }
  }
}
```

## 🐛 Common Issues

### "MCP server not found"
- Ensure `deepflow[mcp]` is installed: `pip install deepflow[mcp]`
- Verify server is running: `ps aux | grep deepflow-mcp-server`

### "Tools not available"
- Check if you're in a Python project directory
- Verify project contains `.py` files to analyze

### "Permission denied"
- Ensure you have read access to the project files
- Check if any files are locked by other processes

### "Analysis taking too long"
- Large projects may need more time
- Consider excluding large directories with `.deepflowignore`

## 🎯 Best Practices

### 1. **Project Scope**
- Run analysis from your project root
- Use `.deepflowignore` to exclude unnecessary files
- Start with smaller modules before analyzing entire codebases

### 2. **Performance**
- The server caches results for 5 minutes by default
- Subsequent analyses of unchanged code are much faster
- Clear cache if you want fresh analysis: restart the server

### 3. **Workflow Integration**
- Use `validate_commit` before making git commits
- Generate documentation after major architectural changes
- Run quality analysis regularly during development

### 4. **AI Assistant Usage**
- Be specific about what you want to analyze
- Ask for explanations of complex dependency relationships
- Request actionable recommendations, not just reports

## 🚀 Next Steps

1. **Try the Example Workflows**: See `examples/workflows/` for common development scenarios
2. **Advanced Configuration**: Check `docs/MCP_ADVANCED_CONFIG.md` for server tuning
3. **Custom Integration**: Learn about extending MCP tools in `docs/MCP_EXTENSION_GUIDE.md`
4. **Troubleshooting**: Full debugging guide in `docs/MCP_TROUBLESHOOTING.md`

## 📖 Learn More

- [MCP Integration Guide](./MCP_INTEGRATION_GUIDE.md) - Complete setup and configuration
- [Example Workflows](../examples/workflows/) - Real-world usage examples
- [API Reference](./API_REFERENCE.md) - Detailed tool documentation
- [Contributing](./CONTRIBUTING.md) - Help improve Deepflow MCP

---

**🎉 You're ready to supercharge your development workflow with Deepflow MCP integration!**

Need help? Check the [troubleshooting guide](./MCP_TROUBLESHOOTING.md) or [open an issue](https://github.com/scs03004/deepflow/issues).