# Deepflow MCP Integration Guide

**Configure Claude Code to use Deepflow's dependency analysis tools directly through the Model Context Protocol (MCP).**

## 🔌 Quick Setup

### Step 1: Verify Deepflow MCP Server
First, ensure the MCP server works on your system:

```bash
# Navigate to deepflow project
cd "C:\Users\Sebastian\PycharmProjects\npcgpt-dependency\dependency-toolkit"

# Test MCP server starts correctly
python -m deepflow.mcp.server
# Press Ctrl+C to stop when you see it's running
```

### Step 2: Find Claude Code Configuration Directory
Locate your Claude Code configuration folder:
- **Windows**: `%APPDATA%\.claude\` or `%USERPROFILE%\.claude\`
- **Mac/Linux**: `~/.claude/`

### Step 3: Configure MCP Server Using Claude Code CLI
Use the Claude Code CLI to automatically configure the MCP server:

```bash
# Navigate to your deepflow project directory
cd "C:\Users\Sebastian\PycharmProjects\npcgpt-dependency\dependency-toolkit"

# Add the deepflow MCP server (use quotes around the full command)
claude mcp add deepflow "python -m deepflow.mcp.server" --scope user
```

**Alternative Manual Configuration** (if CLI method doesn't work):
Create or edit the MCP configuration file in your Claude Code config directory.

**File**: `mcp_servers.json` (or check Claude Code docs for exact filename)

```json
{
  "mcpServers": {
    "deepflow": {
      "command": "python",
      "args": ["-m", "deepflow.mcp.server"],
      "cwd": "C:\\Users\\Sebastian\\PycharmProjects\\npcgpt-dependency\\dependency-toolkit",
      "env": {
        "PYTHONPATH": "C:\\Users\\Sebastian\\PycharmProjects\\npcgpt-dependency\\dependency-toolkit"
      }
    }
  }
}
```

**Important Notes**:
- Adjust the `cwd` path to match your actual deepflow installation location
- When using `claude mcp add`, quote the full command as a single string: `"python -m deepflow.mcp.server"`
- The `--scope user` option makes the server available across all projects

### Step 4: Verify Configuration
Check that the MCP server was configured correctly:

```bash
# List configured MCP servers
claude mcp list

# You should see output like:
# deepflow: python -m deepflow.mcp.server - ✓ Connected
```

### Step 5: Restart Claude Code (if needed)
If you used manual configuration, close and restart Claude Code to load the new MCP server configuration.

## 🛠️ Available MCP Tools

Once configured, you'll have access to these tools in Claude Code:

### 1. `analyze_dependencies`
**Purpose**: Analyze project dependencies and create visualizations

**Usage in Claude Code**:
```
"Use the analyze_dependencies tool to analyze the current project"
"Generate a dependency visualization in HTML format"
"Show me the dependency structure with AI-awareness enabled"
```

**Parameters**:
- `project_path`: Path to analyze (default: current directory)
- `format`: Output format - "text", "html", or "json" (default: "text")
- `ai_awareness`: Enable AI-friendly analysis (default: true)

### 2. `analyze_code_quality`
**Purpose**: Analyze code quality, detect unused imports, coupling issues, and technical debt

**Usage in Claude Code**:
```
"Run code quality analysis on this project"
"Check for unused imports and fix them"
"Analyze technical debt and coupling issues"
```

**Parameters**:
- `project_path`: Path to analyze (default: current directory)
- `analysis_type`: "all", "imports", "coupling", "architecture", "debt", or "ai_context"
- `fix_imports`: Automatically fix unused imports (default: false)

### 3. `validate_commit`
**Purpose**: Validate code changes before commit using pre-commit hooks

**Usage in Claude Code**:
```
"Validate the current changes before committing"
"Check if the code is ready for commit"
"Run pre-commit validation on the current changes"
```

**Parameters**:
- `project_path`: Path to validate (default: current directory)
- `check_dependencies`: Check for dependency issues (default: true)
- `check_patterns`: Check for pattern consistency (default: true)

### 4. `generate_documentation`
**Purpose**: Generate project documentation including dependency maps and architecture overviews

**Usage in Claude Code**:
```
"Generate dependency documentation for this project"
"Create an architecture overview document"
"Generate API documentation"
```

**Parameters**:
- `project_path`: Path to document (default: current directory)
- `doc_type`: "dependency_map", "architecture_overview", or "api_docs"
- `output_path`: Where to save documentation (optional)

## 🧪 Testing the Integration

### Verify MCP Tools are Available
Once configured, verify the connection and try these commands in Claude Code:

```bash
# First, verify the server is connected
claude mcp list

# You should see:
# deepflow: python -m deepflow.mcp.server - ✓ Connected
```

Then try these commands in Claude Code:

```
1. "What MCP tools are available?"
2. "Use the analyze_dependencies tool on the current project"
3. "Run code quality analysis using the analyze_code_quality tool"
```

### Expected Behavior
- `claude mcp list` should show deepflow server as "✓ Connected"
- Claude Code should recognize the deepflow MCP server automatically
- Tools should appear in the available tools list
- You should get structured analysis results in JSON format

## 🚨 Troubleshooting

### Common Issues

**1. MCP Server Not Found**
```bash
# Verify deepflow is installed
python -c "import deepflow.mcp.server; print('MCP server available')"

# Check if command works manually
python -m deepflow.mcp.server
```

**2. Path Issues**
- Ensure the `cwd` path in configuration matches your deepflow installation
- Use forward slashes `/` or escaped backslashes `\\` in JSON
- Verify Python can find the deepflow module from that directory

**3. Claude Code Not Recognizing Server**
- Run `claude mcp list` to check if server is configured and connected
- If using manual configuration: Check Claude Code documentation for correct MCP configuration file name
- If using manual configuration: Verify JSON syntax is valid
- Restart Claude Code after configuration changes
- Check Claude Code logs for MCP-related errors

**4. Command Parsing Issues**
- When using `claude mcp add`, always quote the full command: `"python -m deepflow.mcp.server"`
- Do not split command arguments separately, use a single quoted string
- If getting "unknown option" errors, check that you're quoting the command properly

**5. Tool Execution Failures**
```bash
# Test deepflow tools work independently
deepflow-visualizer . --help
deepflow-analyzer . --help
```

### Debugging Steps
1. **Test MCP server manually**: Run `python -m deepflow.mcp.server` to see if it starts
2. **Check configuration**: Run `claude mcp list` to verify server is configured and connected
3. **Check tool availability**: Verify CLI tools work: `deepflow-visualizer --help`
4. **Validate configuration**: If using manual config, ensure JSON syntax is correct
5. **Check logs**: Look for MCP-related errors in Claude Code logs
6. **Verify paths**: Ensure all paths in configuration are absolute and correct
7. **Test command syntax**: Make sure you're quoting the full command when using `claude mcp add`

## 🎯 Example Usage Session

Once configured, you can have conversations like this in Claude Code:

```
You: "Analyze the dependency structure of this project"
Claude: [Uses analyze_dependencies tool] "I've analyzed your project and found..."

You: "Are there any code quality issues?"
Claude: [Uses analyze_code_quality tool] "I found several unused imports and..."

You: "Generate documentation for this project"
Claude: [Uses generate_documentation tool] "I've generated comprehensive documentation..."
```

## 🔧 Advanced Configuration

### Multiple Projects
You can configure multiple deepflow servers for different projects:

```json
{
  "mcpServers": {
    "deepflow-main": {
      "command": "python",
      "args": ["-m", "deepflow.mcp.server"],
      "cwd": "/path/to/main/project"
    },
    "deepflow-secondary": {
      "command": "python",
      "args": ["-m", "deepflow.mcp.server"],
      "cwd": "/path/to/secondary/project"
    }
  }
}
```

### Environment Variables
Add environment variables if needed:

```json
{
  "mcpServers": {
    "deepflow": {
      "command": "python",
      "args": ["-m", "deepflow.mcp.server"],
      "cwd": "C:\\Users\\Sebastian\\PycharmProjects\\npcgpt-dependency\\dependency-toolkit",
      "env": {
        "PYTHONPATH": "C:\\Users\\Sebastian\\PycharmProjects\\npcgpt-dependency\\dependency-toolkit",
        "DEEPFLOW_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

## 📚 Additional Resources

- **Deepflow Documentation**: See `CLAUDE.md` for comprehensive development commands
- **MCP Protocol**: [Model Context Protocol Specification](https://spec.modelcontextprotocol.io/)
- **Claude Code MCP**: Check Claude Code documentation for MCP configuration details

---

**🎉 Once configured, you'll have the first dependency analysis MCP server integrated directly into your Claude Code workflow!**