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

## 🚀 Real-Time Intelligence Tools ✨ **NEW**

### 5. `start_realtime_monitoring`
**Purpose**: Start live file monitoring with incremental dependency analysis and AI development assistance

**Usage in Claude Code**:
```
"Start real-time monitoring of this project"
"Begin live file monitoring with AI awareness"
"Enable real-time intelligence for the current project"
```

**Parameters**:
- `project_path`: Path to monitor (default: current directory) 
- `ai_awareness`: Enable AI-aware analysis features (default: true)

**What it does**:
- Monitors file changes with 500ms debouncing
- Updates dependency graphs incrementally (10x+ performance)
- Detects AI context window issues (files >1500 tokens)
- Identifies pattern deviations in AI-generated code
- Prevents circular dependencies before they occur
- Suggests file splits for better AI comprehension

### 6. `stop_realtime_monitoring`
**Purpose**: Stop real-time file monitoring

**Usage in Claude Code**:
```
"Stop the real-time monitoring"
"Disable live file watching"
"Turn off real-time intelligence"
```

**Parameters**: None

### 7. `get_realtime_activity`
**Purpose**: Get recent real-time monitoring activity and events

**Usage in Claude Code**:
```
"Show me recent real-time activity"
"What pattern deviations were detected recently?"
"Get the latest file changes and alerts"
```

**Parameters**:
- `limit`: Maximum number of events to return (default: 20)

**Returns**:
- Recent file changes with token estimates
- Pattern deviations (naming, imports, structure)
- Circular dependency alerts 
- File split suggestions
- Duplicate pattern detections
- AI context optimization alerts

### 8. `get_realtime_stats`
**Purpose**: View comprehensive real-time monitoring statistics

**Usage in Claude Code**:
```
"Show me real-time monitoring statistics"
"What are the current monitoring metrics?"
"Give me a summary of real-time intelligence activity"
```

**Parameters**: None

**Returns**:
- Monitoring status and project path
- Performance metrics (files monitored, changes processed)
- Pattern deviation counts
- Circular dependencies prevented
- File split suggestions made
- Duplicate patterns found
- Learned patterns (naming conventions, structures)

## 🧠 AI Session Intelligence Tools ✨ **NEW - Priority 3**

### 9. `start_ai_session`
**Purpose**: Start a new AI development session with context tracking and journaling

**Usage in Claude Code**:
```
"Start a new AI session for feature development"
"Begin session tracking for user authentication feature"  
"Start AI session: refactoring database layer"
```

**Parameters**:
- `session_name`: Name for the AI development session (default: "")
- `session_description`: Description of session goals and context (default: "")
- `session_tags`: Array of tags to categorize the session (default: [])

### 10. `end_ai_session`
**Purpose**: End the current AI development session and save context for future reference

**Usage in Claude Code**:
```
"End the current AI session"
"Complete session and save achievements"
"Finish session with goal: implemented login system"
```

**Parameters**:
- `achievements`: Array of goals achieved during the session (default: [])

### 11. `get_session_context`
**Purpose**: Get current AI session context for continuity and progress tracking

**Usage in Claude Code**:
```
"What's the current session context?"
"Show me the active session details"
"Get session status and progress"
```

**Returns**:
- Current session ID and metadata
- Files modified in this session
- Changes made and AI interactions count
- Session duration and goals

### 12. `restore_session_context`
**Purpose**: Restore a previous AI session context for continuity across development cycles

**Usage in Claude Code**:
```
"Restore session context from yesterday"
"Resume previous session: session_123456"
"Continue from where I left off in session ABC"
```

**Parameters**:
- `session_id`: ID of the session to restore (required)

### 13. `analyze_change_impact`
**Purpose**: Analyze ripple effects and impact of code changes across the project

**Usage in Claude Code**:
```
"Analyze impact of changing user.py"
"What files are affected by modifying the auth system?"
"Show ripple effects of database schema changes"
```

**Parameters**:
- `file_path`: Path to the file that was changed (required)
- `change_type`: Type of change - "addition", "modification", "deletion", "rename" (required)
- `change_details`: Additional details about the change (default: {})

**Returns**:
- Risk assessment (low/medium/high/critical)
- Impact score (0.0 to 1.0)
- Affected files and dependencies
- Test files that need updating
- Documentation that requires changes
- Mitigation suggestions

### 14. `get_session_intelligence`
**Purpose**: Get comprehensive AI session intelligence and analytics

**Usage in Claude Code**:
```
"Show me session intelligence data"
"Get development analytics and insights"
"What patterns have been learned from my coding?"
```

**Parameters**:
- `limit`: Maximum number of entries to return (default: 50)

**Returns**:
- Current and historical session data
- Change impact analyses performed
- Patterns learned from development
- Multi-file coordination progress
- Session journal entries and activities

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

**Core Analysis Tools:**
```
1. "What MCP tools are available?"
2. "Use the analyze_dependencies tool on the current project"
3. "Run code quality analysis using the analyze_code_quality tool"
4. "Generate documentation using the generate_documentation tool"
```

**Real-Time Intelligence Tools:**
```
5. "Start real-time monitoring for this project"
6. "Show me recent real-time activity" 
7. "What are the current monitoring statistics?"
8. "Stop the real-time monitoring"
```

**AI Session Intelligence Tools:**
```
9. "Start AI session: implementing user auth"
10. "End current session with achievements"
11. "What's my current session context?"
12. "Restore session from yesterday"
13. "Analyze impact of changing database.py"
14. "Show me session intelligence and patterns learned"
```

### Expected Behavior
- `claude mcp list` should show deepflow server as "✓ Connected"
- Claude Code should recognize **14 deepflow MCP tools** automatically
- Core tools should provide structured analysis results in JSON format
- Real-time tools should provide live monitoring capabilities
- Pattern deviations and AI context alerts should be detected in real-time
- Session intelligence tools provide development context and impact analysis

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