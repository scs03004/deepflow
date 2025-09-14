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

## 🔧 Requirements Management Tools ✨ **NEW**

### 9. `analyze_requirements`
**Purpose**: Analyze project dependencies and detect missing requirements.txt packages

**Usage in Claude Code**:
```
"Analyze requirements for this project"
"Check for missing packages in requirements.txt"
"What dependencies does this project need?"
```

**Parameters**:
- `project_path`: Path to analyze (default: current directory)
- `check_patterns`: Validate common requirement patterns (default: true)

**Returns**:
- Missing packages with confidence scores
- Current requirements.txt content
- Package mapping suggestions (sklearn→scikit-learn, yaml→pyyaml, etc.)
- Import detection results
- Update recommendations

### 10. `update_requirements`
**Purpose**: Update requirements.txt with missing packages automatically

**Usage in Claude Code**:
```
"Update requirements.txt with missing packages"
"Add detected dependencies to requirements file"
"Apply requirements recommendations"
```

**Parameters**:
- `project_path`: Path to update (default: current directory)
- `backup`: Create backup before updating (default: true)
- `dry_run`: Preview changes without applying (default: false)
- `apply_changes`: Actually update the file (default: false)

**Returns**:
- Requirements file updates applied
- Backup file location
- Update statistics and results

## 🗂️ File Organization Tools ✨ **NEW**

### 11. `analyze_file_organization`
**Purpose**: Analyze project structure and detect messy file organization patterns

**Usage in Claude Code**:
```
"Analyze the file organization of this project"
"Check for root clutter and naming inconsistencies"
"Score the project structure quality"
```

**Parameters**:
- `project_path`: Path to analyze (default: current directory)
- `check_patterns`: Check naming pattern consistency (default: true)

**Returns**:
- Project structure score (0-100)
- Root clutter files that should be moved
- Suggested directory structure
- File relocation recommendations
- Naming pattern inconsistencies
- Organization recommendations with confidence scores

### 12. `organize_files`
**Purpose**: Apply file organization recommendations with safety checks

**Usage in Claude Code**:
```
"Organize files in this project"
"Apply file organization recommendations"
"Clean up the project structure"
```

**Parameters**:
- `project_path`: Path to organize (default: current directory)
- `dry_run`: Preview changes without applying (default: true)
- `backup`: Create backups before moving files (default: true)
- `apply_changes`: Actually reorganize files (default: false)

**Returns**:
- Organization changes applied
- Files moved and directories created
- Backup locations
- Before/after structure scores
- Organization statistics

## 🧠 AI Session Intelligence Tools ✨ **NEW - Priority 3**

### 13. `start_ai_session`
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

### 14. `end_ai_session`
**Purpose**: End the current AI development session and save context for future reference

**Usage in Claude Code**:
```
"End the current AI session"
"Complete session and save achievements"
"Finish session with goal: implemented login system"
```

**Parameters**:
- `achievements`: Array of goals achieved during the session (default: [])

### 15. `get_session_context`
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

### 16. `restore_session_context`
**Purpose**: Restore a previous AI session context for continuity across development cycles

**Usage in Claude Code**:
```
"Restore session context from yesterday"
"Resume previous session: session_123456"
"Continue from where I left off in session ABC"
```

**Parameters**:
- `session_id`: ID of the session to restore (required)

### 17. `analyze_change_impact`
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

### 18. `get_session_intelligence`
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

## 🔧 Smart Refactoring Tools ✨ **NEW**

### 19. `standardize_patterns`
**Purpose**: Standardize code patterns across the project for consistency and maintainability

**Usage in Claude Code**:
```
"Standardize naming patterns in this project"
"Apply consistent code patterns to all modules"
"Fix inconsistent import styles across files"
```

**Parameters**:
- `project_path`: Path to analyze (default: current directory)
- `pattern_type`: Type of patterns to standardize - "naming", "imports", "structure", "all" (default: "all")
- `apply_changes`: Actually apply the standardization (default: false)

**Returns**:
- Pattern inconsistencies detected
- Standardization recommendations
- Files requiring changes
- Preview of proposed changes

### 20. `optimize_imports`
**Purpose**: Optimize import statements by removing unused imports and organizing efficiently

**Usage in Claude Code**:
```
"Optimize import statements in this project"
"Remove unused imports and reorganize"
"Clean up import organization"
```

**Parameters**:
- `project_path`: Path to optimize (default: current directory)
- `remove_unused`: Remove unused imports (default: true)
- `sort_imports`: Sort imports alphabetically (default: true)
- `apply_changes`: Actually apply the optimizations (default: false)

**Returns**:
- Unused imports removed
- Import organization improvements
- Files modified
- Import optimization statistics

### 21. `suggest_file_splits`
**Purpose**: Suggest file splits for large files to improve maintainability and AI comprehension

**Usage in Claude Code**:
```
"Suggest file splits for better organization"
"What large files should be split up?"
"Recommend file organization improvements"
```

**Parameters**:
- `project_path`: Path to analyze (default: current directory)
- `max_lines`: Maximum lines per file threshold (default: 500)
- `ai_context_aware`: Consider AI context window optimization (default: true)

**Returns**:
- Files exceeding size thresholds
- Suggested split points and new file names
- Refactoring recommendations
- AI comprehension improvements

### 22. `remove_dead_code`
**Purpose**: Identify and remove dead code including unused functions, classes, and variables

**Usage in Claude Code**:
```
"Remove dead code from this project"
"Find unused functions and classes"
"Clean up unreachable code"
```

**Parameters**:
- `project_path`: Path to analyze (default: current directory)
- `aggressive`: Use aggressive dead code detection (default: false)
- `apply_changes`: Actually remove dead code (default: false)

**Returns**:
- Dead code locations identified
- Unused functions and classes
- Unreachable code blocks
- Code cleanup recommendations

### 23. `generate_docstrings`
**Purpose**: Automatically generate comprehensive docstrings for functions, classes, and modules

**Usage in Claude Code**:
```
"Generate docstrings for undocumented functions"
"Add comprehensive documentation to this project"
"Create docstrings following project conventions"
```

**Parameters**:
- `project_path`: Path to document (default: current directory)
- `style`: Docstring style - "google", "numpy", "sphinx" (default: "google")
- `apply_changes`: Actually add docstrings (default: false)

**Returns**:
- Undocumented functions and classes
- Generated docstrings preview
- Documentation coverage improvements
- Style consistency recommendations

### 24. `comprehensive_refactor`
**Purpose**: Perform comprehensive refactoring combining multiple optimization techniques

**Usage in Claude Code**:
```
"Perform comprehensive refactoring of this project"
"Apply all code quality improvements"
"Refactor project for better maintainability"
```

**Parameters**:
- `project_path`: Path to refactor (default: current directory)
- `include_patterns`: Include pattern standardization (default: true)
- `include_imports`: Include import optimization (default: true)
- `include_splits`: Include file split suggestions (default: true)
- `include_dead_code`: Include dead code removal (default: true)
- `include_docstrings`: Include docstring generation (default: true)
- `apply_changes`: Actually apply all refactoring (default: false)

**Returns**:
- Comprehensive refactoring plan
- All improvement recommendations
- Risk assessment for changes
- Step-by-step implementation guide

## 🔄 Workflow Orchestration Tools ✨ **NEW**

### 25. `create_analysis_pipeline`
**Purpose**: Create custom analysis pipelines that combine multiple deepflow tools

**Usage in Claude Code**:
```
"Create an analysis pipeline for CI/CD"
"Set up a custom workflow for code quality checks"
"Build a pipeline combining dependency analysis and documentation"
```

**Parameters**:
- `pipeline_name`: Name for the analysis pipeline (required)
- `tools`: Array of tools to include in pipeline (required)
- `configuration`: Pipeline configuration options (default: {})

**Returns**:
- Created pipeline configuration
- Execution order and dependencies
- Pipeline validation results
- Usage instructions

### 26. `execute_workflow`
**Purpose**: Execute a predefined workflow or pipeline with specified parameters

**Usage in Claude Code**:
```
"Execute the CI/CD analysis workflow"
"Run the comprehensive code quality pipeline"
"Execute workflow: pre-commit-validation"
```

**Parameters**:
- `workflow_name`: Name of workflow to execute (required)
- `project_path`: Path to execute workflow on (default: current directory)
- `parameters`: Workflow-specific parameters (default: {})

**Returns**:
- Workflow execution results
- Individual tool outputs
- Execution time and performance metrics
- Success/failure status with details

### 27. `create_conditional_workflow`
**Purpose**: Create workflows with conditional logic based on project characteristics

**Usage in Claude Code**:
```
"Create conditional workflow based on project type"
"Set up workflow that adapts to codebase characteristics"
"Build smart workflow with conditional steps"
```

**Parameters**:
- `workflow_name`: Name for conditional workflow (required)
- `conditions`: Conditions and corresponding actions (required)
- `fallback_actions`: Default actions if no conditions match (default: [])

**Returns**:
- Conditional workflow configuration
- Condition evaluation logic
- Workflow branching structure
- Testing recommendations

### 28. `create_batch_operation`
**Purpose**: Create batch operations for processing multiple projects or directories

**Usage in Claude Code**:
```
"Create batch operation for multiple projects"
"Set up batch analysis across repositories"
"Configure bulk code quality assessment"
```

**Parameters**:
- `operation_name`: Name for batch operation (required)
- `target_pattern`: Pattern for selecting targets (required)
- `operation_config`: Configuration for the batch operation (default: {})

**Returns**:
- Batch operation configuration
- Target selection results
- Estimated execution time
- Resource requirements

### 29. `execute_batch_operation`
**Purpose**: Execute batch operations across multiple projects or directories

**Usage in Claude Code**:
```
"Execute batch analysis on all repositories"
"Run bulk code quality checks"
"Process multiple projects with batch operation"
```

**Parameters**:
- `operation_name`: Name of batch operation to execute (required)
- `parallel`: Execute operations in parallel (default: true)
- `max_workers`: Maximum parallel workers (default: 4)

**Returns**:
- Batch execution results
- Per-project success/failure status
- Aggregated metrics and insights
- Performance and timing information

### 30. `load_custom_workflow`
**Purpose**: Load and register custom workflow definitions from configuration files

**Usage in Claude Code**:
```
"Load custom workflow from config file"
"Import workflow definition from YAML"
"Register new workflow from configuration"
```

**Parameters**:
- `workflow_file`: Path to workflow configuration file (required)
- `validate`: Validate workflow before loading (default: true)
- `overwrite`: Overwrite existing workflow with same name (default: false)

**Returns**:
- Loaded workflow configuration
- Validation results
- Available workflow commands
- Registration status

### 31. `setup_scheduled_hygiene`
**Purpose**: Set up scheduled code hygiene tasks and maintenance workflows

**Usage in Claude Code**:
```
"Set up scheduled code hygiene tasks"
"Configure automated maintenance workflows"
"Schedule regular dependency analysis"
```

**Parameters**:
- `project_path`: Path to set up hygiene for (default: current directory)
- `schedule`: Schedule configuration (cron-like syntax) (required)
- `tasks`: Array of hygiene tasks to schedule (required)

**Returns**:
- Scheduled hygiene configuration
- Task scheduling details
- Integration instructions (GitHub Actions, etc.)
- Monitoring setup

### 32. `get_workflow_status`
**Purpose**: Get status and progress information for running or completed workflows

**Usage in Claude Code**:
```
"What's the status of the current workflow?"
"Check progress of batch analysis operation"
"Get workflow execution details"
```

**Parameters**:
- `workflow_id`: ID of workflow to check (optional)
- `include_history`: Include execution history (default: false)

**Returns**:
- Current workflow status
- Progress and completion information
- Execution history (if requested)
- Performance metrics

### 33. `list_workflows`
**Purpose**: List all available workflows, pipelines, and batch operations

**Usage in Claude Code**:
```
"List all available workflows"
"What workflows are configured?"
"Show me all custom pipelines"
```

**Parameters**:
- `filter_type`: Filter by type - "pipeline", "workflow", "batch", "all" (default: "all")
- `include_custom`: Include user-defined workflows (default: true)

**Returns**:
- Available workflows and pipelines
- Configuration summaries
- Usage statistics
- Recommended workflows for current project

### 34. `get_workflow_metrics`
**Purpose**: Get comprehensive metrics and analytics for workflow executions

**Usage in Claude Code**:
```
"Show workflow execution metrics"
"Get analytics for batch operations"
"What are the performance trends for workflows?"
```

**Parameters**:
- `time_range`: Time range for metrics - "day", "week", "month", "all" (default: "week")
- `workflow_name`: Specific workflow to analyze (optional)

**Returns**:
- Execution frequency and success rates
- Performance trends and bottlenecks
- Resource utilization metrics
- Optimization recommendations

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

**Requirements Management Tools:**
```
5. "Analyze requirements for this project"
6. "Update requirements.txt with missing packages"
```

**File Organization Tools:**
```
7. "Analyze the file organization of this project"
8. "Organize files and clean up the project structure"
```

**Real-Time Intelligence Tools:**
```
9. "Start real-time monitoring for this project"
10. "Show me recent real-time activity" 
11. "What are the current monitoring statistics?"
12. "Stop the real-time monitoring"
```

**AI Session Intelligence Tools:**
```
13. "Start AI session: implementing user auth"
14. "End current session with achievements"
15. "What's my current session context?"
16. "Restore session from yesterday"
17. "Analyze impact of changing database.py"
18. "Show me session intelligence and patterns learned"
```

**Smart Refactoring Tools:**
```
19. "Standardize naming patterns in this project"
20. "Optimize import statements and remove unused imports"
21. "Suggest file splits for better organization"
22. "Remove dead code from this project"
23. "Generate docstrings for undocumented functions"
24. "Perform comprehensive refactoring of this project"
```

**Workflow Orchestration Tools:**
```
25. "Create an analysis pipeline for CI/CD"
26. "Execute the CI/CD analysis workflow"
27. "Create conditional workflow based on project type"
28. "Create batch operation for multiple projects"
29. "Execute batch analysis on all repositories"
30. "Load custom workflow from config file"
31. "Set up scheduled code hygiene tasks"
32. "What's the status of the current workflow?"
33. "List all available workflows"
34. "Show workflow execution metrics"
```

### Expected Behavior
- `claude mcp list` should show deepflow server as "✓ Connected"
- Claude Code should recognize **34 deepflow MCP tools** automatically
- Core tools should provide structured analysis results in JSON format
- Real-time tools should provide live monitoring capabilities
- Pattern deviations and AI context alerts should be detected in real-time
- Session intelligence tools provide development context and impact analysis
- Smart refactoring tools automate code quality improvements
- Workflow orchestration enables custom analysis pipelines and batch operations

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