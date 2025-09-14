# Bug Report: MCP Server Logging Issue Causes Claude Code Startup Hang

## Summary
The Deepflow MCP server causes Claude Code to hang during startup due to logging conflicts with MCP protocol communication streams.

## Issue Details

**Problem:** When Claude Code attempts to initialize the Deepflow MCP server, the application hangs indefinitely and becomes unresponsive. User must press Enter then Ctrl+C to interrupt the hanging process.

**Root Cause:** The MCP server's logging system attempts to write to stdout/stderr streams that have been redirected by Claude Code for MCP protocol communication, resulting in `ValueError: I/O operation on closed file.`

## Error Details

**Location:** `deepflow/mcp/server.py` lines 3433 and 3408
**Error:** 
```
ValueError: I/O operation on closed file.
```

**Stack Trace:**
```
--- Logging error ---
Traceback (most recent call last):
  File "C:\Python313\Lib\logging\__init__.py", line 1154, in emit
    stream.write(msg + self.terminator)
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
ValueError: I/O operation on closed file.
Call stack:
  File "C:\Users\Sebastian\PycharmProjects\npcgpt-dependency\dependency-toolkit\deepflow\mcp\server.py", line 3433, in run
    logger.info(f"Server shutting down - Final stats: {stats}")
```

## Configuration Context

**Claude Code MCP Configuration:**
```json
{
  "mcpServers": {
    "deepflow": {
      "command": "C:\\Users\\Sebastian\\PycharmProjects\\npcgpt-dependency\\NPCGPT\\.venv\\Scripts\\python.exe",
      "args": ["-m", "deepflow.mcp.server"],
      "cwd": "C:\\Users\\Sebastian\\PycharmProjects\\npcgpt-dependency\\dependency-toolkit",
      "env": {
        "PYTHONPATH": "C:\\Users\\Sebastian\\PycharmProjects\\npcgpt-dependency\\dependency-toolkit"
      }
    }
  }
}
```

## Reproduction Steps

1. Configure Deepflow MCP server in Claude Code's `mcp_servers.json`
2. Launch Claude Code CLI: `npx @anthropic-ai/claude-code`
3. Observe Claude Code hangs during startup
4. Press Enter then Ctrl+C to interrupt
5. Check that Claude Code becomes responsive after interrupting MCP initialization

## Manual Testing Results

Running the MCP server standalone shows successful initialization but logging errors during shutdown:
```bash
cd "C:\Users\Sebastian\PycharmProjects\npcgpt-dependency\dependency-toolkit"
python -m deepflow.mcp.server
# Shows logging errors but runs when not connected to MCP protocol streams
```

## Proposed Solution

**Issue:** The logging system is configured to write to stdout/stderr, which conflicts with MCP protocol communication that uses these streams.

**Fix Options:**

1. **File-based logging only for MCP mode:**
   ```python
   # Detect MCP mode and configure logging accordingly
   if running_as_mcp_server():
       # Configure logger to write only to files, not stdout/stderr
       logging.basicConfig(
           filename=log_file_path,
           level=logging.INFO,
           format='%(asctime)s | %(name)s | %(levelname)8s | %(message)s'
       )
   ```

2. **Disable stdout logging in MCP mode:**
   ```python
   # Remove stdout/stderr handlers when running as MCP server
   logger = logging.getLogger()
   for handler in logger.handlers[:]:
       if isinstance(handler, logging.StreamHandler):
           logger.removeHandler(handler)
   ```

3. **Proper cleanup of logging resources:**
   ```python
   def cleanup_logging():
       """Properly close all logging handlers"""
       for handler in logging.getLogger().handlers:
           handler.close()
           logging.getLogger().removeHandler(handler)
   ```

## Environment

- **OS:** Windows 11
- **Python:** 3.13
- **Claude Code:** Latest version
- **MCP Protocol:** stdio transport
- **Deepflow:** Current version from dependency-toolkit

## Workaround

**Temporary fix:** Disable MCP server in Claude Code configuration:
```json
{
  "mcpServers": {}
}
```

## Impact

- **Severity:** High - Blocks Claude Code usage when MCP server is configured
- **User Experience:** Application appears broken/frozen on startup
- **Functionality:** MCP server cannot be used with Claude Code until fixed

## Related Files

- `deepflow/mcp/server.py` (main server implementation)
- `C:\Users\Sebastian\.claude\mcp_servers.json` (Claude Code MCP configuration)
- Logging configuration throughout the Deepflow MCP server

## Test Case

The fix should ensure that:
1. MCP server starts successfully when launched by Claude Code
2. No logging errors occur during MCP protocol communication
3. File-based logging continues to work for debugging
4. Server shutdown is clean without stream errors