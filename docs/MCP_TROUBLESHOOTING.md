# Deepflow MCP Integration - Troubleshooting Guide

Comprehensive guide to diagnosing and fixing common issues with Deepflow MCP integration.

## 🚨 Common Issues & Solutions

### 1. MCP Server Won't Start

#### **Error**: `deepflow-mcp-server: command not found`

**Cause**: Deepflow MCP dependencies not installed

**Solution**:
```bash
# Install MCP support
pip install deepflow[mcp]

# Verify installation
deepflow-mcp-server --help
```

#### **Error**: `ImportError: No module named 'mcp'`

**Cause**: MCP library not installed or wrong version

**Solution**:
```bash
# Install specific MCP version
pip install "mcp>=1.2.0"

# Or reinstall deepflow with MCP
pip uninstall deepflow
pip install deepflow[mcp]
```

#### **Error**: Server starts but immediately exits

**Cause**: Port conflict or permission issues

**Solution**:
```bash
# Check for port conflicts
netstat -tulpn | grep :8000

# Run with debug logging
DEEPFLOW_LOG_LEVEL=DEBUG deepflow-mcp-server

# Check server logs
tail -f ~/.deepflow/logs/mcp_server.log
```

---

### 2. Claude Code Connection Issues

#### **Error**: "MCP server connection failed"

**Cause**: Server not running or configuration mismatch

**Diagnostic Steps**:
```bash
# 1. Check if server is running
ps aux | grep deepflow-mcp-server

# 2. Test server manually
curl http://localhost:8000/health

# 3. Check server logs
cat ~/.deepflow/logs/mcp_server.log
```

**Solutions**:
1. **Server not running**: Start with `deepflow-mcp-server`
2. **Wrong port**: Check Claude Code MCP configuration
3. **Permission issues**: Run with appropriate permissions

#### **Error**: "No MCP tools available in Claude Code"

**Cause**: MCP server not properly registered or tools not loading

**Solution**:
```bash
# 1. Restart MCP server with verbose logging
DEEPFLOW_LOG_LEVEL=DEBUG deepflow-mcp-server

# 2. Check tool registration
deepflow-mcp-server --list-tools

# 3. Verify Claude Code MCP config
# Check: ~/.claude/mcp-servers.json
```

---

### 3. Tool Execution Errors

#### **Error**: "Deepflow tools not available"

**Cause**: Core deepflow modules not found

**Diagnostic**:
```bash
# Test import manually
python -c "from tools.dependency_visualizer import DependencyVisualizer; print('OK')"
```

**Solutions**:
```bash
# 1. Install in development mode
pip install -e .

# 2. Check Python path
python -c "import sys; print('\n'.join(sys.path))"

# 3. Verify tools directory
ls -la tools/
```

#### **Error**: "Project path does not exist"

**Cause**: Invalid project path or permission issues

**Solutions**:
```bash
# 1. Check path exists
ls -la /path/to/project

# 2. Check permissions
stat /path/to/project

# 3. Use absolute paths
deepflow-mcp-client analyze_dependencies --project-path $(pwd)
```

#### **Error**: Analysis times out or hangs

**Cause**: Large project or performance issues

**Solutions**:
```bash
# 1. Increase timeout
export DEEPFLOW_ANALYSIS_TIMEOUT=300

# 2. Exclude large directories
echo "node_modules/" > .deepflowignore
echo "__pycache__/" >> .deepflowignore
echo "venv/" >> .deepflowignore

# 3. Run on smaller scope
deepflow-mcp-client analyze_dependencies --project-path ./src
```

---

### 4. Performance Issues

#### **Issue**: Slow analysis times

**Diagnostic**:
```bash
# Check project size
find . -name "*.py" | wc -l

# Monitor memory usage
top -p $(pgrep deepflow-mcp-server)

# Check server performance stats
curl http://localhost:8000/stats
```

**Optimization Strategies**:

1. **Use .deepflowignore**:
```
# .deepflowignore
__pycache__/
*.pyc
venv/
node_modules/
.git/
build/
dist/
*.egg-info/
```

2. **Enable caching**:
```bash
# Cache results for 10 minutes
export DEEPFLOW_CACHE_TIMEOUT=600
```

3. **Limit analysis scope**:
```bash
# Analyze only source code
deepflow-mcp-client analyze_code_quality --project-path ./src --analysis-type imports
```

#### **Issue**: High memory usage

**Cause**: Large dependency graphs in memory

**Solutions**:
```bash
# 1. Restart server periodically
pkill deepflow-mcp-server
deepflow-mcp-server &

# 2. Use streaming analysis for large projects
export DEEPFLOW_STREAMING_MODE=true

# 3. Limit concurrent analyses
export DEEPFLOW_MAX_CONCURRENT=1
```

---

### 5. Integration Issues

#### **Issue**: Results not appearing in Claude Code

**Cause**: MCP protocol issues or tool response format

**Diagnostic**:
```bash
# Test tool manually
deepflow-mcp-client analyze_dependencies --format json

# Check MCP protocol compliance
deepflow-mcp-server --validate-protocol
```

**Solutions**:
1. **Restart both server and Claude Code**
2. **Check MCP configuration in Claude Code**
3. **Update to latest versions**

#### **Issue**: Inconsistent results between CLI and MCP

**Cause**: Different execution contexts or configurations

**Solutions**:
```bash
# Use same project path
PROJECT_PATH="/full/path/to/project"

# CLI test
cd $PROJECT_PATH
deepflow-visualizer .

# MCP test
deepflow-mcp-client analyze_dependencies --project-path $PROJECT_PATH
```

---

### 6. Error Messages & Solutions

#### **ImportError: cannot import name 'X'**
```bash
# Solution: Check dependency versions
pip list | grep -E "(deepflow|mcp|networkx|matplotlib)"

# Reinstall if needed
pip install --upgrade deepflow[mcp]
```

#### **FileNotFoundError: [Errno 2] No such file or directory**
```bash
# Solution: Verify project structure
find . -name "*.py" -type f | head -5

# Check current directory
pwd
ls -la
```

#### **PermissionError: [Errno 13] Permission denied**
```bash
# Solution: Check file permissions
ls -la /path/to/project
chmod -R 755 /path/to/project

# Or run with appropriate user
sudo -u project-owner deepflow-mcp-server
```

#### **MemoryError: Unable to allocate array**
```bash
# Solution: Reduce analysis scope or increase memory
ulimit -v 4000000  # Limit virtual memory
deepflow-mcp-client analyze_dependencies --project-path ./small-module
```

---

## 🔧 Debugging Tools

### 1. Server Health Check
```bash
#!/bin/bash
# health-check.sh
echo "🏥 Deepflow MCP Health Check"

echo "1. Checking installation..."
if command -v deepflow-mcp-server &> /dev/null; then
    echo "✅ deepflow-mcp-server installed"
else
    echo "❌ deepflow-mcp-server not found"
fi

echo "2. Checking server process..."
if pgrep -f "deepflow-mcp-server" > /dev/null; then
    echo "✅ MCP server running (PID: $(pgrep deepflow-mcp-server))"
else
    echo "❌ MCP server not running"
fi

echo "3. Checking dependencies..."
python -c "
try:
    import mcp
    print('✅ MCP library available')
except ImportError:
    print('❌ MCP library missing')

try:
    from deepflow.mcp.server import DeepflowMCPServer
    print('✅ Deepflow MCP server module available')
except ImportError:
    print('❌ Deepflow MCP server module missing')
"

echo "4. Checking logs..."
if [ -f ~/.deepflow/logs/mcp_server.log ]; then
    echo "✅ Log file exists"
    echo "Recent errors:"
    tail -10 ~/.deepflow/logs/mcp_server.log | grep -i error || echo "No recent errors"
else
    echo "❌ No log file found"
fi
```

### 2. Configuration Validator
```python
#!/usr/bin/env python3
# validate-config.py

import sys
import json
import os
from pathlib import Path

def validate_mcp_config():
    """Validate MCP configuration."""
    print("🔍 Validating MCP Configuration")
    
    # Check Claude Code MCP config
    claude_config = Path.home() / ".claude" / "mcp-servers.json"
    if claude_config.exists():
        try:
            with open(claude_config) as f:
                config = json.load(f)
            
            if "deepflow" in config.get("mcpServers", {}):
                print("✅ Deepflow found in Claude Code MCP config")
                deepflow_config = config["mcpServers"]["deepflow"]
                print(f"   Command: {deepflow_config.get('command')}")
                print(f"   Args: {deepflow_config.get('args', [])}")
            else:
                print("❌ Deepflow not configured in Claude Code")
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in Claude Code config: {e}")
    else:
        print("❌ Claude Code MCP config not found")
    
    # Check environment variables
    env_vars = [
        "DEEPFLOW_CACHE_TIMEOUT",
        "DEEPFLOW_LOG_LEVEL", 
        "DEEPFLOW_MAX_PROJECT_SIZE"
    ]
    
    print("\n🔧 Environment Variables:")
    for var in env_vars:
        value = os.environ.get(var)
        if value:
            print(f"✅ {var}={value}")
        else:
            print(f"   {var}=<not set>")

if __name__ == "__main__":
    validate_mcp_config()
```

### 3. Performance Profiler
```bash
#!/bin/bash
# profile-performance.sh

echo "📊 Profiling Deepflow MCP Performance"

# Start server with profiling
python -m cProfile -o mcp_profile.stats -m deepflow.mcp.server &
SERVER_PID=$!

sleep 3

# Run test analysis
time deepflow-mcp-client analyze_dependencies --project-path .

# Stop server
kill $SERVER_PID

# Analyze profile
python -c "
import pstats
p = pstats.Stats('mcp_profile.stats')
p.sort_stats('cumulative')
p.print_stats(10)
"
```

---

## 🚑 Emergency Recovery

### Complete Reset
```bash
#!/bin/bash
# emergency-reset.sh

echo "🚑 Emergency Deepflow MCP Reset"

# 1. Stop all processes
pkill -f deepflow-mcp-server
pkill -f deepflow

# 2. Clear cache
rm -rf ~/.deepflow/cache/*

# 3. Clear logs (optional)
# rm -rf ~/.deepflow/logs/*

# 4. Reinstall
pip uninstall -y deepflow
pip install deepflow[mcp]

# 5. Restart
deepflow-mcp-server &

echo "✅ Reset complete. Test with: deepflow-mcp-client analyze_dependencies"
```

### Rollback to CLI-Only Mode
```bash
# If MCP is completely broken, use CLI tools
pip install deepflow  # Without [mcp]

# Use CLI commands instead
deepflow-visualizer /path/to/project
deepflow-analyzer /path/to/project --all
```

---

## 📞 Getting Help

### Before Opening Issues

Run the diagnostic suite:
```bash
# Generate diagnostic report
bash health-check.sh > diagnostic-report.txt
python validate-config.py >> diagnostic-report.txt

# Include system info
echo "\nSystem Information:" >> diagnostic-report.txt
python -c "
import sys, platform
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'Architecture: {platform.architecture()}')
" >> diagnostic-report.txt

# Include installed packages
pip list | grep -E "(deepflow|mcp|networkx|matplotlib)" >> diagnostic-report.txt
```

### Support Channels

1. **GitHub Issues**: [Deepflow Issues](https://github.com/scs03004/deepflow/issues)
2. **Documentation**: [Complete MCP Guide](./MCP_INTEGRATION_GUIDE.md)
3. **Discord**: [Deepflow Community](https://discord.gg/deepflow) *(coming soon)*

### Issue Template

```markdown
**Environment**:
- OS: [e.g., Ubuntu 20.04, Windows 11, macOS 12]
- Python Version: [e.g., 3.8.10]
- Deepflow Version: [e.g., 2.1.0]
- MCP Version: [e.g., 1.2.0]

**Issue Description**:
[Clear description of the problem]

**Steps to Reproduce**:
1. [First step]
2. [Second step]
3. [etc.]

**Expected Behavior**:
[What you expected to happen]

**Actual Behavior**:
[What actually happened]

**Diagnostic Output**:
[Paste output from health-check.sh]

**Logs**:
[Relevant log entries from ~/.deepflow/logs/mcp_server.log]
```

---

## 🎯 Prevention Best Practices

### 1. **Regular Maintenance**
```bash
# Weekly cleanup script
#!/bin/bash
# Clear old cache entries
find ~/.deepflow/cache -mtime +7 -delete

# Rotate logs
mv ~/.deepflow/logs/mcp_server.log ~/.deepflow/logs/mcp_server.log.old

# Update dependencies
pip install --upgrade deepflow[mcp]
```

### 2. **Monitoring Setup**
```bash
# Add to crontab for health monitoring
# */5 * * * * /path/to/health-check.sh
```

### 3. **Backup Configuration**
```bash
# Backup MCP configuration
cp ~/.claude/mcp-servers.json ~/.claude/mcp-servers.json.backup
```

Remember: Most issues are solved by ensuring proper installation (`pip install deepflow[mcp]`) and having the MCP server running before connecting from Claude Code.