#!/usr/bin/env python3
"""
Enhanced Error Handling for Deepflow MCP Server
===============================================

Comprehensive error handling, logging, and monitoring system for the MCP server.
Provides structured error reporting, performance monitoring, and debugging capabilities.
"""

import logging
import sys
import traceback
import time
import functools
from typing import Dict, Any, Optional, Callable, List
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

# Configure structured logging
class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class ErrorContext:
    """Context information for errors."""
    tool_name: str
    arguments: Dict[str, Any]
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    request_id: Optional[str] = None

@dataclass
class PerformanceMetrics:
    """Performance tracking metrics."""
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    cache_hit: bool = False
    
    def finish(self):
        """Mark the operation as finished and calculate duration."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time

class MCPErrorHandler:
    """Enhanced error handler for MCP server operations."""
    
    def __init__(self, logger_name: str = "deepflow.mcp"):
        """Initialize the error handler."""
        self.logger = self._setup_logger(logger_name)
        self.error_counts: Dict[str, int] = {}
        self.performance_history: List[PerformanceMetrics] = []
        self.max_history_size = 1000
        
        # Error recovery strategies
        self.recovery_strategies = {
            "ImportError": self._handle_import_error,
            "FileNotFoundError": self._handle_file_not_found,
            "PermissionError": self._handle_permission_error,
            "TimeoutError": self._handle_timeout_error,
            "MemoryError": self._handle_memory_error,
        }
    
    def _setup_logger(self, logger_name: str) -> logging.Logger:
        """Set up structured logging with multiple handlers."""
        logger = logging.getLogger(logger_name)
        
        # Prevent duplicate handlers
        if logger.handlers:
            return logger
            
        logger.setLevel(logging.DEBUG)
        
        # Console handler with colored output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Detailed formatter for console
        console_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)8s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler for detailed debugging (if possible)
        try:
            log_dir = Path.home() / ".deepflow" / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_dir / "mcp_server.log")
            file_handler.setLevel(logging.DEBUG)
            
            # Detailed formatter for file
            file_formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
        except (OSError, PermissionError) as e:
            logger.warning(f"Could not set up file logging: {e}")
        
        return logger
    
    def log_performance(self, metrics: PerformanceMetrics):
        """Log performance metrics."""
        self.performance_history.append(metrics)
        
        # Trim history if it gets too large
        if len(self.performance_history) > self.max_history_size:
            self.performance_history = self.performance_history[-self.max_history_size//2:]
        
        # Log performance info
        cache_status = "HIT" if metrics.cache_hit else "MISS"
        self.logger.info(
            f"Performance: {metrics.duration:.3f}s | Cache: {cache_status} | "
            f"Memory: {metrics.memory_usage_mb or 'N/A'}MB"
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.performance_history:
            return {"message": "No performance data available"}
        
        durations = [m.duration for m in self.performance_history if m.duration]
        cache_hits = sum(1 for m in self.performance_history if m.cache_hit)
        
        if not durations:
            return {"message": "No completed operations"}
        
        return {
            "total_operations": len(self.performance_history),
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "cache_hit_rate": cache_hits / len(self.performance_history),
            "error_counts": self.error_counts.copy()
        }
    
    def _handle_import_error(self, error: ImportError, context: ErrorContext) -> Dict[str, Any]:
        """Handle import errors with helpful suggestions."""
        missing_module = str(error).split("'")[1] if "'" in str(error) else "unknown"
        
        suggestions = {
            "mcp": "pip install mcp>=1.2.0",
            "dependency_visualizer": "Ensure deepflow tools are installed correctly",
            "code_analyzer": "Ensure deepflow tools are installed correctly",
            "matplotlib": "pip install matplotlib",
            "networkx": "pip install networkx"
        }
        
        suggestion = suggestions.get(missing_module, f"pip install {missing_module}")
        
        return {
            "error_type": "ImportError",
            "missing_module": missing_module,
            "suggestion": suggestion,
            "recovery_action": "install_dependency"
        }
    
    def _handle_file_not_found(self, error: FileNotFoundError, context: ErrorContext) -> Dict[str, Any]:
        """Handle file not found errors."""
        return {
            "error_type": "FileNotFoundError",
            "missing_path": str(error).split("'")[1] if "'" in str(error) else "unknown",
            "suggestion": "Check if the project path exists and is accessible",
            "recovery_action": "verify_path"
        }
    
    def _handle_permission_error(self, error: PermissionError, context: ErrorContext) -> Dict[str, Any]:
        """Handle permission errors."""
        return {
            "error_type": "PermissionError",
            "suggestion": "Check file/directory permissions or run with appropriate privileges",
            "recovery_action": "check_permissions"
        }
    
    def _handle_timeout_error(self, error: TimeoutError, context: ErrorContext) -> Dict[str, Any]:
        """Handle timeout errors."""
        return {
            "error_type": "TimeoutError",
            "suggestion": "Try analyzing a smaller project or increase timeout limits",
            "recovery_action": "retry_with_smaller_scope"
        }
    
    def _handle_memory_error(self, error: MemoryError, context: ErrorContext) -> Dict[str, Any]:
        """Handle memory errors."""
        return {
            "error_type": "MemoryError",
            "suggestion": "Project may be too large for available memory. Try excluding large directories",
            "recovery_action": "reduce_scope"
        }
    
    def handle_error(self, error: Exception, context: ErrorContext) -> Dict[str, Any]:
        """Handle errors with context-aware recovery."""
        error_type = type(error).__name__
        error_message = str(error)
        
        # Update error counts
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Get stack trace
        stack_trace = traceback.format_exc()
        
        # Log the error with full context
        self.logger.error(
            f"Error in {context.tool_name}: {error_type}: {error_message}",
            extra={
                "tool_name": context.tool_name,
                "arguments": context.arguments,
                "request_id": context.request_id,
                "error_type": error_type,
                "stack_trace": stack_trace
            }
        )
        
        # Try to get specific recovery information
        recovery_info = {}
        if error_type in self.recovery_strategies:
            try:
                recovery_info = self.recovery_strategies[error_type](error, context)
            except Exception as recovery_error:
                self.logger.warning(f"Recovery strategy failed: {recovery_error}")
        
        return {
            "error": True,
            "error_type": error_type,
            "message": error_message,
            "tool_name": context.tool_name,
            "timestamp": context.timestamp,
            "recovery_info": recovery_info,
            "request_id": context.request_id
        }
    
    def create_user_friendly_error(self, error_info: Dict[str, Any]) -> str:
        """Create user-friendly error messages."""
        error_type = error_info.get("error_type", "Unknown")
        message = error_info.get("message", "")
        tool_name = error_info.get("tool_name", "unknown")
        recovery = error_info.get("recovery_info", {})
        
        user_message = f"Error in {tool_name}: {error_type}"
        
        if recovery.get("suggestion"):
            user_message += f"\n\nSuggestion: {recovery['suggestion']}"
        
        # Add common troubleshooting steps
        user_message += "\n\nCommon solutions:"
        user_message += "\n1. Ensure you're in a valid project directory"
        user_message += "\n2. Check that all dependencies are installed: pip install deepflow[mcp]"
        user_message += "\n3. Verify file permissions and accessibility"
        
        if recovery.get("recovery_action"):
            action_descriptions = {
                "install_dependency": "Install the missing dependency",
                "verify_path": "Verify the project path exists and is readable",
                "check_permissions": "Check file and directory permissions",
                "retry_with_smaller_scope": "Try with a smaller project or timeout",
                "reduce_scope": "Exclude large directories or files"
            }
            action = recovery["recovery_action"]
            if action in action_descriptions:
                user_message += f"\n4. {action_descriptions[action]}"
        
        return user_message

def with_error_handling(error_handler: MCPErrorHandler):
    """Decorator for comprehensive error handling on MCP tool methods."""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract tool name from function name or arguments
            tool_name = func.__name__.replace('_handle_', '').replace('_', ' ')
            if args and hasattr(args[0], '__class__'):
                tool_name = f"{args[0].__class__.__name__}.{tool_name}"
            
            # Extract arguments (usually second parameter after self)
            arguments = kwargs.copy()
            if len(args) > 1 and isinstance(args[1], dict):
                arguments.update(args[1])
            
            context = ErrorContext(
                tool_name=tool_name,
                arguments=arguments,
                request_id=f"{time.time()}_{hash(str(arguments))}"
            )
            
            # Start performance tracking
            metrics = PerformanceMetrics(start_time=time.time())
            
            try:
                # Call the original function
                result = await func(*args, **kwargs)
                
                # Mark successful completion
                metrics.finish()
                error_handler.log_performance(metrics)
                
                return result
                
            except Exception as e:
                # Handle the error
                metrics.finish()
                error_info = error_handler.handle_error(e, context)
                
                # Create user-friendly error message
                friendly_message = error_handler.create_user_friendly_error(error_info)
                
                # Return structured error response
                from mcp.types import TextContent
                return [TextContent(
                    type="text",
                    text=friendly_message
                )]
        
        return wrapper
    return decorator

def setup_mcp_error_handling(logger_name: str = "deepflow.mcp") -> MCPErrorHandler:
    """Set up comprehensive error handling for MCP server."""
    error_handler = MCPErrorHandler(logger_name)
    
    # Log startup information
    error_handler.logger.info("Deepflow MCP Error Handling initialized")
    error_handler.logger.info(f"Log directory: {Path.home() / '.deepflow' / 'logs'}")
    
    return error_handler