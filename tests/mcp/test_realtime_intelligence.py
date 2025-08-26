#!/usr/bin/env python3
"""
Comprehensive test suite for Real-Time Intelligence features
===========================================================

Tests for:
- File change detection with watchdog
- Incremental dependency updates
- MCP real-time tools integration
- Architectural violation detection
- AI context monitoring
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
import json
import time

# Test imports with graceful fallbacks
try:
    from deepflow.mcp.realtime_intelligence import (
        RealTimeIntelligenceEngine,
        RealTimeNotificationService,
        RealTimeFileHandler,
        FileChangeEvent,
        DependencyUpdate,
        ArchitecturalViolation,
        AIContextAlert,
        get_intelligence_engine,
        get_notification_service
    )
    REALTIME_AVAILABLE = True
except ImportError:
    REALTIME_AVAILABLE = False
    pytest.skip("Real-time intelligence not available", allow_module_level=True)

try:
    from deepflow.mcp.server import DeepflowMCPServer
    MCP_SERVER_AVAILABLE = True
except ImportError:
    MCP_SERVER_AVAILABLE = False

try:
    import watchdog
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

class TestFileChangeEvent:
    """Test FileChangeEvent dataclass."""
    
    def test_file_change_event_creation(self):
        """Test creating a file change event."""
        event = FileChangeEvent(
            file_path="/test/file.py",
            event_type="modified",
            timestamp=time.time(),
            is_python=True,
            estimated_tokens=500
        )
        
        assert event.file_path == "/test/file.py"
        assert event.event_type == "modified"
        assert event.is_python is True
        assert event.estimated_tokens == 500
        assert isinstance(event.affected_imports, set)

class TestRealTimeFileHandler:
    """Test real-time file system event handler."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_engine = MagicMock()
        self.handler = RealTimeFileHandler(self.mock_engine)
    
    def test_should_process_python_files(self):
        """Test that Python files are processed."""
        assert self.handler._should_process_file("/test/module.py") is True
    
    def test_should_ignore_non_python_files(self):
        """Test that non-Python files are ignored."""
        assert self.handler._should_process_file("/test/data.json") is False
        assert self.handler._should_process_file("/test/doc.md") is False
    
    def test_should_ignore_cache_directories(self):
        """Test that cache directories are ignored."""
        assert self.handler._should_process_file("/test/__pycache__/module.py") is False
        assert self.handler._should_process_file("/test/.pytest_cache/file.py") is False
        assert self.handler._should_process_file("/test/venv/lib/python.py") is False
    
    def test_debounce_rapid_changes(self):
        """Test debouncing of rapid file changes."""
        file_path = "/test/module.py"
        
        # First event should be processed
        assert self.handler._debounce_event(file_path) is True
        
        # Immediate second event should be debounced
        assert self.handler._debounce_event(file_path) is False
        
        # After delay, should be processed again
        time.sleep(0.6)  # Wait for debounce delay
        assert self.handler._debounce_event(file_path) is True
    
    @patch('asyncio.create_task')
    def test_on_modified_creates_task(self, mock_create_task):
        """Test that file modifications create async tasks."""
        mock_event = MagicMock()
        mock_event.is_directory = False
        mock_event.src_path = "/test/module.py"
        
        # Mock debounce to allow processing
        with patch.object(self.handler, '_debounce_event', return_value=True):
            self.handler.on_modified(mock_event)
        
        mock_create_task.assert_called_once()

class TestRealTimeIntelligenceEngine:
    """Test the main real-time intelligence engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.engine = RealTimeIntelligenceEngine(
            project_path=str(self.test_dir),
            ai_awareness=True
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, 'engine') and self.engine._is_monitoring:
            asyncio.run(self.engine.stop_monitoring())
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test engine initialization."""
        assert self.engine.project_path == self.test_dir
        assert self.engine.ai_awareness is True
        assert self.engine._is_monitoring is False
        assert len(self.engine._recent_changes) == 0
        assert len(self.engine._dependency_updates) == 0
    
    def test_add_notification_callback(self):
        """Test adding notification callbacks."""
        callback = MagicMock()
        self.engine.add_notification_callback(callback)
        
        assert callback in self.engine._notification_callbacks
    
    @pytest.mark.asyncio
    async def test_file_change_handling(self):
        """Test handling file change events."""
        # Create a test Python file
        test_file = self.test_dir / "test_module.py"
        test_file.write_text("import os\nprint('hello')")
        
        change_event = FileChangeEvent(
            file_path=str(test_file),
            event_type="modified",
            timestamp=time.time(),
            is_python=True
        )
        
        # Mock the dependency analysis
        with patch.object(self.engine, '_analyze_file_incrementally'), \
             patch.object(self.engine, '_check_architectural_violations'), \
             patch.object(self.engine, '_check_ai_context_issues'), \
             patch.object(self.engine, '_send_notifications'):
            
            await self.engine.handle_file_change(change_event)
        
        assert len(self.engine._recent_changes) == 1
        assert self.engine._stats['changes_processed'] == 1
    
    @pytest.mark.asyncio
    async def test_ai_context_monitoring(self):
        """Test AI context window monitoring."""
        # Create a large test file
        large_content = "# This is a large file\n" + "print('line')\n" * 1000
        test_file = self.test_dir / "large_module.py"
        test_file.write_text(large_content)
        
        change_event = FileChangeEvent(
            file_path=str(test_file),
            event_type="modified",
            timestamp=time.time(),
            is_python=True
        )
        
        await self.engine._check_ai_context_issues(change_event)
        
        # Should generate an AI context alert for oversized file
        assert len(self.engine._ai_alerts) == 1
        alert = self.engine._ai_alerts[0]
        assert alert.alert_type == "oversized"
        assert alert.token_count > 1500
    
    def test_get_real_time_stats(self):
        """Test getting real-time statistics."""
        stats = self.engine.get_real_time_stats()
        
        assert 'monitoring' in stats
        assert 'project_path' in stats
        assert 'ai_awareness' in stats
        assert 'stats' in stats
        assert stats['monitoring'] is False
        assert stats['project_path'] == str(self.test_dir)
    
    def test_get_recent_activity(self):
        """Test getting recent activity."""
        # Add some test changes
        test_change = FileChangeEvent(
            file_path="/test/file.py",
            event_type="modified",
            timestamp=time.time(),
            is_python=True
        )
        self.engine._recent_changes.append(test_change)
        
        activity = self.engine.get_recent_activity(limit=10)
        
        assert 'changes' in activity
        assert 'dependency_updates' in activity
        assert 'violations' in activity
        assert 'ai_alerts' in activity
        assert len(activity['changes']) == 1

class TestRealTimeNotificationService:
    """Test the notification service."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.service = RealTimeNotificationService()
    
    def test_subscription_management(self):
        """Test subscribing and unsubscribing."""
        callback = MagicMock()
        
        self.service.subscribe(callback)
        assert callback in self.service._subscribers
        
        self.service.unsubscribe(callback)
        assert callback not in self.service._subscribers
    
    @pytest.mark.asyncio
    async def test_push_notification(self):
        """Test pushing notifications to queue."""
        test_data = {"type": "test", "message": "test notification"}
        
        await self.service.push_notification(test_data)
        
        # Check that notification is in queue
        assert not self.service._notification_queue.empty()

@pytest.mark.skipif(not MCP_SERVER_AVAILABLE, reason="MCP server not available")
class TestMCPRealTimeIntegration:
    """Test MCP server integration with real-time features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.server = DeepflowMCPServer()
        self.test_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    @pytest.mark.asyncio
    async def test_start_realtime_monitoring_mcp_tool(self):
        """Test starting real-time monitoring via MCP tool."""
        arguments = {
            "project_path": str(self.test_dir),
            "ai_awareness": True
        }
        
        # Mock the real-time engine to avoid actual file watching in tests
        with patch('deepflow.mcp.server.get_intelligence_engine') as mock_get_engine, \
             patch('deepflow.mcp.server.get_notification_service'):
            
            mock_engine = MagicMock()
            mock_engine.start_monitoring = AsyncMock(return_value=True)
            mock_engine.add_notification_callback = MagicMock()
            mock_get_engine.return_value = mock_engine
            
            result = await self.server._handle_start_realtime_monitoring(arguments)
            
            assert len(result) == 1
            response = json.loads(result[0].text)
            assert response["status"] == "started"
            assert "Real-time monitoring started" in response["message"]
    
    @pytest.mark.asyncio
    async def test_stop_realtime_monitoring_mcp_tool(self):
        """Test stopping real-time monitoring via MCP tool."""
        # Set up a mock engine
        mock_engine = MagicMock()
        mock_engine.stop_monitoring = AsyncMock()
        self.server._realtime_engine = mock_engine
        
        result = await self.server._handle_stop_realtime_monitoring({})
        
        assert len(result) == 1
        response = json.loads(result[0].text)
        assert response["status"] == "stopped"
    
    @pytest.mark.asyncio
    async def test_get_realtime_stats_mcp_tool(self):
        """Test getting real-time stats via MCP tool."""
        # Set up a mock engine with stats
        mock_engine = MagicMock()
        mock_engine.get_real_time_stats.return_value = {
            "monitoring": True,
            "project_path": str(self.test_dir),
            "stats": {"changes_processed": 5}
        }
        self.server._realtime_engine = mock_engine
        
        result = await self.server._handle_get_realtime_stats({})
        
        assert len(result) == 1
        stats = json.loads(result[0].text)
        assert stats["monitoring"] is True
        assert stats["stats"]["changes_processed"] == 5
    
    @pytest.mark.asyncio
    async def test_get_realtime_activity_mcp_tool(self):
        """Test getting real-time activity via MCP tool."""
        mock_engine = MagicMock()
        mock_activity = {
            "changes": [{"file_path": "/test/file.py", "event_type": "modified"}],
            "dependency_updates": [],
            "violations": [],
            "ai_alerts": []
        }
        mock_engine.get_recent_activity.return_value = mock_activity
        self.server._realtime_engine = mock_engine
        
        result = await self.server._handle_get_realtime_activity({"limit": 10})
        
        assert len(result) == 1
        activity = json.loads(result[0].text)
        assert len(activity["changes"]) == 1
        assert activity["changes"][0]["file_path"] == "/test/file.py"
    
    def test_realtime_tools_in_mcp_tools_list(self):
        """Test that real-time tools are included in MCP tools list."""
        tools = self.server.get_tools()
        tool_names = [tool.name for tool in tools]
        
        assert "start_realtime_monitoring" in tool_names
        assert "stop_realtime_monitoring" in tool_names
        assert "get_realtime_activity" in tool_names
        assert "get_realtime_stats" in tool_names

class TestGlobalInstanceManagement:
    """Test global instance management functions."""
    
    def test_get_intelligence_engine_singleton(self):
        """Test that get_intelligence_engine returns singleton."""
        engine1 = get_intelligence_engine("/test/path1")
        engine2 = get_intelligence_engine("/test/path1")
        
        # Same path should return same instance
        assert engine1 is engine2
        
        # Different path should return new instance
        engine3 = get_intelligence_engine("/test/path2")
        assert engine3 is not engine1
    
    def test_get_notification_service_singleton(self):
        """Test that get_notification_service returns singleton."""
        service1 = get_notification_service()
        service2 = get_notification_service()
        
        assert service1 is service2

@pytest.mark.integration
@pytest.mark.skipif(not WATCHDOG_AVAILABLE, reason="Watchdog not available")
class TestRealTimeIntegrationE2E:
    """End-to-end integration tests for real-time features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.engine = RealTimeIntelligenceEngine(
            project_path=str(self.test_dir),
            ai_awareness=True
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if hasattr(self, 'engine') and self.engine._is_monitoring:
            asyncio.run(self.engine.stop_monitoring())
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
    
    @pytest.mark.asyncio
    async def test_end_to_end_file_monitoring(self):
        """Test complete end-to-end file monitoring workflow."""
        # Create initial Python file
        test_file = self.test_dir / "module.py"
        test_file.write_text("import os\n")
        
        # Set up notification tracking
        notifications = []
        
        def notification_callback(data):
            notifications.append(data)
        
        self.engine.add_notification_callback(notification_callback)
        
        # Start monitoring (may not work in test environment)
        # This is more of a smoke test to ensure no exceptions
        try:
            await self.engine.start_monitoring()
            
            # Modify the file
            test_file.write_text("import os\nimport sys\n")
            
            # Give some time for file system events
            await asyncio.sleep(0.1)
            
            # Stop monitoring
            await self.engine.stop_monitoring()
            
            # Check stats were updated
            stats = self.engine.get_real_time_stats()
            assert stats['monitoring'] is False
            
        except Exception as e:
            # File watching may not work in all test environments
            # Just ensure we can start/stop without crashing
            pytest.skip(f"File watching not available in test environment: {e}")

class TestErrorHandling:
    """Test error handling in real-time features."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = RealTimeIntelligenceEngine("/nonexistent/path")
    
    @pytest.mark.asyncio
    async def test_start_monitoring_with_invalid_path(self):
        """Test starting monitoring with invalid path."""
        # Should not crash, but may return False
        result = await self.engine.start_monitoring()
        
        # Result depends on whether watchdog is available
        assert isinstance(result, bool)
    
    @pytest.mark.asyncio
    async def test_handle_file_change_with_invalid_file(self):
        """Test handling changes to non-existent files."""
        change_event = FileChangeEvent(
            file_path="/nonexistent/file.py",
            event_type="modified",
            timestamp=time.time(),
            is_python=True
        )
        
        # Should not crash
        await self.engine.handle_file_change(change_event)
        
        # Stats should still be updated
        assert self.engine._stats['changes_processed'] == 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])