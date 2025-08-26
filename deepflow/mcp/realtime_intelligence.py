#!/usr/bin/env python3
"""
Real-Time Intelligence System for Deepflow MCP
===============================================

This module provides live file monitoring and incremental dependency analysis
for AI development workflows. Features include:

- Live file watching with watchdog
- Incremental dependency graph updates  
- Real-time notifications to MCP clients
- AI context window monitoring
- Pattern deviation detection

Key Features:
- File change detection with debouncing
- Incremental graph updates for performance
- Real-time architectural violation alerts
- Session-aware change tracking
"""

import asyncio
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import logging

# Graceful imports with fallbacks
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent, FileDeletedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    Observer = None
    FileSystemEventHandler = None

try:
    import aiofiles
    AIOFILES_AVAILABLE = True
except ImportError:
    AIOFILES_AVAILABLE = False

# Import deepflow tools
try:
    import sys
    tools_dir = Path(__file__).parent.parent.parent / "tools"
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    
    from dependency_visualizer import DependencyVisualizer, DependencyAnalyzer
    from code_analyzer import CodeAnalyzer
    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class FileChangeEvent:
    """Represents a file change event with metadata."""
    file_path: str
    event_type: str  # 'created', 'modified', 'deleted'
    timestamp: float
    is_python: bool = False
    estimated_tokens: int = 0
    affected_imports: Set[str] = field(default_factory=set)

@dataclass
class DependencyUpdate:
    """Represents an incremental dependency graph update."""
    file_path: str
    old_dependencies: Set[str]
    new_dependencies: Set[str]
    added_deps: Set[str] = field(default_factory=set)
    removed_deps: Set[str] = field(default_factory=set)
    timestamp: float = field(default_factory=time.time)

@dataclass
class ArchitecturalViolation:
    """Represents a detected architectural violation."""
    file_path: str
    violation_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    suggestion: str
    timestamp: float = field(default_factory=time.time)

@dataclass
class AIContextAlert:
    """Represents an AI context optimization alert."""
    file_path: str
    token_count: int
    alert_type: str  # 'oversized', 'split_suggested', 'pattern_deviation'
    recommendation: str
    priority: str = 'medium'
    timestamp: float = field(default_factory=time.time)

class RealTimeFileHandler(FileSystemEventHandler):
    """Handles real-time file system events for deepflow."""
    
    def __init__(self, intelligence_engine):
        self.engine = intelligence_engine
        self.debounce_delay = 0.5  # 500ms debounce
        self._pending_events: Dict[str, float] = {}
        
    def _should_process_file(self, file_path: str) -> bool:
        """Check if file should be processed."""
        path = Path(file_path)
        
        # Only process Python files for now
        if not path.suffix == '.py':
            return False
            
        # Skip common ignored patterns
        ignore_patterns = [
            '__pycache__',
            '.git',
            '.pytest_cache',
            'venv',
            '.venv',
            'build',
            'dist',
            '.tox'
        ]
        
        for pattern in ignore_patterns:
            if pattern in str(path):
                return False
                
        return True
    
    def _debounce_event(self, file_path: str) -> bool:
        """Debounce rapid file changes."""
        current_time = time.time()
        
        if file_path in self._pending_events:
            if current_time - self._pending_events[file_path] < self.debounce_delay:
                return False  # Too soon, skip this event
        
        self._pending_events[file_path] = current_time
        return True
    
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory or not self._should_process_file(event.src_path):
            return
            
        if self._debounce_event(event.src_path):
            change_event = FileChangeEvent(
                file_path=event.src_path,
                event_type='modified',
                timestamp=time.time(),
                is_python=event.src_path.endswith('.py')
            )
            asyncio.create_task(self.engine.handle_file_change(change_event))
    
    def on_created(self, event):
        """Handle file creation events."""
        if event.is_directory or not self._should_process_file(event.src_path):
            return
            
        change_event = FileChangeEvent(
            file_path=event.src_path,
            event_type='created', 
            timestamp=time.time(),
            is_python=event.src_path.endswith('.py')
        )
        asyncio.create_task(self.engine.handle_file_change(change_event))
    
    def on_deleted(self, event):
        """Handle file deletion events."""
        if event.is_directory or not self._should_process_file(event.src_path):
            return
            
        change_event = FileChangeEvent(
            file_path=event.src_path,
            event_type='deleted',
            timestamp=time.time(),
            is_python=event.src_path.endswith('.py')
        )
        asyncio.create_task(self.engine.handle_file_change(change_event))

class RealTimeIntelligenceEngine:
    """Core real-time intelligence engine for deepflow."""
    
    def __init__(self, project_path: str = ".", ai_awareness: bool = True):
        self.project_path = Path(project_path)
        self.ai_awareness = ai_awareness
        
        # Internal state
        self._dependency_graph = None
        self._observer: Optional[Observer] = None
        self._file_handler = None
        self._is_monitoring = False
        
        # Change tracking
        self._recent_changes: List[FileChangeEvent] = []
        self._dependency_updates: List[DependencyUpdate] = []
        self._violations: List[ArchitecturalViolation] = []
        self._ai_alerts: List[AIContextAlert] = []
        
        # Performance tracking
        self._stats = {
            'files_monitored': 0,
            'changes_processed': 0,
            'violations_detected': 0,
            'alerts_generated': 0,
            'incremental_updates': 0
        }
        
        # Notification callbacks
        self._notification_callbacks: List[Callable] = []
        
        logger.info(f"Initialized RealTime Intelligence Engine for {self.project_path}")
    
    def add_notification_callback(self, callback: Callable):
        """Add a callback for real-time notifications."""
        self._notification_callbacks.append(callback)
        logger.debug(f"Added notification callback: {callback.__name__}")
    
    async def start_monitoring(self) -> bool:
        """Start real-time file monitoring."""
        if not WATCHDOG_AVAILABLE:
            logger.error("Watchdog not available. Install with: pip install deepflow[mcp]")
            return False
        
        if self._is_monitoring:
            logger.warning("Already monitoring. Stop first before restarting.")
            return False
        
        try:
            # Initialize dependency graph
            await self._initialize_dependency_graph()
            
            # Set up file system observer
            self._observer = Observer()
            self._file_handler = RealTimeFileHandler(self)
            
            # Monitor project directory recursively
            self._observer.schedule(
                self._file_handler,
                str(self.project_path),
                recursive=True
            )
            
            self._observer.start()
            self._is_monitoring = True
            
            logger.info(f"Started real-time monitoring of {self.project_path}")
            logger.info(f"AI Awareness: {self.ai_awareness}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}", exc_info=True)
            return False
    
    async def stop_monitoring(self):
        """Stop real-time file monitoring."""
        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None
        
        self._is_monitoring = False
        logger.info("Stopped real-time monitoring")
    
    async def _initialize_dependency_graph(self):
        """Initialize the dependency graph for incremental updates."""
        if not TOOLS_AVAILABLE:
            logger.error("Deepflow tools not available")
            return
        
        try:
            analyzer = DependencyAnalyzer(str(self.project_path), ai_awareness=self.ai_awareness)
            self._dependency_graph = analyzer.analyze_project()
            
            # Count monitored files
            self._stats['files_monitored'] = len(self._dependency_graph.nodes)
            
            logger.info(f"Initialized dependency graph with {self._stats['files_monitored']} nodes")
            
        except Exception as e:
            logger.error(f"Failed to initialize dependency graph: {e}", exc_info=True)
    
    async def handle_file_change(self, change_event: FileChangeEvent):
        """Handle a file change event with incremental updates."""
        logger.debug(f"Processing change: {change_event.event_type} {change_event.file_path}")
        
        self._recent_changes.append(change_event)
        self._stats['changes_processed'] += 1
        
        # Trim recent changes to last 100 events
        if len(self._recent_changes) > 100:
            self._recent_changes = self._recent_changes[-100:]
        
        try:
            # Perform incremental analysis
            if change_event.event_type in ['created', 'modified']:
                await self._analyze_file_incrementally(change_event)
            elif change_event.event_type == 'deleted':
                await self._handle_file_deletion(change_event)
            
            # Check for violations and alerts
            await self._check_architectural_violations(change_event)
            await self._check_ai_context_issues(change_event)
            
            # Send notifications
            await self._send_notifications(change_event)
            
        except Exception as e:
            logger.error(f"Error handling file change {change_event.file_path}: {e}", exc_info=True)
    
    async def _analyze_file_incrementally(self, change_event: FileChangeEvent):
        """Perform incremental dependency analysis on changed file."""
        if not TOOLS_AVAILABLE or not self._dependency_graph:
            return
        
        file_path = change_event.file_path
        
        try:
            # Get old dependencies
            old_deps = set()
            if file_path in self._dependency_graph.nodes:
                old_deps = set(self._dependency_graph.nodes[file_path].imports)
            
            # Re-analyze just this file
            analyzer = DependencyAnalyzer(str(self.project_path), ai_awareness=self.ai_awareness)
            new_node = analyzer._analyze_single_file(Path(file_path))
            
            if new_node:
                new_deps = set(new_node.imports)
                
                # Update dependency graph
                self._dependency_graph.nodes[file_path] = new_node
                
                # Create update record
                update = DependencyUpdate(
                    file_path=file_path,
                    old_dependencies=old_deps,
                    new_dependencies=new_deps,
                    added_deps=new_deps - old_deps,
                    removed_deps=old_deps - new_deps
                )
                
                self._dependency_updates.append(update)
                self._stats['incremental_updates'] += 1
                
                # Estimate impact
                if update.added_deps or update.removed_deps:
                    logger.info(f"Dependency update: {file_path}")
                    logger.info(f"  Added: {update.added_deps}")
                    logger.info(f"  Removed: {update.removed_deps}")
                
                # Trim update history
                if len(self._dependency_updates) > 50:
                    self._dependency_updates = self._dependency_updates[-50:]
                    
        except Exception as e:
            logger.error(f"Error in incremental analysis for {file_path}: {e}")
    
    async def _handle_file_deletion(self, change_event: FileChangeEvent):
        """Handle file deletion by removing from dependency graph."""
        if not self._dependency_graph:
            return
        
        file_path = change_event.file_path
        
        if file_path in self._dependency_graph.nodes:
            old_deps = set(self._dependency_graph.nodes[file_path].imports)
            del self._dependency_graph.nodes[file_path]
            
            # Record the deletion
            update = DependencyUpdate(
                file_path=file_path,
                old_dependencies=old_deps,
                new_dependencies=set(),
                removed_deps=old_deps
            )
            
            self._dependency_updates.append(update)
            self._stats['incremental_updates'] += 1
            
            logger.info(f"Removed deleted file from graph: {file_path}")
    
    async def _check_architectural_violations(self, change_event: FileChangeEvent):
        """Check for architectural violations in changed files."""
        if not TOOLS_AVAILABLE or change_event.event_type == 'deleted':
            return
        
        try:
            analyzer = CodeAnalyzer(str(self.project_path))
            violations = analyzer.detect_architecture_violations([change_event.file_path])
            
            for violation in violations:
                arch_violation = ArchitecturalViolation(
                    file_path=violation.file_path,
                    violation_type=violation.violation_type,
                    severity=violation.severity,
                    description=violation.description,
                    suggestion=violation.suggestion
                )
                
                self._violations.append(arch_violation)
                self._stats['violations_detected'] += 1
                
                logger.warning(f"Architectural violation detected: {violation.violation_type} in {violation.file_path}")
            
            # Trim violation history
            if len(self._violations) > 100:
                self._violations = self._violations[-100:]
                
        except Exception as e:
            logger.error(f"Error checking architectural violations: {e}")
    
    async def _check_ai_context_issues(self, change_event: FileChangeEvent):
        """Check for AI context window issues."""
        if not TOOLS_AVAILABLE or not change_event.is_python or change_event.event_type == 'deleted':
            return
        
        try:
            # Estimate token count for the file
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(change_event.file_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
            else:
                with open(change_event.file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            # Simple token estimation (approximately 4 chars per token)
            estimated_tokens = len(content) // 4
            
            # Check for oversized files
            if estimated_tokens > 1500:  # AI context warning threshold
                alert = AIContextAlert(
                    file_path=change_event.file_path,
                    token_count=estimated_tokens,
                    alert_type='oversized',
                    recommendation=f"File has ~{estimated_tokens} tokens. Consider splitting for better AI comprehension.",
                    priority='high' if estimated_tokens > 3000 else 'medium'
                )
                
                self._ai_alerts.append(alert)
                self._stats['alerts_generated'] += 1
                
                logger.warning(f"AI context alert: Oversized file {change_event.file_path} (~{estimated_tokens} tokens)")
            
            # Update change event with token estimate
            change_event.estimated_tokens = estimated_tokens
            
            # Trim alert history
            if len(self._ai_alerts) > 50:
                self._ai_alerts = self._ai_alerts[-50:]
                
        except Exception as e:
            logger.error(f"Error checking AI context issues: {e}")
    
    async def _send_notifications(self, change_event: FileChangeEvent):
        """Send real-time notifications to registered callbacks."""
        notification_data = {
            'type': 'file_change',
            'event': {
                'file_path': change_event.file_path,
                'event_type': change_event.event_type,
                'timestamp': change_event.timestamp,
                'estimated_tokens': change_event.estimated_tokens,
                'is_python': change_event.is_python
            },
            'stats': self._stats.copy(),
            'recent_violations': [
                {
                    'file_path': v.file_path,
                    'type': v.violation_type,
                    'severity': v.severity,
                    'description': v.description
                }
                for v in self._violations[-5:]  # Last 5 violations
            ],
            'recent_alerts': [
                {
                    'file_path': a.file_path,
                    'type': a.alert_type,
                    'priority': a.priority,
                    'recommendation': a.recommendation
                }
                for a in self._ai_alerts[-5:]  # Last 5 alerts
            ]
        }
        
        # Send to all registered callbacks
        for callback in self._notification_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(notification_data)
                else:
                    callback(notification_data)
            except Exception as e:
                logger.error(f"Error in notification callback {callback.__name__}: {e}")
    
    def get_real_time_stats(self) -> Dict[str, Any]:
        """Get current real-time monitoring statistics."""
        return {
            'monitoring': self._is_monitoring,
            'project_path': str(self.project_path),
            'ai_awareness': self.ai_awareness,
            'stats': self._stats.copy(),
            'recent_changes': len(self._recent_changes),
            'dependency_updates': len(self._dependency_updates),
            'violations': len(self._violations),
            'ai_alerts': len(self._ai_alerts),
            'notification_callbacks': len(self._notification_callbacks)
        }
    
    def get_recent_activity(self, limit: int = 20) -> Dict[str, Any]:
        """Get recent monitoring activity."""
        return {
            'changes': [
                {
                    'file_path': c.file_path,
                    'event_type': c.event_type,
                    'timestamp': c.timestamp,
                    'estimated_tokens': c.estimated_tokens
                }
                for c in self._recent_changes[-limit:]
            ],
            'dependency_updates': [
                {
                    'file_path': u.file_path,
                    'added_deps': list(u.added_deps),
                    'removed_deps': list(u.removed_deps),
                    'timestamp': u.timestamp
                }
                for u in self._dependency_updates[-limit:]
            ],
            'violations': [
                {
                    'file_path': v.file_path,
                    'type': v.violation_type,
                    'severity': v.severity,
                    'description': v.description,
                    'timestamp': v.timestamp
                }
                for v in self._violations[-limit:]
            ],
            'ai_alerts': [
                {
                    'file_path': a.file_path,
                    'type': a.alert_type,
                    'priority': a.priority,
                    'recommendation': a.recommendation,
                    'timestamp': a.timestamp
                }
                for a in self._ai_alerts[-limit:]
            ]
        }

class RealTimeNotificationService:
    """Service for handling real-time notifications to MCP clients."""
    
    def __init__(self):
        self._notification_queue = asyncio.Queue()
        self._subscribers = []
        self._is_running = False
    
    def subscribe(self, callback: Callable):
        """Subscribe to real-time notifications."""
        self._subscribers.append(callback)
        logger.info(f"New subscriber: {callback.__name__}")
    
    def unsubscribe(self, callback: Callable):
        """Unsubscribe from notifications."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)
            logger.info(f"Unsubscribed: {callback.__name__}")
    
    async def push_notification(self, notification_data: Dict[str, Any]):
        """Push a notification to all subscribers."""
        await self._notification_queue.put(notification_data)
    
    async def start_service(self):
        """Start the notification service."""
        self._is_running = True
        logger.info("Started real-time notification service")
        
        while self._is_running:
            try:
                # Wait for notifications with timeout
                notification = await asyncio.wait_for(
                    self._notification_queue.get(), 
                    timeout=1.0
                )
                
                # Send to all subscribers
                for subscriber in self._subscribers:
                    try:
                        if asyncio.iscoroutinefunction(subscriber):
                            await subscriber(notification)
                        else:
                            subscriber(notification)
                    except Exception as e:
                        logger.error(f"Error notifying subscriber {subscriber.__name__}: {e}")
                
            except asyncio.TimeoutError:
                # No notifications in the last second, continue
                continue
            except Exception as e:
                logger.error(f"Error in notification service: {e}")
    
    async def stop_service(self):
        """Stop the notification service."""
        self._is_running = False
        logger.info("Stopped real-time notification service")

# Global instances for easy access
_global_intelligence_engine: Optional[RealTimeIntelligenceEngine] = None
_global_notification_service: Optional[RealTimeNotificationService] = None

def get_intelligence_engine(project_path: str = ".", ai_awareness: bool = True) -> RealTimeIntelligenceEngine:
    """Get or create the global intelligence engine instance."""
    global _global_intelligence_engine
    
    if _global_intelligence_engine is None or str(_global_intelligence_engine.project_path) != project_path:
        _global_intelligence_engine = RealTimeIntelligenceEngine(project_path, ai_awareness)
    
    return _global_intelligence_engine

def get_notification_service() -> RealTimeNotificationService:
    """Get or create the global notification service instance."""
    global _global_notification_service
    
    if _global_notification_service is None:
        _global_notification_service = RealTimeNotificationService()
    
    return _global_notification_service