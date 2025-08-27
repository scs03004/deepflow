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

@dataclass
class PatternDeviationAlert:
    """Represents a pattern deviation in AI-generated code."""
    file_path: str
    deviation_type: str  # 'naming', 'structure', 'imports', 'style'
    expected_pattern: str
    actual_pattern: str
    confidence: float  # 0.0 to 1.0
    suggestion: str
    severity: str = 'medium'  # 'low', 'medium', 'high'
    timestamp: float = field(default_factory=time.time)

@dataclass
class CircularDependencyAlert:
    """Represents a potential circular dependency."""
    involved_files: List[str]
    dependency_chain: List[str]
    risk_level: str  # 'potential', 'likely', 'confirmed'
    impact_assessment: str
    prevention_suggestion: str
    timestamp: float = field(default_factory=time.time)

@dataclass
class FileSplitSuggestion:
    """Represents a suggestion to split a large file."""
    file_path: str
    current_size_tokens: int
    suggested_splits: List[Dict[str, Any]]  # Each split contains classes/functions
    split_rationale: str
    estimated_improvement: str
    priority: str = 'medium'
    timestamp: float = field(default_factory=time.time)

@dataclass
class DuplicatePatternAlert:
    """Represents detected duplicate patterns that could be consolidated."""
    pattern_type: str  # 'function', 'class', 'import_block', 'logic_block'
    duplicate_locations: List[Dict[str, Any]]  # file_path, line_range, content_hash
    similarity_score: float  # 0.0 to 1.0
    consolidation_suggestion: str
    estimated_savings: str  # lines saved, complexity reduction
    timestamp: float = field(default_factory=time.time)

# Priority 3: AI Session Intelligence Data Classes

@dataclass
class SessionContext:
    """Represents an AI development session context."""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    session_name: str = ""
    session_description: str = ""
    files_modified: Set[str] = field(default_factory=set)
    changes_made: List[Dict[str, Any]] = field(default_factory=list)
    patterns_learned: Dict[str, Any] = field(default_factory=dict)
    goals_achieved: List[str] = field(default_factory=list)
    session_tags: Set[str] = field(default_factory=set)
    ai_interactions: int = 0
    timestamp: float = field(default_factory=time.time)

@dataclass
class ChangeImpactAnalysis:
    """Represents the ripple effects of current modifications."""
    change_id: str
    affected_file: str
    change_type: str  # 'addition', 'modification', 'deletion', 'rename'
    ripple_effects: List[Dict[str, Any]]  # affected files and their impact
    dependency_impacts: List[str]  # files that depend on this change
    test_impacts: List[str]  # test files that need updating
    documentation_impacts: List[str]  # docs that need updating
    risk_assessment: str  # 'low', 'medium', 'high', 'critical'
    impact_score: float  # 0.0 to 1.0
    mitigation_suggestions: List[str]
    timestamp: float = field(default_factory=time.time)

@dataclass
class PatternLearningData:
    """Represents learned patterns from development sessions."""
    pattern_id: str
    pattern_type: str  # 'naming', 'structure', 'imports', 'style', 'workflow'
    pattern_description: str
    learned_from_files: List[str]
    confidence_score: float  # 0.0 to 1.0
    usage_frequency: int
    pattern_examples: List[Dict[str, Any]]
    project_specificity: float  # how specific to this project (0.0 = universal, 1.0 = project-specific)
    learning_date: float = field(default_factory=time.time)
    last_reinforcement: float = field(default_factory=time.time)

@dataclass
class MultiFileCoordination:
    """Tracks related changes across multiple files."""
    coordination_id: str
    related_files: Set[str]
    coordination_type: str  # 'refactoring', 'feature_development', 'bug_fix', 'pattern_alignment'
    change_sequence: List[Dict[str, Any]]  # ordered changes across files
    dependencies_between_changes: List[Dict[str, str]]  # change_id -> depends_on_change_id
    completion_status: Dict[str, bool]  # file -> is_complete
    coordination_context: str
    estimated_completion: float
    priority: str = 'medium'
    timestamp: float = field(default_factory=time.time)

@dataclass
class SessionJournalEntry:
    """Automatic documentation of AI development activities."""
    entry_id: str
    session_id: str
    entry_type: str  # 'session_start', 'change', 'pattern_learned', 'goal_achieved', 'session_end'
    entry_title: str
    entry_description: str
    affected_files: List[str]
    code_changes: Dict[str, Any]  # file -> change details
    ai_context: str  # what the AI was trying to accomplish
    outcome: str  # what was achieved
    lessons_learned: List[str]
    follow_up_actions: List[str]
    tags: Set[str] = field(default_factory=set)
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
        
        # Priority 2: Proactive AI Development Assistance
        self._pattern_deviations: List[PatternDeviationAlert] = []
        self._circular_dependency_alerts: List[CircularDependencyAlert] = []
        self._file_split_suggestions: List[FileSplitSuggestion] = []
        self._duplicate_patterns: List[DuplicatePatternAlert] = []
        
        # Priority 3: AI Session Intelligence
        self._current_session: Optional[SessionContext] = None
        self._session_history: List[SessionContext] = []
        self._change_impact_analyses: List[ChangeImpactAnalysis] = []
        self._pattern_learning_data: List[PatternLearningData] = []
        self._multi_file_coordinations: List[MultiFileCoordination] = []
        self._session_journal: List[SessionJournalEntry] = []
        
        # Enhanced pattern learning with session intelligence
        self._learned_patterns = {
            'naming_conventions': {},
            'import_patterns': {},
            'class_structures': {},
            'function_signatures': {},
            'workflow_patterns': {},
            'session_patterns': {}
        }
        
        # Performance tracking
        self._stats = {
            'files_monitored': 0,
            'changes_processed': 0,
            'violations_detected': 0,
            'alerts_generated': 0,
            'incremental_updates': 0,
            'pattern_deviations': 0,
            'circular_dependencies_prevented': 0,
            'file_splits_suggested': 0,
            'duplicate_patterns_found': 0,
            # Priority 3: AI Session Intelligence stats
            'sessions_tracked': 0,
            'impact_analyses_performed': 0,
            'patterns_learned': 0,
            'multi_file_coordinations_managed': 0,
            'journal_entries_created': 0,
            'session_context_restorations': 0
        }
        
        # Notification callbacks
        self._notification_callbacks: List[Callable] = []
        
        logger.info(f"Initialized RealTime Intelligence Engine for {self.project_path}")
    
    @property
    def is_monitoring(self) -> bool:
        """Check if real-time monitoring is currently active."""
        return self._is_monitoring
    
    def get_ai_context_stats(self) -> Dict[str, Any]:
        """Get AI context and file size statistics."""
        python_files = list(self.project_path.rglob("*.py"))
        oversized_files = []
        total_tokens = 0
        
        for file_path in python_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Rough token estimation: 1 token ≈ 4 characters
                    file_tokens = len(content) // 4
                    total_tokens += file_tokens
                    
                    # Flag files over 8K tokens (Claude's optimal context window segment)
                    if file_tokens > 8000:
                        oversized_files.append({
                            'file_path': str(file_path),
                            'estimated_tokens': file_tokens,
                            'lines': len(content.splitlines())
                        })
            except Exception:
                continue
        
        return {
            'total_python_files': len(python_files),
            'oversized_files': oversized_files,
            'total_estimated_tokens': total_tokens,
            'optimal_context_violations': len(oversized_files)
        }
    
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
            
            # Priority 2: Proactive AI Development Assistance
            if change_event.event_type in ['created', 'modified']:
                await self._check_pattern_deviations(change_event)
                await self._check_circular_dependencies(change_event)
                await self._suggest_file_splits(change_event)
                await self._detect_duplicate_patterns(change_event)
            
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
    
    async def _check_pattern_deviations(self, change_event: FileChangeEvent):
        """Check for pattern deviations in AI-generated code."""
        if not change_event.is_python or change_event.event_type == 'deleted':
            return
        
        try:
            # Read file content
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(change_event.file_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
            else:
                with open(change_event.file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            # Simple pattern analysis - look for common deviations
            lines = content.split('\n')
            
            # Check naming conventions
            await self._analyze_naming_patterns(change_event.file_path, lines)
            
            # Check import patterns  
            await self._analyze_import_patterns(change_event.file_path, lines)
            
            # Check function/class structure patterns
            await self._analyze_structure_patterns(change_event.file_path, lines)
            
        except Exception as e:
            logger.error(f"Error checking pattern deviations: {e}")
    
    async def _analyze_naming_patterns(self, file_path: str, lines: List[str]):
        """Analyze naming convention patterns."""
        try:
            import re
            
            # Extract function and class names
            functions = []
            classes = []
            variables = []
            
            for line in lines:
                line = line.strip()
                
                # Function definitions
                func_match = re.match(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', line)
                if func_match:
                    functions.append(func_match.group(1))
                
                # Class definitions
                class_match = re.match(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', line)
                if class_match:
                    classes.append(class_match.group(1))
                
                # Variable assignments (simple detection)
                var_match = re.match(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=', line)
                if var_match and not line.startswith('def ') and not line.startswith('class '):
                    variables.append(var_match.group(1))
            
            # Learn patterns from existing project
            if not self._learned_patterns['naming_conventions'].get('functions'):
                self._learned_patterns['naming_conventions']['functions'] = set()
            if not self._learned_patterns['naming_conventions'].get('classes'):
                self._learned_patterns['naming_conventions']['classes'] = set()
            
            # Add to learned patterns
            self._learned_patterns['naming_conventions']['functions'].update(functions)
            self._learned_patterns['naming_conventions']['classes'].update(classes)
            
            # Check for deviations
            for func_name in functions:
                if not re.match(r'^[a-z_][a-z0-9_]*$', func_name):
                    deviation = PatternDeviationAlert(
                        file_path=file_path,
                        deviation_type='naming',
                        expected_pattern='snake_case for functions',
                        actual_pattern=f'Function "{func_name}" uses different naming',
                        confidence=0.8,
                        suggestion=f'Consider renaming "{func_name}" to use snake_case',
                        severity='medium'
                    )
                    self._pattern_deviations.append(deviation)
                    self._stats['pattern_deviations'] += 1
            
            for class_name in classes:
                if not re.match(r'^[A-Z][a-zA-Z0-9]*$', class_name):
                    deviation = PatternDeviationAlert(
                        file_path=file_path,
                        deviation_type='naming',
                        expected_pattern='PascalCase for classes',
                        actual_pattern=f'Class "{class_name}" uses different naming',
                        confidence=0.8,
                        suggestion=f'Consider renaming "{class_name}" to use PascalCase',
                        severity='medium'
                    )
                    self._pattern_deviations.append(deviation)
                    self._stats['pattern_deviations'] += 1
            
            # Trim history
            if len(self._pattern_deviations) > 100:
                self._pattern_deviations = self._pattern_deviations[-100:]
                
        except Exception as e:
            logger.error(f"Error analyzing naming patterns: {e}")
    
    async def _analyze_import_patterns(self, file_path: str, lines: List[str]):
        """Analyze import pattern consistency."""
        try:
            import re
            
            imports = []
            from_imports = []
            
            for line in lines:
                line = line.strip()
                if line.startswith('import '):
                    imports.append(line)
                elif line.startswith('from '):
                    from_imports.append(line)
            
            # Check for common import organization issues
            if imports and from_imports:
                # Look for mixed import styles that could be consolidated
                import_modules = set()
                for imp in imports:
                    module = imp.replace('import ', '').split(' as ')[0].strip()
                    import_modules.add(module.split('.')[0])
                
                for from_imp in from_imports:
                    if ' import ' in from_imp:
                        module = from_imp.split(' import ')[0].replace('from ', '').strip()
                        base_module = module.split('.')[0]
                        
                        if base_module in import_modules:
                            deviation = PatternDeviationAlert(
                                file_path=file_path,
                                deviation_type='imports',
                                expected_pattern='Consistent import style per module',
                                actual_pattern=f'Mixed import styles for {base_module}',
                                confidence=0.7,
                                suggestion=f'Consider using consistent import style for {base_module} module',
                                severity='low'
                            )
                            self._pattern_deviations.append(deviation)
                            self._stats['pattern_deviations'] += 1
                            break
                            
        except Exception as e:
            logger.error(f"Error analyzing import patterns: {e}")
    
    async def _analyze_structure_patterns(self, file_path: str, lines: List[str]):
        """Analyze code structure patterns."""
        try:
            # Check for overly long functions (AI tends to generate these)
            current_function = None
            function_lines = 0
            
            for i, line in enumerate(lines):
                line_stripped = line.strip()
                
                if line_stripped.startswith('def '):
                    if current_function and function_lines > 50:
                        deviation = PatternDeviationAlert(
                            file_path=file_path,
                            deviation_type='structure',
                            expected_pattern='Functions should be concise (<50 lines)',
                            actual_pattern=f'Function {current_function} has {function_lines} lines',
                            confidence=0.6,
                            suggestion=f'Consider breaking down {current_function} into smaller functions',
                            severity='medium'
                        )
                        self._pattern_deviations.append(deviation)
                        self._stats['pattern_deviations'] += 1
                    
                    # Reset for new function
                    current_function = line_stripped.split('(')[0].replace('def ', '').strip()
                    function_lines = 1
                elif current_function:
                    function_lines += 1
                    
        except Exception as e:
            logger.error(f"Error analyzing structure patterns: {e}")
    
    async def _check_circular_dependencies(self, change_event: FileChangeEvent):
        """Check for potential circular dependencies."""
        if not TOOLS_AVAILABLE or not self._dependency_graph:
            return
        
        try:
            current_file = change_event.file_path
            
            # Get current file's dependencies
            if current_file in self._dependency_graph.nodes:
                current_deps = set(self._dependency_graph.nodes[current_file].imports)
                
                # Check for potential circular dependencies
                for dep in current_deps:
                    # Look for files that might import the current file
                    for other_file, node in self._dependency_graph.nodes.items():
                        if other_file != current_file:
                            other_imports = set(node.imports)
                            
                            # Check if other_file imports current_file
                            current_module = Path(current_file).stem
                            if current_module in other_imports:
                                # Check if current_file imports other_file  
                                other_module = Path(other_file).stem
                                if other_module in current_deps:
                                    alert = CircularDependencyAlert(
                                        involved_files=[current_file, other_file],
                                        dependency_chain=[current_file, other_module, current_module],
                                        risk_level='potential',
                                        impact_assessment='May cause import errors or module initialization issues',
                                        prevention_suggestion='Consider extracting shared functionality to a separate module'
                                    )
                                    self._circular_dependency_alerts.append(alert)
                                    self._stats['circular_dependencies_prevented'] += 1
                                    
                                    logger.warning(f"Potential circular dependency: {current_file} <-> {other_file}")
            
            # Trim history
            if len(self._circular_dependency_alerts) > 50:
                self._circular_dependency_alerts = self._circular_dependency_alerts[-50:]
                
        except Exception as e:
            logger.error(f"Error checking circular dependencies: {e}")
    
    async def _suggest_file_splits(self, change_event: FileChangeEvent):
        """Suggest file splits for better AI comprehension."""
        if not change_event.is_python or not hasattr(change_event, 'estimated_tokens'):
            return
        
        try:
            # Only suggest splits for large files
            if change_event.estimated_tokens > 2000:
                # Read file to analyze structure
                if AIOFILES_AVAILABLE:
                    async with aiofiles.open(change_event.file_path, 'r', encoding='utf-8') as f:
                        content = await f.read()
                else:
                    with open(change_event.file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                
                lines = content.split('\n')
                
                # Simple analysis to suggest splits
                classes = []
                functions = []
                current_class = None
                current_function = None
                
                for i, line in enumerate(lines):
                    line_stripped = line.strip()
                    
                    if line_stripped.startswith('class '):
                        class_name = line_stripped.split('(')[0].replace('class ', '').strip().rstrip(':')
                        classes.append({
                            'name': class_name,
                            'start_line': i + 1,
                            'estimated_lines': 50  # Will be calculated properly
                        })
                        current_class = class_name
                    elif line_stripped.startswith('def ') and not line.startswith('    '):
                        func_name = line_stripped.split('(')[0].replace('def ', '').strip()
                        functions.append({
                            'name': func_name,
                            'start_line': i + 1,
                            'estimated_lines': 20  # Will be calculated properly
                        })
                
                # Suggest splits if we have multiple classes or many functions
                suggested_splits = []
                
                if len(classes) > 1:
                    for cls in classes:
                        suggested_splits.append({
                            'type': 'class',
                            'name': cls['name'],
                            'suggested_filename': f"{cls['name'].lower()}.py",
                            'rationale': f"Extract {cls['name']} class to separate file for better organization"
                        })
                
                if len(functions) > 10:
                    # Group related functions
                    suggested_splits.append({
                        'type': 'functions',
                        'name': 'utility_functions',
                        'suggested_filename': 'utils.py',
                        'rationale': f"Extract {len(functions)} utility functions to separate module"
                    })
                
                if suggested_splits:
                    suggestion = FileSplitSuggestion(
                        file_path=change_event.file_path,
                        current_size_tokens=change_event.estimated_tokens,
                        suggested_splits=suggested_splits,
                        split_rationale='Large file detected - splitting will improve AI comprehension and maintainability',
                        estimated_improvement=f'Reduce file size from ~{change_event.estimated_tokens} to <1000 tokens per file',
                        priority='high' if change_event.estimated_tokens > 4000 else 'medium'
                    )
                    self._file_split_suggestions.append(suggestion)
                    self._stats['file_splits_suggested'] += 1
                    
                    logger.info(f"File split suggested for {change_event.file_path} ({change_event.estimated_tokens} tokens)")
            
            # Trim history
            if len(self._file_split_suggestions) > 30:
                self._file_split_suggestions = self._file_split_suggestions[-30:]
                
        except Exception as e:
            logger.error(f"Error suggesting file splits: {e}")
    
    async def _detect_duplicate_patterns(self, change_event: FileChangeEvent):
        """Detect duplicate patterns that could be consolidated."""
        if not change_event.is_python:
            return
        
        try:
            # Read current file content
            if AIOFILES_AVAILABLE:
                async with aiofiles.open(change_event.file_path, 'r', encoding='utf-8') as f:
                    content = await f.read()
            else:
                with open(change_event.file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            lines = content.split('\n')
            
            # Simple duplicate detection - look for similar function structures
            functions = []
            current_function = None
            function_body = []
            
            for line in lines:
                line_stripped = line.strip()
                
                if line_stripped.startswith('def '):
                    # Save previous function
                    if current_function and function_body:
                        functions.append({
                            'name': current_function,
                            'body_hash': hash('\n'.join(function_body)),
                            'body': function_body.copy(),
                            'file_path': change_event.file_path
                        })
                    
                    # Start new function
                    current_function = line_stripped.split('(')[0].replace('def ', '').strip()
                    function_body = []
                elif current_function and line.startswith('    '):
                    # Function body line
                    function_body.append(line_stripped)
                elif current_function and not line.strip():
                    # Empty line in function
                    function_body.append('')
                elif current_function:
                    # End of function
                    if function_body:
                        functions.append({
                            'name': current_function,
                            'body_hash': hash('\n'.join(function_body)),
                            'body': function_body.copy(),
                            'file_path': change_event.file_path
                        })
                    current_function = None
                    function_body = []
            
            # Check for duplicates within this file
            seen_hashes = {}
            for func in functions:
                if func['body_hash'] in seen_hashes:
                    # Found duplicate
                    original = seen_hashes[func['body_hash']]
                    
                    duplicate_alert = DuplicatePatternAlert(
                        pattern_type='function',
                        duplicate_locations=[
                            {'file_path': original['file_path'], 'function': original['name']},
                            {'file_path': func['file_path'], 'function': func['name']}
                        ],
                        similarity_score=1.0,  # Exact match
                        consolidation_suggestion=f'Functions "{original["name"]}" and "{func["name"]}" have identical implementations',
                        estimated_savings=f'{len(func["body"])} lines could be consolidated'
                    )
                    self._duplicate_patterns.append(duplicate_alert)
                    self._stats['duplicate_patterns_found'] += 1
                    
                    logger.info(f"Duplicate pattern detected: {original['name']} and {func['name']}")
                else:
                    seen_hashes[func['body_hash']] = func
            
            # Trim history
            if len(self._duplicate_patterns) > 50:
                self._duplicate_patterns = self._duplicate_patterns[-50:]
                
        except Exception as e:
            logger.error(f"Error detecting duplicate patterns: {e}")
    
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
            ],
            'pattern_deviations': [
                {
                    'file_path': p.file_path,
                    'type': p.deviation_type,
                    'severity': p.severity,
                    'suggestion': p.suggestion,
                    'confidence': p.confidence
                }
                for p in self._pattern_deviations[-3:]  # Last 3 pattern deviations
            ],
            'circular_dependencies': [
                {
                    'files': c.involved_files,
                    'risk_level': c.risk_level,
                    'suggestion': c.prevention_suggestion
                }
                for c in self._circular_dependency_alerts[-3:]  # Last 3 circular dependency alerts
            ],
            'file_splits': [
                {
                    'file_path': f.file_path,
                    'tokens': f.current_size_tokens,
                    'priority': f.priority,
                    'rationale': f.split_rationale
                }
                for f in self._file_split_suggestions[-3:]  # Last 3 file split suggestions
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
            'pattern_deviations': len(self._pattern_deviations),
            'circular_dependency_alerts': len(self._circular_dependency_alerts),
            'file_split_suggestions': len(self._file_split_suggestions),
            'duplicate_patterns': len(self._duplicate_patterns),
            # Priority 3: AI Session Intelligence metrics
            'current_session': self._current_session.session_id if self._current_session else None,
            'session_history_count': len(self._session_history),
            'change_impact_analyses': len(self._change_impact_analyses),
            'pattern_learning_data': len(self._pattern_learning_data),
            'multi_file_coordinations': len(self._multi_file_coordinations),
            'session_journal_entries': len(self._session_journal),
            'notification_callbacks': len(self._notification_callbacks),
            'learned_patterns': {
                'naming_functions': len(self._learned_patterns['naming_conventions'].get('functions', [])),
                'naming_classes': len(self._learned_patterns['naming_conventions'].get('classes', [])),
                'import_patterns': len(self._learned_patterns['import_patterns']),
                'class_structures': len(self._learned_patterns['class_structures']),
                'function_signatures': len(self._learned_patterns['function_signatures']),
                'workflow_patterns': len(self._learned_patterns['workflow_patterns']),
                'session_patterns': len(self._learned_patterns['session_patterns'])
            }
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
            ],
            'pattern_deviations': [
                {
                    'file_path': p.file_path,
                    'deviation_type': p.deviation_type,
                    'expected_pattern': p.expected_pattern,
                    'actual_pattern': p.actual_pattern,
                    'confidence': p.confidence,
                    'suggestion': p.suggestion,
                    'severity': p.severity,
                    'timestamp': p.timestamp
                }
                for p in self._pattern_deviations[-limit:]
            ],
            'circular_dependency_alerts': [
                {
                    'involved_files': c.involved_files,
                    'dependency_chain': c.dependency_chain,
                    'risk_level': c.risk_level,
                    'impact_assessment': c.impact_assessment,
                    'prevention_suggestion': c.prevention_suggestion,
                    'timestamp': c.timestamp
                }
                for c in self._circular_dependency_alerts[-limit:]
            ],
            'file_split_suggestions': [
                {
                    'file_path': f.file_path,
                    'current_size_tokens': f.current_size_tokens,
                    'suggested_splits': f.suggested_splits,
                    'split_rationale': f.split_rationale,
                    'estimated_improvement': f.estimated_improvement,
                    'priority': f.priority,
                    'timestamp': f.timestamp
                }
                for f in self._file_split_suggestions[-limit:]
            ],
            'duplicate_patterns': [
                {
                    'pattern_type': d.pattern_type,
                    'duplicate_locations': d.duplicate_locations,
                    'similarity_score': d.similarity_score,
                    'consolidation_suggestion': d.consolidation_suggestion,
                    'estimated_savings': d.estimated_savings,
                    'timestamp': d.timestamp
                }
                for d in self._duplicate_patterns[-limit:]
            ]
        }
    
    # Priority 3: AI Session Intelligence Methods
    
    def start_ai_session(self, session_name: str = "", session_description: str = "", session_tags: Optional[Set[str]] = None) -> str:
        """Start a new AI development session."""
        session_id = f"session_{int(time.time())}_{hashlib.md5(session_name.encode()).hexdigest()[:8]}"
        
        # End current session if one exists
        if self._current_session:
            self.end_ai_session()
        
        self._current_session = SessionContext(
            session_id=session_id,
            start_time=time.time(),
            session_name=session_name,
            session_description=session_description,
            session_tags=session_tags or set()
        )
        
        # Create journal entry for session start
        self._add_journal_entry(
            entry_type="session_start",
            entry_title=f"Started AI Session: {session_name}",
            entry_description=session_description,
            ai_context="Beginning new AI development session",
            outcome=f"Session {session_id} initialized"
        )
        
        self._stats['sessions_tracked'] += 1
        logger.info(f"Started AI session: {session_id} - {session_name}")
        return session_id
    
    def end_ai_session(self, achievements: Optional[List[str]] = None) -> Optional[SessionContext]:
        """End the current AI development session."""
        if not self._current_session:
            return None
        
        self._current_session.end_time = time.time()
        self._current_session.goals_achieved = achievements or []
        
        # Create journal entry for session end
        duration = self._current_session.end_time - self._current_session.start_time
        self._add_journal_entry(
            entry_type="session_end",
            entry_title=f"Ended AI Session: {self._current_session.session_name}",
            entry_description=f"Session completed in {duration:.1f} seconds",
            ai_context="Concluding AI development session",
            outcome=f"Session completed with {len(self._current_session.goals_achieved)} goals achieved",
            lessons_learned=[f"Modified {len(self._current_session.files_modified)} files"]
        )
        
        # Add to session history
        completed_session = self._current_session
        self._session_history.append(completed_session)
        self._current_session = None
        
        logger.info(f"Ended AI session: {completed_session.session_id}")
        return completed_session
    
    def get_session_context(self) -> Optional[SessionContext]:
        """Get current session context for continuity."""
        return self._current_session
    
    def restore_session_context(self, session_id: str) -> bool:
        """Restore a previous session context for continuity."""
        for session in self._session_history:
            if session.session_id == session_id:
                # Create a new session based on the historical one
                restored_session = SessionContext(
                    session_id=f"restored_{session_id}_{int(time.time())}",
                    start_time=time.time(),
                    session_name=f"Restored: {session.session_name}",
                    session_description=f"Restored from session {session_id}",
                    files_modified=session.files_modified.copy(),
                    patterns_learned=session.patterns_learned.copy(),
                    session_tags=session.session_tags.copy()
                )
                
                self._current_session = restored_session
                self._stats['session_context_restorations'] += 1
                
                self._add_journal_entry(
                    entry_type="session_restoration",
                    entry_title=f"Restored Session Context",
                    entry_description=f"Restored context from {session_id}",
                    ai_context="Resuming previous development context",
                    outcome=f"Session context restored successfully"
                )
                
                logger.info(f"Restored session context from {session_id}")
                return True
        
        logger.warning(f"Session {session_id} not found in history")
        return False
    
    async def analyze_change_impact(self, file_path: str, change_type: str, change_details: Optional[Dict[str, Any]] = None) -> ChangeImpactAnalysis:
        """Analyze the ripple effects of current modifications."""
        change_id = f"impact_{int(time.time())}_{hashlib.md5(file_path.encode()).hexdigest()[:8]}"
        
        ripple_effects = []
        dependency_impacts = []
        test_impacts = []
        documentation_impacts = []
        risk_assessment = 'low'
        impact_score = 0.0
        mitigation_suggestions = []
        
        try:
            # Analyze dependency impacts
            if self._dependency_graph and hasattr(self._dependency_graph, 'nodes'):
                # Find files that depend on the changed file
                for node_path, node_data in self._dependency_graph.nodes.items():
                    if hasattr(node_data, 'imports') and file_path in [str(imp) for imp in node_data.imports]:
                        dependency_impacts.append(node_path)
                        ripple_effects.append({
                            'affected_file': node_path,
                            'impact_type': 'dependency',
                            'description': f'Imports from {file_path}',
                            'severity': 'medium'
                        })
            
            # Identify test files that might be affected
            file_stem = Path(file_path).stem
            for test_pattern in [f"test_{file_stem}.py", f"{file_stem}_test.py", f"tests/{file_stem}.py"]:
                test_file_path = self.project_path / test_pattern
                if test_file_path.exists():
                    test_impacts.append(str(test_file_path))
                    ripple_effects.append({
                        'affected_file': str(test_file_path),
                        'impact_type': 'testing',
                        'description': f'Test file for {file_path}',
                        'severity': 'high'
                    })
            
            # Identify documentation that might need updates
            for doc_pattern in [f"docs/{file_stem}.md", f"README.md", f"CLAUDE.md"]:
                doc_file_path = self.project_path / doc_pattern
                if doc_file_path.exists():
                    documentation_impacts.append(str(doc_file_path))
                    ripple_effects.append({
                        'affected_file': str(doc_file_path),
                        'impact_type': 'documentation',
                        'description': f'Documentation for {file_path}',
                        'severity': 'low'
                    })
            
            # Calculate impact score and risk assessment
            num_deps = len(dependency_impacts)
            num_tests = len(test_impacts)
            impact_score = min(1.0, (num_deps * 0.3 + num_tests * 0.5) / 10)
            
            if impact_score > 0.7:
                risk_assessment = 'critical'
            elif impact_score > 0.5:
                risk_assessment = 'high'
            elif impact_score > 0.2:
                risk_assessment = 'medium'
            
            # Generate mitigation suggestions
            if dependency_impacts:
                mitigation_suggestions.append(f"Review {len(dependency_impacts)} dependent files for compatibility")
            if test_impacts:
                mitigation_suggestions.append(f"Update {len(test_impacts)} test files to match changes")
            if documentation_impacts:
                mitigation_suggestions.append(f"Update {len(documentation_impacts)} documentation files")
            if risk_assessment in ['high', 'critical']:
                mitigation_suggestions.append("Consider incremental deployment and thorough testing")
            
        except Exception as e:
            logger.error(f"Error during impact analysis: {e}")
            mitigation_suggestions.append("Manual review recommended due to analysis limitations")
        
        # Create impact analysis
        impact_analysis = ChangeImpactAnalysis(
            change_id=change_id,
            affected_file=file_path,
            change_type=change_type,
            ripple_effects=ripple_effects,
            dependency_impacts=dependency_impacts,
            test_impacts=test_impacts,
            documentation_impacts=documentation_impacts,
            risk_assessment=risk_assessment,
            impact_score=impact_score,
            mitigation_suggestions=mitigation_suggestions
        )
        
        self._change_impact_analyses.append(impact_analysis)
        self._stats['impact_analyses_performed'] += 1
        
        # Add to session context if active
        if self._current_session:
            self._current_session.changes_made.append({
                'file_path': file_path,
                'change_type': change_type,
                'impact_analysis_id': change_id,
                'timestamp': time.time()
            })
            self._current_session.files_modified.add(file_path)
        
        # Create journal entry
        self._add_journal_entry(
            entry_type="change",
            entry_title=f"Impact Analysis: {Path(file_path).name}",
            entry_description=f"{change_type} change with {risk_assessment} risk",
            affected_files=[file_path] + dependency_impacts + test_impacts,
            ai_context=f"Analyzing impact of {change_type} change",
            outcome=f"Impact score: {impact_score:.2f}, {len(ripple_effects)} effects identified"
        )
        
        logger.info(f"Completed impact analysis for {file_path}: {risk_assessment} risk, {impact_score:.2f} score")
        return impact_analysis
    
    def learn_pattern(self, pattern_type: str, pattern_data: Dict[str, Any], learned_from_files: List[str]) -> PatternLearningData:
        """Learn and store patterns from development sessions."""
        pattern_id = f"pattern_{pattern_type}_{int(time.time())}_{hashlib.md5(str(pattern_data).encode()).hexdigest()[:8]}"
        
        # Calculate project specificity based on how unique this pattern is
        project_specificity = 0.8  # Default to project-specific
        
        # Create pattern learning data
        pattern_learning = PatternLearningData(
            pattern_id=pattern_id,
            pattern_type=pattern_type,
            pattern_description=str(pattern_data),
            learned_from_files=learned_from_files,
            confidence_score=0.7,  # Initial confidence
            usage_frequency=1,
            pattern_examples=[pattern_data],
            project_specificity=project_specificity
        )
        
        self._pattern_learning_data.append(pattern_learning)
        self._stats['patterns_learned'] += 1
        
        # Update learned patterns storage
        if pattern_type not in self._learned_patterns:
            self._learned_patterns[pattern_type] = []
        self._learned_patterns[pattern_type].append(pattern_learning)
        
        # Add to session context if active
        if self._current_session:
            self._current_session.patterns_learned[pattern_id] = pattern_data
        
        # Create journal entry
        self._add_journal_entry(
            entry_type="pattern_learned",
            entry_title=f"Learned {pattern_type} Pattern",
            entry_description=f"Pattern learned from {len(learned_from_files)} files",
            affected_files=learned_from_files,
            ai_context=f"Learning {pattern_type} patterns from codebase",
            outcome=f"Pattern {pattern_id} added with {pattern_learning.confidence_score:.1%} confidence"
        )
        
        logger.info(f"Learned {pattern_type} pattern: {pattern_id}")
        return pattern_learning
    
    def start_multi_file_coordination(self, coordination_type: str, related_files: Set[str], context: str) -> str:
        """Track related changes across multiple files."""
        coordination_id = f"coord_{coordination_type}_{int(time.time())}_{hashlib.md5(context.encode()).hexdigest()[:8]}"
        
        coordination = MultiFileCoordination(
            coordination_id=coordination_id,
            related_files=related_files,
            coordination_type=coordination_type,
            change_sequence=[],
            dependencies_between_changes=[],
            completion_status={file_path: False for file_path in related_files},
            coordination_context=context,
            estimated_completion=time.time() + 3600  # Default 1 hour
        )
        
        self._multi_file_coordinations.append(coordination)
        self._stats['multi_file_coordinations_managed'] += 1
        
        # Create journal entry
        self._add_journal_entry(
            entry_type="coordination_start",
            entry_title=f"Started Multi-File Coordination: {coordination_type}",
            entry_description=context,
            affected_files=list(related_files),
            ai_context=f"Coordinating {coordination_type} across multiple files",
            outcome=f"Coordination {coordination_id} initialized for {len(related_files)} files"
        )
        
        logger.info(f"Started multi-file coordination: {coordination_id} ({coordination_type})")
        return coordination_id
    
    def update_file_coordination(self, coordination_id: str, file_path: str, change_details: Dict[str, Any]) -> bool:
        """Update progress on multi-file coordination."""
        for coordination in self._multi_file_coordinations:
            if coordination.coordination_id == coordination_id:
                # Add change to sequence
                change_entry = {
                    'file_path': file_path,
                    'change_details': change_details,
                    'timestamp': time.time(),
                    'change_id': f"change_{len(coordination.change_sequence)}"
                }
                coordination.change_sequence.append(change_entry)
                
                # Update completion status
                coordination.completion_status[file_path] = change_details.get('completed', False)
                
                logger.info(f"Updated coordination {coordination_id}: {file_path}")
                return True
        
        logger.warning(f"Coordination {coordination_id} not found")
        return False
    
    def _add_journal_entry(self, entry_type: str, entry_title: str, entry_description: str, 
                          affected_files: Optional[List[str]] = None, code_changes: Optional[Dict[str, Any]] = None,
                          ai_context: str = "", outcome: str = "", lessons_learned: Optional[List[str]] = None,
                          follow_up_actions: Optional[List[str]] = None, tags: Optional[Set[str]] = None) -> str:
        """Add an entry to the session journal."""
        entry_id = f"journal_{entry_type}_{int(time.time())}_{hashlib.md5(entry_title.encode()).hexdigest()[:8]}"
        session_id = self._current_session.session_id if self._current_session else "no_session"
        
        journal_entry = SessionJournalEntry(
            entry_id=entry_id,
            session_id=session_id,
            entry_type=entry_type,
            entry_title=entry_title,
            entry_description=entry_description,
            affected_files=affected_files or [],
            code_changes=code_changes or {},
            ai_context=ai_context,
            outcome=outcome,
            lessons_learned=lessons_learned or [],
            follow_up_actions=follow_up_actions or [],
            tags=tags or set()
        )
        
        self._session_journal.append(journal_entry)
        self._stats['journal_entries_created'] += 1
        
        return entry_id
    
    def get_session_intelligence(self, limit: int = 50) -> Dict[str, Any]:
        """Get comprehensive AI session intelligence data."""
        current_session_data = None
        if self._current_session:
            current_session_data = {
                'session_id': self._current_session.session_id,
                'session_name': self._current_session.session_name,
                'start_time': self._current_session.start_time,
                'duration': time.time() - self._current_session.start_time,
                'files_modified': list(self._current_session.files_modified),
                'changes_made': len(self._current_session.changes_made),
                'patterns_learned': len(self._current_session.patterns_learned),
                'goals_achieved': self._current_session.goals_achieved,
                'ai_interactions': self._current_session.ai_interactions
            }
        
        return {
            'current_session': current_session_data,
            'session_history': [
                {
                    'session_id': s.session_id,
                    'session_name': s.session_name,
                    'start_time': s.start_time,
                    'end_time': s.end_time,
                    'duration': (s.end_time - s.start_time) if s.end_time else None,
                    'files_modified': len(s.files_modified),
                    'goals_achieved': len(s.goals_achieved)
                }
                for s in self._session_history[-10:]  # Last 10 sessions
            ],
            'impact_analyses': [
                {
                    'change_id': i.change_id,
                    'affected_file': i.affected_file,
                    'change_type': i.change_type,
                    'risk_assessment': i.risk_assessment,
                    'impact_score': i.impact_score,
                    'ripple_effects_count': len(i.ripple_effects),
                    'timestamp': i.timestamp
                }
                for i in self._change_impact_analyses[-limit:]
            ],
            'learned_patterns': [
                {
                    'pattern_id': p.pattern_id,
                    'pattern_type': p.pattern_type,
                    'confidence_score': p.confidence_score,
                    'usage_frequency': p.usage_frequency,
                    'project_specificity': p.project_specificity,
                    'files_learned_from': len(p.learned_from_files),
                    'learning_date': p.learning_date
                }
                for p in self._pattern_learning_data[-limit:]
            ],
            'multi_file_coordinations': [
                {
                    'coordination_id': c.coordination_id,
                    'coordination_type': c.coordination_type,
                    'related_files_count': len(c.related_files),
                    'changes_made': len(c.change_sequence),
                    'completion_rate': sum(c.completion_status.values()) / len(c.completion_status) if c.completion_status else 0,
                    'priority': c.priority,
                    'timestamp': c.timestamp
                }
                for c in self._multi_file_coordinations[-limit:]
            ],
            'journal_entries': [
                {
                    'entry_id': j.entry_id,
                    'entry_type': j.entry_type,
                    'entry_title': j.entry_title,
                    'affected_files_count': len(j.affected_files),
                    'ai_context': j.ai_context,
                    'outcome': j.outcome,
                    'timestamp': j.timestamp
                }
                for j in self._session_journal[-limit:]
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