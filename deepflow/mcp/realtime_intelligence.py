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
        
        # Project pattern learning
        self._learned_patterns = {
            'naming_conventions': {},
            'import_patterns': {},
            'class_structures': {},
            'function_signatures': {}
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
            'duplicate_patterns_found': 0
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
            'notification_callbacks': len(self._notification_callbacks),
            'learned_patterns': {
                'naming_functions': len(self._learned_patterns['naming_conventions'].get('functions', [])),
                'naming_classes': len(self._learned_patterns['naming_conventions'].get('classes', [])),
                'import_patterns': len(self._learned_patterns['import_patterns']),
                'class_structures': len(self._learned_patterns['class_structures']),
                'function_signatures': len(self._learned_patterns['function_signatures'])
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