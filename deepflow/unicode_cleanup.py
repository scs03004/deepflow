#!/usr/bin/env python3
"""
Unicode Cleanup Tool for Deepflow
===================================

This module provides AI-aware Unicode character cleanup functionality for codebases.
Designed specifically for fixing issues introduced by AI code generation that includes
emoji and special Unicode characters that cause encoding errors on different systems.

Features:
- Comprehensive Unicode detection and classification
- AI-coding-aware replacement mappings
- Configurable cleanup strategies
- Integration with Deepflow MCP protocol
- Performance optimized for large codebases
"""

import os
import re
import sys
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

# Comprehensive Unicode replacement mappings optimized for AI-generated code
UNICODE_REPLACEMENTS = {
    # AI Status indicators (very common in AI-generated code)
    '✅': '[PASS]',
    '❌': '[FAIL]', 
    '⚠️': '[WARN]',
    '🎯': '[TARGET]',
    '🚀': '[LAUNCH]',
    '⭐': '[STAR]',
    '✨': '[SPARKLE]',
    '🔥': '[FIRE]',
    '💯': '[100]',
    
    # Checkmarks and symbols (common in tests and logging)
    '✓': '[OK]',
    '✗': '[ERROR]',
    '→': '->',
    '←': '<-',
    '↓': 'v',
    '↑': '^',
    '↔': '<->',
    
    # AI-generated emoji categories (common in documentation/comments)
    '🤖': '[BOT]',
    '👤': '[USER]',
    '📊': '[STATS]',
    '⚙️': '[CONFIG]',
    '🎲': '[DICE]',
    '⚔️': '[COMBAT]',
    '🏰': '[CASTLE]',
    '👥': '[PEOPLE]',
    '🎮': '[GAME]',
    '📝': '[NOTE]',
    '💾': '[SAVE]',
    '🔍': '[SEARCH]',
    '🔧': '[TOOL]',
    '📋': '[LIST]',
    '📈': '[CHART]',
    '💡': '[IDEA]',
    '🎭': '[THEATER]',
    '🗺️': '[MAP]',
    '🚨': '[ALERT]',
    '🎨': '[ART]',
    '📚': '[BOOKS]',
    '🎪': '[CIRCUS]',
    '🌐': '[WEB]',
    
    # Additional emojis commonly found in AI-generated code
    '🌍': '[GLOBE]',
    '🎉': '[PARTY]',
    '🔄': '[RELOAD]',
    '🔗': '[LINK]',
    '⚡': '[LIGHTNING]',
    '💬': '[CHAT]',
    '📁': '[FOLDER]',
    '📍': '[PIN]',
    '📖': '[BOOK]',
    '📱': '[PHONE]',
    '🏛️': '[BUILDING]',
    '🎬': '[MOVIE]',
    '👋': '[WAVE]',
    '👦': '[BOY]',
    '👧': '[GIRL]',
    '👨': '[MAN]',
    '👩': '[WOMAN]',
    '💀': '[SKULL]',
    '🤝': '[HANDSHAKE]',
    '🧠': '[BRAIN]',
    '🗡️': '[SWORD]',
    '🗣️': '[SPEAKING]',
    '🗺️': '[MAP]',
    '🕹️': '[JOYSTICK]',
    '🕊️': '[DOVE]',
    
    # Warning and alert symbols (common in AI debugging code)
    '⚠': '[WARN]',
    '⚔': '[SWORDS]',
    '⚙': '[GEAR]',
    'ℹ️': '[INFO]',
    '': '',               # U+200D zero-width joiner - remove
    '️': '',               # U+FE0F variation selector - remove
    
    # Mathematical/Technical (common in AI-generated formulas)
    '±': '+/-',
    '≥': '>=',
    '≤': '<=',
    '≠': '!=',
    '×': 'x',
    '÷': '/',
    '∞': 'infinity',
    '°': 'deg',
    
    # Quotation marks (common encoding issues)
    '"': '"',
    '"': '"',
    ''': "'",
    ''': "'",
    
    # Dashes and spaces (encoding artifacts)
    '–': '--',
    '—': '--',
    ' ': ' ',  # Non-breaking space
    '	': '    ',  # Tab character replacement
    
    # Other common Unicode (in AI-generated content)
    '©': '(c)',
    '®': '(r)',
    '™': '(tm)',
    '…': '...',
    '•': '*',
    '◦': 'o',
    '‰': 'per-mil',
    '§': 'section',
    
    # Accented characters (found in international text)
    'á': 'a',
    'í': 'i',
    'é': 'e',
    'ó': 'o',
    'ú': 'u',
    'ñ': 'n',
}

@dataclass
class UnicodeIssue:
    """Represents a Unicode issue found in a file."""
    file_path: Path
    line_number: int
    line_content: str
    unicode_chars: Set[str]
    severity: str  # 'high', 'medium', 'low'
    category: str  # 'emoji', 'symbol', 'accented', 'technical', 'unknown'

@dataclass
class CleanupResults:
    """Results from Unicode cleanup operation."""
    files_processed: int
    files_changed: int
    total_replacements: int
    issues_found: List[UnicodeIssue]
    unmapped_chars: Set[str]
    cleanup_successful: bool
    error_message: Optional[str] = None

class AIAwareUnicodeCleanup:
    """
    AI-aware Unicode cleanup utility optimized for deepflow integration.
    
    This class provides comprehensive Unicode character detection and cleanup
    specifically designed for AI-generated code that often contains emoji
    and special Unicode characters.
    """
    
    def __init__(self, root_dir: Path = None, exclude_patterns: List[str] = None):
        """
        Initialize the Unicode cleanup utility.
        
        Args:
            root_dir: Root directory to process (default: current directory)
            exclude_patterns: List of patterns to exclude (default: common exclusions)
        """
        self.root_dir = Path(root_dir) if root_dir else Path.cwd()
        self.replacements = UNICODE_REPLACEMENTS
        self.excluded_dirs = {'.venv', '__pycache__', '.git', '.idea', 'node_modules', '.pytest_cache'}
        self.excluded_files = {'README.md', 'CHANGELOG.md', 'LICENSE.md'}  # Keep Unicode in docs
        self.exclude_patterns = exclude_patterns or []
        
        # Performance tracking
        self.stats = {
            'files_scanned': 0,
            'unicode_chars_found': 0,
            'replacements_made': 0,
            'files_modified': 0
        }
    
    def find_python_files(self) -> List[Path]:
        """Find all Python files in the project, respecting exclusion patterns."""
        python_files = []
        
        for root, dirs, files in os.walk(self.root_dir):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in self.excluded_dirs]
            
            for file in files:
                if file.endswith('.py') and file not in self.excluded_files:
                    file_path = Path(root) / file
                    
                    # Check exclude patterns
                    if self._should_exclude_file(file_path):
                        continue
                        
                    python_files.append(file_path)
        
        return python_files
    
    def _should_exclude_file(self, file_path: Path) -> bool:
        """Check if file should be excluded based on patterns."""
        file_str = str(file_path)
        for pattern in self.exclude_patterns:
            if pattern in file_str:
                return True
        return False
    
    def find_unicode_in_file(self, file_path: Path) -> List[UnicodeIssue]:
        """Find Unicode characters in a file and classify them."""
        unicode_issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    # Find non-ASCII characters
                    unicode_chars = set()
                    for char in line:
                        if ord(char) > 127:  # Non-ASCII
                            unicode_chars.add(char)
                    
                    if unicode_chars:
                        severity = self._classify_severity(unicode_chars)
                        category = self._classify_category(unicode_chars)
                        
                        issue = UnicodeIssue(
                            file_path=file_path,
                            line_number=line_num,
                            line_content=line.rstrip(),
                            unicode_chars=unicode_chars,
                            severity=severity,
                            category=category
                        )
                        unicode_issues.append(issue)
                        
                        self.stats['unicode_chars_found'] += len(unicode_chars)
        
        except UnicodeDecodeError as e:
            logger.error(f"Error reading {file_path}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error reading {file_path}: {e}")
            
        return unicode_issues
    
    def _classify_severity(self, unicode_chars: Set[str]) -> str:
        """Classify the severity of Unicode issues."""
        high_risk_chars = {'✅', '❌', '⚠️', '🚀', '🔥'}  # Common in AI code
        
        if any(char in high_risk_chars for char in unicode_chars):
            return 'high'
        elif any(ord(char) > 0x1F000 for char in unicode_chars):  # Emoji range
            return 'medium'
        else:
            return 'low'
    
    def _classify_category(self, unicode_chars: Set[str]) -> str:
        """Classify Unicode characters by category."""
        emoji_ranges = [
            (0x1F600, 0x1F64F),  # Emoticons
            (0x1F300, 0x1F5FF),  # Symbols & Pictographs
            (0x1F680, 0x1F6FF),  # Transport & Map
            (0x1F1E0, 0x1F1FF),  # Flags
        ]
        
        for char in unicode_chars:
            char_code = ord(char)
            if any(start <= char_code <= end for start, end in emoji_ranges):
                return 'emoji'
            elif char_code in range(0x2000, 0x206F):  # General Punctuation
                return 'symbol'
            elif char_code in range(0x00C0, 0x017F):  # Latin Extended
                return 'accented'
            elif char_code in range(0x2200, 0x22FF):  # Mathematical Operators
                return 'technical'
        
        return 'unknown'
    
    def scan_for_unicode(self) -> Dict[Path, List[UnicodeIssue]]:
        """Scan all Python files for Unicode characters."""
        logger.info(f"Scanning for Unicode characters in {self.root_dir}")
        
        results = {}
        python_files = self.find_python_files()
        
        for file_path in python_files:
            self.stats['files_scanned'] += 1
            unicode_issues = self.find_unicode_in_file(file_path)
            if unicode_issues:
                results[file_path] = unicode_issues
        
        logger.info(f"Scanned {len(python_files)} files, found Unicode in {len(results)} files")
        return results
    
    def clean_file(self, file_path: Path, dry_run: bool = True) -> Dict[str, Any]:
        """Clean Unicode characters from a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            changes_made = False
            replacements_count = 0
            
            # Apply replacements
            for unicode_char, replacement in self.replacements.items():
                if unicode_char in content:
                    content = content.replace(unicode_char, replacement)
                    changes_made = True
                    replacements_count += content.count(replacement) - original_content.count(replacement)
            
            # Check for remaining Unicode
            remaining_unicode = set()
            for char in content:
                if ord(char) > 127:
                    remaining_unicode.add(char)
            
            # Write file if changes were made and not dry run
            if changes_made and not dry_run:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.stats['files_modified'] += 1
                self.stats['replacements_made'] += replacements_count
            
            return {
                'file_path': file_path,
                'changes_made': changes_made,
                'replacements_count': replacements_count,
                'remaining_unicode': remaining_unicode,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            return {
                'file_path': file_path,
                'changes_made': False,
                'replacements_count': 0,
                'remaining_unicode': set(),
                'success': False,
                'error': str(e)
            }
    
    def clean_all_files(self, dry_run: bool = True) -> CleanupResults:
        """Clean Unicode characters from all Python files."""
        logger.info(f"{'[DRY-RUN] ' if dry_run else ''}Starting Unicode cleanup")
        
        python_files = self.find_python_files()
        files_changed = 0
        total_replacements = 0
        issues_found = []
        unmapped_chars = set()
        
        for file_path in python_files:
            result = self.clean_file(file_path, dry_run)
            
            if result['changes_made']:
                files_changed += 1
                total_replacements += result['replacements_count']
            
            if result['remaining_unicode']:
                unmapped_chars.update(result['remaining_unicode'])
                
                # Create issues for remaining Unicode
                try:
                    unicode_issues = self.find_unicode_in_file(file_path)
                    issues_found.extend(unicode_issues)
                except Exception as e:
                    logger.error(f"Error finding Unicode issues in {file_path}: {e}")
        
        cleanup_results = CleanupResults(
            files_processed=len(python_files),
            files_changed=files_changed,
            total_replacements=total_replacements,
            issues_found=issues_found,
            unmapped_chars=unmapped_chars,
            cleanup_successful=len(unmapped_chars) == 0
        )
        
        logger.info(f"Unicode cleanup completed: {files_changed} files changed, {total_replacements} replacements")
        return cleanup_results
    
    def generate_report(self, results: CleanupResults, format: str = 'text') -> str:
        """Generate a comprehensive cleanup report."""
        if format == 'json':
            return self._generate_json_report(results)
        elif format == 'html':
            return self._generate_html_report(results)
        else:
            return self._generate_text_report(results)
    
    def _generate_text_report(self, results: CleanupResults) -> str:
        """Generate a text-based cleanup report."""
        report = []
        report.append("Unicode Cleanup Report")
        report.append("=" * 50)
        report.append(f"Files processed: {results.files_processed}")
        report.append(f"Files changed: {results.files_changed}")
        report.append(f"Total replacements: {results.total_replacements}")
        report.append(f"Cleanup successful: {'Yes' if results.cleanup_successful else 'No'}")
        
        if results.unmapped_chars:
            report.append("\nUnmapped Unicode characters found:")
            for char in sorted(results.unmapped_chars):
                report.append(f"  U+{ord(char):04X} ({char})")
        
        if results.issues_found:
            report.append(f"\nDetailed issues ({len(results.issues_found)}):")
            for issue in results.issues_found[:10]:  # Show first 10
                rel_path = issue.file_path.relative_to(self.root_dir)
                report.append(f"  {rel_path}:{issue.line_number} [{issue.severity}] {issue.category}")
        
        return "\n".join(report)
    
    def _generate_json_report(self, results: CleanupResults) -> str:
        """Generate a JSON-based cleanup report."""
        report_data = {
            'summary': {
                'files_processed': results.files_processed,
                'files_changed': results.files_changed,
                'total_replacements': results.total_replacements,
                'cleanup_successful': results.cleanup_successful
            },
            'unmapped_chars': [
                {'char': char, 'unicode_point': f"U+{ord(char):04X}"}
                for char in sorted(results.unmapped_chars)
            ],
            'issues': [
                {
                    'file': str(issue.file_path.relative_to(self.root_dir)),
                    'line': issue.line_number,
                    'severity': issue.severity,
                    'category': issue.category,
                    'chars': [f"U+{ord(char):04X}" for char in issue.unicode_chars]
                }
                for issue in results.issues_found
            ],
            'statistics': self.stats
        }
        return json.dumps(report_data, indent=2)
    
    def _generate_html_report(self, results: CleanupResults) -> str:
        """Generate an HTML-based cleanup report."""
        html = f"""
        <html>
        <head>
            <title>Unicode Cleanup Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .summary {{ background: #f5f5f5; padding: 15px; border-radius: 5px; }}
                .success {{ color: green; }}
                .warning {{ color: orange; }}
                .error {{ color: red; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Unicode Cleanup Report</h1>
            <div class="summary">
                <h2>Summary</h2>
                <p>Files processed: <strong>{results.files_processed}</strong></p>
                <p>Files changed: <strong>{results.files_changed}</strong></p>
                <p>Total replacements: <strong>{results.total_replacements}</strong></p>
                <p class="{'success' if results.cleanup_successful else 'error'}">
                    Status: <strong>{'Complete' if results.cleanup_successful else 'Issues Remaining'}</strong>
                </p>
            </div>
        """
        
        if results.unmapped_chars:
            html += "<h2>Unmapped Characters</h2><ul>"
            for char in sorted(results.unmapped_chars):
                html += f"<li>U+{ord(char):04X} ({char})</li>"
            html += "</ul>"
        
        html += "</body></html>"
        return html

# Additional utility functions for deepflow integration

def quick_unicode_scan(project_path: str = ".") -> Dict[str, Any]:
    """Quick Unicode scan for MCP integration."""
    cleanup = AIAwareUnicodeCleanup(Path(project_path))
    results = cleanup.scan_for_unicode()
    
    return {
        'total_files_with_unicode': len(results),
        'total_files_scanned': cleanup.stats['files_scanned'],
        'unicode_chars_found': cleanup.stats['unicode_chars_found'],
        'has_issues': len(results) > 0,
        'severity_breakdown': _get_severity_breakdown(results)
    }

def _get_severity_breakdown(results: Dict[Path, List[UnicodeIssue]]) -> Dict[str, int]:
    """Get breakdown of issues by severity."""
    breakdown = {'high': 0, 'medium': 0, 'low': 0}
    for issues in results.values():
        for issue in issues:
            breakdown[issue.severity] += 1
    return breakdown

def apply_unicode_cleanup(project_path: str = ".", dry_run: bool = True) -> Dict[str, Any]:
    """Apply Unicode cleanup and return results for MCP integration."""
    cleanup = AIAwareUnicodeCleanup(Path(project_path))
    results = cleanup.clean_all_files(dry_run=dry_run)
    
    return {
        'files_processed': results.files_processed,
        'files_changed': results.files_changed,
        'total_replacements': results.total_replacements,
        'cleanup_successful': results.cleanup_successful,
        'unmapped_chars': list(results.unmapped_chars),
        'error_message': results.error_message,
        'report': cleanup.generate_report(results, 'text')
    }