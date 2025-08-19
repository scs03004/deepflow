#!/usr/bin/env python3
"""
AI Session Tracker
==================

Track and analyze AI development sessions to maintain context and consistency
across multiple AI interactions. Helps prevent session fragmentation and 
architectural drift during AI-assisted development.

Usage:
    python ai_session_tracker.py start "feature-name" [--description "desc"]
    python ai_session_tracker.py end [--generate-report]
    python ai_session_tracker.py status
    python ai_session_tracker.py list [--recent N]
    python ai_session_tracker.py analyze /path/to/project
"""

import json
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import hashlib
import sys

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    import git
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Install with: pip install rich GitPython")
    sys.exit(1)


@dataclass
class FileChange:
    """Represents a file change during an AI session."""
    
    file_path: str
    change_type: str  # 'added', 'modified', 'deleted'
    lines_added: int
    lines_removed: int
    tokens_before: int
    tokens_after: int
    timestamp: str


@dataclass
class AISession:
    """Represents an AI development session."""
    
    session_id: str
    name: str
    description: str
    start_time: str
    end_time: Optional[str]
    project_path: str
    initial_commit: Optional[str]
    final_commit: Optional[str]
    files_changed: List[FileChange]
    total_tokens_added: int
    total_tokens_removed: int
    pattern_consistency_scores: Dict[str, float]
    architecture_violations: List[str]
    ai_suggestions_followed: List[str]
    session_notes: List[str]


class AISessionTracker:
    """Core AI session tracking engine."""
    
    def __init__(self, project_path: str = None):
        self.project_path = Path(project_path or ".").resolve()
        self.sessions_dir = self.project_path / ".ai-sessions"
        self.sessions_dir.mkdir(exist_ok=True)
        self.console = Console()
        self.current_session_file = self.sessions_dir / "current_session.json"
        
        # Initialize git repo if available
        try:
            self.repo = git.Repo(self.project_path)
        except (git.InvalidGitRepositoryError, git.GitCommandError):
            self.repo = None
    
    def start_session(self, name: str, description: str = "") -> str:
        """Start a new AI development session."""
        if self.current_session_file.exists():
            current = self.get_current_session()
            if current:
                self.console.print(f"[yellow]Warning:[/yellow] Session '{current.name}' is already active")
                self.console.print("End current session before starting a new one")
                return current.session_id
        
        # Generate unique session ID
        timestamp = datetime.datetime.now()
        session_id = hashlib.md5(f"{name}{timestamp}".encode()).hexdigest()[:8]
        
        # Get initial commit hash if git is available
        initial_commit = None
        if self.repo:
            try:
                initial_commit = self.repo.head.commit.hexsha
            except:
                pass
        
        session = AISession(
            session_id=session_id,
            name=name,
            description=description,
            start_time=timestamp.isoformat(),
            end_time=None,
            project_path=str(self.project_path),
            initial_commit=initial_commit,
            final_commit=None,
            files_changed=[],
            total_tokens_added=0,
            total_tokens_removed=0,
            pattern_consistency_scores={},
            architecture_violations=[],
            ai_suggestions_followed=[],
            session_notes=[]
        )
        
        # Save current session
        with open(self.current_session_file, 'w') as f:
            json.dump(asdict(session), f, indent=2)
        
        self.console.print(f"[green]Started AI session:[/green] {name} ({session_id})")
        self.console.print(f"[blue]Project:[/blue] {self.project_path}")
        if initial_commit:
            self.console.print(f"[blue]Initial commit:[/blue] {initial_commit[:8]}")
        
        return session_id
    
    def end_session(self, generate_report: bool = False) -> Optional[AISession]:
        """End the current AI development session."""
        if not self.current_session_file.exists():
            self.console.print("[red]No active session found[/red]")
            return None
        
        current = self.get_current_session()
        if not current:
            return None
        
        # Update session with end time and final state
        current.end_time = datetime.datetime.now().isoformat()
        
        # Get final commit hash if git is available
        if self.repo:
            try:
                current.final_commit = self.repo.head.commit.hexsha
            except:
                pass
        
        # Analyze changes made during session
        current.files_changed = self._analyze_session_changes(current)
        current.total_tokens_added = sum(fc.tokens_after - fc.tokens_before for fc in current.files_changed if fc.tokens_after > fc.tokens_before)
        current.total_tokens_removed = sum(fc.tokens_before - fc.tokens_after for fc in current.files_changed if fc.tokens_before > fc.tokens_after)
        
        # Save completed session to history
        session_file = self.sessions_dir / f"session_{current.session_id}.json"
        with open(session_file, 'w') as f:
            json.dump(asdict(current), f, indent=2)
        
        # Remove current session file
        self.current_session_file.unlink()
        
        self.console.print(f"[green]Ended AI session:[/green] {current.name}")
        self.console.print(f"[blue]Duration:[/blue] {self._format_duration(current.start_time, current.end_time)}")
        self.console.print(f"[blue]Files changed:[/blue] {len(current.files_changed)}")
        self.console.print(f"[blue]Tokens added:[/blue] {current.total_tokens_added:,}")
        
        if generate_report:
            self._generate_session_report(current)
        
        return current
    
    def get_current_session(self) -> Optional[AISession]:
        """Get the current active session."""
        if not self.current_session_file.exists():
            return None
        
        try:
            with open(self.current_session_file, 'r') as f:
                data = json.load(f)
                return AISession(**data)
        except:
            return None
    
    def list_sessions(self, recent: int = 10) -> List[AISession]:
        """List recent AI sessions."""
        sessions = []
        session_files = sorted(
            self.sessions_dir.glob("session_*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        for session_file in session_files[:recent]:
            try:
                with open(session_file, 'r') as f:
                    data = json.load(f)
                    # Handle FileChange objects
                    if 'files_changed' in data:
                        data['files_changed'] = [
                            FileChange(**fc) if isinstance(fc, dict) else fc 
                            for fc in data['files_changed']
                        ]
                    sessions.append(AISession(**data))
            except:
                continue
        
        return sessions
    
    def analyze_sessions(self) -> Dict:
        """Analyze patterns across all sessions."""
        sessions = self.list_sessions(recent=50)  # Analyze last 50 sessions
        
        if not sessions:
            return {"error": "No sessions found"}
        
        analysis = {
            "total_sessions": len(sessions),
            "avg_files_per_session": sum(len(s.files_changed) for s in sessions) / len(sessions),
            "avg_tokens_per_session": sum(s.total_tokens_added for s in sessions) / len(sessions),
            "most_changed_files": self._get_most_changed_files(sessions),
            "session_frequency": self._analyze_session_frequency(sessions),
            "architecture_drift_risk": self._assess_architecture_drift_risk(sessions)
        }
        
        return analysis
    
    def _analyze_session_changes(self, session: AISession) -> List[FileChange]:
        """Analyze file changes made during the session."""
        changes = []
        
        if not self.repo or not session.initial_commit:
            # Fallback: analyze all Python files for recent changes
            cutoff_time = datetime.datetime.fromisoformat(session.start_time)
            
            for py_file in self.project_path.rglob("*.py"):
                if self._should_skip_file(py_file):
                    continue
                    
                try:
                    stat = py_file.stat()
                    mod_time = datetime.datetime.fromtimestamp(stat.st_mtime)
                    
                    if mod_time > cutoff_time:
                        # File was modified during session
                        with open(py_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        tokens = self._estimate_tokens(content)
                        changes.append(FileChange(
                            file_path=str(py_file.relative_to(self.project_path)),
                            change_type="modified",
                            lines_added=0,  # Can't determine without git
                            lines_removed=0,
                            tokens_before=tokens,  # Approximate
                            tokens_after=tokens,
                            timestamp=mod_time.isoformat()
                        ))
                except:
                    continue
        else:
            # Use git to get actual changes
            try:
                # Get diff between initial commit and current state
                diff = self.repo.git.diff(session.initial_commit, name_only=True)
                changed_files = diff.strip().split('\n') if diff.strip() else []
                
                for file_path in changed_files:
                    if not file_path.endswith('.py'):
                        continue
                    
                    try:
                        # Get detailed diff for this file
                        file_diff = self.repo.git.diff(session.initial_commit, '--', file_path, numstat=True)
                        if file_diff:
                            parts = file_diff.strip().split('\t')
                            if len(parts) >= 2:
                                lines_added = int(parts[0]) if parts[0].isdigit() else 0
                                lines_removed = int(parts[1]) if parts[1].isdigit() else 0
                            else:
                                lines_added = lines_removed = 0
                        else:
                            lines_added = lines_removed = 0
                        
                        # Estimate tokens before/after
                        full_path = self.project_path / file_path
                        if full_path.exists():
                            with open(full_path, 'r', encoding='utf-8') as f:
                                current_content = f.read()
                            tokens_after = self._estimate_tokens(current_content)
                        else:
                            tokens_after = 0
                        
                        # Rough estimation of tokens before
                        tokens_before = max(0, tokens_after - (lines_added * 4) + (lines_removed * 4))
                        
                        changes.append(FileChange(
                            file_path=file_path,
                            change_type="modified",
                            lines_added=lines_added,
                            lines_removed=lines_removed,
                            tokens_before=tokens_before,
                            tokens_after=tokens_after,
                            timestamp=datetime.datetime.now().isoformat()
                        ))
                    except:
                        continue
            except:
                pass
        
        return changes
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped in analysis."""
        skip_patterns = {
            "__pycache__", ".git", ".pytest_cache", "node_modules", 
            "venv", ".venv", "env", ".env", "build", "dist", ".ai-sessions"
        }
        return any(part in skip_patterns for part in file_path.parts)
    
    def _estimate_tokens(self, content: str) -> int:
        """Estimate token count for content."""
        return len(content) // 4  # Rough estimation
    
    def _get_most_changed_files(self, sessions: List[AISession]) -> List[Tuple[str, int]]:
        """Get files that change most frequently across sessions."""
        file_counts = {}
        
        for session in sessions:
            for change in session.files_changed:
                file_counts[change.file_path] = file_counts.get(change.file_path, 0) + 1
        
        return sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    def _analyze_session_frequency(self, sessions: List[AISession]) -> Dict:
        """Analyze frequency of AI development sessions."""
        if not sessions:
            return {}
        
        # Group sessions by date
        dates = []
        for session in sessions:
            date = datetime.datetime.fromisoformat(session.start_time).date()
            dates.append(date)
        
        from collections import Counter
        date_counts = Counter(dates)
        
        return {
            "sessions_per_day_avg": len(sessions) / max(1, (max(dates) - min(dates)).days + 1),
            "most_active_days": date_counts.most_common(5),
            "total_days_active": len(date_counts)
        }
    
    def _assess_architecture_drift_risk(self, sessions: List[AISession]) -> str:
        """Assess risk of architecture drift based on session patterns."""
        if not sessions:
            return "LOW"
        
        # Calculate risk factors
        total_changes = sum(len(s.files_changed) for s in sessions)
        avg_changes_per_session = total_changes / len(sessions)
        
        # Check for sessions with many file changes (fragmentation risk)
        large_sessions = sum(1 for s in sessions if len(s.files_changed) > 10)
        large_session_ratio = large_sessions / len(sessions)
        
        if avg_changes_per_session > 15 or large_session_ratio > 0.3:
            return "HIGH"
        elif avg_changes_per_session > 8 or large_session_ratio > 0.15:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _format_duration(self, start_time: str, end_time: str) -> str:
        """Format session duration."""
        try:
            start = datetime.datetime.fromisoformat(start_time)
            end = datetime.datetime.fromisoformat(end_time)
            duration = end - start
            
            hours = duration.seconds // 3600
            minutes = (duration.seconds % 3600) // 60
            
            if hours > 0:
                return f"{hours}h {minutes}m"
            else:
                return f"{minutes}m"
        except:
            return "Unknown"
    
    def _generate_session_report(self, session: AISession):
        """Generate detailed session report."""
        report_path = self.sessions_dir / f"report_{session.session_id}.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# AI Session Report: {session.name}\n\n")
            f.write(f"**Session ID:** {session.session_id}\n")
            f.write(f"**Description:** {session.description}\n")
            f.write(f"**Duration:** {self._format_duration(session.start_time, session.end_time or session.start_time)}\n")
            f.write(f"**Project:** {session.project_path}\n\n")
            
            if session.initial_commit and session.final_commit:
                f.write(f"**Git Range:** {session.initial_commit[:8]}..{session.final_commit[:8]}\n\n")
            
            f.write(f"## Summary\n\n")
            f.write(f"- Files changed: {len(session.files_changed)}\n")
            f.write(f"- Tokens added: {session.total_tokens_added:,}\n")
            f.write(f"- Tokens removed: {session.total_tokens_removed:,}\n")
            f.write(f"- Net tokens: {session.total_tokens_added - session.total_tokens_removed:,}\n\n")
            
            if session.files_changed:
                f.write(f"## Files Changed\n\n")
                for change in session.files_changed:
                    f.write(f"- **{change.file_path}** ({change.change_type})\n")
                    f.write(f"  - Lines: +{change.lines_added}/-{change.lines_removed}\n")
                    f.write(f"  - Tokens: {change.tokens_before} -> {change.tokens_after}\n")
                f.write("\n")
            
            if session.session_notes:
                f.write(f"## Session Notes\n\n")
                for note in session.session_notes:
                    f.write(f"- {note}\n")
                f.write("\n")
        
        self.console.print(f"[green]Session report generated:[/green] {report_path}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Track AI development sessions")
    parser.add_argument("command", choices=["start", "end", "status", "list", "analyze"], help="Command to execute")
    parser.add_argument("name", nargs="?", help="Session name (for start command)")
    parser.add_argument("--description", help="Session description")
    parser.add_argument("--generate-report", action="store_true", help="Generate detailed report when ending session")
    parser.add_argument("--recent", type=int, default=10, help="Number of recent sessions to show")
    parser.add_argument("--project", help="Project path (default: current directory)")
    
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = AISessionTracker(args.project)
    console = Console()
    
    try:
        if args.command == "start":
            if not args.name:
                console.print("[red]Error:[/red] Session name is required for start command")
                sys.exit(1)
            
            tracker.start_session(args.name, args.description or "")
        
        elif args.command == "end":
            session = tracker.end_session(args.generate_report)
            if not session:
                sys.exit(1)
        
        elif args.command == "status":
            current = tracker.get_current_session()
            
            if current:
                console.print(Panel(
                    f"[bold]{current.name}[/bold]\n"
                    f"ID: {current.session_id}\n"
                    f"Started: {current.start_time}\n"
                    f"Description: {current.description or 'None'}",
                    title="Active AI Session"
                ))
            else:
                console.print("[yellow]No active AI session[/yellow]")
        
        elif args.command == "list":
            sessions = tracker.list_sessions(args.recent)
            
            if not sessions:
                console.print("[yellow]No sessions found[/yellow]")
                return
            
            table = Table(title=f"Recent AI Sessions (last {len(sessions)})")
            table.add_column("ID")
            table.add_column("Name")
            table.add_column("Started")
            table.add_column("Duration")
            table.add_column("Files")
            table.add_column("Tokens")
            
            for session in sessions:
                duration = tracker._format_duration(session.start_time, session.end_time or session.start_time)
                table.add_row(
                    session.session_id[:8],
                    session.name,
                    session.start_time.split('T')[0],  # Just date
                    duration,
                    str(len(session.files_changed)),
                    f"{session.total_tokens_added:,}"
                )
            
            console.print(table)
        
        elif args.command == "analyze":
            analysis = tracker.analyze_sessions()
            
            if "error" in analysis:
                console.print(f"[red]{analysis['error']}[/red]")
                return
            
            console.print(Panel(
                f"Total Sessions: {analysis['total_sessions']}\n"
                f"Avg Files/Session: {analysis['avg_files_per_session']:.1f}\n"
                f"Avg Tokens/Session: {analysis['avg_tokens_per_session']:,.0f}\n"
                f"Architecture Drift Risk: {analysis['architecture_drift_risk']}",
                title="AI Development Analysis"
            ))
            
            if analysis['most_changed_files']:
                console.print("\n[bold]Most Frequently Changed Files:[/bold]")
                for file_path, count in analysis['most_changed_files'][:5]:
                    console.print(f"  {file_path}: {count} sessions")
    
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()