#!/usr/bin/env python3
"""
Design Document Lifecycle Management Utility

This script helps manage the lifecycle of design documents by:
- Moving documents between lifecycle folders
- Updating status fields in document headers
- Validating document structure
- Generating lifecycle reports
"""

import os
import re
import shutil
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

class DesignDocManager:
    def __init__(self, docs_path: str = "."):
        self.docs_path = Path(docs_path)
        self.folders = {
            "draft": self.docs_path,
            "in_progress": self.docs_path / "in-progress",
            "completed": self.docs_path / "completed", 
            "cancelled": self.docs_path / "cancelled",
            "archived": self.docs_path / "archived"
        }
        
    def list_documents(self, status: Optional[str] = None) -> Dict[str, List[str]]:
        """List all design documents by status."""
        docs = {}
        
        if status and status in self.folders:
            folder = self.folders[status]
            docs[status] = [f.name for f in folder.glob("*.md") if f.name != "README.md" and f.name != "_TEMPLATE.md"]
        else:
            for status_name, folder in self.folders.items():
                if folder.exists():
                    docs[status_name] = [f.name for f in folder.glob("*.md") if f.name != "README.md" and f.name != "_TEMPLATE.md"]
        
        return docs
    
    def move_document(self, filename: str, from_status: str, to_status: str, reason: str = "") -> bool:
        """Move a document from one status folder to another."""
        if from_status not in self.folders or to_status not in self.folders:
            print(f"❌ Invalid status. Valid statuses: {list(self.folders.keys())}")
            return False
            
        from_path = self.folders[from_status] / filename
        to_path = self.folders[to_status] / filename
        
        if not from_path.exists():
            print(f"❌ Document '{filename}' not found in {from_status}")
            return False
            
        # Ensure target directory exists
        self.folders[to_status].mkdir(parents=True, exist_ok=True)
        
        # Update document status
        self._update_document_status(from_path, to_status, reason)
        
        # Move the file
        shutil.move(str(from_path), str(to_path))
        print(f"✅ Moved '{filename}' from {from_status} to {to_status}")
        return True
    
    def _update_document_status(self, doc_path: Path, new_status: str, reason: str = ""):
        """Update the status field in a design document."""
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Status mapping
            status_map = {
                "draft": "Draft",
                "in_progress": "In Review", 
                "completed": "Implemented",
                "cancelled": "Cancelled",
                "archived": "Archived"
            }
            
            # Update status line
            status_pattern = r'(\*\*Status:\*\*\s+)([^|\n]+)'
            new_status_text = status_map.get(new_status, new_status.title())
            content = re.sub(status_pattern, f'\\1{new_status_text}', content)
            
            # Add timestamp to post-implementation if moving to completed
            if new_status == "completed":
                impl_date = datetime.now().strftime("%Y-%m-%d")
                content = re.sub(
                    r'\*\*Implementation Completed:\*\*\s+\[YYYY-MM-DD\]',
                    f'**Implementation Completed:** {impl_date}',
                    content
                )
                content = re.sub(
                    r'\*\*Final Status:\*\*\s+\[.*?\]',
                    '**Final Status:** Fully Implemented',
                    content
                )
            
            # Add cancellation reason if moving to cancelled
            if new_status == "cancelled" and reason:
                content = re.sub(
                    r'\*\*Final Status:\*\*\s+\[.*?\]',
                    f'**Final Status:** Cancelled - {reason}',
                    content
                )
            
            with open(doc_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
        except Exception as e:
            print(f"⚠️ Warning: Could not update status in document: {e}")
    
    def generate_report(self) -> str:
        """Generate a status report of all design documents."""
        docs = self.list_documents()
        report = ["# Design Documents Status Report", ""]
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        total_docs = sum(len(doc_list) for doc_list in docs.values())
        report.append(f"**Total Documents:** {total_docs}")
        report.append("")
        
        for status, doc_list in docs.items():
            if doc_list:
                report.append(f"## {status.title()} ({len(doc_list)})")
                for doc in sorted(doc_list):
                    report.append(f"- {doc}")
                report.append("")
        
        return "\n".join(report)
    
    def validate_document(self, filename: str, status: str = "draft") -> List[str]:
        """Validate a design document structure."""
        doc_path = self.folders[status] / filename
        if not doc_path.exists():
            return [f"Document '{filename}' not found in {status}"]
        
        issues = []
        try:
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            required_sections = [
                "Problem Statement",
                "Proposed Solution", 
                "Implementation Plan",
                "Testing Strategy",
                "Risk Assessment"
            ]
            
            for section in required_sections:
                if f"## {section}" not in content and f"### {section}" not in content:
                    issues.append(f"Missing section: {section}")
            
            # Check for status field
            if "**Status:**" not in content:
                issues.append("Missing Status field in header")
            
            # Check for complexity field  
            if "**Complexity:**" not in content:
                issues.append("Missing Complexity field in header")
                
        except Exception as e:
            issues.append(f"Error reading document: {e}")
        
        return issues

def main():
    parser = argparse.ArgumentParser(description="Manage design document lifecycle")
    parser.add_argument("--docs-path", default=".", help="Path to design docs directory")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List documents by status")
    list_parser.add_argument("--status", choices=["draft", "in_progress", "completed", "cancelled", "archived"], help="Filter by status")
    
    # Move command
    move_parser = subparsers.add_parser("move", help="Move document between statuses")
    move_parser.add_argument("filename", help="Document filename")
    move_parser.add_argument("from_status", choices=["draft", "in_progress", "completed", "cancelled", "archived"])
    move_parser.add_argument("to_status", choices=["draft", "in_progress", "completed", "cancelled", "archived"])
    move_parser.add_argument("--reason", help="Reason for the move (especially for cancellations)")
    
    # Report command
    report_parser = subparsers.add_parser("report", help="Generate status report")
    report_parser.add_argument("--output", help="Output file for report")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate document structure")
    validate_parser.add_argument("filename", help="Document filename")
    validate_parser.add_argument("--status", default="draft", help="Status folder to check")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = DesignDocManager(args.docs_path)
    
    if args.command == "list":
        docs = manager.list_documents(args.status)
        for status, doc_list in docs.items():
            if doc_list:
                print(f"\n📁 {status.title()} ({len(doc_list)}):")
                for doc in sorted(doc_list):
                    print(f"  • {doc}")
    
    elif args.command == "move":
        reason = args.reason or ""
        success = manager.move_document(args.filename, args.from_status, args.to_status, reason)
        if success:
            print(f"📋 Remember to commit this change: git add . && git commit -m \"Move {args.filename} to {args.to_status}\"")
    
    elif args.command == "report":
        report = manager.generate_report()
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"📊 Report saved to {args.output}")
        else:
            print(report)
    
    elif args.command == "validate":
        issues = manager.validate_document(args.filename, args.status)
        if issues:
            print(f"⚠️ Validation issues for {args.filename}:")
            for issue in issues:
                print(f"  • {issue}")
        else:
            print(f"✅ {args.filename} passes validation")

if __name__ == "__main__":
    main()