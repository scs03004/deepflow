#!/usr/bin/env python3
"""
Demo Script for Priority 4: Smart Refactoring & Code Quality Features

This script demonstrates all Priority 4 capabilities including:
- Pattern Standardization
- Import Optimization
- Automated File Splitting
- Dead Code Removal
- Documentation Generation
- Comprehensive Refactoring

Usage:
    python demo_priority4_features.py
    python demo_priority4_features.py --apply-changes  # Actually apply refactoring
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Add project to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from deepflow.smart_refactoring_engine import SmartRefactoringEngine
    from deepflow.mcp.server import DeepflowMCPServer
    SMART_REFACTORING_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: Smart Refactoring Engine not available: {e}")
    sys.exit(1)


def create_demo_project():
    """Create a demo project with various refactoring opportunities."""
    print("🏗️  Creating demo project with refactoring opportunities...")
    
    temp_dir = tempfile.mkdtemp(prefix="priority4_demo_")
    project_path = Path(temp_dir)
    
    # Create main.py with mixed patterns and issues
    (project_path / "main.py").write_text("""
import os
import sys
import json
import unused_module
import json  # Duplicate import
from collections import defaultdict

def camelCaseFunction(param1, param2):
    return param1 + param2

def snake_case_function(arg_one, arg_two):
    if arg_one:
        result = process_data(arg_one)
        return result
    return None

class UserManager:
    def __init__(self):
        self.users = []
        self.cache = {}
    
    def addUser(self, user):  # Mixed naming
        self.users.append(user)
    
    def get_user_count(self):
        return len(self.users)

def unused_helper_function():
    '''This function is never called and should be removed.'''
    return "unused"

def process_data(data):
    '''Process data with some logic.'''
    if isinstance(data, str):
        return data.upper()
    return str(data)

# Unused variables
unused_var = "this is never used"
another_unused = {"key": "value"}
""")
    
    # Create utils.py with import issues
    (project_path / "utils.py").write_text("""
import re
import time
import datetime
import unused_utility_import
from typing import List, Dict, Optional

class UtilityHelper:
    def format_string(self, text):
        return text.strip().lower()

class ConfigManager:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = {}
    
    def load_config(self):
        pass
    
    def save_config(self):
        pass

def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def unused_validation_function():
    return True

# More unused code
UNUSED_CONSTANT = "never referenced"
""")
    
    # Create large_file.py that needs splitting
    large_content = '''
"""Large file that should be split into smaller modules."""

import json
import sys
import os
from typing import Any, Dict, List

# This file has too many classes and functions and should be split
'''
    
    # Add many classes
    for i in range(15):
        large_content += f'''

class Component{i}:
    """Component {i} for demonstration."""
    
    def __init__(self, name: str = "component_{i}"):
        self.name = name
        self.id = {i}
        self.active = True
    
    def process(self) -> str:
        return f"Processing {{self.name}} with ID {{self.id}}"
    
    def activate(self):
        self.active = True
    
    def deactivate(self):
        self.active = False
    
    def get_status(self) -> Dict[str, Any]:
        return {{
            "name": self.name,
            "id": self.id,
            "active": self.active
        }}
'''
    
    # Add many standalone functions
    for i in range(20):
        large_content += f'''

def utility_function_{i}(data):
    """Utility function {i}."""
    if data:
        return f"processed_{{data}}_{i}"
    return None
'''
    
    (project_path / "large_file.py").write_text(large_content)
    
    # Create models.py with missing documentation
    (project_path / "models.py").write_text("""
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class User:
    name: str
    email: str
    age: Optional[int] = None
    
    def validate(self):
        return "@" in self.email

class DatabaseManager:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.connection = None
    
    def connect(self):
        pass
    
    def execute_query(self, query, params=None):
        pass
    
    def close(self):
        pass

def create_user(name, email, age=None):
    return User(name=name, email=email, age=age)

def validate_user_data(user_data):
    required_fields = ["name", "email"]
    return all(field in user_data for field in required_fields)
""")
    
    print(f"✅ Demo project created at: {project_path}")
    print("   Files created:")
    for file in project_path.glob("*.py"):
        print(f"   - {file.name}")
    
    return str(project_path)


def print_header(title: str, emoji: str = "🔧"):
    """Print a formatted header."""
    print(f"\n{emoji} {title}")
    print("=" * (len(title) + 4))


def print_analysis_results(title: str, results: dict, emoji: str = "📊"):
    """Print analysis results in a formatted way."""
    print(f"\n{emoji} {title}")
    print("-" * (len(title) + 4))
    print(json.dumps(results, indent=2))


async def demo_mcp_integration(project_path: str, apply_changes: bool = False):
    """Demonstrate MCP integration for Priority 4 features."""
    print_header("MCP Integration Demo", "🔌")
    
    try:
        # Create MCP server instance
        server = DeepflowMCPServer()
        print("✅ MCP Server created successfully")
        
        # Test each Priority 4 MCP tool
        tools_to_test = [
            ("standardize_patterns", {"project_path": project_path, "apply_changes": apply_changes}),
            ("optimize_imports", {"project_path": project_path, "apply_changes": apply_changes}),
            ("suggest_file_splits", {"project_path": project_path}),
            ("remove_dead_code", {"project_path": project_path, "apply_changes": apply_changes}),
            ("generate_docstrings", {"project_path": project_path, "apply_changes": apply_changes}),
            ("comprehensive_refactor", {"project_path": project_path, "apply_changes": apply_changes})
        ]
        
        for tool_name, arguments in tools_to_test:
            print(f"\n🔧 Testing MCP tool: {tool_name}")
            try:
                # Get the handler method
                handler_name = f"_handle_{tool_name}"
                if hasattr(server, handler_name):
                    handler = getattr(server, handler_name)
                    start_time = time.time()
                    result = await handler(arguments)
                    end_time = time.time()
                    
                    if result and len(result) > 0:
                        # Parse the result content
                        result_text = result[0].text
                        try:
                            result_json = json.loads(result_text)
                            print(f"   ✅ Completed in {end_time - start_time:.2f}s")
                            
                            # Show key metrics
                            if tool_name == "standardize_patterns":
                                print(f"   📈 Pattern consistency: {result_json.get('consistency_score', 0):.2%}")
                                print(f"   🔍 Violations found: {len(result_json.get('violations', []))}")
                            elif tool_name == "optimize_imports":
                                print(f"   🗑️  Unused imports: {len(result_json.get('unused_imports', []))}")
                                print(f"   🔄 Duplicates: {len(result_json.get('duplicate_imports', []))}")
                            elif tool_name == "suggest_file_splits":
                                print(f"   📂 Files needing splits: {result_json.get('files_needing_splits', 0)}")
                            elif tool_name == "remove_dead_code":
                                print(f"   🧹 Total removals: {result_json.get('total_removals', 0)}")
                            elif tool_name == "generate_docstrings":
                                print(f"   📝 Functions documented: {result_json.get('functions_documented', 0)}")
                            elif tool_name == "comprehensive_refactor":
                                safety_score = result_json.get('safety_score', 0)
                                print(f"   🛡️  Safety score: {safety_score:.2%}")
                                plan = result_json.get('refactoring_plan', [])
                                print(f"   📋 Refactoring tasks: {len(plan)}")
                        except json.JSONDecodeError:
                            print(f"   ✅ Tool executed (non-JSON response)")
                            print(f"   📝 Response: {result_text[:100]}...")
                    else:
                        print("   ❌ No result returned")
                else:
                    print(f"   ❌ Handler {handler_name} not found")
                    
            except Exception as e:
                print(f"   ❌ Error testing {tool_name}: {e}")
        
        # Show server performance stats
        stats = server.get_performance_stats()
        print(f"\n📊 Server Performance Stats:")
        for key, value in stats.items():
            print(f"   {key}: {value}")
            
    except Exception as e:
        print(f"❌ MCP integration demo failed: {e}")


def demo_direct_engine(project_path: str, apply_changes: bool = False):
    """Demonstrate direct Smart Refactoring Engine usage."""
    print_header("Direct Engine Demo", "⚙️")
    
    engine = SmartRefactoringEngine(project_path)
    
    # 1. Pattern Standardization
    print("\n1️⃣ Pattern Standardization Analysis")
    pattern_analysis = engine.standardize_patterns()
    print(f"   📊 Consistency Score: {pattern_analysis.consistency_score:.2%}")
    print(f"   🚨 Violations Found: {len(pattern_analysis.violations)}")
    for violation in pattern_analysis.violations[:3]:  # Show first 3
        print(f"   - {violation.get('type', 'unknown')}: {violation.get('description', 'No description')}")
    
    # 2. Import Optimization
    print("\n2️⃣ Import Optimization Analysis")
    import_analysis = engine.optimize_imports()
    print(f"   🗑️  Unused Imports: {len(import_analysis.unused_imports)}")
    print(f"   🔄 Duplicate Imports: {len(import_analysis.duplicate_imports)}")
    print(f"   ⭕ Circular Imports: {len(import_analysis.circular_imports)}")
    
    # Show some examples
    if import_analysis.unused_imports:
        print("   Examples of unused imports:")
        for unused in import_analysis.unused_imports[:3]:
            print(f"   - {unused}")
    
    # 3. File Split Suggestions
    print("\n3️⃣ File Split Analysis")
    split_analyses = engine.suggest_file_splits()
    files_needing_splits = [a for a in split_analyses if a.split_recommendations]
    print(f"   📂 Files Analyzed: {len(split_analyses)}")
    print(f"   ✂️  Files Needing Splits: {len(files_needing_splits)}")
    
    for analysis in files_needing_splits:
        print(f"   - {Path(analysis.file_path).name}: Size={analysis.size_score:.2f}, Complexity={analysis.complexity_score:.2f}")
    
    # 4. Dead Code Detection
    print("\n4️⃣ Dead Code Detection")
    dead_code_analysis = engine.detect_dead_code()
    print(f"   🧟 Unused Functions: {len(dead_code_analysis.unused_functions)}")
    print(f"   🏗️  Unused Classes: {len(dead_code_analysis.unused_classes)}")
    print(f"   📦 Unused Variables: {len(dead_code_analysis.unused_variables)}")
    
    # Show examples
    if dead_code_analysis.unused_functions:
        print("   Examples of unused functions:")
        for func in dead_code_analysis.unused_functions[:3]:
            function_name = func.split(":")[-1]
            print(f"   - {function_name}")
    
    # 5. Documentation Generation
    print("\n5️⃣ Documentation Generation")
    doc_analysis = engine.generate_documentation()
    print(f"   📝 Missing Docstrings: {len(doc_analysis.missing_docstrings)}")
    print(f"   ✍️  Generated Docstrings: {len(doc_analysis.generated_docstrings)}")
    
    # Show examples of missing documentation
    if doc_analysis.missing_docstrings:
        print("   Functions/Classes needing documentation:")
        for item in doc_analysis.missing_docstrings[:3]:
            print(f"   - {item['type']} {item['name']} in {Path(item['file']).name}")
    
    # 6. Comprehensive Analysis Summary
    print("\n6️⃣ Comprehensive Analysis Summary")
    total_issues = (
        len(pattern_analysis.violations) +
        len(import_analysis.unused_imports) +
        len(import_analysis.duplicate_imports) +
        len(files_needing_splits) +
        len(dead_code_analysis.unused_functions) +
        len(dead_code_analysis.unused_classes) +
        len(dead_code_analysis.unused_variables) +
        len(doc_analysis.missing_docstrings)
    )
    
    print(f"   📈 Total Issues Found: {total_issues}")
    print(f"   🎯 Pattern Consistency: {pattern_analysis.consistency_score:.1%}")
    print(f"   📊 Refactoring Priority: {'High' if total_issues > 20 else 'Medium' if total_issues > 10 else 'Low'}")
    
    # 7. Apply Changes if requested
    if apply_changes:
        print("\n7️⃣ Applying Refactoring Changes")
        all_analyses = {
            "pattern_analysis": pattern_analysis,
            "import_analysis": import_analysis,
            "file_split_analysis": split_analyses,
            "dead_code_analysis": dead_code_analysis,
            "documentation_analysis": doc_analysis
        }
        
        refactor_results = engine.apply_refactoring(all_analyses, dry_run=False)
        print("   Changes Applied:")
        for key, count in refactor_results.items():
            if key != 'errors' and count > 0:
                print(f"   - {key.replace('_', ' ').title()}: {count}")
        
        if refactor_results.get('errors'):
            print(f"   ❌ Errors: {len(refactor_results['errors'])}")
    else:
        print("\n7️⃣ Dry Run Mode (Use --apply-changes to actually refactor)")


def demo_performance_benchmarks(project_path: str):
    """Demonstrate performance characteristics of Priority 4 features."""
    print_header("Performance Benchmarks", "🚀")
    
    engine = SmartRefactoringEngine(project_path)
    
    benchmarks = [
        ("Pattern Standardization", lambda: engine.standardize_patterns()),
        ("Import Optimization", lambda: engine.optimize_imports()),
        ("File Split Analysis", lambda: engine.suggest_file_splits()),
        ("Dead Code Detection", lambda: engine.detect_dead_code()),
        ("Documentation Generation", lambda: engine.generate_documentation())
    ]
    
    for name, func in benchmarks:
        start_time = time.time()
        try:
            result = func()
            end_time = time.time()
            duration = end_time - start_time
            print(f"   ⚡ {name}: {duration:.3f}s")
        except Exception as e:
            print(f"   ❌ {name}: Failed ({e})")
    
    # Overall benchmark
    start_time = time.time()
    try:
        # Run all analyses
        engine.standardize_patterns()
        engine.optimize_imports()
        engine.suggest_file_splits()
        engine.detect_dead_code()
        engine.generate_documentation()
        end_time = time.time()
        total_duration = end_time - start_time
        print(f"\n   🏁 Total Analysis Time: {total_duration:.3f}s")
        print(f"   📊 Average per Analysis: {total_duration / 5:.3f}s")
    except Exception as e:
        print(f"   ❌ Comprehensive benchmark failed: {e}")


async def main():
    """Main demo function."""
    parser = argparse.ArgumentParser(description="Demo Priority 4: Smart Refactoring Features")
    parser.add_argument("--apply-changes", action="store_true", 
                       help="Actually apply refactoring changes (default: dry run)")
    parser.add_argument("--skip-mcp", action="store_true",
                       help="Skip MCP integration demo")
    parser.add_argument("--project-path", type=str,
                       help="Use existing project path instead of creating demo project")
    
    args = parser.parse_args()
    
    print("🎯 Priority 4: Smart Refactoring & Code Quality Demo")
    print("=" * 55)
    
    if args.apply_changes:
        print("⚠️  WARNING: --apply-changes is enabled. Files will be modified!")
        print("   Press Ctrl+C within 3 seconds to cancel...")
        try:
            time.sleep(3)
        except KeyboardInterrupt:
            print("\n❌ Cancelled by user")
            return
    else:
        print("ℹ️  Running in dry-run mode (no files will be modified)")
    
    # Create or use project
    if args.project_path and Path(args.project_path).exists():
        project_path = args.project_path
        print(f"📁 Using existing project: {project_path}")
    else:
        project_path = create_demo_project()
    
    try:
        # Demo 1: Direct Engine Usage
        demo_direct_engine(project_path, args.apply_changes)
        
        # Demo 2: MCP Integration (if not skipped)
        if not args.skip_mcp:
            await demo_mcp_integration(project_path, args.apply_changes)
        else:
            print("\n🔌 MCP Integration Demo (Skipped)")
            print("   Use without --skip-mcp to see MCP features")
        
        # Demo 3: Performance Benchmarks
        demo_performance_benchmarks(project_path)
        
        print("\n✅ Priority 4 Demo Completed Successfully!")
        print(f"📁 Demo project location: {project_path}")
        
        if not args.apply_changes:
            print("\n💡 Tips:")
            print("   - Run with --apply-changes to see actual refactoring")
            print("   - Use --project-path to test on your own project")
            print("   - Inspect the demo project files to see the issues detected")
            
    except KeyboardInterrupt:
        print("\n❌ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup demo project if we created it
        if not args.project_path:
            try:
                import shutil
                shutil.rmtree(project_path)
                print(f"🧹 Cleaned up demo project: {project_path}")
            except:
                print(f"⚠️  Could not clean up demo project: {project_path}")


if __name__ == "__main__":
    asyncio.run(main())