#!/usr/bin/env python3
"""
Demo: Requirements Management for AI Coding Workflows
=====================================================

This demo shows how deepflow can automatically detect missing packages
from imports and update requirements.txt - perfect for AI-assisted development!

This addresses the common pain point where AI coding tools generate code
with imports but forget to update requirements.txt.
"""

import tempfile
import os
from pathlib import Path
import sys

# Add deepflow to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from deepflow.smart_refactoring_engine import SmartRefactoringEngine
    print("[OK] SmartRefactoringEngine imported successfully")
except ImportError as e:
    print(f"[ERROR] Import failed: {e}")
    sys.exit(1)

def create_sample_ai_project():
    """Create a sample project that AI might generate - with missing requirements."""
    temp_dir = Path(tempfile.mkdtemp())
    print(f"📁 Created sample project at: {temp_dir}")
    
    # Create a typical AI-generated project structure
    
    # main.py - AI often generates this with multiple imports
    (temp_dir / "main.py").write_text("""
#!/usr/bin/env python3
# AI-generated main application

import os
import sys
import json
import requests  # Missing from requirements.txt
from flask import Flask, jsonify  # Missing from requirements.txt
import pandas as pd  # Missing from requirements.txt
from sklearn.model_selection import train_test_split  # Missing from requirements.txt

app = Flask(__name__)

@app.route('/api/data')
def get_data():
    # AI generated some data processing
    df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
    return jsonify(df.to_dict())

@app.route('/api/predict')
def predict():
    # AI added ML functionality
    response = requests.get('https://api.example.com/data')
    return jsonify({'prediction': 'AI result'})

if __name__ == '__main__':
    app.run(debug=True)
""")
    
    # utils.py - AI often creates utility modules
    (temp_dir / "utils.py").write_text("""
# AI-generated utilities

import datetime  # Built-in, shouldn't be in requirements
import numpy as np  # Missing from requirements.txt
import yaml  # Missing from requirements.txt (maps to pyyaml)
from rich.console import Console  # Missing from requirements.txt
import asyncio  # Built-in

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def print_status(message):
    console = Console()
    console.print(f"[green]{message}[/green]")

async def process_data(data):
    # AI loves async functions
    await asyncio.sleep(0.1)
    return np.array(data).mean()
""")
    
    # analysis.py - AI might generate data analysis code
    (temp_dir / "analysis.py").write_text("""
# AI-generated analysis module

import matplotlib.pyplot as plt  # Missing from requirements.txt
import seaborn as sns  # Missing from requirements.txt
import plotly.express as px  # Missing from requirements.txt
from sklearn.ensemble import RandomForestRegressor  # Missing (sklearn)
import tensorflow as tf  # Missing from requirements.txt

def create_plots(data):
    # AI loves generating visualization code
    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr())
    plt.show()
    
    fig = px.scatter(data, x='x', y='y')
    fig.show()

def train_model(X, y):
    # AI often adds ML without checking requirements
    model = RandomForestRegressor()
    model.fit(X, y)
    return model

def deep_learning_model(input_shape):
    # AI loves adding TensorFlow
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.Dense(1)
    ])
    return model
""")
    
    # Create an incomplete requirements.txt (typical AI oversight)
    (temp_dir / "requirements.txt").write_text("""# Incomplete requirements.txt
# AI forgot to add many packages!

fastapi==0.104.1
uvicorn==0.24.0
""")
    
    return temp_dir

def demo_requirements_analysis():
    """Demonstrate the requirements analysis feature."""
    print("\n🚀 DEMO: AI Coding Requirements Management")
    print("=" * 60)
    
    # Create sample project
    project_path = create_sample_ai_project()
    
    print(f"\n📝 Created sample AI-generated project with intentionally incomplete requirements.txt")
    print(f"Project contains Python files with imports that aren't in requirements.txt")
    
    # Initialize the requirements analyzer
    engine = SmartRefactoringEngine(str(project_path))
    
    print("\n🔍 ANALYZING REQUIREMENTS...")
    print("-" * 40)
    
    # Analyze requirements
    analysis = engine.analyze_requirements()
    
    print(f"\n📊 ANALYSIS RESULTS:")
    print(f"• Files analyzed: {len(analysis.detected_imports)} Python files")
    print(f"• Current requirements: {len(analysis.current_requirements)}")
    print(f"• Missing packages detected: {len(analysis.missing_packages)}")
    print(f"• Unused packages: {len(analysis.unused_packages)}")
    
    # Show current requirements
    print(f"\n📋 CURRENT REQUIREMENTS.TXT:")
    for req in analysis.current_requirements:
        print(f"  • {req}")
    
    # Show missing packages
    print(f"\n🔍 MISSING PACKAGES DETECTED:")
    high_confidence_count = 0
    for pkg in analysis.missing_packages:
        if not pkg['is_standard_library']:
            confidence_icon = "🟢" if pkg['confidence'] >= 0.9 else "🟡" if pkg['confidence'] >= 0.7 else "🔴"
            print(f"  {confidence_icon} {pkg['package_name']} (from import: {pkg['import_name']}, confidence: {pkg['confidence']:.1%})")
            print(f"    Used in: {', '.join(pkg['files_using'])}")
            if pkg['confidence'] >= 0.9:
                high_confidence_count += 1
    
    print(f"\n🎯 HIGH-CONFIDENCE RECOMMENDATIONS:")
    print(f"  • {high_confidence_count} packages should definitely be added")
    
    # Show recommendations
    print(f"\n💡 SMART RECOMMENDATIONS:")
    for rec in analysis.update_recommendations:
        priority_icon = "🚨" if rec['priority'] == 'high' else "⚠️" if rec['priority'] == 'medium' else "ℹ️"
        print(f"  {priority_icon} [{rec['priority'].upper()}] {rec['action']}")
        for pkg in rec['packages']:
            print(f"    - {pkg}")
        print(f"    Rationale: {rec['rationale']}")
    
    # Demonstrate dry run update
    print(f"\n🔄 DRY RUN - REQUIREMENTS UPDATE PREVIEW:")
    print("-" * 50)
    
    update_results = engine.update_requirements_file(analysis, dry_run=True)
    
    print(f"📦 PROPOSED CHANGES:")
    print(f"  • Packages to add: {update_results['packages_added']}")
    print(f"  • Packages to remove: {update_results['packages_removed']}")
    print(f"  • Original count: {update_results['original_count']}")
    print(f"  • New count: {update_results['new_count']}")
    
    print(f"\n📝 UPDATED REQUIREMENTS.TXT WOULD CONTAIN:")
    for req in sorted(update_results['new_requirements']):
        if req not in analysis.current_requirements:
            print(f"  + {req}  # 🆕 Added by deepflow")
        else:
            print(f"    {req}")
    
    print(f"\n✅ DEMO COMPLETE!")
    print(f"🎯 This shows how deepflow can automatically detect missing packages")
    print(f"   from AI-generated code and update requirements.txt intelligently!")
    
    # Cleanup
    import shutil
    shutil.rmtree(project_path)
    print(f"\n🧹 Cleaned up demo project")

def demo_mcp_integration():
    """Show how this integrates with MCP for Claude Code."""
    print(f"\n🔌 MCP INTEGRATION FOR CLAUDE CODE:")
    print("=" * 50)
    
    print("📋 NEW MCP TOOLS AVAILABLE:")
    print("  • analyze_requirements - Detect missing packages from imports")
    print("  • update_requirements - Update requirements.txt automatically")
    
    print(f"\n💻 CLAUDE CODE USAGE EXAMPLES:")
    print("1. Analyze current project:")
    print("   Call: analyze_requirements")
    print("   Args: {'project_path': '.', 'check_installed': true}")
    
    print("\n2. Update requirements.txt (dry run):")
    print("   Call: update_requirements") 
    print("   Args: {'project_path': '.', 'dry_run': true}")
    
    print("\n3. Actually update requirements.txt:")
    print("   Call: update_requirements")
    print("   Args: {'project_path': '.', 'apply_changes': true}")
    
    print(f"\n🎯 PERFECT FOR AI WORKFLOWS:")
    print("  ✅ Automatically detects imports in AI-generated code")
    print("  ✅ Maps imports to correct PyPI package names")
    print("  ✅ Handles 130+ common packages (Flask, pandas, sklearn, etc.)")
    print("  ✅ Confidence scoring for reliability")
    print("  ✅ Conservative unused package detection")
    print("  ✅ Safe dry-run mode by default")
    print("  ✅ Creates backups before changes")

if __name__ == "__main__":
    print("🎯 DEEPFLOW REQUIREMENTS MANAGEMENT DEMO")
    print("Solving the AI coding workflow pain point!")
    print("=" * 60)
    
    try:
        demo_requirements_analysis()
        demo_mcp_integration()
        
        print(f"\n🎉 SUCCESS! Requirements management feature is ready!")
        print(f"   This solves the common AI coding problem of missing requirements.txt updates.")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()