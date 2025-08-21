#!/usr/bin/env python3
"""
Setup script for Deepflow
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
requirements_path = this_directory / "requirements.txt"
if requirements_path.exists():
    requirements = requirements_path.read_text().strip().split('\n')

setup(
    name="deepflow",
    version="2.1.0",
    author="Deepflow Team",
    author_email="team@deepflow.dev",
    description="Keep AI-assisted codebases clean, consistent, and maintainable. Specialized tools for Claude Code, Cursor, GitHub Copilot users.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/scs03004/deepflow",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Quality Assurance",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0", 
            "black>=23.0.0",
            "mypy>=1.0.0",
            "flake8>=6.0.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.18.0",
        ],
        "mcp": [
            "mcp>=1.2.0",
        ],
        "all": [
            "pytest>=7.0.0", "pytest-cov>=4.0.0", "black>=23.0.0", "mypy>=1.0.0", 
            "flake8>=6.0.0", "pre-commit>=3.0.0", "sphinx>=5.0.0", 
            "sphinx-rtd-theme>=1.0.0", "myst-parser>=0.18.0", "mcp>=1.2.0"
        ],
    },
    entry_points={
        "console_scripts": [
            "deepflow-visualizer=tools.dependency_visualizer:main",
            "deepflow-validator=tools.pre_commit_validator:main",
            "deepflow-docs=tools.doc_generator:main",
            "deepflow-ci=tools.ci_cd_integrator:main",
            "deepflow-monitor=tools.monitoring_dashboard:main",
            "deepflow-analyzer=tools.code_analyzer:main",
            "ai-session-tracker=tools.ai_session_tracker:main",
            "deepflow-mcp-server=deepflow.mcp.server:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["templates/*.md", "templates/*.html", "*.yml", "*.yaml"],
    },
    keywords="ai-development codebase-hygiene pattern-consistency context-window architecture-drift claude-code cursor copilot",
    project_urls={
        "Bug Reports": "https://github.com/scs03004/deepflow/issues",
        "Source": "https://github.com/scs03004/deepflow",
        "Documentation": "https://github.com/scs03004/deepflow/blob/main/README.md",
    },
)