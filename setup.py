#!/usr/bin/env python3
"""
Setup script for Dependency Toolkit
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
    name="dependency-toolkit",
    version="1.0.0",
    author="Dependency Toolkit Team",
    author_email="team@dependency-toolkit.dev",
    description="A comprehensive suite of tools for managing, visualizing, and validating project dependencies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/scs03004/dependency-toolkit",
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
    entry_points={
        "console_scripts": [
            "dependency-visualizer=tools.dependency_visualizer:main",
            "dependency-validator=tools.pre_commit_validator:main",
            "dependency-docs=tools.doc_generator:main",
            "dependency-ci=tools.ci_cd_integrator:main",
            "dependency-monitor=tools.monitoring_dashboard:main",
            "dependency-analyzer=tools.code_analyzer:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["templates/*.md", "templates/*.html", "*.yml", "*.yaml"],
    },
    keywords="dependency analysis visualization monitoring code-quality ci-cd",
    project_urls={
        "Bug Reports": "https://github.com/scs03004/dependency-toolkit/issues",
        "Source": "https://github.com/scs03004/dependency-toolkit",
        "Documentation": "https://github.com/scs03004/dependency-toolkit/blob/main/README.md",
    },
)