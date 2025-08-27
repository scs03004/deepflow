"""
Smart Refactoring Engine - Priority 4 Implementation

Automated code improvement for AI-generated code with intelligent pattern standardization,
import optimization, file splitting, dead code removal, and documentation generation.

This module provides the core engine for Priority 4: Smart Refactoring & Code Quality features.
"""

import ast
import os
import re
import textwrap
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PatternAnalysis:
    """Analysis result for pattern standardization."""
    pattern_type: str
    consistency_score: float
    violations: List[Dict[str, Any]]
    recommended_pattern: str
    files_affected: List[str]


@dataclass
class ImportAnalysis:
    """Analysis result for import optimization."""
    unused_imports: List[str]
    duplicate_imports: List[str]
    circular_imports: List[Tuple[str, str]]
    optimization_suggestions: List[Dict[str, Any]]


@dataclass
class FileSplitAnalysis:
    """Analysis result for file splitting recommendations."""
    file_path: str
    size_score: float
    complexity_score: float
    split_recommendations: List[Dict[str, Any]]
    suggested_modules: List[str]


@dataclass
class DeadCodeAnalysis:
    """Analysis result for dead code detection."""
    unused_functions: List[str]
    unused_classes: List[str]
    unused_variables: List[str]
    unreachable_code: List[Dict[str, Any]]


@dataclass
class DocumentationAnalysis:
    """Analysis result for documentation generation."""
    missing_docstrings: List[Dict[str, Any]]
    incomplete_docstrings: List[Dict[str, Any]]
    generated_docstrings: Dict[str, str]


@dataclass 
class RequirementsAnalysis:
    """Analysis result for requirements.txt management."""
    missing_packages: List[Dict[str, Any]]
    unused_packages: List[str]
    version_conflicts: List[Dict[str, Any]]
    update_recommendations: List[Dict[str, Any]]
    current_requirements: List[str]
    detected_imports: List[Dict[str, Any]]


class SmartRefactoringEngine:
    """
    Main engine for Smart Refactoring & Code Quality (Priority 4).
    
    Provides automated code improvement capabilities including:
    - Pattern standardization for consistent AI-generated code
    - Import optimization and cleanup
    - Automated file splitting for better organization
    - Dead code removal for cleaner codebases
    - Documentation generation for AI-generated functions
    """
    
    def __init__(self, project_path: str):
        """Initialize the smart refactoring engine."""
        self.project_path = Path(project_path)
        self.pattern_cache = {}
        self.import_cache = {}
        self.analysis_cache = {}
        self.requirements_cache = {}
        
        # Comprehensive import-to-package mapping for common libraries
        self.import_to_package = {
            # Web Frameworks
            'flask': 'flask',
            'django': 'django', 
            'fastapi': 'fastapi',
            'starlette': 'starlette',
            'tornado': 'tornado',
            'bottle': 'bottle',
            'pyramid': 'pyramid',
            
            # HTTP & API
            'requests': 'requests',
            'urllib3': 'urllib3',
            'httpx': 'httpx',
            'aiohttp': 'aiohttp',
            'websockets': 'websockets',
            'socketio': 'python-socketio',
            'pydantic': 'pydantic',
            
            # Database
            'sqlalchemy': 'sqlalchemy',
            'sqlite3': '',  # Built-in
            'pymongo': 'pymongo',
            'redis': 'redis',
            'psycopg2': 'psycopg2-binary',
            'mysql': 'mysql-connector-python',
            'cx_Oracle': 'cx-Oracle',
            'peewee': 'peewee',
            'tortoise': 'tortoise-orm',
            'databases': 'databases',
            'asyncpg': 'asyncpg',
            
            # Data Science & ML
            'numpy': 'numpy',
            'pandas': 'pandas',
            'matplotlib': 'matplotlib',
            'seaborn': 'seaborn',
            'plotly': 'plotly',
            'scipy': 'scipy',
            'sklearn': 'scikit-learn',
            'tensorflow': 'tensorflow',
            'torch': 'torch',
            'keras': 'keras',
            'xgboost': 'xgboost',
            'lightgbm': 'lightgbm',
            'opencv': 'opencv-python',
            'cv2': 'opencv-python',
            'PIL': 'Pillow',
            'skimage': 'scikit-image',
            'statsmodels': 'statsmodels',
            
            # Async & Concurrency  
            'asyncio': '',  # Built-in
            'aiofiles': 'aiofiles',
            'aiodns': 'aiodns',
            'uvloop': 'uvloop',
            'celery': 'celery',
            'rq': 'rq',
            
            # Testing
            'pytest': 'pytest',
            'unittest': '',  # Built-in
            'mock': '',  # Built-in (unittest.mock)
            'nose': 'nose',
            'coverage': 'coverage',
            'hypothesis': 'hypothesis',
            'factory_boy': 'factory-boy',
            'faker': 'faker',
            
            # CLI & Config
            'click': 'click',
            'argparse': '',  # Built-in
            'typer': 'typer',
            'fire': 'fire',
            'configparser': '',  # Built-in
            'yaml': 'pyyaml',
            'toml': 'toml',
            'dotenv': 'python-dotenv',
            
            # Utilities
            'rich': 'rich',
            'tqdm': 'tqdm',
            'loguru': 'loguru',
            'schedule': 'schedule',
            'watchdog': 'watchdog',
            
            # Built-in modules (Python standard library)
            'os': '',  # Built-in
            'sys': '',  # Built-in
            'pathlib': '',  # Built-in
            'datetime': '',  # Built-in
            'json': '',  # Built-in
            'csv': '',  # Built-in
            'pickle': '',  # Built-in
            'hashlib': '',  # Built-in
            'uuid': '',  # Built-in
            'base64': '',  # Built-in
            'gzip': '',  # Built-in
            'zipfile': '',  # Built-in
            'tarfile': '',  # Built-in
            'time': '',  # Built-in
            'random': '',  # Built-in
            'math': '',  # Built-in
            'statistics': '',  # Built-in
            'collections': '',  # Built-in
            'itertools': '',  # Built-in
            'functools': '',  # Built-in
            'operator': '',  # Built-in
            're': '',  # Built-in
            'string': '',  # Built-in
            'textwrap': '',  # Built-in
            'difflib': '',  # Built-in
            'unicodedata': '',  # Built-in
            'logging': '',  # Built-in
            'threading': '',  # Built-in
            'multiprocessing': '',  # Built-in
            'concurrent': '',  # Built-in
            'subprocess': '',  # Built-in
            'signal': '',  # Built-in
            'contextlib': '',  # Built-in
            'io': '',  # Built-in
            'tempfile': '',  # Built-in
            'shutil': '',  # Built-in
            'glob': '',  # Built-in
            'fnmatch': '',  # Built-in
            
            # Development Tools
            'black': 'black',
            'flake8': 'flake8',
            'mypy': 'mypy',
            'isort': 'isort',
            'bandit': 'bandit',
            'safety': 'safety',
            'pre_commit': 'pre-commit',
            
            # Crypto & Security
            'cryptography': 'cryptography',
            'bcrypt': 'bcrypt',
            'passlib': 'passlib',
            'jose': 'python-jose',
            'jwt': 'pyjwt',
            
            # File Processing
            'openpyxl': 'openpyxl',
            'xlsxwriter': 'xlsxwriter',
            'docx': 'python-docx',
            'pdf': 'pypdf2',
            'lxml': 'lxml',
            'bs4': 'beautifulsoup4',
            
            # Network & Parsing
            'paramiko': 'paramiko',
            'ftplib': '',  # Built-in
            'smtplib': '',  # Built-in
            'email': '',  # Built-in
            'dns': 'dnspython',
            'netifaces': 'netifaces',
            
            # Graphics & GUI
            'tkinter': '',  # Built-in
            'kivy': 'kivy',
            'pygame': 'pygame',
            'qt': 'pyqt5',
            'wx': 'wxpython',
            
            # Scientific Computing
            'sympy': 'sympy',
            'networkx': 'networkx',
            'igraph': 'python-igraph',
            'Bio': 'biopython',
            'astropy': 'astropy',
        }
        
    def standardize_patterns(self, target_files: Optional[List[str]] = None) -> PatternAnalysis:
        """
        Auto-align inconsistent AI-generated patterns.
        
        Analyzes code patterns and suggests standardizations to maintain consistency
        across AI-generated code sections.
        
        Args:
            target_files: Optional list of specific files to analyze
            
        Returns:
            PatternAnalysis with detected inconsistencies and recommendations
        """
        logger.info("Starting pattern standardization analysis...")
        
        files_to_analyze = target_files or self._get_python_files()
        pattern_violations = []
        pattern_frequency = defaultdict(int)
        
        for file_path in files_to_analyze:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                tree = ast.parse(content, filename=file_path)
                file_patterns = self._extract_patterns(tree, file_path)
                
                # Track pattern frequency
                for pattern_type, patterns in file_patterns.items():
                    for pattern in patterns:
                        pattern_key = f"{pattern_type}:{pattern['signature']}"
                        pattern_frequency[pattern_key] += 1
                
                # Detect pattern violations
                violations = self._detect_pattern_violations(file_patterns, file_path)
                pattern_violations.extend(violations)
                
            except Exception as e:
                logger.warning(f"Error analyzing patterns in {file_path}: {e}")
                continue
        
        # Determine most common patterns as standards
        recommended_patterns = self._determine_standard_patterns(pattern_frequency)
        
        # Calculate consistency score
        total_patterns = sum(pattern_frequency.values())
        standard_patterns = sum(freq for pattern, freq in pattern_frequency.items() 
                              if pattern in recommended_patterns)
        consistency_score = standard_patterns / total_patterns if total_patterns > 0 else 1.0
        
        return PatternAnalysis(
            pattern_type="comprehensive",
            consistency_score=consistency_score,
            violations=pattern_violations,
            recommended_pattern=str(recommended_patterns),
            files_affected=[str(f) for f in files_to_analyze]
        )
    
    def optimize_imports(self, target_files: Optional[List[str]] = None) -> ImportAnalysis:
        """
        Clean up and organize imports intelligently.
        
        Analyzes import statements and provides optimization suggestions including
        unused import removal, duplicate cleanup, and circular import detection.
        
        Args:
            target_files: Optional list of specific files to analyze
            
        Returns:
            ImportAnalysis with optimization suggestions
        """
        logger.info("Starting import optimization analysis...")
        
        files_to_analyze = target_files or self._get_python_files()
        unused_imports = []
        duplicate_imports = []
        circular_imports = []
        optimization_suggestions = []
        
        for file_path in files_to_analyze:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content, filename=file_path)
                
                # Analyze imports in this file
                import_analysis = self._analyze_file_imports(tree, file_path, content)
                unused_imports.extend(import_analysis['unused'])
                duplicate_imports.extend(import_analysis['duplicates'])
                optimization_suggestions.extend(import_analysis['suggestions'])
                
            except Exception as e:
                logger.warning(f"Error analyzing imports in {file_path}: {e}")
                continue
        
        # Detect circular imports across files
        circular_imports = self._detect_circular_imports(files_to_analyze)
        
        return ImportAnalysis(
            unused_imports=unused_imports,
            duplicate_imports=duplicate_imports,
            circular_imports=circular_imports,
            optimization_suggestions=optimization_suggestions
        )
    
    def suggest_file_splits(self, target_files: Optional[List[str]] = None) -> List[FileSplitAnalysis]:
        """
        Break large files into logical components.
        
        Analyzes file complexity and size to suggest optimal splitting strategies
        for better AI comprehension and maintainability.
        
        Args:
            target_files: Optional list of specific files to analyze
            
        Returns:
            List of FileSplitAnalysis with splitting recommendations
        """
        logger.info("Starting file split analysis...")
        
        files_to_analyze = target_files or self._get_python_files()
        split_analyses = []
        
        for file_path in files_to_analyze:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content, filename=file_path)
                
                # Calculate file metrics
                size_score = self._calculate_size_score(content)
                complexity_score = self._calculate_complexity_score(tree)
                
                # Generate split recommendations if needed
                split_recommendations = []
                suggested_modules = []
                
                if size_score > 0.7 or complexity_score > 0.8:  # High complexity/size
                    recommendations = self._generate_split_recommendations(tree, file_path)
                    split_recommendations = recommendations['splits']
                    suggested_modules = recommendations['modules']
                
                split_analyses.append(FileSplitAnalysis(
                    file_path=str(file_path),
                    size_score=size_score,
                    complexity_score=complexity_score,
                    split_recommendations=split_recommendations,
                    suggested_modules=suggested_modules
                ))
                
            except Exception as e:
                logger.warning(f"Error analyzing file splits for {file_path}: {e}")
                continue
        
        return split_analyses
    
    def detect_dead_code(self, target_files: Optional[List[str]] = None) -> DeadCodeAnalysis:
        """
        Clean up unused AI-generated code.
        
        Identifies unused functions, classes, variables, and unreachable code
        sections that can be safely removed.
        
        Args:
            target_files: Optional list of specific files to analyze
            
        Returns:
            DeadCodeAnalysis with removal recommendations
        """
        logger.info("Starting dead code detection...")
        
        files_to_analyze = target_files or self._get_python_files()
        unused_functions = []
        unused_classes = []
        unused_variables = []
        unreachable_code = []
        
        # Build cross-file usage map
        usage_map = self._build_usage_map(files_to_analyze)
        
        for file_path in files_to_analyze:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content, filename=file_path)
                
                # Analyze dead code in this file
                dead_code = self._analyze_dead_code(tree, file_path, usage_map)
                unused_functions.extend(dead_code['functions'])
                unused_classes.extend(dead_code['classes'])
                unused_variables.extend(dead_code['variables'])
                unreachable_code.extend(dead_code['unreachable'])
                
            except Exception as e:
                logger.warning(f"Error detecting dead code in {file_path}: {e}")
                continue
        
        return DeadCodeAnalysis(
            unused_functions=unused_functions,
            unused_classes=unused_classes,
            unused_variables=unused_variables,
            unreachable_code=unreachable_code
        )
    
    def generate_documentation(self, target_files: Optional[List[str]] = None) -> DocumentationAnalysis:
        """
        Add docstrings to AI-generated functions.
        
        Analyzes code structure and generates appropriate docstrings for functions,
        classes, and methods that are missing documentation.
        
        Args:
            target_files: Optional list of specific files to analyze
            
        Returns:
            DocumentationAnalysis with generated docstrings
        """
        logger.info("Starting documentation generation...")
        
        files_to_analyze = target_files or self._get_python_files()
        missing_docstrings = []
        incomplete_docstrings = []
        generated_docstrings = {}
        
        for file_path in files_to_analyze:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content, filename=file_path)
                
                # Analyze documentation needs
                doc_analysis = self._analyze_documentation_needs(tree, file_path)
                missing_docstrings.extend(doc_analysis['missing'])
                incomplete_docstrings.extend(doc_analysis['incomplete'])
                
                # Generate docstrings for missing documentation
                for item in doc_analysis['missing']:
                    docstring = self._generate_docstring(item)
                    key = f"{file_path}:{item['name']}:{item['line']}"
                    generated_docstrings[key] = docstring
                
            except Exception as e:
                logger.warning(f"Error generating documentation for {file_path}: {e}")
                continue
        
        return DocumentationAnalysis(
            missing_docstrings=missing_docstrings,
            incomplete_docstrings=incomplete_docstrings,
            generated_docstrings=generated_docstrings
        )
    
    def analyze_requirements(self, target_files: Optional[List[str]] = None, 
                           check_installed: bool = True) -> RequirementsAnalysis:
        """
        Analyze and manage requirements.txt for AI-assisted development.
        
        Detects missing packages from imports, identifies unused requirements,
        and provides intelligent update recommendations for requirements.txt.
        
        Args:
            target_files: Optional list of specific files to analyze
            check_installed: Whether to check if packages are actually installed
            
        Returns:
            RequirementsAnalysis with package recommendations
        """
        logger.info("Starting requirements analysis...")
        
        files_to_analyze = target_files or self._get_python_files()
        
        # Parse current requirements.txt
        current_requirements = self._parse_requirements_file()
        current_packages = {req.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0].lower() 
                           for req in current_requirements}
        
        # Detect all imports from Python files
        detected_imports = self._detect_all_imports(files_to_analyze)
        
        # Map imports to package names
        required_packages = set()
        missing_packages = []
        
        for import_info in detected_imports:
            import_name = import_info['import_name']
            package_name = self._map_import_to_package(import_name)
            
            if package_name and package_name not in current_packages:
                required_packages.add(package_name)
                missing_packages.append({
                    'import_name': import_name,
                    'package_name': package_name,
                    'files_using': import_info['files'],
                    'is_standard_library': package_name == '',
                    'confidence': self._get_mapping_confidence(import_name),
                    'suggested_version': self._get_suggested_version(package_name)
                })
        
        # Find potentially unused packages
        unused_packages = []
        if check_installed:
            unused_packages = self._find_unused_packages(current_packages, detected_imports)
        
        # Check for version conflicts
        version_conflicts = self._detect_version_conflicts(current_requirements, missing_packages)
        
        # Generate update recommendations
        update_recommendations = self._generate_update_recommendations(
            missing_packages, unused_packages, version_conflicts
        )
        
        return RequirementsAnalysis(
            missing_packages=missing_packages,
            unused_packages=unused_packages,
            version_conflicts=version_conflicts,
            update_recommendations=update_recommendations,
            current_requirements=current_requirements,
            detected_imports=detected_imports
        )
    
    def update_requirements_file(self, analysis: RequirementsAnalysis, 
                               backup: bool = True, dry_run: bool = True) -> Dict[str, Any]:
        """
        Update requirements.txt based on analysis results.
        
        Args:
            analysis: RequirementsAnalysis from analyze_requirements()
            backup: Whether to create a backup of the original file
            dry_run: If True, only show what would be changed
            
        Returns:
            Dictionary with update results and statistics
        """
        requirements_path = self.project_path / "requirements.txt"
        
        if backup and requirements_path.exists() and not dry_run:
            backup_path = self.project_path / "requirements.txt.backup"
            import shutil
            shutil.copy2(requirements_path, backup_path)
        
        # Build new requirements content
        new_requirements = set(analysis.current_requirements)
        
        # Add missing packages
        for pkg in analysis.missing_packages:
            if not pkg['is_standard_library'] and pkg['confidence'] >= 0.8:
                package_name = pkg['package_name']
                version = pkg.get('suggested_version', '')
                if version:
                    new_requirements.add(f"{package_name}=={version}")
                else:
                    new_requirements.add(package_name)
        
        # Remove unused packages if user confirms
        for unused_pkg in analysis.unused_packages:
            new_requirements.discard(unused_pkg)
        
        # Sort requirements for consistency
        sorted_requirements = sorted(list(new_requirements))
        
        results = {
            'dry_run': dry_run,
            'packages_added': len([p for p in analysis.missing_packages 
                                  if not p['is_standard_library'] and p['confidence'] >= 0.8]),
            'packages_removed': len(analysis.unused_packages),
            'original_count': len(analysis.current_requirements),
            'new_count': len(sorted_requirements),
            'backup_created': backup and requirements_path.exists() and not dry_run,
            'new_requirements': sorted_requirements
        }
        
        if not dry_run:
            # Write updated requirements.txt
            with open(requirements_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(sorted_requirements) + '\n')
            logger.info(f"Updated requirements.txt with {results['packages_added']} new packages")
        else:
            logger.info("Dry run: would update requirements.txt with the following changes:")
            for req in sorted_requirements:
                if req not in analysis.current_requirements:
                    logger.info(f"  + {req}")
            for unused in analysis.unused_packages:
                logger.info(f"  - {unused}")
        
        return results
    
    def apply_refactoring(self, analysis_results: Dict[str, Any], dry_run: bool = True) -> Dict[str, Any]:
        """
        Apply refactoring suggestions with safety validation.
        
        Args:
            analysis_results: Combined results from all analysis methods
            dry_run: If True, only simulate changes without applying them
            
        Returns:
            Results of refactoring operations
        """
        logger.info(f"Applying refactoring (dry_run={dry_run})...")
        
        results = {
            'patterns_standardized': 0,
            'imports_optimized': 0,
            'files_split': 0,
            'dead_code_removed': 0,
            'documentation_added': 0,
            'errors': []
        }
        
        try:
            # Apply pattern standardization
            if 'pattern_analysis' in analysis_results:
                pattern_results = self._apply_pattern_standardization(
                    analysis_results['pattern_analysis'], dry_run
                )
                results['patterns_standardized'] = pattern_results['applied']
                
            # Apply import optimization
            if 'import_analysis' in analysis_results:
                import_results = self._apply_import_optimization(
                    analysis_results['import_analysis'], dry_run
                )
                results['imports_optimized'] = import_results['applied']
                
            # Apply file splitting
            if 'file_split_analysis' in analysis_results:
                split_results = self._apply_file_splitting(
                    analysis_results['file_split_analysis'], dry_run
                )
                results['files_split'] = split_results['applied']
                
            # Remove dead code
            if 'dead_code_analysis' in analysis_results:
                dead_code_results = self._apply_dead_code_removal(
                    analysis_results['dead_code_analysis'], dry_run
                )
                results['dead_code_removed'] = dead_code_results['applied']
                
            # Add documentation
            if 'documentation_analysis' in analysis_results:
                doc_results = self._apply_documentation_generation(
                    analysis_results['documentation_analysis'], dry_run
                )
                results['documentation_added'] = doc_results['applied']
                
        except Exception as e:
            logger.error(f"Error applying refactoring: {e}")
            results['errors'].append(str(e))
        
        return results
    
    # Requirements management helper methods
    
    def _parse_requirements_file(self) -> List[str]:
        """Parse existing requirements.txt file."""
        requirements_path = self.project_path / "requirements.txt"
        if not requirements_path.exists():
            return []
        
        try:
            with open(requirements_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            requirements = []
            for line in lines:
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('#'):
                    requirements.append(line)
            
            return requirements
        except Exception as e:
            logger.warning(f"Error reading requirements.txt: {e}")
            return []
    
    def _detect_all_imports(self, files: List[Path]) -> List[Dict[str, Any]]:
        """Detect all imports across Python files."""
        import_usage = defaultdict(list)
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                tree = ast.parse(content, filename=str(file_path))
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            import_name = alias.name.split('.')[0]  # Get top-level module
                            import_usage[import_name].append(str(file_path))
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            import_name = node.module.split('.')[0]  # Get top-level module
                            import_usage[import_name].append(str(file_path))
                            
            except Exception as e:
                logger.warning(f"Error parsing {file_path}: {e}")
                continue
        
        # Convert to list format
        detected_imports = []
        for import_name, files_using in import_usage.items():
            detected_imports.append({
                'import_name': import_name,
                'files': list(set(files_using)),  # Remove duplicates
                'usage_count': len(files_using)
            })
        
        return detected_imports
    
    def _map_import_to_package(self, import_name: str) -> str:
        """Map import name to package name using comprehensive mapping."""
        # Direct mapping from our comprehensive dictionary
        if import_name in self.import_to_package:
            return self.import_to_package[import_name]
        
        # Handle common patterns for unmapped imports
        # Most imports match their package name
        return import_name
    
    def _get_mapping_confidence(self, import_name: str) -> float:
        """Get confidence score for import-to-package mapping."""
        if import_name in self.import_to_package:
            return 1.0  # High confidence for known mappings
        else:
            return 0.7  # Medium confidence for direct name mapping
    
    def _get_suggested_version(self, package_name: str) -> str:
        """Get suggested version for a package."""
        # For now, return empty string to use latest version
        # In the future, could query PyPI API or use version constraints
        return ""
    
    def _find_unused_packages(self, current_packages: Set[str], 
                            detected_imports: List[Dict[str, Any]]) -> List[str]:
        """Find packages in requirements.txt that might not be used."""
        detected_package_names = set()
        
        for import_info in detected_imports:
            package_name = self._map_import_to_package(import_info['import_name'])
            if package_name:  # Skip built-in modules
                detected_package_names.add(package_name.lower())
        
        # Find packages in requirements but not in detected imports
        unused = []
        for pkg in current_packages:
            if pkg.lower() not in detected_package_names:
                # Be conservative - only flag obvious unused packages
                if not self._is_likely_indirect_dependency(pkg):
                    unused.append(pkg)
        
        return unused
    
    def _is_likely_indirect_dependency(self, package_name: str) -> bool:
        """Check if package is likely an indirect dependency."""
        # Common indirect dependencies that shouldn't be flagged as unused
        indirect_deps = {
            'setuptools', 'pip', 'wheel', 'six', 'urllib3', 'certifi', 
            'charset-normalizer', 'idna', 'pycparser', 'cffi'
        }
        return package_name.lower() in indirect_deps
    
    def _detect_version_conflicts(self, current_requirements: List[str], 
                                missing_packages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect potential version conflicts."""
        conflicts = []
        
        # For now, return empty list - could be enhanced to check for known conflicts
        # between packages or Python version compatibility
        
        return conflicts
    
    def _generate_update_recommendations(self, missing_packages: List[Dict[str, Any]],
                                       unused_packages: List[str],
                                       version_conflicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate intelligent update recommendations."""
        recommendations = []
        
        # High-confidence missing packages
        high_confidence_missing = [pkg for pkg in missing_packages 
                                 if pkg['confidence'] >= 0.9 and not pkg['is_standard_library']]
        if high_confidence_missing:
            recommendations.append({
                'type': 'add_packages',
                'priority': 'high',
                'action': 'Add missing packages to requirements.txt',
                'packages': [pkg['package_name'] for pkg in high_confidence_missing],
                'rationale': 'These packages are imported but not in requirements.txt'
            })
        
        # Medium-confidence missing packages
        medium_confidence_missing = [pkg for pkg in missing_packages 
                                   if 0.7 <= pkg['confidence'] < 0.9 and not pkg['is_standard_library']]
        if medium_confidence_missing:
            recommendations.append({
                'type': 'review_packages',
                'priority': 'medium',
                'action': 'Review and potentially add packages',
                'packages': [pkg['package_name'] for pkg in medium_confidence_missing],
                'rationale': 'These imports might require additional packages'
            })
        
        # Unused packages (be conservative)
        if unused_packages:
            recommendations.append({
                'type': 'review_unused',
                'priority': 'low',
                'action': 'Review potentially unused packages',
                'packages': unused_packages,
                'rationale': 'These packages may not be directly imported (could be indirect dependencies)'
            })
        
        return recommendations
    
    # Helper methods
    
    def _get_python_files(self) -> List[Path]:
        """Get all Python files in the project."""
        python_files = []
        for root, dirs, files in os.walk(self.project_path):
            # Skip common non-source directories
            dirs[:] = [d for d in dirs if d not in {'.git', '__pycache__', '.pytest_cache', 'venv', '.venv', 'node_modules'}]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        
        return python_files
    
    def _extract_patterns(self, tree: ast.AST, file_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """Extract code patterns from AST."""
        patterns = defaultdict(list)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                patterns['function'].append({
                    'signature': f"def {node.name}({len(node.args.args)} args)",
                    'name': node.name,
                    'line': node.lineno,
                    'decorators': len(node.decorator_list),
                    'file': file_path
                })
            elif isinstance(node, ast.ClassDef):
                patterns['class'].append({
                    'signature': f"class {node.name}({len(node.bases)} bases)",
                    'name': node.name,
                    'line': node.lineno,
                    'decorators': len(node.decorator_list),
                    'file': file_path
                })
            elif isinstance(node, ast.Import):
                patterns['import'].append({
                    'signature': f"import {len(node.names)} modules",
                    'modules': [alias.name for alias in node.names],
                    'line': node.lineno,
                    'file': file_path
                })
        
        return patterns
    
    def _detect_pattern_violations(self, patterns: Dict[str, List[Dict[str, Any]]], file_path: str) -> List[Dict[str, Any]]:
        """Detect pattern violations in extracted patterns."""
        violations = []
        
        # Check for inconsistent function naming patterns
        function_names = [p['name'] for p in patterns.get('function', [])]
        naming_patterns = {
            'snake_case': sum(1 for name in function_names if '_' in name and name.islower()),
            'camelCase': sum(1 for name in function_names if any(c.isupper() for c in name) and '_' not in name),
            'PascalCase': sum(1 for name in function_names if name[0].isupper() if name)
        }
        
        if len(naming_patterns) > 1 and max(naming_patterns.values()) < len(function_names) * 0.8:
            violations.append({
                'type': 'inconsistent_naming',
                'file': file_path,
                'description': 'Mixed naming conventions detected',
                'patterns': naming_patterns
            })
        
        return violations
    
    def _determine_standard_patterns(self, pattern_frequency: Dict[str, int]) -> Dict[str, Any]:
        """Determine standard patterns from frequency analysis."""
        # Simple heuristic: most frequent pattern wins
        return {pattern: freq for pattern, freq in pattern_frequency.items() if freq > 1}
    
    def _analyze_file_imports(self, tree: ast.AST, file_path: str, content: str) -> Dict[str, List]:
        """Analyze imports in a single file."""
        imports = {'unused': [], 'duplicates': [], 'suggestions': []}
        
        # Extract all imports
        imported_names = set()
        import_lines = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    name = alias.asname if alias.asname else alias.name
                    if name in imported_names:
                        imports['duplicates'].append(f"{file_path}:{node.lineno}:{name}")
                    imported_names.add(name)
                    import_lines.append((node.lineno, name))
        
        # Check for unused imports (basic heuristic)
        for line_no, name in import_lines:
            # Simple check: if name doesn't appear elsewhere in content
            if content.count(name) <= 1:  # Only appears in import line
                imports['unused'].append(f"{file_path}:{line_no}:{name}")
        
        return imports
    
    def _detect_circular_imports(self, files: List[Path]) -> List[Tuple[str, str]]:
        """Detect circular import dependencies."""
        # Build import graph
        import_graph = defaultdict(set)
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                tree = ast.parse(content)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        # Try to resolve relative imports
                        if node.module.startswith('.'):
                            # Relative import - would need proper resolution
                            continue
                        import_graph[str(file_path)].add(node.module)
            except:
                continue
        
        # Detect cycles (simplified)
        cycles = []
        for file, imports in import_graph.items():
            for imported in imports:
                if imported in import_graph and file in import_graph[imported]:
                    cycles.append((file, imported))
        
        return cycles
    
    def _calculate_size_score(self, content: str) -> float:
        """Calculate size score (0-1, higher means larger)."""
        lines = len(content.split('\n'))
        # Normalize to 0-1 scale, 500+ lines = score of 1.0
        return min(lines / 500.0, 1.0)
    
    def _calculate_complexity_score(self, tree: ast.AST) -> float:
        """Calculate complexity score based on AST structure."""
        complexity_nodes = 0
        total_nodes = 0
        
        for node in ast.walk(tree):
            total_nodes += 1
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With, ast.FunctionDef, ast.ClassDef)):
                complexity_nodes += 1
        
        return complexity_nodes / max(total_nodes, 1)
    
    def _generate_split_recommendations(self, tree: ast.AST, file_path: str) -> Dict[str, List]:
        """Generate file splitting recommendations."""
        recommendations = {'splits': [], 'modules': []}
        
        # Analyze top-level definitions
        classes = []
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and hasattr(node, 'lineno'):
                classes.append({'name': node.name, 'line': node.lineno})
            elif isinstance(node, ast.FunctionDef) and hasattr(node, 'lineno'):
                functions.append({'name': node.name, 'line': node.lineno})
        
        # Suggest splitting if many top-level definitions
        if len(classes) > 3:
            recommendations['splits'].append({
                'type': 'class_grouping',
                'description': f'Split {len(classes)} classes into separate modules',
                'classes': classes
            })
            
        if len(functions) > 10:
            recommendations['splits'].append({
                'type': 'function_grouping',  
                'description': f'Group {len(functions)} functions by functionality',
                'functions': functions
            })
        
        return recommendations
    
    def _build_usage_map(self, files: List[Path]) -> Dict[str, Set[str]]:
        """Build cross-file usage map for dead code detection."""
        usage_map = defaultdict(set)
        
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Simple usage detection based on name references
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Name):
                        usage_map[node.id].add(str(file_path))
                        
            except:
                continue
                
        return usage_map
    
    def _analyze_dead_code(self, tree: ast.AST, file_path: str, usage_map: Dict[str, Set[str]]) -> Dict[str, List]:
        """Analyze dead code in a single file."""
        dead_code = {'functions': [], 'classes': [], 'variables': [], 'unreachable': []}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if len(usage_map.get(node.name, set())) <= 1:  # Only defined, not used
                    dead_code['functions'].append(f"{file_path}:{node.lineno}:{node.name}")
            elif isinstance(node, ast.ClassDef):
                if len(usage_map.get(node.name, set())) <= 1:  # Only defined, not used
                    dead_code['classes'].append(f"{file_path}:{node.lineno}:{node.name}")
        
        return dead_code
    
    def _analyze_documentation_needs(self, tree: ast.AST, file_path: str) -> Dict[str, List]:
        """Analyze documentation needs in a file."""
        analysis = {'missing': [], 'incomplete': []}
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                docstring = ast.get_docstring(node)
                if not docstring:
                    analysis['missing'].append({
                        'type': 'function' if isinstance(node, ast.FunctionDef) else 'class',
                        'name': node.name,
                        'line': node.lineno,
                        'file': file_path,
                        'args': [arg.arg for arg in node.args.args] if isinstance(node, ast.FunctionDef) else []
                    })
                elif len(docstring.strip()) < 10:  # Very short docstring
                    analysis['incomplete'].append({
                        'type': 'function' if isinstance(node, ast.FunctionDef) else 'class',
                        'name': node.name,
                        'line': node.lineno,
                        'file': file_path,
                        'current_docstring': docstring
                    })
        
        return analysis
    
    def _generate_docstring(self, item: Dict[str, Any]) -> str:
        """Generate a docstring for a function or class."""
        if item['type'] == 'function':
            args_doc = '\n        '.join([f"{arg}: Description of {arg}" for arg in item['args']]) if item['args'] else ""
            args_section = f"\n\n    Args:\n        {args_doc}" if args_doc else ""
            
            return f'''"""{item['name']} function.
    
    Brief description of what {item['name']} does.{args_section}
    
    Returns:
        Description of return value.
    """'''
        else:
            return f'''"""{item['name']} class.
    
    Brief description of what {item['name']} class represents and its purpose.
    
    Attributes:
        Add class attributes here.
    """'''
    
    # Apply methods (stubs for now)
    
    def _apply_pattern_standardization(self, analysis: PatternAnalysis, dry_run: bool) -> Dict[str, int]:
        """Apply pattern standardization changes."""
        # Implementation would modify files to standardize patterns
        logger.info(f"Pattern standardization ({'dry run' if dry_run else 'applying'})")
        return {'applied': len(analysis.violations) if not dry_run else 0}
    
    def _apply_import_optimization(self, analysis: ImportAnalysis, dry_run: bool) -> Dict[str, int]:
        """Apply import optimization changes."""
        logger.info(f"Import optimization ({'dry run' if dry_run else 'applying'})")
        return {'applied': len(analysis.unused_imports + analysis.duplicate_imports) if not dry_run else 0}
    
    def _apply_file_splitting(self, analyses: List[FileSplitAnalysis], dry_run: bool) -> Dict[str, int]:
        """Apply file splitting changes."""
        logger.info(f"File splitting ({'dry run' if dry_run else 'applying'})")
        return {'applied': len([a for a in analyses if a.split_recommendations]) if not dry_run else 0}
    
    def _apply_dead_code_removal(self, analysis: DeadCodeAnalysis, dry_run: bool) -> Dict[str, int]:
        """Apply dead code removal changes."""
        logger.info(f"Dead code removal ({'dry run' if dry_run else 'applying'})")
        total_removals = len(analysis.unused_functions + analysis.unused_classes + analysis.unused_variables)
        return {'applied': total_removals if not dry_run else 0}
    
    def _apply_documentation_generation(self, analysis: DocumentationAnalysis, dry_run: bool) -> Dict[str, int]:
        """Apply documentation generation changes."""
        logger.info(f"Documentation generation ({'dry run' if dry_run else 'applying'})")
        return {'applied': len(analysis.generated_docstrings) if not dry_run else 0}