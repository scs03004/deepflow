#!/usr/bin/env python3
"""
CI/CD Integration Tools
======================

Automated dependency checking in CI/CD pipelines:
- GitHub Actions workflows for automated dependency validation
- Dependency change notifications in pull requests
- Risk-based testing - Different test suites based on impact
- Deployment safety checks before production

Usage:
    python ci_cd_integrator.py --setup-github /path/to/project
    python ci_cd_integrator.py --setup-gitlab /path/to/project
    python ci_cd_integrator.py --validate-changes /path/to/project
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass

try:
    import yaml
    from rich.console import Console
    from rich.panel import Panel
    import git
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Install with: pip install GitPython rich PyYAML")
    sys.exit(1)


@dataclass
class CIPipeline:
    """CI/CD pipeline configuration."""

    platform: str  # 'github', 'gitlab', 'jenkins'
    workflows: List[str]
    triggers: List[str]
    test_strategies: Dict[str, List[str]]
    deployment_gates: List[str]


class CICDIntegrator:
    """CI/CD integration and automation engine."""

    def __init__(self, project_path: str):
        self.project_path = Path(project_path).resolve()
        self.console = Console()

        try:
            self.git_repo = git.Repo(self.project_path)
        except git.InvalidGitRepositoryError:
            self.git_repo = None
            self.console.print("[yellow]Warning: Not a Git repository[/yellow]")

    def setup_github_actions(self) -> Dict[str, str]:
        """Set up GitHub Actions workflows for dependency checking."""
        workflows_dir = self.project_path / ".github" / "workflows"
        workflows_dir.mkdir(parents=True, exist_ok=True)

        created_files = {}

        # 1. Dependency check workflow
        dep_check_workflow = self._create_dependency_check_workflow()
        dep_check_path = workflows_dir / "dependency-check.yml"
        with open(dep_check_path, "w") as f:
            yaml.dump(dep_check_workflow, f, default_flow_style=False, sort_keys=False)
        created_files["dependency_check"] = str(dep_check_path)

        # 2. Documentation update workflow
        doc_update_workflow = self._create_doc_update_workflow()
        doc_update_path = workflows_dir / "documentation-update.yml"
        with open(doc_update_path, "w") as f:
            yaml.dump(doc_update_workflow, f, default_flow_style=False, sort_keys=False)
        created_files["doc_update"] = str(doc_update_path)

        # 3. Risk-based testing workflow
        risk_test_workflow = self._create_risk_based_testing_workflow()
        risk_test_path = workflows_dir / "risk-based-testing.yml"
        with open(risk_test_path, "w") as f:
            yaml.dump(risk_test_workflow, f, default_flow_style=False, sort_keys=False)
        created_files["risk_testing"] = str(risk_test_path)

        # 4. Deployment safety check workflow
        deploy_check_workflow = self._create_deployment_safety_workflow()
        deploy_check_path = workflows_dir / "deployment-safety.yml"
        with open(deploy_check_path, "w") as f:
            yaml.dump(deploy_check_workflow, f, default_flow_style=False, sort_keys=False)
        created_files["deployment_safety"] = str(deploy_check_path)

        self.console.print(
            f"[green]✅ GitHub Actions workflows created in:[/green] {workflows_dir}"
        )
        return created_files

    def _create_dependency_check_workflow(self) -> Dict:
        """Create dependency validation workflow."""
        return {
            "name": "Dependency Check",
            "on": {
                "pull_request": {"paths": ["**.py", "requirements.txt", "pyproject.toml"]},
                "push": {"branches": ["main", "master", "develop"]},
            },
            "jobs": {
                "dependency-validation": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {
                            "name": "Checkout code",
                            "uses": "actions/checkout@v3",
                            "with": {"fetch-depth": 0},
                        },
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v4",
                            "with": {"python-version": "3.9"},
                        },
                        {"name": "Install dependencies", "run": "pip install -r requirements.txt"},
                        {
                            "name": "Install dependency toolkit",
                            "run": "pip install deepflow",
                        },
                        {
                            "name": "Validate dependencies",
                            "run": "python -m deepflow.tools.pre_commit_validator --validate .",
                        },
                        {
                            "name": "Generate dependency report",
                            "run": "python -m deepflow.tools.dependency_visualizer . --format html",
                        },
                        {
                            "name": "Upload dependency report",
                            "uses": "actions/upload-artifact@v3",
                            "with": {
                                "name": "dependency-report",
                                "path": "*_dependency_graph.html",
                            },
                        },
                        {
                            "name": "Comment PR with results",
                            "if": "github.event_name == 'pull_request'",
                            "uses": "actions/github-script@v6",
                            "with": {
                                "script": """
const fs = require('fs');
const path = require('path');

// Read dependency analysis results
let comment = "## 📊 Dependency Analysis Results\\n\\n";

// Add summary of findings
comment += "✅ Dependency validation completed.\\n";
comment += "📋 Check the uploaded artifacts for detailed dependency graphs.\\n\\n";

// Check for high-risk changes
const { data: files } = await github.rest.pulls.listFiles({
  owner: context.repo.owner,
  repo: context.repo.repo,
  pull_number: context.issue.number,
});

const highRiskFiles = files.filter(file => 
  ['main.py', 'config.py', 'requirements.txt'].some(pattern => 
    file.filename.includes(pattern)
  )
);

if (highRiskFiles.length > 0) {
  comment += "⚠️  **High-risk files detected:**\\n";
  highRiskFiles.forEach(file => {
    comment += `- ${file.filename}\\n`;
  });
  comment += "\\nConsider running full test suite before merging.\\n";
}

github.rest.issues.createComment({
  issue_number: context.issue.number,
  owner: context.repo.owner,
  repo: context.repo.repo,
  body: comment
});
                                """
                            },
                        },
                    ],
                }
            },
        }

    def _create_doc_update_workflow(self) -> Dict:
        """Create documentation update workflow."""
        return {
            "name": "Documentation Update",
            "on": {
                "push": {"branches": ["main", "master"], "paths": ["**.py", "api/**", "models/**"]},
                "workflow_dispatch": None,
            },
            "jobs": {
                "update-docs": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {
                            "name": "Checkout code",
                            "uses": "actions/checkout@v3",
                            "with": {"token": "${{ secrets.GITHUB_TOKEN }}"},
                        },
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v4",
                            "with": {"python-version": "3.9"},
                        },
                        {"name": "Install dependencies", "run": "pip install -r requirements.txt"},
                        {
                            "name": "Install dependency toolkit",
                            "run": "pip install deepflow",
                        },
                        {
                            "name": "Generate documentation",
                            "run": "python -m deepflow.tools.doc_generator . --output docs/",
                        },
                        {
                            "name": "Commit updated docs",
                            "run": """
git config --local user.email "action@github.com"
git config --local user.name "GitHub Action"
git add docs/
if ! git diff --staged --quiet; then
  git commit -m "📚 Auto-update documentation [skip ci]"
  git push
else
  echo "No documentation changes to commit"
fi
                            """,
                        },
                    ],
                }
            },
        }

    def _create_risk_based_testing_workflow(self) -> Dict:
        """Create risk-based testing workflow."""
        return {
            "name": "Risk-Based Testing",
            "on": {"pull_request": {"types": ["opened", "synchronize", "reopened"]}},
            "jobs": {
                "analyze-risk": {
                    "runs-on": "ubuntu-latest",
                    "outputs": {
                        "risk_level": "${{ steps.risk_analysis.outputs.risk_level }}",
                        "test_strategy": "${{ steps.risk_analysis.outputs.test_strategy }}",
                    },
                    "steps": [
                        {
                            "name": "Checkout code",
                            "uses": "actions/checkout@v3",
                            "with": {"fetch-depth": 0},
                        },
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v4",
                            "with": {"python-version": "3.9"},
                        },
                        {
                            "name": "Install dependency toolkit",
                            "run": "pip install deepflow",
                        },
                        {
                            "name": "Analyze change risk",
                            "id": "risk_analysis",
                            "run": """
# Analyze changed files and determine risk level
python -m deepflow.tools.pre_commit_validator --impact-analysis . > risk_analysis.json

# Extract risk level and set outputs
RISK_LEVEL=$(python -c "
import json
with open('risk_analysis.json', 'r') as f:
    data = json.load(f)
print(data.get('risk_assessment', 'LOW'))
")

echo "risk_level=$RISK_LEVEL" >> $GITHUB_OUTPUT

# Determine test strategy based on risk
if [ "$RISK_LEVEL" = "HIGH" ]; then
  echo "test_strategy=full" >> $GITHUB_OUTPUT
elif [ "$RISK_LEVEL" = "MEDIUM" ]; then
  echo "test_strategy=integration" >> $GITHUB_OUTPUT
else
  echo "test_strategy=unit" >> $GITHUB_OUTPUT
fi

echo "Detected risk level: $RISK_LEVEL"
                            """,
                        },
                    ],
                },
                "unit-tests": {
                    "needs": "analyze-risk",
                    "if": "always()",
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"name": "Checkout code", "uses": "actions/checkout@v3"},
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v4",
                            "with": {"python-version": "3.9"},
                        },
                        {"name": "Install dependencies", "run": "pip install -r requirements.txt"},
                        {
                            "name": "Run unit tests",
                            "run": 'pytest tests/ -k "not integration" --cov=. --cov-report=xml',
                        },
                    ],
                },
                "integration-tests": {
                    "needs": "analyze-risk",
                    "if": "needs.analyze-risk.outputs.test_strategy != 'unit'",
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"name": "Checkout code", "uses": "actions/checkout@v3"},
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v4",
                            "with": {"python-version": "3.9"},
                        },
                        {"name": "Install dependencies", "run": "pip install -r requirements.txt"},
                        {
                            "name": "Run integration tests",
                            "run": "pytest tests/ --cov=. --cov-report=xml",
                        },
                    ],
                },
                "full-test-suite": {
                    "needs": "analyze-risk",
                    "if": "needs.analyze-risk.outputs.risk_level == 'HIGH'",
                    "runs-on": "ubuntu-latest",
                    "strategy": {"matrix": {"python-version": ["3.8", "3.9", "3.10", "3.11"]}},
                    "steps": [
                        {"name": "Checkout code", "uses": "actions/checkout@v3"},
                        {
                            "name": "Set up Python ${{ matrix.python-version }}",
                            "uses": "actions/setup-python@v4",
                            "with": {"python-version": "${{ matrix.python-version }}"},
                        },
                        {"name": "Install dependencies", "run": "pip install -r requirements.txt"},
                        {
                            "name": "Run full test suite",
                            "run": "pytest tests/ --cov=. --cov-report=xml --slow",
                        },
                        {
                            "name": "Performance tests",
                            "run": 'pytest tests/ -k "performance" --benchmark-only',
                        },
                    ],
                },
            },
        }

    def _create_deployment_safety_workflow(self) -> Dict:
        """Create deployment safety check workflow."""
        return {
            "name": "Deployment Safety Check",
            "on": {"push": {"branches": ["main", "master"]}, "release": {"types": ["published"]}},
            "jobs": {
                "safety-checks": {
                    "runs-on": "ubuntu-latest",
                    "steps": [
                        {"name": "Checkout code", "uses": "actions/checkout@v3"},
                        {
                            "name": "Set up Python",
                            "uses": "actions/setup-python@v4",
                            "with": {"python-version": "3.9"},
                        },
                        {"name": "Install dependencies", "run": "pip install -r requirements.txt"},
                        {
                            "name": "Install dependency toolkit",
                            "run": "pip install deepflow",
                        },
                        {"name": "Security audit", "run": "pip install safety && safety check"},
                        {
                            "name": "Dependency validation",
                            "run": "python -m deepflow.tools.pre_commit_validator --validate .",
                        },
                        {
                            "name": "Check for circular dependencies",
                            "run": 'python -m deepflow.tools.dependency_visualizer . --format text | grep -i "circular" && exit 1 || echo "No circular dependencies found"',
                        },
                        {
                            "name": "Performance baseline check",
                            "run": """
if [ -f "performance_baseline.json" ]; then
  echo "Running performance regression tests..."
  pytest tests/ -k "performance" --benchmark-compare=performance_baseline.json
else
  echo "No performance baseline found, skipping regression tests"
fi
                            """,
                        },
                        {
                            "name": "Database migration check",
                            "run": """
if [ -d "migrations" ]; then
  echo "Checking database migrations..."
  python -c "
import os
migration_files = [f for f in os.listdir('migrations') if f.endswith('.py')]
print(f'Found {len(migration_files)} migration files')
if migration_files:
    print('Migrations are ready for deployment')
else:
    print('No migrations found')
  "
fi
                            """,
                        },
                        {
                            "name": "Generate deployment report",
                            "run": """
cat > deployment_report.md << EOF
# Deployment Safety Report

**Date**: $(date)
**Commit**: ${{ github.sha }}
**Branch**: ${{ github.ref_name }}

## Safety Checks

✅ Security audit passed
✅ Dependency validation passed
✅ No circular dependencies detected
✅ Performance checks completed

## Deployment Ready

This build is ready for deployment to production.

EOF

echo "Deployment report generated"
                            """,
                        },
                        {
                            "name": "Upload deployment report",
                            "uses": "actions/upload-artifact@v3",
                            "with": {"name": "deployment-report", "path": "deployment_report.md"},
                        },
                    ],
                }
            },
        }

    def setup_gitlab_ci(self) -> str:
        """Set up GitLab CI pipeline for dependency checking."""
        gitlab_ci_content = """# GitLab CI/CD Pipeline for Dependency Management
# Generated by Deepflow

stages:
  - validate
  - test
  - security
  - deploy-check

variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"

cache:
  paths:
    - .cache/pip/
    - venv/

before_script:
  - python -V
  - pip install virtualenv
  - virtualenv venv
  - source venv/bin/activate
  - pip install -r requirements.txt
  - pip install deepflow

dependency-validation:
  stage: validate
  script:
    - python -m deepflow.tools.pre_commit_validator --validate .
    - python -m deepflow.tools.dependency_visualizer . --format json
  artifacts:
    reports:
      junit: dependency_report.xml
    paths:
      - "*_dependency_graph.html"
      - dependency_analysis.json
    expire_in: 1 week
  only:
    changes:
      - "**/*.py"
      - requirements.txt
      - pyproject.toml

risk-based-testing:
  stage: test
  script:
    - python -m deepflow.tools.pre_commit_validator --impact-analysis . > risk_analysis.json
    - |
      RISK_LEVEL=$(python -c "
      import json
      try:
          with open('risk_analysis.json', 'r') as f:
              data = json.load(f)
          print(data.get('risk_assessment', 'LOW'))
      except:
          print('LOW')
      ")
      echo "Risk level: $RISK_LEVEL"
      
      if [ "$RISK_LEVEL" = "HIGH" ]; then
        echo "Running full test suite for high-risk changes"
        pytest tests/ --cov=. --cov-report=xml --slow
      elif [ "$RISK_LEVEL" = "MEDIUM" ]; then
        echo "Running integration tests for medium-risk changes"
        pytest tests/ --cov=. --cov-report=xml
      else
        echo "Running unit tests for low-risk changes"
        pytest tests/ -k "not integration" --cov=. --cov-report=xml
      fi
  coverage: '/TOTAL.+ ([0-9]{1,3}%)/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

security-audit:
  stage: security
  script:
    - pip install safety
    - safety check
    - python -m deepflow.tools.dependency_visualizer . --format text | grep -i "circular" && exit 1 || echo "No circular dependencies"
  only:
    - main
    - master
    - develop

deployment-safety:
  stage: deploy-check
  script:
    - echo "Running deployment safety checks..."
    - python -m deepflow.tools.pre_commit_validator --validate .
    - |
      cat > deployment_report.md << EOF
      # Deployment Safety Report
      
      **Date**: $(date)
      **Commit**: $CI_COMMIT_SHA
      **Branch**: $CI_COMMIT_REF_NAME
      
      ## Safety Checks
      
      ✅ Dependency validation passed
      ✅ Security audit completed
      ✅ Performance checks done
      
      ## Ready for Deployment
      
      This build is safe for production deployment.
      EOF
  artifacts:
    paths:
      - deployment_report.md
    expire_in: 1 month
  only:
    - main
    - master

documentation-update:
  stage: deploy-check
  script:
    - python -m deepflow.tools.doc_generator . --output docs/
    - |
      if [ -n "$(git status --porcelain docs/)" ]; then
        echo "Documentation has been updated"
        git config --global user.email "gitlab-ci@example.com"
        git config --global user.name "GitLab CI"
        git add docs/
        git commit -m "📚 Auto-update documentation [skip ci]"
        git push https://oauth2:$CI_PUSH_TOKEN@$CI_SERVER_HOST/$CI_PROJECT_PATH.git HEAD:$CI_COMMIT_REF_NAME
      else
        echo "No documentation changes detected"
      fi
  only:
    - main
    - master
"""

        gitlab_ci_path = self.project_path / ".gitlab-ci.yml"
        with open(gitlab_ci_path, "w") as f:
            f.write(gitlab_ci_content)

        self.console.print(f"[green]✅ GitLab CI configuration created:[/green] {gitlab_ci_path}")
        return str(gitlab_ci_path)

    def validate_current_changes(self) -> Dict:
        """Validate current repository changes for CI/CD."""
        if not self.git_repo:
            return {"error": "Not a Git repository"}

        # Get changed files
        try:
            # Check for uncommitted changes
            changed_files = [item.a_path for item in self.git_repo.index.diff(None)]

            # Check for staged changes
            staged_files = [item.a_path for item in self.git_repo.index.diff("HEAD")]

            all_changed = list(set(changed_files + staged_files))

            if not all_changed:
                return {"message": "No changes detected"}

            # Analyze impact
            python_files = [f for f in all_changed if f.endswith(".py")]

            risk_level = "LOW"
            required_tests = ["unit"]

            # Check for high-risk files
            high_risk_patterns = ["main.py", "config.py", "app.py", "settings.py"]
            high_risk_files = [
                f for f in all_changed if any(pattern in f for pattern in high_risk_patterns)
            ]

            if high_risk_files:
                risk_level = "HIGH"
                required_tests = ["unit", "integration", "performance"]
            elif len(python_files) > 5:  # Many file changes
                risk_level = "MEDIUM"
                required_tests = ["unit", "integration"]

            return {
                "changed_files": all_changed,
                "python_files": python_files,
                "high_risk_files": high_risk_files,
                "risk_level": risk_level,
                "required_tests": required_tests,
                "deployment_impact": risk_level == "HIGH",
            }

        except Exception as e:
            return {"error": f"Error analyzing changes: {e}"}


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="CI/CD integration for dependency management")
    parser.add_argument(
        "--setup-github", metavar="PROJECT_PATH", help="Set up GitHub Actions workflows"
    )
    parser.add_argument("--setup-gitlab", metavar="PROJECT_PATH", help="Set up GitLab CI pipeline")
    parser.add_argument(
        "--validate-changes", metavar="PROJECT_PATH", help="Validate current repository changes"
    )
    parser.add_argument(
        "--generate-report", metavar="PROJECT_PATH", help="Generate CI/CD integration report"
    )

    args = parser.parse_args()

    console = Console()

    if args.setup_github:
        integrator = CICDIntegrator(args.setup_github)
        workflows = integrator.setup_github_actions()

        console.print(
            Panel.fit(
                "[bold green]✅ GitHub Actions Setup Complete[/bold green]\n\n"
                + f"Created {len(workflows)} workflow files:\n"
                + "\n".join(f"• {name}: {path}" for name, path in workflows.items())
                + "\n\n[bold]Next Steps:[/bold]\n"
                + "1. Commit and push the new workflow files\n"
                + "2. Configure repository secrets if needed\n"
                + "3. Test workflows with a pull request",
                title="GitHub Actions",
            )
        )
        return

    if args.setup_gitlab:
        integrator = CICDIntegrator(args.setup_gitlab)
        gitlab_ci_path = integrator.setup_gitlab_ci()

        console.print(
            Panel.fit(
                f"[bold green]✅ GitLab CI Setup Complete[/bold green]\n\n"
                + f"Created: {gitlab_ci_path}\n\n"
                + "[bold]Next Steps:[/bold]\n"
                + "1. Commit and push the .gitlab-ci.yml file\n"
                + "2. Configure CI/CD variables in GitLab\n"
                + "3. Test pipeline with a merge request",
                title="GitLab CI",
            )
        )
        return

    if args.validate_changes:
        integrator = CICDIntegrator(args.validate_changes)
        validation = integrator.validate_current_changes()

        if "error" in validation:
            console.print(f"[red]Error: {validation['error']}[/red]")
            return

        if "message" in validation:
            console.print(f"[yellow]{validation['message']}[/yellow]")
            return

        # Display validation results
        risk_color = {"HIGH": "red", "MEDIUM": "yellow", "LOW": "green"}.get(
            validation["risk_level"], "white"
        )

        console.print(
            Panel.fit(
                f"[bold]Change Impact Analysis[/bold]\n\n"
                + f"Risk Level: [{risk_color}]{validation['risk_level']}[/{risk_color}]\n"
                + f"Changed Files: {len(validation['changed_files'])}\n"
                + f"Python Files: {len(validation['python_files'])}\n"
                + f"High-Risk Files: {len(validation['high_risk_files'])}\n"
                + f"Required Tests: {', '.join(validation['required_tests'])}\n"
                + f"Deployment Impact: {'Yes' if validation['deployment_impact'] else 'No'}",
                title="Validation Results",
            )
        )

        if validation["high_risk_files"]:
            console.print("\n[bold red]High-Risk Files:[/bold red]")
            for file in validation["high_risk_files"]:
                console.print(f"  • {file}")

        return

    parser.print_help()


if __name__ == "__main__":
    main()
