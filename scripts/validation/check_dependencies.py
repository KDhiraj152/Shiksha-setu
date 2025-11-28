#!/usr/bin/env python3
"""
Dependency Checker for ShikshaSetu

Verifies all required dependencies are installed before testing.
"""

import sys
import importlib.util
from typing import List, Tuple, Dict
import subprocess

# Required dependencies grouped by category
DEPENDENCIES = {
    "Core Framework": [
        "fastapi",
        "uvicorn",
        "pydantic",
        "sqlalchemy",
        "alembic",
    ],
    "Database": [
        "psycopg2",
        "asyncpg",
        "redis",
    ],
    "Task Queue": [
        "celery",
        "kombu",
    ],
    "AI/ML": [
        "torch",
        "transformers",
        "sentence_transformers",
    ],
    "Testing": [
        "pytest",
        "pytest_asyncio",
        "pytest_cov",
        "httpx",
    ],
    "Monitoring": [
        "prometheus_client",
        "sentry_sdk",
    ],
    "Utilities": [
        "python_dotenv",
        "pyyaml",
        "numpy",
        "pandas",
    ]
}


def check_package(package_name: str) -> Tuple[bool, str]:
    """Check if a package is installed."""
    try:
        spec = importlib.util.find_spec(package_name)
        if spec is not None:
            # Try to get version
            try:
                module = importlib.import_module(package_name)
                version = getattr(module, '__version__', 'unknown')
                return True, version
            except Exception:
                return True, 'installed'
        return False, 'not found'
    except (ImportError, ModuleNotFoundError):
        return False, 'not found'


def check_pip_package(package_name: str) -> Tuple[bool, str]:
    """Check package via pip list."""
    try:
        result = subprocess.run(
            ['pip', 'list'],
            capture_output=True,
            text=True,
            check=True
        )
        for line in result.stdout.split('\n'):
            if line.lower().startswith(package_name.lower().replace('_', '-')):
                parts = line.split()
                if len(parts) >= 2:
                    return True, parts[1]
        return False, 'not found'
    except Exception as e:
        return False, f'error: {str(e)}'


def main():
    """Check all dependencies."""
    print("=" * 80)
    print("ShikshaSetu Dependency Checker")
    print("=" * 80)
    print()
    
    all_installed = True
    missing_packages = []
    results: Dict[str, List[Tuple[str, bool, str]]] = {}
    
    for category, packages in DEPENDENCIES.items():
        print(f"\n{category}:")
        print("-" * 40)
        
        category_results = []
        for package in packages:
            installed, version = check_package(package)
            
            # If not found via importlib, try pip
            if not installed:
                installed, version = check_pip_package(package)
            
            status = "✓" if installed else "✗"
            color = "\033[92m" if installed else "\033[91m"
            reset = "\033[0m"
            
            print(f"  {color}{status}{reset} {package:30s} {version}")
            
            category_results.append((package, installed, version))
            
            if not installed:
                all_installed = False
                missing_packages.append(package)
        
        results[category] = category_results
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total_packages = sum(len(pkgs) for pkgs in DEPENDENCIES.values())
    installed_count = total_packages - len(missing_packages)
    
    print(f"\nTotal Packages: {total_packages}")
    print(f"Installed: {installed_count}")
    print(f"Missing: {len(missing_packages)}")
    print(f"Coverage: {(installed_count/total_packages)*100:.1f}%")
    
    if missing_packages:
        print("\n\033[91mMissing Packages:\033[0m")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\n\033[93mTo install missing packages:\033[0m")
        print(f"  pip install -r requirements.txt")
        print()
        return 1
    else:
        print("\n\033[92m✓ All required dependencies are installed!\033[0m")
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
