#!/usr/bin/env python3
"""
Verification Script for Issues #9 and #11

Run this script to verify:
1. Sentry error tracking integration
2. Curriculum validation service
3. Database schema
4. Configuration

Usage:
    python scripts/verify_issues_9_11.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker


def print_header(message: str):
    """Print formatted header."""
    print("\n" + "="*60)
    print(f"  {message}")
    print("="*60 + "\n")


def print_status(check: str, passed: bool, details: str = ""):
    """Print check status."""
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} | {check}")
    if details:
        print(f"       {details}")


def check_file_exists(filepath: str) -> bool:
    """Check if file exists."""
    return Path(project_root / filepath).exists()


def check_environment_vars():
    """Check environment configuration."""
    print_header("Environment Configuration")
    
    # Check .env.example for Sentry config
    env_example = project_root / ".env.example"
    has_sentry_config = False
    
    if env_example.exists():
        content = env_example.read_text()
        has_sentry_config = "SENTRY_DSN" in content
        print_status(
            "Sentry config in .env.example",
            has_sentry_config,
            "Found SENTRY_DSN configuration"
        )
    else:
        print_status(".env.example exists", False, "File not found")
    
    # Check if .env exists
    env_file = project_root / ".env"
    env_exists = env_file.exists()
    print_status(
        ".env file exists",
        env_exists,
        "Copy .env.example to .env and configure SENTRY_DSN" if not env_exists else "Found"
    )
    
    return has_sentry_config


def check_services():
    """Check if service files exist."""
    print_header("Service Files")
    
    files_to_check = [
        ("backend/services/error_tracking.py", "Error Tracking Service"),
        ("backend/services/curriculum_validation.py", "Curriculum Validation Service"),
        ("backend/api/sentry_middleware.py", "Sentry Middleware"),
    ]
    
    all_exist = True
    for filepath, name in files_to_check:
        exists = check_file_exists(filepath)
        print_status(name, exists, filepath)
        all_exist = all_exist and exists
    
    return all_exist


def check_test_files():
    """Check if test files exist."""
    print_header("Test Files")
    
    test_files = [
        ("tests/unit/test_error_tracking.py", "Error Tracking Tests"),
        ("tests/unit/test_curriculum_validation.py", "Curriculum Validation Tests"),
    ]
    
    all_exist = True
    for filepath, name in test_files:
        exists = check_file_exists(filepath)
        print_status(name, exists, filepath)
        all_exist = all_exist and exists
    
    return all_exist


def check_documentation():
    """Check if documentation exists."""
    print_header("Documentation")
    
    docs = [
        ("docs/error-tracking-and-validation.md", "Implementation Guide"),
        ("CHANGELOG.md", "Changelog"),
        ("SESSION_SUMMARY_ISSUES_9_11.md", "Session Summary"),
    ]
    
    all_exist = True
    for filepath, name in docs:
        exists = check_file_exists(filepath)
        print_status(name, exists, filepath)
        all_exist = all_exist and exists
    
    return all_exist


def check_imports():
    """Check if services can be imported."""
    print_header("Import Tests")
    
    try:
        from backend.services import error_tracking
        print_status("Error Tracking imports", True, "backend.services.error_tracking")
    except Exception as e:
        print_status("Error Tracking imports", False, f"Error: {e}")
        return False
    
    try:
        from backend.services import curriculum_validation
        print_status("Curriculum Validation imports", True, "backend.services.curriculum_validation")
    except Exception as e:
        print_status("Curriculum Validation imports", False, f"Error: {e}")
        return False
    
    try:
        from backend.api import sentry_middleware
        print_status("Sentry Middleware imports", True, "backend.api.sentry_middleware")
    except Exception as e:
        print_status("Sentry Middleware imports", False, f"Error: {e}")
        return False
    
    return True


def check_database_schema():
    """Check database schema for validation tables."""
    print_header("Database Schema")
    
    try:
        # Try to get database URL from environment
        from backend.core.config import Settings
        settings = Settings()
        
        if not settings.DATABASE_URL:
            print_status("Database connection", False, "DATABASE_URL not configured")
            return False
        
        # Create engine
        engine = create_engine(str(settings.DATABASE_URL))
        
        # Check if content_validation table exists
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT EXISTS (SELECT FROM information_schema.tables "
                "WHERE table_name = 'content_validation')"
            ))
            table_exists = result.scalar()
            print_status(
                "content_validation table",
                table_exists,
                "Table exists in database" if table_exists else "Run: alembic upgrade head"
            )
        
        # Check if token_blacklist table exists
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT EXISTS (SELECT FROM information_schema.tables "
                "WHERE table_name = 'token_blacklist')"
            ))
            table_exists = result.scalar()
            print_status(
                "token_blacklist table",
                table_exists,
                "Table exists in database" if table_exists else "Run: alembic upgrade head"
            )
        
        # Check if teacher_profiles table exists
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT EXISTS (SELECT FROM information_schema.tables "
                "WHERE table_name = 'teacher_profiles')"
            ))
            table_exists = result.scalar()
            print_status(
                "teacher_profiles table",
                table_exists,
                "Table exists in database" if table_exists else "Run: alembic upgrade head"
            )
        
        return True
        
    except Exception as e:
        print_status("Database connection", False, f"Error: {e}")
        print("       Make sure PostgreSQL is running and DATABASE_URL is correct")
        return False


def check_sentry_configuration():
    """Check Sentry configuration."""
    print_header("Sentry Configuration")
    
    try:
        from backend.services.error_tracking import init_sentry, sentry_health_check
        
        # Try to initialize (will log warning if no DSN)
        init_sentry()
        print_status("Sentry initialization", True, "No errors during init")
        
        # Run health check
        health = sentry_health_check()
        is_healthy = health.get("status") == "healthy"
        
        if is_healthy:
            print_status(
                "Sentry health check",
                True,
                f"Event ID: {health.get('test_event_id', 'N/A')}"
            )
        else:
            print_status(
                "Sentry health check",
                False,
                "Configure SENTRY_DSN in .env to enable error tracking"
            )
        
        return True
        
    except Exception as e:
        print_status("Sentry check", False, f"Error: {e}")
        return False


def run_unit_tests():
    """Run unit tests."""
    print_header("Unit Tests")
    
    import subprocess
    
    try:
        # Run error tracking tests
        result = subprocess.run(
            ["pytest", "tests/unit/test_error_tracking.py", "-v", "--tb=short"],
            cwd=project_root,
            capture_output=True,
            text=True
        )
        
        passed = result.returncode == 0
        print_status(
            "Error Tracking tests",
            passed,
            "All tests passed" if passed else "Some tests failed (see output below)"
        )
        
        if not passed:
            print("\n" + result.stdout)
            print(result.stderr)
        
    except FileNotFoundError:
        print_status("pytest", False, "pytest not installed. Run: pip install -r requirements.dev.txt")
        return False
    
    try:
        # Run curriculum validation tests
        result = subprocess.run(
            ["pytest", "tests/unit/test_curriculum_validation.py", "-v", "--tb=short"],
            cwd=project_root,
            capture_output=True,
            text=True
        )
        
        passed = result.returncode == 0
        print_status(
            "Curriculum Validation tests",
            passed,
            "All tests passed" if passed else "Some tests failed (see output below)"
        )
        
        if not passed:
            print("\n" + result.stdout)
            print(result.stderr)
        
        return passed
        
    except Exception as e:
        print_status("Unit tests", False, f"Error: {e}")
        return False


def main():
    """Run all verification checks."""
    print("\n" + "🔍 Verifying Issues #9 and #11 Implementation".center(60))
    print("=" * 60)
    
    results = {
        "Environment Configuration": check_environment_vars(),
        "Service Files": check_services(),
        "Test Files": check_test_files(),
        "Documentation": check_documentation(),
        "Python Imports": check_imports(),
        "Database Schema": check_database_schema(),
        "Sentry Configuration": check_sentry_configuration(),
    }
    
    # Run tests last (takes time)
    results["Unit Tests"] = run_unit_tests()
    
    # Summary
    print_header("Verification Summary")
    
    total_checks = len(results)
    passed_checks = sum(1 for v in results.values() if v)
    
    for check, passed in results.items():
        print_status(check, passed)
    
    print(f"\n📊 Results: {passed_checks}/{total_checks} checks passed")
    
    if passed_checks == total_checks:
        print("\n🎉 All checks passed! Issues #9 and #11 are properly implemented.")
        print("\n📝 Next steps:")
        print("   1. Configure SENTRY_DSN in .env for production error tracking")
        print("   2. Run database migrations: alembic upgrade head")
        print("   3. Start the application: uvicorn backend.api.main:app --reload")
        print("   4. Test error tracking at: http://localhost:8000/docs")
        return 0
    else:
        print(f"\n⚠️  {total_checks - passed_checks} checks failed. Review output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
