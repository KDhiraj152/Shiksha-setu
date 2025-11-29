#!/usr/bin/env python3
"""
Database Migration Management Script

Provides utilities for:
- Auto-generating migrations from model changes
- Applying migrations with rollback support
- Migration status checking
- Database backup before migrations

Usage:
    python scripts/manage_migrations.py status
    python scripts/manage_migrations.py generate "add user preferences"
    python scripts/manage_migrations.py upgrade
    python scripts/manage_migrations.py downgrade --steps 1
    python scripts/manage_migrations.py backup
"""
import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_database_url():
    """Get database URL from environment."""
    from dotenv import load_dotenv
    load_dotenv()
    
    url = os.getenv('DATABASE_URL', '')
    if not url:
        print("❌ DATABASE_URL not set")
        sys.exit(1)
    return url


def run_alembic(args: list, capture_output: bool = False):
    """Run alembic command."""
    cmd = ["alembic", "-c", str(PROJECT_ROOT / "config" / "alembic.ini")] + args
    
    if capture_output:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
        return result.stdout, result.stderr, result.returncode
    else:
        return subprocess.run(cmd, cwd=PROJECT_ROOT)


def check_status():
    """Check current migration status."""
    print("🔍 Checking migration status...\n")
    
    # Current head
    print("Current database version:")
    run_alembic(["current"])
    
    print("\n" + "="*50 + "\n")
    
    # Pending migrations
    print("Migration history:")
    run_alembic(["history", "--verbose", "-r", "-3:"])
    
    print("\n" + "="*50 + "\n")
    
    # Check for pending
    stdout, stderr, code = run_alembic(["heads"], capture_output=True)
    heads = stdout.strip().split('\n') if stdout.strip() else []
    
    stdout, stderr, code = run_alembic(["current"], capture_output=True)
    current = stdout.strip().split('\n') if stdout.strip() else []
    
    if heads and current:
        head_revs = {h.split()[0] for h in heads if h}
        current_revs = {c.split()[0] for c in current if c and '(head)' not in c}
        
        if head_revs == current_revs or '(head)' in str(current):
            print("✅ Database is up to date")
        else:
            print(f"⚠️  Pending migrations available")
            print(f"   Current: {current_revs}")
            print(f"   Head: {head_revs}")


def generate_migration(message: str, autogenerate: bool = True):
    """Generate a new migration."""
    print(f"🔧 Generating migration: {message}\n")
    
    # Sanitize message for filename
    safe_message = message.lower().replace(' ', '_').replace('-', '_')
    safe_message = ''.join(c for c in safe_message if c.isalnum() or c == '_')[:50]
    
    args = ["revision", "-m", message]
    if autogenerate:
        args.append("--autogenerate")
    
    result = run_alembic(args)
    
    if result.returncode == 0:
        print("\n✅ Migration generated successfully")
        print("\n⚠️  Review the generated migration file before applying!")
    else:
        print("\n❌ Failed to generate migration")
        sys.exit(1)


def upgrade_database(revision: str = "head"):
    """Apply migrations."""
    print(f"⬆️  Upgrading database to: {revision}\n")
    
    result = run_alembic(["upgrade", revision])
    
    if result.returncode == 0:
        print("\n✅ Database upgraded successfully")
    else:
        print("\n❌ Failed to upgrade database")
        sys.exit(1)


def downgrade_database(steps: int = 1, revision: str = None):
    """Rollback migrations."""
    target = revision if revision else f"-{steps}"
    
    print(f"⬇️  Downgrading database by: {target}\n")
    
    # Confirm downgrade
    confirm = input("Are you sure you want to downgrade? This may cause data loss. (yes/no): ")
    if confirm.lower() != 'yes':
        print("Cancelled.")
        return
    
    result = run_alembic(["downgrade", target])
    
    if result.returncode == 0:
        print("\n✅ Database downgraded successfully")
    else:
        print("\n❌ Failed to downgrade database")
        sys.exit(1)


def backup_database():
    """Create database backup before migration."""
    print("💾 Creating database backup...\n")
    
    url = get_database_url()
    
    # Parse database URL
    if 'postgresql' in url:
        # Extract connection details
        # postgresql://user:pass@host:port/dbname
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = PROJECT_ROOT / "data" / "backups" / f"backup_{timestamp}.sql"
            backup_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Use pg_dump
            env = os.environ.copy()
            env['PGPASSWORD'] = parsed.password or ''
            
            cmd = [
                'pg_dump',
                '-h', parsed.hostname or 'localhost',
                '-p', str(parsed.port or 5432),
                '-U', parsed.username or 'postgres',
                '-d', parsed.path.lstrip('/'),
                '-f', str(backup_file),
                '--format=custom'
            ]
            
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Backup created: {backup_file}")
                return str(backup_file)
            else:
                print(f"⚠️  pg_dump not available or failed: {result.stderr}")
                
        except Exception as e:
            print(f"⚠️  Backup failed: {e}")
    
    elif 'sqlite' in url:
        # Simple file copy for SQLite
        import shutil
        
        db_path = url.replace('sqlite:///', '')
        if os.path.exists(db_path):
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f"{db_path}.backup_{timestamp}"
            shutil.copy2(db_path, backup_path)
            print(f"✅ Backup created: {backup_path}")
            return backup_path
        else:
            print("⚠️  SQLite database file not found")
    
    return None


def show_diff():
    """Show what changes would be made by autogenerate."""
    print("🔍 Checking for model changes...\n")
    
    # Generate migration but don't save
    stdout, stderr, code = run_alembic(
        ["revision", "--autogenerate", "-m", "temp_check", "--sql"],
        capture_output=True
    )
    
    if stdout:
        print("Detected changes:")
        print(stdout)
    else:
        print("No changes detected between models and database")


def verify_migrations():
    """Verify all migrations can be applied cleanly."""
    print("🔍 Verifying migrations...\n")
    
    # Check for multiple heads (branched migrations)
    stdout, stderr, code = run_alembic(["heads"], capture_output=True)
    heads = [h for h in stdout.strip().split('\n') if h]
    
    if len(heads) > 1:
        print("⚠️  Multiple migration heads detected!")
        print("   This indicates branched migrations that need to be merged.")
        for head in heads:
            print(f"   - {head}")
        return False
    
    # Verify migration chain
    stdout, stderr, code = run_alembic(["check"], capture_output=True)
    
    if "Target database is not up to date" in stdout or "Target database is not up to date" in stderr:
        print("⚠️  Database is not up to date with migrations")
        return False
    
    print("✅ Migrations are valid")
    return True


def main():
    parser = argparse.ArgumentParser(description="Database Migration Management")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Status command
    subparsers.add_parser("status", help="Show migration status")
    
    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate new migration")
    gen_parser.add_argument("message", help="Migration description")
    gen_parser.add_argument("--manual", action="store_true", help="Don't autogenerate")
    
    # Upgrade command
    up_parser = subparsers.add_parser("upgrade", help="Apply migrations")
    up_parser.add_argument("--revision", default="head", help="Target revision")
    up_parser.add_argument("--backup", action="store_true", help="Create backup first")
    
    # Downgrade command
    down_parser = subparsers.add_parser("downgrade", help="Rollback migrations")
    down_parser.add_argument("--steps", type=int, default=1, help="Number of steps")
    down_parser.add_argument("--revision", help="Target revision")
    down_parser.add_argument("--backup", action="store_true", help="Create backup first")
    
    # Backup command
    subparsers.add_parser("backup", help="Create database backup")
    
    # Diff command
    subparsers.add_parser("diff", help="Show pending model changes")
    
    # Verify command
    subparsers.add_parser("verify", help="Verify migration integrity")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print(f"  ShikshaSetu Database Migration Manager")
    print(f"{'='*60}\n")
    
    if args.command == "status":
        check_status()
    
    elif args.command == "generate":
        generate_migration(args.message, autogenerate=not args.manual)
    
    elif args.command == "upgrade":
        if args.backup:
            backup_database()
            print()
        upgrade_database(args.revision)
    
    elif args.command == "downgrade":
        if args.backup:
            backup_database()
            print()
        downgrade_database(steps=args.steps, revision=args.revision)
    
    elif args.command == "backup":
        backup_database()
    
    elif args.command == "diff":
        show_diff()
    
    elif args.command == "verify":
        verify_migrations()


if __name__ == "__main__":
    main()
