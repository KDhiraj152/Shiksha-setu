#!/usr/bin/env python
"""
Simple script to run Alembic migrations without connection pool caching issues.
"""
import os
import sys
import time

# Set environment variable
os.environ['DATABASE_URL'] = 'postgresql://shiksha_user:shiksha_pass@127.0.0.1:5432/shiksha_setu'

# Wait a bit for any connection resets
time.sleep(2)

# Import after setting env
from alembic.config import Config
from alembic import command

# Create Alembic configuration
alembic_cfg = Config("alembic.ini")
alembic_cfg.set_main_option("sqlalchemy.url", os.environ['DATABASE_URL'])

# Run upgrade
try:
    print("Running migrations...")
    command.upgrade(alembic_cfg, "head")
    print("✅ Migrations completed successfully!")
except Exception as e:
    print(f"❌ Migration failed: {e}")
    sys.exit(1)
