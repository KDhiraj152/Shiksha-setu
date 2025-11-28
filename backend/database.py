"""
Simplified database connection and session management.

Supports both local PostgreSQL and Supabase.
Uses FastAPI's Depends pattern for session management.
"""

from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import OperationalError, DBAPIError
from typing import Generator
from contextlib import contextmanager
import os
import logging
import time

logger = logging.getLogger(__name__)

# Create base class for models
Base = declarative_base()

# Database configuration - supports Supabase or local PostgreSQL
DATABASE_URL = os.getenv(
    'DATABASE_URL',
    'postgresql://shiksha_user:shiksha_pass@localhost:5432/shiksha_setu'
)

# Supabase configuration (optional)
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

# Create engine with connection pooling and retry logic
# Handle SQLite vs PostgreSQL differences
is_sqlite = DATABASE_URL.startswith('sqlite')

if is_sqlite:
    # SQLite configuration
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=os.getenv('SQL_ECHO', 'false').lower() == 'true'
    )
else:
    # PostgreSQL configuration
    engine = create_engine(
        DATABASE_URL,
        poolclass=QueuePool,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,  # Verify connections before using
        pool_recycle=3600,  # Recycle connections after 1 hour
        pool_timeout=30,  # Timeout for getting connection from pool
        echo=os.getenv('SQL_ECHO', 'false').lower() == 'true',
        connect_args={
            "connect_timeout": 10,
            "options": "-c statement_timeout=30000"  # 30s query timeout
        }
    )


# Add connection pool event listeners for monitoring
@event.listens_for(engine, "connect")
def receive_connect(dbapi_conn, connection_record):
    """Log new connections."""
    logger.debug("New database connection established")


@event.listens_for(engine, "checkout")
def receive_checkout(dbapi_conn, connection_record, connection_proxy):
    """Log connection checkout from pool."""
    logger.debug(f"Connection checked out from pool (size: {engine.pool.size()})")


@event.listens_for(engine, "checkin")
def receive_checkin(dbapi_conn, connection_record):
    """Log connection return to pool."""
    logger.debug("Connection returned to pool")


# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Get database session for FastAPI Depends with retry logic.
    
    Usage:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        db = None
        try:
            db = SessionLocal()
            yield db
            db.commit()
            return
        except (OperationalError, DBAPIError) as e:
            if db:
                db.rollback()
            
            if attempt < max_retries - 1:
                logger.warning(
                    f"Database operation failed (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error(f"Database operation failed after {max_retries} attempts: {e}")
                raise
        except Exception as e:
            if db:
                db.rollback()
            logger.error(f"Unexpected database error: {e}")
            raise
        finally:
            if db:
                db.close()


@contextmanager
def get_db_session():
    """
    Get database session as a context manager with retry logic.
    
    Usage:
        with get_db_session() as session:
            session.query(Item).all()
    """
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        session = None
        try:
            session = SessionLocal()
            yield session
            session.commit()
            return
        except (OperationalError, DBAPIError) as e:
            if session:
                session.rollback()
            
            if attempt < max_retries - 1:
                logger.warning(f"Database operation failed, retrying... ({e})")
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise
        except Exception:
            if session:
                session.rollback()
            raise
        finally:
            if session:
                session.close()


def create_tables():
    """Create all database tables"""
    Base.metadata.create_all(bind=engine)


def init_db():
    """
    Initialize database - create all tables and enable pgvector.
    Safe to call multiple times (idempotent).
    """
    # Test database connection first
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            logger.info("Database connection successful")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        logger.error(f"DATABASE_URL: {DATABASE_URL[:30]}...")
        raise
    
    # Try to enable pgvector extension for Supabase/PostgreSQL
    try:
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
            logger.info("pgvector extension enabled")
    except Exception as e:
        logger.warning(f"Could not enable pgvector extension: {e}")
        logger.warning("RAG/Q&A features will have limited functionality without pgvector")
    
    # Create all tables
    create_tables()
    logger.info("Database tables created/verified")


def drop_tables():
    """Drop all database tables"""
    Base.metadata.drop_all(bind=engine)


def get_supabase_client():
    """
    Get Supabase client for additional features (storage, realtime, etc.).
    Returns None if Supabase credentials are not configured.
    """
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            from supabase import create_client
            return create_client(SUPABASE_URL, SUPABASE_KEY)
        except ImportError:
            logger.warning("supabase package not installed")
            return None
    return None


__all__ = ['Base', 'engine', 'SessionLocal', 'get_db', 'get_db_session', 'create_tables', 'init_db', 'drop_tables', 'get_supabase_client']
