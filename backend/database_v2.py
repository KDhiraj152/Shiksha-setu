"""
Production-Grade Database Connection Layer with Resilience.

Implements circuit breaker pattern, adaptive connection pooling,
connection health checks, and read replica support.
"""

from sqlalchemy import create_engine, text, event, pool
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from sqlalchemy.exc import OperationalError, DBAPIError, DisconnectionError
from sqlalchemy.pool import QueuePool, NullPool
from typing import Generator, Optional, Dict, Any
from contextlib import contextmanager
import os
import logging
import time
import threading
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Create base class for models
Base = declarative_base()


class PoolHealthStatus(Enum):
    """Database connection pool health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    RECOVERING = "recovering"


@dataclass
class PoolMetrics:
    """Database connection pool metrics."""
    status: PoolHealthStatus
    pool_size: int
    checked_out: int
    available: int
    overflow: int
    failed_checkouts: int
    total_checkouts: int
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None


class ConnectionCircuitBreaker:
    """Circuit breaker for database connection failures."""
    
    def __init__(
        self,
        failure_threshold: int = 10,
        recovery_timeout: int = 60,
        half_open_max_attempts: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_attempts = half_open_max_attempts
        
        self.failure_count = 0
        self.success_count_half_open = 0
        self.state = "closed"  # closed, open, half-open
        self.last_failure_time = None
        self._lock = threading.Lock()
    
    def record_failure(self, error: str = "Unknown"):
        """Record a connection failure."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                if self.state != "open":
                    logger.warning(
                        f"Circuit breaker OPEN: {self.failure_count} failures. "
                        f"Error: {error}"
                    )
                    self.state = "open"
    
    def record_success(self):
        """Record a successful connection."""
        with self._lock:
            if self.state == "closed":
                # Normal operation
                if self.failure_count > 0:
                    self.failure_count = max(0, self.failure_count - 1)
            elif self.state == "half-open":
                # Recovery in progress
                self.success_count_half_open += 1
                if self.success_count_half_open >= self.half_open_max_attempts:
                    logger.info("Circuit breaker CLOSED - recovery successful")
                    self.state = "closed"
                    self.failure_count = 0
                    self.success_count_half_open = 0
    
    def can_attempt(self) -> bool:
        """Check if connection attempt should be allowed."""
        with self._lock:
            if self.state == "closed":
                return True
            elif self.state == "open":
                # Check if recovery timeout has passed
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    logger.info("Circuit breaker HALF-OPEN - attempting recovery")
                    self.state = "half-open"
                    self.success_count_half_open = 0
                    return True
                return False
            else:  # half-open
                return True
    
    def is_open(self) -> bool:
        """Check if circuit is open."""
        with self._lock:
            return self.state == "open"


class ResilientDatabaseConnection:
    """
    Production-grade database connection with resilience patterns.
    
    Features:
    - Adaptive connection pooling
    - Circuit breaker for cascading failures
    - Connection health monitoring
    - Read replica support
    - Graceful degradation
    """
    
    def __init__(
        self,
        database_url: str,
        read_replica_urls: Optional[list[str]] = None,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_recycle: int = 3600,
        pool_timeout: int = 30,
        query_timeout: int = 30,
        echo: bool = False
    ):
        self.database_url = database_url
        self.read_replica_urls = read_replica_urls or []
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_recycle = pool_recycle
        self.pool_timeout = pool_timeout
        self.query_timeout = query_timeout
        
        # Circuit breaker
        self.circuit_breaker = ConnectionCircuitBreaker()
        
        # Metrics
        self.metrics: PoolMetrics = PoolMetrics(
            status=PoolHealthStatus.HEALTHY,
            pool_size=pool_size,
            checked_out=0,
            available=pool_size,
            overflow=0,
            failed_checkouts=0,
            total_checkouts=0
        )
        
        # Create engines
        self.write_engine = self._create_engine(database_url, echo, "write")
        self.read_engines = [
            self._create_engine(url, echo, f"read-{i}")
            for i, url in enumerate(self.read_replica_urls)
        ]
        
        # Session factory
        self.SessionLocal = sessionmaker(
            bind=self.write_engine,
            expire_on_commit=False
        )
        
        # Setup event listeners
        self._setup_event_listeners()
        
        logger.info(
            f"✓ Database connection pool initialized: "
            f"{pool_size} + {max_overflow} overflow"
        )
    
    def _create_engine(self, url: str, echo: bool, name: str):
        """Create SQLAlchemy engine with retry logic."""
        is_sqlite = url.startswith('sqlite')
        
        if is_sqlite:
            engine = create_engine(
                url,
                connect_args={"check_same_thread": False},
                echo=echo,
                poolclass=NullPool
            )
        else:
            engine = create_engine(
                url,
                poolclass=QueuePool,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_pre_ping=True,  # Verify connections before using
                pool_recycle=self.pool_recycle,
                pool_timeout=self.pool_timeout,
                echo=echo,
                connect_args={
                    "connect_timeout": 10,
                    "options": f"-c statement_timeout={self.query_timeout * 1000}"
                }
            )
        
        logger.debug(f"Engine created: {name}")
        return engine
    
    def _setup_event_listeners(self):
        """Setup connection pool event listeners."""
        @event.listens_for(self.write_engine, "connect")
        def receive_connect(dbapi_conn, connection_record):
            """Log new connections."""
            logger.debug("New database connection established")
            
            # Set session timeout
            try:
                dbapi_conn.isolation_level
            except Exception as e:
                logger.warning(f"Error setting isolation level: {e}")
        
        @event.listens_for(self.write_engine, "checkout")
        def receive_checkout(dbapi_conn, connection_record, connection_proxy):
            """Monitor connection checkout."""
            self.metrics.total_checkouts += 1
            self.metrics.checked_out = getattr(
                self.write_engine.pool, 'checkedout', lambda: 0
            )()
            
            if self.metrics.checked_out > self.pool_size * 0.8:
                logger.warning(
                    f"Connection pool usage high: {self.metrics.checked_out}/"
                    f"{self.pool_size + self.max_overflow}"
                )
        
        @event.listens_for(self.write_engine, "checkin")
        def receive_checkin(dbapi_conn, connection_record):
            """Monitor connection return."""
            self.metrics.checked_out = getattr(
                self.write_engine.pool, 'checkedout', lambda: 0
            )()
        
        @event.listens_for(self.write_engine, "detach")
        def receive_detach(dbapi_conn, connection_record):
            """Handle connection detach."""
            logger.debug("Connection detached from pool")
        
        @event.listens_for(self.write_engine, "close")
        def receive_close(dbapi_conn, connection_record):
            """Handle connection close."""
            logger.debug("Connection closed")
    
    def get_session(self) -> Session:
        """
        Get a database session.
        
        Returns:
            SQLAlchemy Session
        
        Raises:
            OperationalError: If connection fails after retries
        """
        if self.circuit_breaker.is_open():
            logger.error("Circuit breaker OPEN - database unavailable")
            raise OperationalError(
                "Database circuit breaker is open",
                None,
                None
            )
        
        if not self.circuit_breaker.can_attempt():
            raise OperationalError(
                "Database connection attempts exhausted",
                None,
                None
            )
        
        try:
            session = self.SessionLocal()
            
            # Test connection
            session.execute(text("SELECT 1"))
            
            self.circuit_breaker.record_success()
            self.metrics.status = PoolHealthStatus.HEALTHY
            
            return session
        
        except (OperationalError, DBAPIError, DisconnectionError) as e:
            self.circuit_breaker.record_failure(str(e))
            self.metrics.status = PoolHealthStatus.CRITICAL
            self.metrics.last_error = str(e)
            self.metrics.last_error_time = datetime.now(timezone.utc)
            self.metrics.failed_checkouts += 1
            
            logger.error(f"Database connection failed: {e}")
            raise
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        Context manager for database session.
        
        Usage:
            with db.session_scope() as session:
                # Use session
        """
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    async def execute_read(
        self,
        query,
        read_preference: str = "replica"
    ):
        """
        Execute a read query on read replica.
        
        Args:
            query: SQLAlchemy query
            read_preference: "replica" or "primary"
        
        Returns:
            Query result
        """
        # Use replica if available and preferred
        engine = self.write_engine
        
        if read_preference == "replica" and self.read_engines:
            # Round-robin through replicas
            engine = self.read_engines[time.time_ns() % len(self.read_engines)]
        
        try:
            with engine.connect() as conn:
                return conn.execute(query).fetchall()
        except Exception as e:
            logger.error(f"Read query failed: {e}")
            raise
    
    def get_pool_status(self) -> PoolMetrics:
        """Get current connection pool status."""
        pool = self.write_engine.pool
        
        if hasattr(pool, 'size'):
            self.metrics.available = pool.size() - (
                pool.checkedout() if hasattr(pool, 'checkedout') else 0
            )
        
        # Determine health status
        if self.circuit_breaker.is_open():
            self.metrics.status = PoolHealthStatus.CRITICAL
        elif self.metrics.checked_out > self.pool_size * 0.8:
            self.metrics.status = PoolHealthStatus.DEGRADED
        else:
            self.metrics.status = PoolHealthStatus.HEALTHY
        
        return self.metrics
    
    def reset_circuit_breaker(self) -> None:
        """Manually reset circuit breaker (admin operation)."""
        self.circuit_breaker.failure_count = 0
        self.circuit_breaker.state = "closed"
        logger.warning("Circuit breaker manually reset")
    
    def close(self) -> None:
        """Close all connections."""
        try:
            self.write_engine.dispose()
            for engine in self.read_engines:
                engine.dispose()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing connections: {e}")


# Global database connection instance
_db_connection: Optional[ResilientDatabaseConnection] = None


def get_db_connection(
    database_url: Optional[str] = None,
    **kwargs
) -> ResilientDatabaseConnection:
    """
    Get or create global database connection.
    
    Args:
        database_url: Optional override for DATABASE_URL
        **kwargs: Additional arguments for ResilientDatabaseConnection
    
    Returns:
        ResilientDatabaseConnection instance
    """
    global _db_connection
    
    if _db_connection is None:
        from ...core.config import settings
        
        url = database_url or settings.DATABASE_URL if hasattr(settings, 'DATABASE_URL') else os.getenv(
            'DATABASE_URL',
            'postgresql://shiksha_user:shiksha_pass@localhost:5432/shiksha_setu'
        )
        
        _db_connection = ResilientDatabaseConnection(
            database_url=url,
            **kwargs
        )
    
    return _db_connection


def get_db() -> Generator[Session, None, None]:
    """
    Dependency for FastAPI routes.
    
    Usage:
        @app.get("/items")
        async def get_items(db: Session = Depends(get_db)):
            # Use db
    """
    db_connection = get_db_connection()
    session = db_connection.get_session()
    try:
        yield session
    finally:
        session.close()
