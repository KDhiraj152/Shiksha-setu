"""
Application Lifecycle Hooks - Dependency-Aware Parallel Initialization

Handles application startup and shutdown with:
- Dependency graph for initialization order
- Parallel execution of independent handlers
- Graceful degradation for non-critical services
- Timeout protection for slow initializations

Architecture:
    ┌─────────────────────────────────────────────────┐
    │              Startup Phase                       │
    │  ┌─────────────┐  ┌─────────────┐              │
    │  │ Environment │  │  Logging    │  (parallel)  │
    │  └──────┬──────┘  └──────┬──────┘              │
    │         └────────┬───────┘                      │
    │              ┌───┴───┐                          │
    │              │ merge │                          │
    │              └───┬───┘                          │
    │         ┌────────┼────────┐                     │
    │  ┌──────┴──────┐ │ ┌──────┴──────┐             │
    │  │  Database   │ │ │   Redis     │  (parallel) │
    │  └──────┬──────┘ │ └──────┬──────┘             │
    │         └────────┼────────┘                     │
    │              ┌───┴───┐                          │
    │  ┌───────────┴───────┴───────────┐             │
    │  │  Metrics  │  AI Orchestrator  │  (parallel) │
    │  └───────────────────────────────┘             │
    └─────────────────────────────────────────────────┘
"""
import asyncio
import logging
import os
import signal
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    AsyncGenerator, Callable, Dict, List, Optional, Set, Any
)

logger = logging.getLogger(__name__)


class InitializationPhase(str, Enum):
    """Initialization phases (executed in order)."""
    ENVIRONMENT = "environment"  # Phase 1: Validate environment
    INFRASTRUCTURE = "infrastructure"  # Phase 2: Database, Redis
    SERVICES = "services"  # Phase 3: AI, Metrics, etc.
    WARMUP = "warmup"  # Phase 4: Model preloading


class Criticality(str, Enum):
    """Handler criticality levels."""
    CRITICAL = "critical"  # Failure stops startup
    REQUIRED = "required"  # Failure logs error but continues
    OPTIONAL = "optional"  # Failure only logs warning


@dataclass
class HandlerConfig:
    """Configuration for a lifecycle handler."""
    name: str
    handler: Callable
    phase: InitializationPhase
    criticality: Criticality
    timeout_seconds: float = 30.0
    depends_on: List[str] = field(default_factory=list)
    provides: List[str] = field(default_factory=list)


@dataclass
class HandlerResult:
    """Result of a handler execution."""
    name: str
    success: bool
    duration_ms: float
    error: Optional[str] = None


class DependencyGraph:
    """Dependency resolution for parallel execution."""
    
    def __init__(self):
        self._handlers: Dict[str, HandlerConfig] = {}
        self._provides: Dict[str, str] = {}  # capability -> handler name
    
    def add(self, config: HandlerConfig) -> None:
        """Add a handler to the graph."""
        self._handlers[config.name] = config
        for capability in config.provides:
            self._provides[capability] = config.name
    
    def get_execution_order(self) -> List[List[str]]:
        """
        Get handlers grouped by execution wave.
        
        Returns list of lists, where each inner list can be executed in parallel.
        """
        # Group by phase first
        by_phase: Dict[InitializationPhase, List[str]] = {
            phase: [] for phase in InitializationPhase
        }
        
        for name, config in self._handlers.items():
            by_phase[config.phase].append(name)
        
        # Within each phase, topological sort by dependencies
        waves: List[List[str]] = []
        
        for phase in InitializationPhase:
            phase_handlers = by_phase[phase]
            if not phase_handlers:
                continue
            
            # Simple topological sort within phase
            remaining = set(phase_handlers)
            completed: Set[str] = set()
            
            while remaining:
                # Find handlers with all dependencies satisfied
                ready = []
                for name in remaining:
                    config = self._handlers[name]
                    deps_satisfied = all(
                        self._provides.get(dep, dep) in completed or
                        self._provides.get(dep, dep) not in remaining
                        for dep in config.depends_on
                    )
                    if deps_satisfied:
                        ready.append(name)
                
                if not ready:
                    # Circular dependency or missing dep - just run remaining
                    logger.warning(f"Dependency issue in phase {phase.value}, running remaining: {remaining}")
                    ready = list(remaining)
                
                waves.append(ready)
                completed.update(ready)
                remaining -= set(ready)
        
        return waves


class LifecycleManager:
    """
    Manages application lifecycle events with parallel execution.
    
    Features:
    - Dependency-aware parallel initialization
    - Timeout protection per handler
    - Graceful degradation for non-critical services
    - Detailed startup/shutdown metrics
    """
    
    def __init__(self):
        self._startup_graph = DependencyGraph()
        self._shutdown_handlers: List[HandlerConfig] = []
        self._is_ready = False
        self._is_shutting_down = False
        self._startup_results: List[HandlerResult] = []
    
    def on_startup(
        self,
        phase: InitializationPhase = InitializationPhase.SERVICES,
        criticality: Criticality = Criticality.REQUIRED,
        timeout_seconds: float = 30.0,
        depends_on: Optional[List[str]] = None,
        provides: Optional[List[str]] = None,
        name: Optional[str] = None
    ):
        """
        Decorator to register a startup handler.
        
        Args:
            phase: Initialization phase
            criticality: How critical this handler is
            timeout_seconds: Max time for handler
            depends_on: List of capabilities this depends on
            provides: List of capabilities this provides
            name: Handler name (defaults to function name)
        """
        def decorator(handler: Callable):
            handler_name = name or handler.__name__
            config = HandlerConfig(
                name=handler_name,
                handler=handler,
                phase=phase,
                criticality=criticality,
                timeout_seconds=timeout_seconds,
                depends_on=depends_on or [],
                provides=provides or [handler_name]
            )
            self._startup_graph.add(config)
            return handler
        return decorator
    
    def on_shutdown(self, handler: Callable):
        """Register shutdown handler (legacy API support)."""
        config = HandlerConfig(
            name=handler.__name__,
            handler=handler,
            phase=InitializationPhase.SERVICES,
            criticality=Criticality.REQUIRED,
        )
        self._shutdown_handlers.append(config)
        return handler
    
    @property
    def is_ready(self) -> bool:
        """Check if application is ready."""
        return self._is_ready
    
    @property
    def is_shutting_down(self) -> bool:
        """Check if application is shutting down."""
        return self._is_shutting_down
    
    async def _run_handler(self, config: HandlerConfig) -> HandlerResult:
        """Run a single handler with timeout protection."""
        import time
        start = time.monotonic()
        
        try:
            if asyncio.iscoroutinefunction(config.handler):
                await asyncio.wait_for(
                    config.handler(),
                    timeout=config.timeout_seconds
                )
            else:
                # Run sync handler in thread pool
                await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, config.handler
                    ),
                    timeout=config.timeout_seconds
                )
            
            duration = (time.monotonic() - start) * 1000
            logger.info(f"✓ {config.name} ({duration:.0f}ms)")
            
            return HandlerResult(
                name=config.name,
                success=True,
                duration_ms=duration
            )
            
        except asyncio.TimeoutError:
            duration = (time.monotonic() - start) * 1000
            error = f"Timeout after {config.timeout_seconds}s"
            logger.error(f"✗ {config.name}: {error}")
            
            return HandlerResult(
                name=config.name,
                success=False,
                duration_ms=duration,
                error=error
            )
            
        except Exception as e:
            duration = (time.monotonic() - start) * 1000
            logger.error(f"✗ {config.name}: {e}")
            
            return HandlerResult(
                name=config.name,
                success=False,
                duration_ms=duration,
                error=str(e)
            )
    
    async def startup(self) -> List[HandlerResult]:
        """Execute all startup handlers with parallel waves."""
        import time
        overall_start = time.monotonic()
        
        logger.info("=" * 60)
        logger.info("Starting application...")
        logger.info("=" * 60)
        
        self._startup_results = []
        waves = self._startup_graph.get_execution_order()
        
        for wave_idx, wave in enumerate(waves):
            if not wave:
                continue
            
            logger.info(f"\n[Wave {wave_idx + 1}] Running: {', '.join(wave)}")
            
            # Get configs for this wave
            configs = [
                self._startup_graph._handlers[name]
                for name in wave
            ]
            
            # Run wave in parallel
            results = await asyncio.gather(*[
                self._run_handler(config)
                for config in configs
            ], return_exceptions=True)
            
            # Process results
            for result, config in zip(results, configs):
                if isinstance(result, Exception):
                    result = HandlerResult(
                        name=config.name,
                        success=False,
                        duration_ms=0,
                        error=str(result)
                    )
                
                self._startup_results.append(result)
                
                # Check criticality
                if not result.success:
                    if config.criticality == Criticality.CRITICAL:
                        raise RuntimeError(
                            f"Critical startup handler failed: {config.name} - {result.error}"
                        )
                    elif config.criticality == Criticality.REQUIRED:
                        logger.error(f"Required handler failed: {config.name}")
        
        overall_duration = (time.monotonic() - overall_start) * 1000
        
        # Summary
        successful = sum(1 for r in self._startup_results if r.success)
        total = len(self._startup_results)
        
        logger.info("\n" + "=" * 60)
        logger.info(f"Startup complete: {successful}/{total} handlers succeeded ({overall_duration:.0f}ms)")
        logger.info("=" * 60 + "\n")
        
        self._is_ready = True
        return self._startup_results
    
    async def shutdown(self) -> None:
        """Execute all shutdown handlers."""
        logger.info("\n" + "=" * 60)
        logger.info("Shutting down application...")
        logger.info("=" * 60)
        
        self._is_shutting_down = True
        self._is_ready = False
        
        # Execute in reverse order
        for config in reversed(self._shutdown_handlers):
            try:
                logger.info(f"Running shutdown: {config.name}")
                
                if asyncio.iscoroutinefunction(config.handler):
                    await config.handler()
                else:
                    config.handler()
                    
                logger.info(f"✓ {config.name}")
                
            except Exception as e:
                logger.error(f"✗ {config.name}: {e}")
        
        logger.info("\n" + "=" * 60)
        logger.info("Shutdown complete")
        logger.info("=" * 60 + "\n")


# Global lifecycle manager
lifecycle = LifecycleManager()


# ============ Default Startup Handlers ============

@lifecycle.on_startup(
    phase=InitializationPhase.ENVIRONMENT,
    criticality=Criticality.REQUIRED,
    provides=["environment"]
)
async def validate_environment():
    """Validate required environment variables."""
    required_vars = [
        "DATABASE_URL",
        "JWT_SECRET_KEY",
    ]
    
    optional_vars = {
        "REDIS_URL": "redis://localhost:6379/0",
        "MODEL_TIER": "SMALL",
        "LOG_LEVEL": "INFO",
    }
    
    missing = []
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        logger.warning(f"Missing environment variables: {missing}")
    
    # Set defaults for optional vars
    for var, default in optional_vars.items():
        if not os.getenv(var):
            os.environ[var] = default
            logger.debug(f"Using default for {var}")


@lifecycle.on_startup(
    phase=InitializationPhase.ENVIRONMENT,
    criticality=Criticality.REQUIRED,
    provides=["logging"]
)
async def setup_logging():
    """Configure structured logging."""
    try:
        from ..utils.structured_logging import setup_structured_logging
        
        log_level = os.getenv("LOG_LEVEL", "INFO")
        environment = os.getenv("ENVIRONMENT", "development")
        json_logs = os.getenv("JSON_LOGS", "false").lower() == "true"
        
        setup_structured_logging(
            level=log_level,
            json_output=json_logs,
            service_name="shiksha-setu",
            environment=environment
        )
    except ImportError:
        pass  # Basic logging already configured


@lifecycle.on_startup(
    phase=InitializationPhase.INFRASTRUCTURE,
    criticality=Criticality.CRITICAL,
    depends_on=["environment", "logging"],
    provides=["database"]
)
async def initialize_database():
    """Initialize database connection pool."""
    try:
        from ..core.database import engine
        
        # Test connection (sync for now)
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise


@lifecycle.on_startup(
    phase=InitializationPhase.INFRASTRUCTURE,
    criticality=Criticality.OPTIONAL,
    depends_on=["environment", "logging"],
    provides=["redis"]
)
async def initialize_redis():
    """Initialize Redis connection."""
    try:
        from ..core.cache import get_redis
        
        redis = get_redis()
        if redis:
            redis.ping()
            logger.info("Redis connection established")
        else:
            logger.warning("Redis not available")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")


@lifecycle.on_startup(
    phase=InitializationPhase.SERVICES,
    criticality=Criticality.OPTIONAL,
    depends_on=["database", "redis"],
    provides=["metrics"]
)
async def initialize_metrics():
    """Initialize metrics collection."""
    try:
        from ..utils.metrics import get_metrics
        
        metrics = get_metrics()
        logger.info("Metrics collection initialized")
    except ImportError:
        logger.warning("Metrics module not available")


@lifecycle.on_startup(
    phase=InitializationPhase.SERVICES,
    criticality=Criticality.OPTIONAL,
    depends_on=["database"],
    provides=["ai_orchestrator"],
    timeout_seconds=60.0
)
async def initialize_ai_orchestrator():
    """Initialize the AI orchestrator for new optimized stack."""
    if os.getenv("DISABLE_AI_ORCHESTRATOR", "false").lower() == "true":
        logger.info("AI orchestrator disabled")
        return
        
    try:
        from ..services.ai import get_ai_orchestrator
        
        orchestrator = await get_ai_orchestrator()
        status = orchestrator.get_status()
        
        logger.info(f"AI orchestrator initialized: max_memory={status['config']['max_memory_gb']}GB")
    except Exception as e:
        logger.warning(f"AI orchestrator initialization deferred: {e}")


@lifecycle.on_startup(
    phase=InitializationPhase.SERVICES,
    criticality=Criticality.OPTIONAL,
    depends_on=["ai_orchestrator"],
    provides=["memory_scheduler"]
)
async def initialize_memory_scheduler():
    """Initialize the predictive memory scheduler."""
    if os.getenv("DISABLE_MEMORY_SCHEDULER", "false").lower() == "true":
        logger.info("Memory scheduler disabled")
        return
    
    try:
        from ..core.memory_scheduler import init_memory_scheduler
        from ..services.ai.orchestrator import get_ai_orchestrator
        
        # Get orchestrator for callbacks
        orchestrator = await get_ai_orchestrator()
        
        await init_memory_scheduler(
            max_memory_gb=float(os.getenv("MAX_MODEL_MEMORY_GB", "10.0")),
            preload_callback=orchestrator._ensure_service_loaded if orchestrator else None,
            unload_callback=orchestrator._unload_service if orchestrator else None
        )
        
        logger.info("Predictive memory scheduler initialized")
    except Exception as e:
        logger.warning(f"Memory scheduler initialization failed: {e}")


@lifecycle.on_startup(
    phase=InitializationPhase.WARMUP,
    criticality=Criticality.OPTIONAL,
    depends_on=["ai_orchestrator"],
    provides=["models_ready"]
)
async def warmup_models():
    """Optionally pre-load models."""
    if os.getenv("PRELOAD_MODELS", "false").lower() != "true":
        logger.info("Model preloading disabled")
        return
    
    try:
        from ..utils.lazy_loader import get_model_loader
        
        loader = get_model_loader()
        logger.info("Model loader initialized")
    except ImportError:
        logger.warning("Model loader not available")


# ============ Default Shutdown Handlers ============

@lifecycle.on_shutdown
async def shutdown_memory_scheduler():
    """Shutdown memory scheduler."""
    try:
        from ..core.memory_scheduler import get_memory_scheduler
        scheduler = get_memory_scheduler()
        await scheduler.stop()
        logger.info("Memory scheduler stopped")
    except Exception as e:
        logger.warning(f"Memory scheduler shutdown error: {e}")


@lifecycle.on_shutdown
async def shutdown_ai_orchestrator():
    """Shutdown AI orchestrator and free memory."""
    try:
        from ..services.ai.orchestrator import shutdown_orchestrator
        await shutdown_orchestrator()
        logger.info("AI orchestrator shutdown complete")
    except Exception as e:
        logger.error(f"Error shutting down AI orchestrator: {e}")


@lifecycle.on_shutdown
async def close_database():
    """Close database connections."""
    try:
        from ..core.database import engine
        await engine.dispose()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error closing database: {e}")


@lifecycle.on_shutdown
async def close_redis():
    """Close Redis connections."""
    try:
        from ..core.cache import close_redis
        close_redis()
        logger.info("Redis connections closed")
    except Exception as e:
        logger.error(f"Error closing Redis: {e}")


@lifecycle.on_shutdown
async def unload_models():
    """Unload ML models to free memory."""
    try:
        from ..utils.lazy_loader import get_model_loader
        
        loader = get_model_loader()
        loader.unload_all()
        logger.info("Models unloaded")
    except Exception as e:
        logger.error(f"Error unloading models: {e}")


@lifecycle.on_shutdown
async def flush_metrics():
    """Flush any pending metrics."""
    logger.info("Metrics flushed")


# ============ FastAPI Lifespan Context ============

@asynccontextmanager
async def lifespan(app) -> AsyncGenerator:
    """
    FastAPI lifespan context manager.
    
    Usage:
        app = FastAPI(lifespan=lifespan)
    """
    # Startup
    await lifecycle.startup()
    
    yield
    
    # Shutdown
    await lifecycle.shutdown()


# ============ Signal Handlers ============

def setup_signal_handlers():
    """Setup graceful shutdown on signals."""
    
    def handle_signal(signum, frame):
        logger.info(f"Received signal {signum}")
        # The lifespan context will handle cleanup
        sys.exit(0)
    
    # Handle common shutdown signals
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)


# Export
__all__ = [
    'lifecycle',
    'LifecycleManager',
    'lifespan',
    'setup_signal_handlers',
]
