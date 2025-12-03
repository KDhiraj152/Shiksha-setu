"""
ShikshaSetu Main Application - V2 Optimized API Only
=====================================================

Streamlined application using ONLY the optimized V2 API.
All legacy V1 routes have been consolidated into the v2_api module.

Features:
- Single unified API at /api/v2/*
- Native Apple Silicon (M4) optimization
- Multi-tier caching (L1 memory, L2 Redis)
- Concurrent processing with batching
- No middleware patching - native optimizations
- OpenTelemetry distributed tracing
- Circuit breakers for resilience
- Consistent validation error handling
- API versioning headers
- Configurable policy engine for content filtering

Policy Modes (controlled via ALLOW_UNRESTRICTED_MODE env var):
- RESTRICTED (default): Full curriculum/safety enforcement
- UNRESTRICTED: Educational filters bypassed, system safety active
- EXTERNAL_ALLOWED: Unrestricted + external API calls enabled
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..core.config import settings
from ..core.exceptions import ShikshaSetuException
from ..utils.logging import setup_logging
from ..services.error_tracking import init_sentry
from .middleware import (
    RequestIDMiddleware,
    SecurityHeadersMiddleware,
    RequestTimingMiddleware,
    RequestLoggingMiddleware,
    exception_handler,
    generic_exception_handler
)
from .sentry_middleware import SentryContextMiddleware
from .validation_middleware import register_validation_handlers
from .version_middleware import APIVersionMiddleware
from ..core.optimized.rate_limiter import RateLimitMiddleware as UnifiedRateLimitMiddleware
from ..cache import get_redis

# Policy module for configurable content filtering
try:
    from ..policy import get_policy_engine, print_startup_banner
    _POLICY_AVAILABLE = True
except ImportError:
    _POLICY_AVAILABLE = False
    def print_startup_banner():
        pass

# V2 API - Modular router structure
from .routes.v2 import router as v2_router

from .metrics import metrics_endpoint

# Initialize logging
logger = setup_logging()

# Initialize FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="Production-grade multilingual education content processing with AI/ML pipeline - V2 Optimized API",
    version="2.0.0",
    debug=settings.DEBUG,
    docs_url="/docs",
    redoc_url="/redoc"
)

# ==================== MIDDLEWARE CONFIGURATION ====================
# Order matters - they execute in reverse order of addition

# GZip compression for responses > 500 bytes (reduces bandwidth ~60-80%)
from starlette.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=500, compresslevel=6)

# Request ID middleware (first - generates ID before other middleware needs it)
app.add_middleware(RequestIDMiddleware)

# Sentry context middleware
app.add_middleware(SentryContextMiddleware)

# Security headers middleware
app.add_middleware(SecurityHeadersMiddleware)

# API versioning middleware
app.add_middleware(APIVersionMiddleware)

# Timing middleware
app.add_middleware(RequestTimingMiddleware)

# Logging middleware (only in development)
if settings.ENVIRONMENT != "production":
    app.add_middleware(RequestLoggingMiddleware)

# CORS middleware (added last to apply first)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=settings.ALLOW_CREDENTIALS,
    allow_methods=settings.ALLOWED_METHODS,
    allow_headers=settings.ALLOWED_HEADERS,
    expose_headers=["X-Process-Time", "X-RateLimit-Limit", "X-RateLimit-Remaining", "X-API-Version"]
)

# Rate limiting middleware (production ready)
if settings.RATE_LIMIT_ENABLED:
    try:
        redis_client = get_redis()
        app.add_middleware(UnifiedRateLimitMiddleware, redis_client=redis_client)
        logger.info("Rate limiting enabled with Redis backend")
    except Exception as e:
        logger.warning(f"Failed to initialize Redis for rate limiting, using in-memory: {e}")
        app.add_middleware(UnifiedRateLimitMiddleware, redis_client=None)
        logger.info("Rate limiting enabled with in-memory backend")

# ==================== EXCEPTION HANDLERS ====================
app.add_exception_handler(ShikshaSetuException, exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# Register validation error handlers for consistent error format
register_validation_handlers(app)

# ==================== ROUTE REGISTRATION ====================
# V2 Modular API - all endpoints at /api/v2/*
app.include_router(v2_router, prefix="/api/v2")

logger.info("V2 Modular API registered - all endpoints at /api/v2/*")

# ==================== ROOT & METRICS ENDPOINTS ====================

@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint with API info."""
    # Include policy mode in root response
    policy_mode = "unknown"
    if _POLICY_AVAILABLE:
        try:
            engine = get_policy_engine()
            policy_mode = engine.mode.value
        except Exception:
            pass
    
    return {
        "name": settings.APP_NAME,
        "version": "2.0.0",
        "api_version": "v2",
        "docs": "/docs",
        "health": "/api/v2/health",
        "status": "operational",
        "policy_mode": policy_mode
    }


@app.get("/metrics", include_in_schema=False)
async def metrics():
    """Prometheus metrics endpoint."""
    return metrics_endpoint()


# ==================== LIFECYCLE EVENTS ====================

@app.on_event("startup")
async def startup_event():
    """Application startup tasks."""
    logger.info(f"Starting {settings.APP_NAME} v2.0.0")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug mode: {settings.DEBUG}")
    
    # Print Policy Engine startup banner (FIRST for visibility)
    if _POLICY_AVAILABLE:
        try:
            print_startup_banner()
        except Exception as e:
            logger.warning(f"Policy banner failed: {e}")
    
    # Initialize Global Memory Coordinator FIRST (before any model loading)
    try:
        from ..core.optimized.memory_coordinator import get_memory_coordinator
        memory_coordinator = get_memory_coordinator()
        app.state.memory_coordinator = memory_coordinator
        
        # Register memory pressure callback for graceful degradation
        def on_memory_pressure(current_usage: float, threshold: float):
            logger.warning(
                f"Memory pressure detected: {current_usage:.1f}GB / {threshold:.1f}GB - "
                "triggering model eviction"
            )
        
        memory_coordinator.register_pressure_callback(on_memory_pressure)
        logger.info(
            f"Memory coordinator initialized: "
            f"{memory_coordinator.available_memory_gb:.1f}GB available for models"
        )
    except Exception as e:
        logger.warning(f"Memory coordinator initialization failed: {e}")
    
    # Initialize OpenTelemetry tracing
    try:
        from ..core.tracing import init_tracing
        init_tracing()
        logger.info("OpenTelemetry tracing initialized")
    except Exception as e:
        logger.warning(f"Tracing initialization failed (continuing without tracing): {e}")
    
    # Initialize circuit breakers
    try:
        from ..core.circuit_breaker import (
            get_database_breaker,
            get_redis_breaker,
            get_ml_breaker
        )
        get_database_breaker()
        get_redis_breaker()
        get_ml_breaker()
        logger.info("Circuit breakers initialized (database, redis, ml_model)")
    except Exception as e:
        logger.warning(f"Circuit breaker initialization failed: {e}")
    
    # Initialize Sentry error tracking
    try:
        init_sentry()
        logger.info("Sentry error tracking initialized")
    except Exception as e:
        logger.warning(f"Sentry initialization failed: {e}")
    
    # Validate JWT secret
    try:
        if not settings.SECRET_KEY or len(settings.SECRET_KEY) < 32:
            msg = "JWT secret key is too short or missing. Use a strong secret (>=32 chars)."
            if settings.ENVIRONMENT == "production":
                logger.error(msg)
                raise RuntimeError(msg)
            else:
                logger.warning(msg)
    except Exception as e:
        logger.exception(f"Startup validation error: {e}")
        raise
    
    # Initialize database
    from ..database import init_db
    try:
        init_db()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
    
    # Initialize device router and cache
    try:
        from ..core.optimized import DeviceRouter
        from ..cache.unified import UnifiedCache
        
        device_router = DeviceRouter()
        caps = device_router.capabilities
        logger.info(f"Device router initialized: {caps.chip_name} ({caps.device_type}) with {caps.memory_gb:.1f}GB memory")
        
        app.state.device_router = device_router
        app.state.cache = UnifiedCache()
        logger.info("Unified multi-tier cache initialized")
    except Exception as e:
        logger.warning(f"Optimized components initialization failed: {e}")
    
    # Initialize GPU Pipeline Scheduler
    try:
        from ..core.optimized.gpu_pipeline import get_gpu_scheduler
        
        scheduler = get_gpu_scheduler()
        app.state.gpu_scheduler = scheduler
        logger.info("GPU Pipeline Scheduler initialized (will start with warmup)")
    except Exception as e:
        logger.warning(f"GPU Scheduler initialization failed: {e}")
    
    # Start memory coordinator background monitor
    try:
        if hasattr(app.state, 'memory_coordinator'):
            import asyncio
            asyncio.create_task(app.state.memory_coordinator.start_monitor(interval=10.0))
            logger.info("Memory coordinator monitor started")
    except Exception as e:
        logger.warning(f"Memory monitor start failed: {e}")
    
    # Background model warm-up
    if settings.ENVIRONMENT != "test":
        import asyncio
        app.state.warmup_task = asyncio.create_task(_warm_up_models_async())
    
    logger.info("V2 API startup complete - all systems operational")


async def _warm_up_models_async():
    """Background task to warm up AI models with progressive batching for M4.
    
    Progressive warmup strategy (memory-coordinated):
    1. MLX LLM - single inference to prime Metal buffers (~4.5GB)
    2. BGE-M3 Embedder - progressive batch sizes (1 -> 8 -> 32 -> 64) (~2.0GB)
    3. Reranker - optional, loaded on-demand (~1.5GB)
    
    This eliminates cold-start latency spikes for production traffic.
    Memory coordinator ensures we stay within 16GB unified memory budget.
    """
    import time
    import asyncio
    import gc
    
    await asyncio.sleep(0.5)  # Let server start first
    
    total_start = time.perf_counter()
    
    # Get memory coordinator for coordinated loading
    try:
        from ..core.optimized.memory_coordinator import get_memory_coordinator
        coordinator = get_memory_coordinator()
    except Exception as e:
        logger.warning(f"Memory coordinator not available for warmup: {e}")
        coordinator = None
    
    # 1. Pre-load MLX LLM model (with memory coordination)
    start = time.perf_counter()
    logger.info("Starting MLX model pre-load for Apple Silicon...")
    
    try:
        # Acquire memory slot before loading
        if coordinator:
            acquired = await coordinator.acquire("llm")
            if not acquired:
                logger.warning("Could not acquire memory for LLM - skipping warmup")
                raise RuntimeError("Memory acquisition failed")
        
        from ..services.inference import get_inference_engine
        
        engine = get_inference_engine(auto_load=True)
        app.state.inference_engine = engine
        
        elapsed = time.perf_counter() - start
        logger.info(f"✓ MLX model pre-loaded in {elapsed:.2f}s")
        
    except Exception as e:
        logger.warning(f"MLX model pre-load failed: {e}")
        if coordinator:
            coordinator.release("llm")  # Release on failure
    
    # Force GC between model loads to keep memory clean
    gc.collect()
    
    # 2. Pre-load RAG embedding model (BGE-M3) with progressive warmup
    start2 = time.perf_counter()
    logger.info("Starting RAG embedder pre-load with progressive warmup...")
    
    try:
        # Acquire memory slot before loading
        if coordinator:
            acquired = await coordinator.acquire("embedder")
            if not acquired:
                logger.warning("Could not acquire memory for embedder - skipping warmup")
                raise RuntimeError("Memory acquisition failed")
        
        from ..services.rag import get_embedder
        
        embedder = get_embedder()
        embedder._load_model()  # Force immediate model load
        
        # Progressive batch warmup for optimal Metal buffer allocation
        # This prevents first-request latency spikes
        warmup_texts = ["warmup test"] * 64
        for batch_size in [1, 8, 32, 64]:
            batch = warmup_texts[:batch_size]
            _ = embedder.encode(batch, batch_size=batch_size)
            logger.debug(f"  Embedder warmup: batch_size={batch_size}")
        
        elapsed2 = time.perf_counter() - start2
        logger.info(f"✓ RAG embedder pre-loaded in {elapsed2:.2f}s - ready for fast search")
        
    except Exception as e:
        logger.warning(f"RAG embedder pre-load failed (will load on first request): {e}")
        if coordinator:
            coordinator.release("embedder")  # Release on failure
    
    # 3. Pre-initialize AI Engine with RAG service
    try:
        from ..services.ai_core.engine import get_ai_engine
        
        ai_engine = get_ai_engine()
        ai_engine._ensure_initialized()
        app.state.ai_engine = ai_engine
        
        logger.info("✓ AI Engine with RAG initialized")
        
    except Exception as e:
        logger.warning(f"AI Engine pre-init failed: {e}")
    
    # 4. Warm up TTS for common Indian languages (M4 optimization)
    # This eliminates 2-5s latency on first TTS request per language
    try:
        from ..services.tts.mms_tts_service import get_mms_tts_service, is_mms_available
        
        if is_mms_available():
            tts_service = get_mms_tts_service()
            # Warm up Hindi and English (most common), others load on demand
            tts_service.warmup(languages=['hi', 'en'])
            logger.info("✓ TTS warmed up for common languages (hi, en)")
    except Exception as e:
        logger.warning(f"TTS warmup failed (will load on first request): {e}")
    
    # Log memory status after warmup
    if coordinator:
        stats = coordinator.get_memory_stats()
        logger.info(
            f"Memory after warmup: {stats['used_memory_gb']:.1f}GB used, "
            f"{stats['available_memory_gb']:.1f}GB available, "
            f"models loaded: {list(stats['loaded_models'].keys())}"
        )
    
    # 4. Initialize and start GPU Pipeline Scheduler with queues
    try:
        from ..core.optimized.gpu_pipeline import get_gpu_scheduler, QueuePriority
        
        scheduler = get_gpu_scheduler()
        
        # Register inference queue if engine is available
        if hasattr(app.state, 'inference_engine') and app.state.inference_engine:
            engine = app.state.inference_engine
            
            def llm_executor(batch):
                """Execute LLM inference batch."""
                results = []
                for prompt in batch:
                    result = engine.generate_sync(prompt, max_tokens=512)
                    results.append(result)
                return results
            
            scheduler.register_queue(
                name="llm",
                executor=llm_executor,
                max_batch_size=4,  # LLM batching limited by memory
                max_wait_ms=50.0,
            )
        
        # Register embedding queue
        if embedder is not None:
            def embed_executor(batch):
                """Execute embedding batch."""
                return embedder.encode(batch)
            
            scheduler.register_queue(
                name="embedding",
                executor=embed_executor,
                max_batch_size=64,
                max_wait_ms=10.0,
            )
        
        # Start all queues
        await scheduler.start_all()
        logger.info("✓ GPU Pipeline Scheduler started with queues")
        
    except Exception as e:
        logger.warning(f"GPU Scheduler queue setup failed: {e}")
    
    total_elapsed = time.perf_counter() - total_start
    logger.info(f"✓ All models warmed up in {total_elapsed:.2f}s - system ready for production traffic")


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks."""
    logger.info(f"Shutting down {settings.APP_NAME}")
    
    # Cancel warmup task
    if hasattr(app.state, 'warmup_task') and app.state.warmup_task:
        if not app.state.warmup_task.done():
            app.state.warmup_task.cancel()
            logger.info("Warmup task cancelled")
    
    # Stop GPU Pipeline Scheduler first (drains queues gracefully)
    if hasattr(app.state, 'gpu_scheduler') and app.state.gpu_scheduler:
        try:
            await app.state.gpu_scheduler.stop_all()
            logger.info("GPU Pipeline Scheduler stopped")
        except Exception as e:
            logger.warning(f"GPU Scheduler shutdown failed: {e}")
    
    # Gracefully unload all models via memory coordinator
    if hasattr(app.state, 'memory_coordinator') and app.state.memory_coordinator:
        try:
            coordinator = app.state.memory_coordinator
            loaded_models = [
                name for name, info in coordinator._loaded_models.items()
            ]
            if loaded_models:
                logger.info(f"Unloading {len(loaded_models)} models: {loaded_models}")
            
            # Unload inference engine
            if hasattr(app.state, 'inference_engine') and app.state.inference_engine:
                try:
                    app.state.inference_engine.unload()
                    logger.info("Inference engine unloaded")
                except Exception as e:
                    logger.warning(f"Inference engine unload failed: {e}")
            
            # Unload RAG embedder and reranker
            try:
                from ..services.rag import get_embedder, get_reranker
                
                embedder = get_embedder()
                if embedder.is_loaded:
                    embedder.unload()
                    logger.info("RAG embedder unloaded")
                
                reranker = get_reranker()
                if reranker.is_loaded:
                    reranker.unload()
                    logger.info("RAG reranker unloaded")
            except Exception as e:
                logger.warning(f"RAG model unload failed: {e}")
            
            # Unload TTS models
            try:
                from ..services.tts.mms_tts_service import unload_mms_tts_service
                unload_mms_tts_service()
                logger.info("TTS models unloaded")
            except Exception as e:
                logger.warning(f"TTS unload failed: {e}")
            
            # Force final memory cleanup
            coordinator.force_cleanup()
            
            final_stats = coordinator.get_memory_stats()
            logger.info(f"Final memory stats: {final_stats}")
        except Exception as e:
            logger.warning(f"Memory coordinator shutdown failed: {e}")
    
    # Log cache stats
    if hasattr(app.state, 'cache') and app.state.cache:
        try:
            stats = app.state.cache.get_stats()
            logger.info(f"Cache stats at shutdown: {stats}")
        except Exception:
            pass
    
    logger.info("V2 API shutdown complete")


# Export
__all__ = ["app"]
