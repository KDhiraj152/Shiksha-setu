"""
Health check endpoints.

Provides basic and detailed health checks for monitoring and orchestration.
"""
import time
import platform
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class ComponentHealth(BaseModel):
    """Health status of a single component."""
    status: str  # "healthy", "degraded", "unhealthy"
    latency_ms: Optional[float] = None
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Complete health response."""
    status: str
    timestamp: str
    version: str
    uptime_seconds: float
    components: Dict[str, ComponentHealth]


# Track startup time
_startup_time = time.time()


def _check_database() -> ComponentHealth:
    """Check database connectivity and performance."""
    start = time.time()
    try:
        from ...core.database import SessionLocal
        from sqlalchemy import text
        
        db = SessionLocal()
        try:
            result = db.execute(text("SELECT 1"))
            result.fetchone()
            
            # Check connection pool stats if available
            pool_info = {}
            try:
                engine = db.get_bind()
                pool = engine.pool
                pool_info = {
                    "pool_size": pool.size(),
                    "checked_in": pool.checkedin(),
                    "checked_out": pool.checkedout(),
                    "overflow": pool.overflow(),
                }
            except Exception:
                pass
            
            latency = (time.time() - start) * 1000
            return ComponentHealth(
                status="healthy" if latency < 100 else "degraded",
                latency_ms=round(latency, 2),
                message="Connected",
                details=pool_info if pool_info else None
            )
        finally:
            db.close()
            
    except Exception as e:
        return ComponentHealth(
            status="unhealthy",
            latency_ms=round((time.time() - start) * 1000, 2),
            message=f"Connection failed: {str(e)[:100]}"
        )


def _check_redis() -> ComponentHealth:
    """Check Redis connectivity."""
    start = time.time()
    try:
        from ...core.cache import get_redis
        
        client = get_redis()
        if client is None:
            return ComponentHealth(
                status="degraded",
                message="Redis not configured"
            )
        
        client.ping()
        info = client.info("memory")
        
        latency = (time.time() - start) * 1000
        return ComponentHealth(
            status="healthy" if latency < 50 else "degraded",
            latency_ms=round(latency, 2),
            message="Connected",
            details={
                "used_memory_human": info.get("used_memory_human", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
            }
        )
    except Exception as e:
        return ComponentHealth(
            status="unhealthy",
            latency_ms=round((time.time() - start) * 1000, 2),
            message=f"Connection failed: {str(e)[:100]}"
        )


def _check_celery() -> ComponentHealth:
    """Check Celery worker connectivity."""
    start = time.time()
    try:
        from ...tasks.celery_app import celery_app
        
        # Check if any workers are alive
        inspector = celery_app.control.inspect(timeout=2)
        active_workers = inspector.active()
        
        if active_workers is None:
            return ComponentHealth(
                status="unhealthy",
                latency_ms=round((time.time() - start) * 1000, 2),
                message="No workers responding"
            )
        
        worker_count = len(active_workers)
        task_count = sum(len(tasks) for tasks in active_workers.values())
        
        latency = (time.time() - start) * 1000
        return ComponentHealth(
            status="healthy",
            latency_ms=round(latency, 2),
            message=f"{worker_count} worker(s) active",
            details={
                "workers": worker_count,
                "active_tasks": task_count,
            }
        )
    except Exception as e:
        return ComponentHealth(
            status="degraded",
            latency_ms=round((time.time() - start) * 1000, 2),
            message=f"Cannot reach workers: {str(e)[:100]}"
        )


def _check_ml_models() -> ComponentHealth:
    """Check ML model availability."""
    start = time.time()
    try:
        from ...core.model_tier_router import get_router
        
        router = get_router()
        if router is None:
            return ComponentHealth(
                status="degraded",
                message="Model router not initialized"
            )
        
        # Get loaded models info
        loaded_models = getattr(router, 'loaded_models', {})
        available_memory = getattr(router, 'available_memory_gb', 0)
        
        latency = (time.time() - start) * 1000
        return ComponentHealth(
            status="healthy",
            latency_ms=round(latency, 2),
            message=f"{len(loaded_models)} model(s) loaded",
            details={
                "models_loaded": list(loaded_models.keys()) if loaded_models else [],
                "available_memory_gb": round(available_memory, 2),
            }
        )
    except ImportError:
        return ComponentHealth(
            status="degraded",
            message="ML module not available"
        )
    except Exception as e:
        return ComponentHealth(
            status="degraded",
            latency_ms=round((time.time() - start) * 1000, 2),
            message=f"Check failed: {str(e)[:100]}"
        )


def _check_disk_space() -> ComponentHealth:
    """Check available disk space."""
    try:
        import shutil
        
        # Check data directory
        data_dir = os.environ.get("UPLOAD_DIR", "data/uploads")
        if not os.path.exists(data_dir):
            data_dir = "."
        
        usage = shutil.disk_usage(data_dir)
        free_gb = usage.free / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        used_percent = (usage.used / usage.total) * 100
        
        status = "healthy"
        if used_percent > 90:
            status = "unhealthy"
        elif used_percent > 80:
            status = "degraded"
        
        return ComponentHealth(
            status=status,
            message=f"{free_gb:.1f}GB free of {total_gb:.1f}GB",
            details={
                "free_gb": round(free_gb, 2),
                "total_gb": round(total_gb, 2),
                "used_percent": round(used_percent, 1),
            }
        )
    except Exception as e:
        return ComponentHealth(
            status="degraded",
            message=f"Check failed: {str(e)[:100]}"
        )


def _check_memory() -> ComponentHealth:
    """Check system memory usage."""
    try:
        import psutil
        
        memory = psutil.virtual_memory()
        used_percent = memory.percent
        available_gb = memory.available / (1024 ** 3)
        
        status = "healthy"
        if used_percent > 90:
            status = "unhealthy"
        elif used_percent > 80:
            status = "degraded"
        
        return ComponentHealth(
            status=status,
            message=f"{available_gb:.1f}GB available ({100-used_percent:.1f}% free)",
            details={
                "available_gb": round(available_gb, 2),
                "total_gb": round(memory.total / (1024 ** 3), 2),
                "used_percent": round(used_percent, 1),
            }
        )
    except ImportError:
        return ComponentHealth(
            status="degraded",
            message="psutil not installed"
        )
    except Exception as e:
        return ComponentHealth(
            status="degraded",
            message=f"Check failed: {str(e)[:100]}"
        )


@router.get("/health")
async def health_check():
    """
    Basic health check for load balancer/orchestrator probes.
    
    Returns 200 if the service is running and database is accessible.
    """
    from ...core.config import settings
    
    db_health = _check_database()
    
    return {
        "status": "healthy" if db_health.status == "healthy" else "degraded",
        "database": db_health.status,
        "version": settings.APP_VERSION,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@router.get("/health/detailed", response_model=HealthResponse)
async def detailed_health_check():
    """
    Detailed health check with all component statuses.
    
    Checks:
    - Database connectivity and pool stats
    - Redis connectivity and memory
    - Celery worker availability
    - ML model loading status
    - Disk space
    - System memory
    """
    from ...core.config import settings
    
    components = {
        "database": _check_database(),
        "redis": _check_redis(),
        "celery": _check_celery(),
        "ml_models": _check_ml_models(),
        "disk": _check_disk_space(),
        "memory": _check_memory(),
    }
    
    # Determine overall status
    statuses = [c.status for c in components.values()]
    if "unhealthy" in statuses:
        overall_status = "unhealthy"
    elif "degraded" in statuses:
        overall_status = "degraded"
    else:
        overall_status = "healthy"
    
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now(timezone.utc).isoformat(),
        version=settings.APP_VERSION,
        uptime_seconds=round(time.time() - _startup_time, 2),
        components=components
    )


@router.get("/health/ready")
async def readiness_check():
    """
    Kubernetes-style readiness probe.
    
    Returns 200 only if the service is ready to accept traffic.
    """
    db_health = _check_database()
    
    if db_health.status == "unhealthy":
        from fastapi import HTTPException
        raise HTTPException(status_code=503, detail="Database unavailable")
    
    return {"ready": True}


@router.get("/health/live")
async def liveness_check():
    """
    Kubernetes-style liveness probe.
    
    Returns 200 if the service is alive (not deadlocked).
    """
    return {"alive": True}


@router.get("/health/ai")
async def ai_services_health():
    """
    Health check for AI services (new optimized stack).
    
    Checks:
    - AI Orchestrator status
    - NLLB-200 translator (loaded/not loaded)
    - Ollama simplifier (llama3.2:3b availability)
    - Edge TTS connectivity
    - BGE-M3 embeddings model
    - Memory usage by AI services
    """
    try:
        from ...services.ai import get_ai_orchestrator
        
        orchestrator = await get_ai_orchestrator()
        status = orchestrator.get_status()
        health = await orchestrator.health_check()
        
        # Determine overall AI status
        service_statuses = list(health.get("services", {}).values())
        has_error = any("error" in str(s) for s in service_statuses)
        all_healthy = all(s in ["healthy", "not_loaded"] for s in service_statuses)
        
        if has_error:
            ai_status = "degraded"
        elif all_healthy:
            ai_status = "healthy"
        else:
            ai_status = "degraded"
        
        return {
            "status": ai_status,
            "orchestrator": "running" if status["running"] else "stopped",
            "memory": status["memory"],
            "services": health["services"],
            "config": {
                "device": status["config"]["device"],
                "compute_type": status["config"]["compute_type"],
                "max_memory_gb": status["config"]["max_memory_gb"],
            },
            "loaded_models": {
                "translator": status["services"]["translator_loaded"],
                "tts": status["services"]["tts_loaded"],
                "simplifier": status["services"]["simplifier_loaded"],
                "embeddings": status["services"]["embeddings_loaded"],
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unavailable",
            "error": str(e),
            "message": "AI orchestrator not initialized. Services will load on first request.",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


@router.post("/health/ai/preload")
async def preload_ai_services(
    services: list = None
):
    """
    Preload AI services into memory.
    
    Args:
        services: List of services to preload. Options:
            - "translation": Load NLLB-200 translator
            - "simplification": Load Ollama client
            - "tts": Initialize Edge TTS
            - "embeddings": Load BGE-M3 model
            
    If no services specified, all services will be preloaded.
    """
    try:
        from ...services.ai import get_ai_orchestrator
        
        orchestrator = await get_ai_orchestrator()
        loaded = []
        errors = []
        
        services_to_load = services or ["translation", "simplification", "tts", "embeddings"]
        
        for service in services_to_load:
            try:
                if service == "translation":
                    await orchestrator._ensure_translator()
                    loaded.append("translation")
                elif service == "simplification":
                    await orchestrator._ensure_simplifier()
                    loaded.append("simplification")
                elif service == "tts":
                    await orchestrator._ensure_tts()
                    loaded.append("tts")
                elif service == "embeddings":
                    await orchestrator._ensure_embeddings()
                    loaded.append("embeddings")
            except Exception as e:
                errors.append({"service": service, "error": str(e)})
        
        status = orchestrator.get_status()
        
        return {
            "success": len(errors) == 0,
            "loaded": loaded,
            "errors": errors if errors else None,
            "memory_usage": status["memory"]
        }
        
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=str(e))
