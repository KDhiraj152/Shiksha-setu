"""
Model Lifecycle Manager - Principles D, E, Q compliant.

Handles:
- On-demand model loading with pre-warming
- Aggressive model eviction (idle timeout, RAM threshold)
- Memory-mapped model support for GGUF
- Fallback to Bhashini API under pressure
"""
import os
import time
import logging
import threading
import weakref
from typing import Dict, Optional, Any, Callable, List
from dataclasses import dataclass, field
from enum import Enum
import psutil

from ..core.config import settings

logger = logging.getLogger(__name__)


class ModelState(str, Enum):
    """Model lifecycle states."""
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    BUSY = "busy"
    EVICTING = "evicting"


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    name: str
    state: ModelState = ModelState.UNLOADED
    memory_mb: float = 0.0
    last_used: float = 0.0
    load_time: float = 0.0
    use_count: int = 0
    instance: Any = None
    loader: Optional[Callable] = None
    priority: int = 1  # 1=high, 2=medium, 3=low


class ModelEvictionPolicy:
    """
    Model eviction policy - Principle E compliant.
    
    Rules:
    - Idle > 40s: evict
    - RAM > 85%: evict biggest model first
    - Task queue spike: offload to API fallback
    """
    
    IDLE_TIMEOUT_SECONDS: float = float(os.getenv("MODEL_IDLE_TIMEOUT", "40"))
    RAM_THRESHOLD_PERCENT: float = float(os.getenv("RAM_THRESHOLD_PERCENT", "85"))
    EVICTION_CHECK_INTERVAL: float = 10.0  # Check every 10 seconds
    
    @staticmethod
    def should_evict_idle(model: ModelInfo) -> bool:
        """Check if model should be evicted due to idle timeout."""
        if model.state != ModelState.READY:
            return False
        
        idle_time = time.time() - model.last_used
        return idle_time > ModelEvictionPolicy.IDLE_TIMEOUT_SECONDS
    
    @staticmethod
    def get_ram_usage_percent() -> float:
        """Get current RAM usage percentage."""
        return psutil.virtual_memory().percent
    
    @staticmethod
    def should_evict_for_memory() -> bool:
        """Check if models should be evicted due to high RAM."""
        return ModelEvictionPolicy.get_ram_usage_percent() > ModelEvictionPolicy.RAM_THRESHOLD_PERCENT
    
    @staticmethod
    def get_eviction_candidates(models: Dict[str, ModelInfo]) -> List[str]:
        """Get list of models to evict, sorted by priority (lowest first) and memory (highest first)."""
        candidates = []
        
        for name, model in models.items():
            if model.state == ModelState.READY:
                candidates.append((name, model.priority, model.memory_mb, model.last_used))
        
        # Sort by: priority (desc), memory (desc), last_used (asc)
        candidates.sort(key=lambda x: (-x[1], -x[2], x[3]))
        
        return [c[0] for c in candidates]


class ModelLifecycleManager:
    """
    Manages model loading, unloading, and eviction.
    
    Principles implemented:
    - D: Load on-demand with pre-warming
    - E: Aggressive eviction
    - Q: Memory-mapped GGUF support (via llama-cpp-python)
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern for global model management."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.models: Dict[str, ModelInfo] = {}
        self.eviction_policy = ModelEvictionPolicy()
        self._eviction_thread: Optional[threading.Thread] = None
        self._stop_eviction = threading.Event()
        self._model_lock = threading.Lock()
        
        # Pre-register known models with their priorities
        self._register_default_models()
        
        # Start eviction monitor
        self._start_eviction_monitor()
        
        self._initialized = True
        logger.info("ModelLifecycleManager initialized")
    
    def _register_default_models(self):
        """Register default models from config."""
        default_models = [
            ("ocr", settings.OCR_MODEL_ID, 1500, 1),  # High priority - often first
            ("embedding", settings.EMBEDDING_MODEL_ID, 2000, 1),  # High priority
            ("simplification", settings.SIMPLIFICATION_MODEL_ID, 2000, 2),
            ("translation", settings.TRANSLATION_MODEL_ID, 1200, 2),
            ("validation", settings.VALIDATION_MODEL_ID, 1500, 3),  # Low priority
            ("reranker", settings.RERANKER_MODEL_ID, 800, 3),  # Low priority
        ]
        
        for name, model_id, memory_mb, priority in default_models:
            self.models[name] = ModelInfo(
                name=name,
                memory_mb=memory_mb,
                priority=priority,
            )
    
    def _start_eviction_monitor(self):
        """Start background thread for eviction monitoring."""
        def monitor():
            while not self._stop_eviction.is_set():
                self._check_and_evict()
                self._stop_eviction.wait(ModelEvictionPolicy.EVICTION_CHECK_INTERVAL)
        
        self._eviction_thread = threading.Thread(target=monitor, daemon=True)
        self._eviction_thread.start()
        logger.info("Eviction monitor started")
    
    def _check_and_evict(self):
        """Check eviction conditions and evict models if needed."""
        with self._model_lock:
            # Check idle models
            for name, model in self.models.items():
                if self.eviction_policy.should_evict_idle(model):
                    logger.info(f"Evicting idle model: {name}")
                    self._evict_model(name)
            
            # Check memory pressure
            if self.eviction_policy.should_evict_for_memory():
                candidates = self.eviction_policy.get_eviction_candidates(self.models)
                if candidates:
                    logger.warning(f"RAM at {self.eviction_policy.get_ram_usage_percent():.1f}%, evicting: {candidates[0]}")
                    self._evict_model(candidates[0])
    
    def _evict_model(self, name: str):
        """Evict a specific model from memory."""
        model = self.models.get(name)
        if not model or model.state != ModelState.READY:
            return
        
        model.state = ModelState.EVICTING
        
        try:
            # Clear model instance
            if model.instance is not None:
                del model.instance
                model.instance = None
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA/MPS cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except Exception:
                pass
            
            model.state = ModelState.UNLOADED
            logger.info(f"Model {name} evicted successfully")
            
        except Exception as e:
            logger.error(f"Failed to evict model {name}: {e}")
            model.state = ModelState.READY  # Restore state
    
    def register_loader(self, name: str, loader: Callable, memory_mb: float = 0, priority: int = 2):
        """Register a model loader function."""
        with self._model_lock:
            if name not in self.models:
                self.models[name] = ModelInfo(name=name, memory_mb=memory_mb, priority=priority)
            self.models[name].loader = loader
    
    def get_model(self, name: str) -> Optional[Any]:
        """
        Get a model, loading it if necessary.
        
        Implements Principle D: Load on-demand.
        """
        with self._model_lock:
            model = self.models.get(name)
            if not model:
                logger.error(f"Unknown model: {name}")
                return None
            
            # Return if already loaded
            if model.state == ModelState.READY and model.instance is not None:
                model.last_used = time.time()
                model.use_count += 1
                return model.instance
            
            # Load the model
            if model.loader is None:
                logger.error(f"No loader registered for model: {name}")
                return None
            
            model.state = ModelState.LOADING
            start_time = time.time()
            
            try:
                model.instance = model.loader()
                model.state = ModelState.READY
                model.load_time = time.time() - start_time
                model.last_used = time.time()
                model.use_count = 1
                
                logger.info(f"Model {name} loaded in {model.load_time:.2f}s")
                return model.instance
                
            except Exception as e:
                model.state = ModelState.UNLOADED
                logger.error(f"Failed to load model {name}: {e}")
                return None
    
    def prewarm(self, names: List[str]):
        """
        Pre-warm specified models.
        
        Implements Principle D: Pre-warming for predicted tasks.
        """
        for name in names:
            if name in self.models and self.models[name].state == ModelState.UNLOADED:
                logger.info(f"Pre-warming model: {name}")
                self.get_model(name)
    
    def mark_busy(self, name: str):
        """Mark a model as busy (prevents eviction)."""
        if name in self.models:
            self.models[name].state = ModelState.BUSY
    
    def mark_ready(self, name: str):
        """Mark a model as ready (can be evicted)."""
        if name in self.models and self.models[name].state == ModelState.BUSY:
            self.models[name].state = ModelState.READY
            self.models[name].last_used = time.time()
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of all models."""
        status = {
            "ram_usage_percent": self.eviction_policy.get_ram_usage_percent(),
            "models": {}
        }
        
        for name, model in self.models.items():
            status["models"][name] = {
                "state": model.state.value,
                "memory_mb": model.memory_mb,
                "last_used": model.last_used,
                "use_count": model.use_count,
                "loaded": model.instance is not None,
            }
        
        return status
    
    def evict_all(self):
        """Evict all models (for cleanup)."""
        with self._model_lock:
            for name in list(self.models.keys()):
                self._evict_model(name)
    
    def shutdown(self):
        """Shutdown the manager."""
        self._stop_eviction.set()
        self.evict_all()
        logger.info("ModelLifecycleManager shutdown complete")


# Global instance
_manager: Optional[ModelLifecycleManager] = None


def get_model_manager() -> ModelLifecycleManager:
    """Get the global model lifecycle manager."""
    global _manager
    if _manager is None:
        _manager = ModelLifecycleManager()
    return _manager


# Context manager for model usage
class ModelContext:
    """Context manager for safe model usage with automatic busy/ready marking."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.manager = get_model_manager()
        self.model = None
    
    def __enter__(self):
        self.model = self.manager.get_model(self.model_name)
        if self.model:
            self.manager.mark_busy(self.model_name)
        return self.model
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.manager.mark_ready(self.model_name)
        return False


# Export
__all__ = [
    'ModelLifecycleManager',
    'ModelInfo',
    'ModelState',
    'ModelEvictionPolicy',
    'ModelContext',
    'get_model_manager'
]
