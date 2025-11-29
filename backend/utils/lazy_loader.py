"""
Lazy Model Loader with Memory Management

Provides on-demand model loading with:
- Memory-aware caching
- Automatic model eviction when memory is low
- Graceful fallback to smaller models
- Thread-safe loading
"""
import gc
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, TypeVar
from functools import wraps

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class LoadedModel:
    """Container for a loaded model with metadata."""
    model: Any
    tokenizer: Any
    name: str
    memory_gb: float
    loaded_at: float
    last_used: float
    use_count: int = 0


class LazyModelLoader:
    """
    Thread-safe lazy model loader with memory management.
    
    Features:
    - Models loaded on first use
    - LRU eviction when memory is constrained
    - Automatic fallback to smaller models
    - Memory tracking and monitoring
    """
    
    def __init__(
        self,
        max_memory_gb: float = 8.0,
        eviction_threshold: float = 0.9,  # Start evicting at 90% capacity
        device: str = "cpu"
    ):
        """
        Initialize the lazy model loader.
        
        Args:
            max_memory_gb: Maximum memory budget for models
            eviction_threshold: Fraction of max memory before eviction starts
            device: Default device for model loading (cpu, cuda, mps)
        """
        self.max_memory_gb = max_memory_gb
        self.eviction_threshold = eviction_threshold
        self.device = device
        
        self._models: Dict[str, LoadedModel] = {}
        self._lock = threading.RLock()
        self._current_memory_gb = 0.0
        
        # Model size estimates (in GB)
        self._model_sizes = {
            "sentence-transformers/all-MiniLM-L6-v2": 0.1,
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 0.3,
            "google/flan-t5-small": 0.4,
            "google/flan-t5-base": 1.0,
            "google/flan-t5-large": 3.0,
            "ai4bharat/indictrans2-en-indic-dist-200M": 0.5,
            "ai4bharat/indictrans2-en-indic-1B": 2.0,
            "Qwen/Qwen2.5-1.5B-Instruct": 1.5,
            "Qwen/Qwen2.5-7B-Instruct": 4.0,  # With 4-bit quantization
        }
        
        logger.info(f"LazyModelLoader initialized: max_memory={max_memory_gb}GB, device={device}")
    
    @property
    def available_memory_gb(self) -> float:
        """Get available memory budget."""
        return self.max_memory_gb - self._current_memory_gb
    
    @property
    def loaded_models(self) -> Dict[str, float]:
        """Get dictionary of loaded model names and their memory usage."""
        with self._lock:
            return {name: m.memory_gb for name, m in self._models.items()}
    
    def estimate_model_size(self, model_name: str) -> float:
        """
        Estimate memory size for a model.
        
        Args:
            model_name: Model identifier (HuggingFace model ID)
            
        Returns:
            Estimated size in GB
        """
        # Check known sizes
        if model_name in self._model_sizes:
            return self._model_sizes[model_name]
        
        # Heuristic based on model name patterns
        name_lower = model_name.lower()
        
        if "mini" in name_lower or "tiny" in name_lower:
            return 0.2
        elif "small" in name_lower:
            return 0.5
        elif "base" in name_lower:
            return 1.0
        elif "large" in name_lower:
            return 3.0
        elif "xl" in name_lower:
            return 5.0
        elif "7b" in name_lower:
            return 4.0  # Quantized
        elif "13b" in name_lower or "14b" in name_lower:
            return 8.0  # Quantized
        else:
            return 1.0  # Default estimate
    
    def _should_evict(self) -> bool:
        """Check if eviction is needed based on memory usage."""
        return self._current_memory_gb > (self.max_memory_gb * self.eviction_threshold)
    
    def _evict_lru(self) -> bool:
        """
        Evict least recently used model.
        
        Returns:
            True if a model was evicted
        """
        with self._lock:
            if not self._models:
                return False
            
            # Find LRU model
            lru_name = min(
                self._models.keys(),
                key=lambda k: self._models[k].last_used
            )
            
            return self._unload_model(lru_name)
    
    def _unload_model(self, model_name: str) -> bool:
        """
        Unload a specific model.
        
        Args:
            model_name: Name of model to unload
            
        Returns:
            True if model was unloaded
        """
        with self._lock:
            if model_name not in self._models:
                return False
            
            loaded = self._models[model_name]
            memory_freed = loaded.memory_gb
            
            # Delete model and tokenizer
            del loaded.model
            if loaded.tokenizer:
                del loaded.tokenizer
            del self._models[model_name]
            
            # Update memory tracking
            self._current_memory_gb -= memory_freed
            
            # Force garbage collection
            gc.collect()
            
            # Clear GPU/MPS cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
            except ImportError:
                pass
            
            logger.info(f"Unloaded model '{model_name}', freed {memory_freed:.2f}GB")
            return True
    
    def load_model(
        self,
        model_name: str,
        loader_fn: Callable[[], tuple],
        memory_estimate: Optional[float] = None,
        force_reload: bool = False
    ) -> LoadedModel:
        """
        Load a model lazily with memory management.
        
        Args:
            model_name: Unique identifier for the model
            loader_fn: Function that returns (model, tokenizer) tuple
            memory_estimate: Optional memory estimate override
            force_reload: Force reload even if cached
            
        Returns:
            LoadedModel container
            
        Raises:
            MemoryError: If model cannot fit in available memory
        """
        with self._lock:
            # Check if already loaded
            if model_name in self._models and not force_reload:
                loaded = self._models[model_name]
                loaded.last_used = time.time()
                loaded.use_count += 1
                logger.debug(f"Using cached model '{model_name}'")
                return loaded
            
            # Estimate memory requirement
            required_memory = memory_estimate or self.estimate_model_size(model_name)
            
            # Check if we need to evict
            while self._current_memory_gb + required_memory > self.max_memory_gb:
                if not self._models:
                    raise MemoryError(
                        f"Cannot load model '{model_name}' ({required_memory:.2f}GB): "
                        f"exceeds max memory budget ({self.max_memory_gb}GB)"
                    )
                
                # Evict LRU model
                evicted = self._evict_lru()
                if not evicted:
                    raise MemoryError(
                        f"Cannot free enough memory for '{model_name}'"
                    )
            
            # Load the model
            logger.info(f"Loading model '{model_name}' (estimated {required_memory:.2f}GB)...")
            start_time = time.time()
            
            try:
                model, tokenizer = loader_fn()
                
                # Track actual memory if possible
                actual_memory = required_memory
                try:
                    import torch
                    if hasattr(model, 'get_memory_footprint'):
                        actual_memory = model.get_memory_footprint() / (1024 ** 3)
                except Exception:
                    pass
                
                loaded = LoadedModel(
                    model=model,
                    tokenizer=tokenizer,
                    name=model_name,
                    memory_gb=actual_memory,
                    loaded_at=time.time(),
                    last_used=time.time(),
                    use_count=1
                )
                
                self._models[model_name] = loaded
                self._current_memory_gb += actual_memory
                
                load_time = time.time() - start_time
                logger.info(
                    f"Loaded model '{model_name}' in {load_time:.2f}s "
                    f"(memory: {actual_memory:.2f}GB, total: {self._current_memory_gb:.2f}GB)"
                )
                
                return loaded
                
            except Exception as e:
                logger.error(f"Failed to load model '{model_name}': {e}")
                raise
    
    def get_model(self, model_name: str) -> Optional[LoadedModel]:
        """
        Get a loaded model if available.
        
        Args:
            model_name: Model identifier
            
        Returns:
            LoadedModel or None if not loaded
        """
        with self._lock:
            if model_name in self._models:
                loaded = self._models[model_name]
                loaded.last_used = time.time()
                loaded.use_count += 1
                return loaded
            return None
    
    def is_loaded(self, model_name: str) -> bool:
        """Check if a model is loaded."""
        return model_name in self._models
    
    def unload_all(self):
        """Unload all models and free memory."""
        with self._lock:
            model_names = list(self._models.keys())
            for name in model_names:
                self._unload_model(name)
            
            logger.info("All models unloaded")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get loader statistics."""
        with self._lock:
            return {
                "loaded_models": len(self._models),
                "memory_used_gb": round(self._current_memory_gb, 2),
                "memory_available_gb": round(self.available_memory_gb, 2),
                "memory_max_gb": self.max_memory_gb,
                "models": {
                    name: {
                        "memory_gb": round(m.memory_gb, 2),
                        "use_count": m.use_count,
                        "loaded_at": m.loaded_at,
                        "last_used": m.last_used,
                    }
                    for name, m in self._models.items()
                }
            }


# Global instance
_loader: Optional[LazyModelLoader] = None
_loader_lock = threading.Lock()


def get_lazy_loader(
    max_memory_gb: Optional[float] = None,
    device: Optional[str] = None
) -> LazyModelLoader:
    """
    Get or create the global lazy model loader.
    
    Args:
        max_memory_gb: Optional memory limit override
        device: Optional device override
        
    Returns:
        LazyModelLoader instance
    """
    global _loader
    
    with _loader_lock:
        if _loader is None:
            # Get defaults from config
            try:
                from ..core.config import settings
                max_memory = max_memory_gb or getattr(settings, 'MAX_MODEL_MEMORY_GB', 8.0)
            except ImportError:
                max_memory = max_memory_gb or 8.0
            
            # Detect device
            if device is None:
                try:
                    import torch
                    if torch.cuda.is_available():
                        device = "cuda"
                    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                        device = "mps"
                    else:
                        device = "cpu"
                except ImportError:
                    device = "cpu"
            
            _loader = LazyModelLoader(
                max_memory_gb=max_memory,
                device=device
            )
        
        return _loader


def lazy_load(
    model_name: str,
    memory_estimate: Optional[float] = None
):
    """
    Decorator for lazy model loading.
    
    Usage:
        @lazy_load("my-model", memory_estimate=1.0)
        def get_my_model():
            return load_model(), load_tokenizer()
        
        # First call loads the model, subsequent calls return cached
        model = get_my_model()
    """
    def decorator(loader_fn: Callable[[], tuple]):
        @wraps(loader_fn)
        def wrapper(*args, **kwargs) -> LoadedModel:
            loader = get_lazy_loader()
            return loader.load_model(
                model_name=model_name,
                loader_fn=loader_fn,
                memory_estimate=memory_estimate
            )
        return wrapper
    return decorator
