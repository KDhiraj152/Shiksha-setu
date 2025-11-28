"""Device detection and management for ML models across platforms."""
import os
import logging
import torch
from typing import Literal, Optional, Tuple

logger = logging.getLogger(__name__)

DeviceType = Literal["cuda", "mps", "cpu"]


class DeviceManager:
    """Manages device selection and validation for ML model inference."""
    
    def __init__(self):
        """Initialize device manager with automatic detection."""
        self._device: Optional[DeviceType] = None
        self._device_count: int = 0
        self._device_name: Optional[str] = None
        self._total_memory: int = 0
        
        self._detect_device()
    
    def _detect_device(self) -> None:
        """Detect available compute devices in priority order: CUDA > MPS > CPU."""
        # Check CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            self._device = "cuda"
            self._device_count = torch.cuda.device_count()
            self._device_name = torch.cuda.get_device_name(0)
            self._total_memory = torch.cuda.get_device_properties(0).total_memory
            logger.info(
                f"CUDA detected: {self._device_count}x {self._device_name} "
                f"({self._total_memory / 1e9:.1f}GB total)"
            )
            return
        
        # Check MPS (Apple Silicon)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = "mps"
            self._device_count = 1
            self._device_name = "Apple Silicon (MPS)"
            # Estimate unified memory (cannot query directly)
            self._total_memory = 8 * 1024**3  # Conservative 8GB estimate
            logger.info(f"MPS detected: {self._device_name}")
            return
        
        # Fallback to CPU
        self._device = "cpu"
        self._device_count = 1
        self._device_name = "CPU"
        self._total_memory = 0
        logger.warning("No GPU detected, using CPU (performance will be degraded)")
    
    @property
    def device(self) -> DeviceType:
        """Get current device type."""
        return self._device
    
    @property
    def device_str(self) -> str:
        """Get device string for torch.device()."""
        return str(self._device)
    
    @property
    def device_count(self) -> int:
        """Get number of available devices."""
        return self._device_count
    
    @property
    def device_name(self) -> str:
        """Get device name/model."""
        return self._device_name
    
    @property
    def is_gpu(self) -> bool:
        """Check if GPU (CUDA or MPS) is available."""
        return self._device in ["cuda", "mps"]
    
    @property
    def supports_quantization(self) -> bool:
        """Check if device supports quantization (8bit/4bit)."""
        # BitsAndBytes only works on CUDA
        return self._device == "cuda"
    
    def get_device_info(self) -> dict:
        """Get detailed device information."""
        info = {
            "device_type": self._device,
            "device_str": self.device_str,
            "device_name": self._device_name,
            "device_count": self._device_count,
            "is_gpu": self.is_gpu,
            "supports_quantization": self.supports_quantization,
            "supports_flash_attention": self.supports_flash_attention,
        }
        
        if self._device == "cuda":
            info["cuda_available"] = True
            info["available_memory"] = f"{self._total_memory / 1024**3:.2f} GB"
        elif self._device == "mps":
            info["mps_available"] = True
            info["available_memory"] = "Shared with system RAM"
            info["mps_optimizations"] = self.get_mps_optimizations()
        else:
            info["available_memory"] = "N/A"
        
        return info
    
    def get_mps_optimizations(self) -> dict:
        """Get MPS-specific optimization settings."""
        if self._device != "mps":
            return {}
        
        return {
            "use_float16": True,  # FP16 faster on Apple Silicon
            "max_batch_size": 4,  # Lower than CUDA due to unified memory
            "max_sequence_length": 2048,  # Conservative for memory
            "enable_fallback": True,  # Allow CPU fallback for unsupported ops
        }
    
    def configure_mps_environment(self):
        """Configure environment variables for MPS optimization."""
        if self._device != "mps":
            return
        
        # Enable MPS fallback for unsupported operations
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
        # Set default dtype to FP16 for better MPS performance
        torch.set_default_dtype(torch.float16)
        
        logger.info("MPS environment configured: FP16 default, fallback enabled")
    
    def empty_cache(self):
        """Empty device cache to free memory."""
        if self._device == "cuda":
            torch.cuda.empty_cache()
            logger.debug("Cleared CUDA cache")
        elif self._device == "mps":
            torch.mps.empty_cache()
            logger.debug("Cleared MPS cache")
    
    def get_memory_stats(self) -> dict:
        """Get current memory usage statistics."""
        stats = {"device": self._device}
        
        if self._device == "cuda":
            stats["allocated"] = f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
            stats["reserved"] = f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
            stats["max_allocated"] = f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
        else:
            stats["note"] = "Memory stats only available for CUDA"
        
        return stats
    
    @property
    def supports_flash_attention(self) -> bool:
        """Check if device supports Flash Attention 2."""
        if self._device != "cuda":
            return False
        # Flash Attention requires Ampere (SM 8.0+) or newer
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability(0)
            return capability[0] >= 8
        return False
    
    def get_optimal_batch_size(self, model_size_gb: float) -> int:
        """Calculate optimal batch size based on available memory."""
        if not self.is_gpu:
            return 1
        
        available_memory = self._total_memory * 0.8  # 80% safety margin
        model_memory = model_size_gb * 1024**3
        
        # Reserve 2GB for activation and overhead
        inference_memory = available_memory - model_memory - (2 * 1024**3)
        
        if inference_memory <= 0:
            logger.warning("Model may not fit in GPU memory")
            return 1
        
        # Estimate ~1GB per batch item for typical transformer
        batch_size = max(1, int(inference_memory / 1024**3))
        return min(batch_size, 32)  # Cap at 32
    
    def get_quantization_config(self) -> Optional[dict]:
        """Get BitsAndBytes quantization config if supported."""
        if not self.supports_quantization:
            return None
        
        from transformers import BitsAndBytesConfig
        
        # Use 4-bit quantization for max memory savings
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        logger.info("Quantization enabled: 4-bit NF4")
        return config
    
    def move_to_device(self, model: torch.nn.Module) -> torch.nn.Module:
        """Move model to optimal device."""
        try:
            model = model.to(self.device_str)
            logger.info(f"Model moved to {self.device}")
            return model
        except Exception as e:
            logger.error(f"Failed to move model to {self.device}: {e}")
            if self._device != "cpu":
                logger.warning("Falling back to CPU")
                self._device = "cpu"
                return model.to("cpu")
            raise
    
    def clear_cache(self) -> None:
        """Clear device cache to free memory."""
        if self._device == "cuda":
            torch.cuda.empty_cache()
            logger.debug("CUDA cache cleared")
        elif self._device == "mps":
            torch.mps.empty_cache()
            logger.debug("MPS cache cleared")


# Global singleton instance
_device_manager: Optional[DeviceManager] = None


def get_device_manager() -> DeviceManager:
    """Get or create global device manager instance."""
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
    return _device_manager


def get_device() -> DeviceType:
    """Convenience function to get current device type."""
    return get_device_manager().device


def get_device_string() -> str:
    """Convenience function to get device string for torch.device()."""
    return get_device_manager().device_str


__all__ = [
    "DeviceManager",
    "DeviceType",
    "get_device_manager",
    "get_device",
    "get_device_string"
]
