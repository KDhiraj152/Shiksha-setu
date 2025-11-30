"""Device detection and optimization utilities for Apple Silicon and CUDA."""
import os
import logging
from typing import Dict, Any, Literal

logger = logging.getLogger(__name__)

DeviceType = Literal["cuda", "mps", "cpu"]


def detect_optimal_device() -> DeviceType:
    """
    Detect the optimal compute device available.
    
    Priority:
    1. CUDA GPU (production)
    2. Apple Silicon MPS (local M4)
    3. CPU (fallback)
    
    Returns:
        Device type string
    """
    try:
        import torch
        
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"CUDA GPU detected: {device_name}")
            return "cuda"
        
        if torch.backends.mps.is_available():
            logger.info("Apple Silicon MPS detected - M4 optimization available")
            return "mps"
        
        logger.info("No GPU acceleration available - using CPU")
        return "cpu"
        
    except ImportError:
        logger.warning("PyTorch not installed - defaulting to CPU")
        return "cpu"


def get_device_info() -> Dict[str, Any]:
    """
    Get detailed information about available compute devices.
    
    Returns:
        Dictionary with device information
    """
    info = {
        "optimal_device": "cpu",
        "cuda_available": False,
        "mps_available": False,
        "cpu_count": os.cpu_count() or 1
    }
    
    try:
        import torch
        
        info["optimal_device"] = detect_optimal_device()
        info["cuda_available"] = torch.cuda.is_available()
        info["mps_available"] = torch.backends.mps.is_available()
        
        if info["cuda_available"]:
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_devices"] = [
                {
                    "id": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": f"{torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB"
                }
                for i in range(torch.cuda.device_count())
            ]
        
        if info["mps_available"]:
            info["apple_silicon"] = True
            info["mps_optimization"] = "enabled"
            
    except ImportError:
        # PyTorch not available, skip MPS detection
        logger.debug("PyTorch not available for MPS detection")
    
    return info


def should_use_quantization(device: DeviceType) -> bool:
    """
    Determine if quantization should be used for the given device.
    
    Args:
        device: Device type
        
    Returns:
        True if quantization is recommended
    """
    # Check environment override
    env_value = os.getenv("USE_QUANTIZATION", "true").lower()
    if env_value in ("false", "0", "no"):
        return False
    
    # CUDA: Use 8-bit quantization for memory efficiency
    if device == "cuda":
        return True
    
    # MPS: Currently doesn't support bitsandbytes quantization
    if device == "mps":
        return False
    
    # CPU: Optional 8-bit quantization
    return env_value == "true"


def get_optimal_batch_size(device: DeviceType, model_size: str = "base") -> int:
    """
    Get optimal batch size for the given device and model size.
    
    Args:
        device: Device type
        model_size: Model size (base, large, xl)
        
    Returns:
        Recommended batch size
    """
    # Check environment override
    env_batch = os.getenv("BATCH_SIZE")
    if env_batch:
        try:
            return int(env_batch)
        except ValueError:
            logger.warning(f"Invalid BATCH_SIZE value: {env_batch}, using defaults")
    
    # Default batch sizes
    batch_sizes = {
        "cuda": {"base": 32, "large": 16, "xl": 8},
        "mps": {"base": 16, "large": 8, "xl": 4},
        "cpu": {"base": 4, "large": 2, "xl": 1}
    }
    
    return batch_sizes.get(device, {}).get(model_size, 1)


def get_optimal_workers(device: DeviceType) -> int:
    """
    Get optimal number of worker processes for the given device.
    
    Args:
        device: Device type
        
    Returns:
        Number of workers
    """
    # Check environment override
    env_workers = os.getenv("WORKER_COUNT")
    if env_workers:
        try:
            return int(env_workers)
        except ValueError:
            logger.warning(f"Invalid WORKER_COUNT value: {env_workers}, using defaults")
    
    cpu_count = os.cpu_count() or 1
    
    if device == "cuda":
        # One worker per GPU
        try:
            import torch
            return torch.cuda.device_count()
        except (ImportError, RuntimeError):
            return 1
    
    elif device == "mps":
        # Apple Silicon: Limited by unified memory
        return min(4, cpu_count)
    
    else:
        # CPU: Use half of available cores
        return max(1, cpu_count // 2)


def log_device_status():
    """Log current device configuration for debugging."""
    info = get_device_info()
    
    logger.info("="*60)
    logger.info("DEVICE CONFIGURATION")
    logger.info("="*60)
    logger.info(f"Optimal Device: {info['optimal_device'].upper()}")
    logger.info(f"CUDA Available: {info['cuda_available']}")
    logger.info(f"MPS Available: {info['mps_available']}")
    logger.info(f"CPU Count: {info['cpu_count']}")
    
    if info.get("cuda_devices"):
        logger.info("\nCUDA Devices:")
        for device in info["cuda_devices"]:
            logger.info(f"  GPU {device['id']}: {device['name']} ({device['memory_total']})")
    
    if info.get("apple_silicon"):
        logger.info("\nApple Silicon Optimizations: ENABLED")
    
    logger.info("="*60)


__all__ = [
    'DeviceType',
    'detect_optimal_device',
    'get_device_info',
    'should_use_quantization',
    'get_optimal_batch_size',
    'get_optimal_workers',
    'log_device_status'
]
