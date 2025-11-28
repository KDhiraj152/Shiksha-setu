"""
Model Serving Architecture Configuration

Implements production-grade model serving using:
- vLLM for LLM inference (text generation, translation)
- NVIDIA Triton for ONNX models (embeddings, classification)
- Ollama for local development/offline mode

Issue: CODE-REVIEW-GPT #3 (CRITICAL)
Problem: Current setup violates production ML best practices:
- Models loaded in-process (memory bloat, no scaling)
- No batch inference (inefficient GPU utilization)
- No model versioning (A/B testing impossible)
- Single-threaded blocking calls (API slowdown)

Solution Architecture:
1. Separate model serving layer (microservice pattern)
2. Load balancing across model replicas
3. Request batching and queueing
4. Model version control with DVC
"""

from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
import os


class ModelBackend(str, Enum):
    """Supported model serving backends."""
    VLLM = "vllm"  # Production LLM serving
    TRITON = "triton"  # Production ONNX/TensorRT serving
    OLLAMA = "ollama"  # Local development/offline
    HUGGINGFACE = "huggingface"  # Fallback direct inference


class ModelConfig(BaseModel):
    """Configuration for a served model."""
    
    # Model identification
    model_id: str = Field(..., description="Unique model identifier")
    model_path: str = Field(..., description="Path to model weights (local or S3)")
    model_type: str = Field(..., description="Model task type (llm, embedding, translation)")
    
    # Serving backend
    backend: ModelBackend = Field(ModelBackend.VLLM, description="Model serving backend")
    
    # Resource allocation
    gpu_memory_utilization: float = Field(0.9, ge=0.1, le=1.0, description="GPU memory usage fraction")
    tensor_parallel_size: int = Field(1, ge=1, description="Number of GPUs for tensor parallelism")
    max_batch_size: int = Field(32, ge=1, description="Maximum batch size for inference")
    
    # Performance tuning
    max_num_seqs: int = Field(256, ge=1, description="Maximum number of sequences in memory")
    max_model_len: Optional[int] = Field(None, description="Maximum sequence length")
    quantization: Optional[str] = Field(None, description="Quantization method (awq, gptq, bitsandbytes)")
    
    # Endpoint configuration
    port: int = Field(8000, ge=1024, le=65535, description="Serving port")
    host: str = Field("0.0.0.0", description="Serving host")
    
    # Health check
    health_check_path: str = Field("/health", description="Health check endpoint")
    
    class Config:
        use_enum_values = True


# =============================================================================
# PRODUCTION MODEL CONFIGURATIONS
# =============================================================================

# LLM for content generation (simplification, translation)
LLM_CONFIG = ModelConfig(
    model_id="llm-content-generator",
    model_path=os.getenv("LLM_MODEL_PATH", "models/mistral-7b-instruct"),
    model_type="llm",
    backend=ModelBackend.VLLM,
    gpu_memory_utilization=0.85,
    tensor_parallel_size=1,
    max_batch_size=32,
    max_num_seqs=256,
    max_model_len=8192,
    quantization="awq",  # 4-bit quantization for efficiency
    port=8001,
)

# Embedding model for RAG/Q&A
EMBEDDING_CONFIG = ModelConfig(
    model_id="embedding-rag",
    model_path=os.getenv("EMBEDDING_MODEL_PATH", "models/all-MiniLM-L6-v2.onnx"),
    model_type="embedding",
    backend=ModelBackend.TRITON,
    gpu_memory_utilization=0.3,
    max_batch_size=128,  # Embeddings benefit from large batches
    port=8002,
)

# Translation model (Indic languages)
TRANSLATION_CONFIG = ModelConfig(
    model_id="translation-indic",
    model_path=os.getenv("TRANSLATION_MODEL_PATH", "models/indictrans2.onnx"),
    model_type="translation",
    backend=ModelBackend.TRITON,
    gpu_memory_utilization=0.4,
    max_batch_size=64,
    port=8003,
)

# TTS model (speech synthesis)
TTS_CONFIG = ModelConfig(
    model_id="tts-multilingual",
    model_path=os.getenv("TTS_MODEL_PATH", "models/vits-indic-tts"),
    model_type="tts",
    backend=ModelBackend.TRITON,
    gpu_memory_utilization=0.3,
    max_batch_size=16,
    port=8004,
)

# Local development configuration (Ollama)
OLLAMA_DEV_CONFIG = ModelConfig(
    model_id="ollama-mistral",
    model_path="mistral:7b-instruct",
    model_type="llm",
    backend=ModelBackend.OLLAMA,
    gpu_memory_utilization=0.9,
    max_batch_size=8,
    port=11434,  # Default Ollama port
)


# =============================================================================
# MODEL REGISTRY
# =============================================================================

MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "llm": LLM_CONFIG,
    "embedding": EMBEDDING_CONFIG,
    "translation": TRANSLATION_CONFIG,
    "tts": TTS_CONFIG,
    "ollama_dev": OLLAMA_DEV_CONFIG,
}


def get_model_config(model_type: str) -> ModelConfig:
    """Get model configuration by type."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_type]


# =============================================================================
# DEPLOYMENT SETTINGS
# =============================================================================

class DeploymentConfig(BaseModel):
    """Deployment configuration for model serving."""
    
    # Environment
    environment: str = Field(os.getenv("ENVIRONMENT", "development"), description="Deployment environment")
    
    # Model serving endpoints
    vllm_endpoint: str = Field(os.getenv("VLLM_ENDPOINT", "http://localhost:8001"), description="vLLM service endpoint")
    triton_endpoint: str = Field(os.getenv("TRITON_ENDPOINT", "http://localhost:8002"), description="Triton service endpoint")
    ollama_endpoint: str = Field(os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434"), description="Ollama service endpoint")
    
    # Load balancing
    enable_load_balancing: bool = Field(True, description="Enable load balancing across replicas")
    num_replicas: int = Field(2, ge=1, description="Number of model replicas")
    
    # Request handling
    request_timeout: int = Field(60, ge=1, description="Request timeout in seconds")
    max_retries: int = Field(3, ge=0, description="Maximum retry attempts")
    retry_delay: float = Field(1.0, ge=0, description="Delay between retries in seconds")
    
    # Monitoring
    enable_metrics: bool = Field(True, description="Enable Prometheus metrics")
    enable_tracing: bool = Field(True, description="Enable OpenTelemetry tracing")
    
    # Model versioning (DVC)
    dvc_remote: str = Field(os.getenv("DVC_REMOTE", "s3://shiksha-setu-models"), description="DVC remote storage")
    model_version_tag: str = Field(os.getenv("MODEL_VERSION", "latest"), description="Model version to deploy")


# Global deployment configuration
DEPLOYMENT_CONFIG = DeploymentConfig()


# =============================================================================
# OFFLINE MODE CONFIGURATION
# =============================================================================

class OfflineModeConfig(BaseModel):
    """Configuration for offline-first architecture (PS3)."""
    
    # Model optimization for offline
    use_onnx_runtime: bool = Field(True, description="Use ONNX Runtime for faster CPU inference")
    use_quantized_models: bool = Field(True, description="Use 8-bit quantized models")
    
    # Caching
    cache_inference_results: bool = Field(True, description="Cache inference results locally")
    cache_ttl_hours: int = Field(24, ge=1, description="Cache TTL in hours")
    
    # Progressive Web App (PWA)
    enable_pwa: bool = Field(True, description="Enable PWA features")
    cache_strategy: str = Field("cache-first", description="Service worker cache strategy")
    
    # Sync
    enable_background_sync: bool = Field(True, description="Enable background sync when online")
    sync_interval_minutes: int = Field(15, ge=1, description="Sync interval in minutes")


OFFLINE_CONFIG = OfflineModeConfig()
