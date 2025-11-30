"""Centralized configuration management - Optimal 2025 Model Stack."""
import os
import logging
import secrets
from typing import List, Literal, Optional
from pathlib import Path
from enum import Enum


DeploymentTier = Literal["local", "production"]


class ModelBackend(str, Enum):
    """Model inference backends."""
    TRANSFORMERS = "transformers"
    VLLM = "vllm"
    TGI = "tgi"
    OLLAMA = "ollama"


class Settings:
    """Application settings with optimal 2025 model stack."""
    
    # =================================================================
    # APPLICATION
    # =================================================================
    APP_NAME: str = "ShikshaSetu AI Education API"
    APP_VERSION: str = "3.0.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"
    
    # Deployment Configuration
    DEPLOYMENT_TIER: DeploymentTier = os.getenv("DEPLOYMENT_TIER", "local")
    
    # =================================================================
    # DEVICE & COMPUTE
    # =================================================================
    DEVICE: str = os.getenv("DEVICE", "auto")  # auto | cuda | mps | cpu
    USE_QUANTIZATION: bool = os.getenv("USE_QUANTIZATION", "true").lower() == "true"
    QUANTIZATION_TYPE: str = os.getenv("QUANTIZATION_TYPE", "int4")  # int4 | int8 | fp16
    USE_FLASH_ATTENTION: bool = os.getenv("USE_FLASH_ATTENTION", "true").lower() == "true"
    MAX_GPU_MEMORY_GB: float = float(os.getenv("MAX_GPU_MEMORY_GB", "16.0"))
    
    # =================================================================
    # DIRECTORIES
    # =================================================================
    MODEL_CACHE_DIR: Path = Path(os.getenv("MODEL_CACHE_DIR", "data/models"))
    UPLOAD_DIR: Path = Path(os.getenv("UPLOAD_DIR", "data/uploads"))
    LOG_DIR: Path = Path(os.getenv("LOG_DIR", "logs"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "shiksha_setu.log")
    
    # =================================================================
    # OPTIMAL MODEL STACK (2025)
    # =================================================================
    
    # --- Text Simplification: Llama-3.2-3B-Instruct ---
    SIMPLIFICATION_MODEL_ID: str = os.getenv(
        "SIMPLIFICATION_MODEL_ID",
        "meta-llama/Llama-3.2-3B-Instruct"
    )
    SIMPLIFICATION_BACKEND: str = os.getenv("SIMPLIFICATION_BACKEND", "vllm")
    SIMPLIFICATION_MAX_LENGTH: int = int(os.getenv("SIMPLIFICATION_MAX_LENGTH", "2048"))
    SIMPLIFICATION_TEMPERATURE: float = float(os.getenv("SIMPLIFICATION_TEMPERATURE", "0.7"))
    
    # --- Translation: IndicTrans2-1B ---
    TRANSLATION_MODEL_ID: str = os.getenv(
        "TRANSLATION_MODEL_ID",
        "ai4bharat/indictrans2-en-indic-1B"
    )
    TRANSLATION_BACKEND: str = os.getenv("TRANSLATION_BACKEND", "transformers")
    TRANSLATION_MAX_LENGTH: int = int(os.getenv("TRANSLATION_MAX_LENGTH", "512"))
    
    # Supported Indian languages
    SUPPORTED_LANGUAGES: List[str] = [
        "Hindi", "Tamil", "Telugu", "Bengali", "Marathi",
        "Gujarati", "Kannada", "Malayalam", "Punjabi", "Odia"
    ]
    
    # --- Embeddings: BGE-M3 ---
    EMBEDDING_MODEL_ID: str = os.getenv(
        "EMBEDDING_MODEL_ID",
        "BAAI/bge-m3"
    )
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
    EMBEDDING_MAX_LENGTH: int = int(os.getenv("EMBEDDING_MAX_LENGTH", "8192"))
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    
    # --- Reranker: BGE-Reranker-v2-M3 ---
    RERANKER_MODEL_ID: str = os.getenv(
        "RERANKER_MODEL_ID",
        "BAAI/bge-reranker-v2-m3"
    )
    RERANKER_TOP_K: int = int(os.getenv("RERANKER_TOP_K", "10"))
    
    # --- TTS: AI4Bharat Indic-TTS ---
    TTS_MODEL_ID: str = os.getenv(
        "TTS_MODEL_ID",
        "ai4bharat/indic-tts"
    )
    TTS_SAMPLE_RATE: int = int(os.getenv("TTS_SAMPLE_RATE", "22050"))
    
    # --- Validation: Gemma-2-2B-it ---
    VALIDATION_MODEL_ID: str = os.getenv(
        "VALIDATION_MODEL_ID",
        "google/gemma-2-2b-it"
    )
    VALIDATION_BACKEND: str = os.getenv("VALIDATION_BACKEND", "transformers")
    
    # --- OCR: GOT-OCR2 ---
    OCR_MODEL_ID: str = os.getenv(
        "OCR_MODEL_ID",
        "ucaslcl/GOT-OCR2_0"
    )
    OCR_BACKEND: str = os.getenv("OCR_BACKEND", "transformers")
    OCR_MAX_IMAGE_SIZE: int = int(os.getenv("OCR_MAX_IMAGE_SIZE", "2048"))
    
    # =================================================================
    # vLLM PRODUCTION SERVING
    # =================================================================
    VLLM_ENABLED: bool = os.getenv("VLLM_ENABLED", "true").lower() == "true"
    VLLM_HOST: str = os.getenv("VLLM_HOST", "localhost")
    VLLM_PORT: int = int(os.getenv("VLLM_PORT", "8001"))
    VLLM_API_KEY: str = os.getenv("VLLM_API_KEY", "")
    VLLM_TENSOR_PARALLEL_SIZE: int = int(os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1"))
    VLLM_GPU_MEMORY_UTILIZATION: float = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.90"))
    VLLM_MAX_MODEL_LEN: int = int(os.getenv("VLLM_MAX_MODEL_LEN", "4096"))
    VLLM_QUANTIZATION: str = os.getenv("VLLM_QUANTIZATION", "awq")  # awq | gptq | None
    
    # --- Principle C: Generous Swap Space ---
    VLLM_SWAP_SPACE_GB: int = int(os.getenv("VLLM_SWAP_SPACE_GB", "4"))  # 4GB swap for KV cache overflow
    VLLM_CPU_OFFLOAD_GB: float = float(os.getenv("VLLM_CPU_OFFLOAD_GB", "0"))  # CPU offload (set > 0 to enable)
    
    # --- Principle D: Tensor Parallelism for Multi-GPU ---
    VLLM_PIPELINE_PARALLEL_SIZE: int = int(os.getenv("VLLM_PIPELINE_PARALLEL_SIZE", "1"))
    VLLM_ENFORCE_EAGER: bool = os.getenv("VLLM_ENFORCE_EAGER", "false").lower() == "true"  # Disable CUDA graph
    VLLM_MAX_PARALLEL_LOADING_WORKERS: int = int(os.getenv("VLLM_MAX_PARALLEL_LOADING_WORKERS", "2"))
    
    # --- Principle H: KV Cache Configuration ---
    VLLM_BLOCK_SIZE: int = int(os.getenv("VLLM_BLOCK_SIZE", "16"))  # KV cache block size
    VLLM_ENABLE_PREFIX_CACHING: bool = os.getenv("VLLM_ENABLE_PREFIX_CACHING", "true").lower() == "true"
    
    # --- Principle O: Threadpool for Non-Model I/O ---
    THREADPOOL_MAX_WORKERS: int = int(os.getenv("THREADPOOL_MAX_WORKERS", "4"))  # For file I/O, etc.
    ASYNC_POOL_SIZE: int = int(os.getenv("ASYNC_POOL_SIZE", "8"))  # For async operations
    
    # =================================================================
    # BHASHINI API (Fallback)
    # =================================================================
    BHASHINI_API_KEY: str = os.getenv("BHASHINI_API_KEY", "")
    BHASHINI_API_URL: str = os.getenv(
        "BHASHINI_API_URL",
        "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"
    )
    USE_BHASHINI_FALLBACK: bool = os.getenv("USE_BHASHINI_FALLBACK", "true").lower() == "true"
    
    # =================================================================
    # API CONFIGURATION
    # =================================================================
    API_V1_PREFIX: str = "/api/v1"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # =================================================================
    # SECURITY
    # =================================================================
    SECRET_KEY: str = ""  # Set in __init__
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))
    BCRYPT_ROUNDS: int = int(os.getenv("BCRYPT_ROUNDS", "12"))
    
    # Password Requirements
    MIN_PASSWORD_LENGTH: int = 12
    PASSWORD_MIN_LENGTH: int = 12
    PASSWORD_REQUIRE_UPPERCASE: bool = True
    PASSWORD_REQUIRE_LOWERCASE: bool = True
    PASSWORD_REQUIRE_DIGIT: bool = True
    PASSWORD_REQUIRE_SPECIAL: bool = True
    
    # =================================================================
    # STORAGE
    # =================================================================
    STORAGE_TYPE: str = os.getenv("STORAGE_TYPE", "local")  # local | s3
    AWS_ACCESS_KEY_ID: str = os.getenv("AWS_ACCESS_KEY_ID", "")
    AWS_SECRET_ACCESS_KEY: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "shiksha-setu-uploads")
    
    # =================================================================
    # DATABASE
    # =================================================================
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://shiksha_user:shiksha_pass@127.0.0.1:5432/shiksha_setu"
    )
    DB_POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "10"))
    DB_MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "20"))
    DB_POOL_TIMEOUT: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    DB_POOL_RECYCLE: int = int(os.getenv("DB_POOL_RECYCLE", "3600"))
    DB_POOL_PRE_PING: bool = True
    
    # =================================================================
    # REDIS & CELERY
    # =================================================================
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CELERY_BROKER_URL: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
    CELERY_TASK_ALWAYS_EAGER: bool = os.getenv("CELERY_TASK_ALWAYS_EAGER", "false").lower() == "true"
    
    # Celery Task Configuration
    TASK_RESULT_EXPIRES: int = 3600
    TASK_ACKS_LATE: bool = True
    TASK_REJECT_ON_WORKER_LOST: bool = True
    WORKER_PREFETCH_MULTIPLIER: int = 1
    
    # =================================================================
    # CORS
    # =================================================================
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
    ]
    ALLOW_CREDENTIALS: bool = True
    ALLOWED_METHODS: List[str] = ["*"]
    ALLOWED_HEADERS: List[str] = ["*"]
    
    # =================================================================
    # RATE LIMITING
    # =================================================================
    RATE_LIMIT_ENABLED: bool = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
    RATE_LIMIT_PER_MINUTE: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    RATE_LIMIT_PER_HOUR: int = int(os.getenv("RATE_LIMIT_PER_HOUR", "1000"))
    RATE_LIMIT_BURST_MULTIPLIER: int = int(os.getenv("RATE_LIMIT_BURST_MULTIPLIER", "2"))
    RATE_LIMIT_STORAGE: str = os.getenv("RATE_LIMIT_STORAGE", "redis")
    
    # =================================================================
    # LOGGING
    # =================================================================
    LOG_MAX_BYTES: int = 10485760  # 10 MB
    LOG_BACKUP_COUNT: int = 5
    SLOW_REQUEST_THRESHOLD: float = float(os.getenv("SLOW_REQUEST_THRESHOLD", "5.0"))
    
    def __init__(self):
        """Initialize settings and create necessary directories."""
        # Initialize SECRET_KEY
        key = os.getenv("JWT_SECRET_KEY", "")
        if not key or len(key) < 64:
            key = secrets.token_urlsafe(64)
            logger = logging.getLogger(__name__)
            logger.warning(
                "JWT_SECRET_KEY not set or too short. Generated temporary key. "
                "Set JWT_SECRET_KEY environment variable (64+ chars) for production!"
            )
            if self.ENVIRONMENT == "production":
                raise ValueError(
                    "JWT_SECRET_KEY must be set explicitly in production (min 64 characters). "
                    "Generate with: python -c 'import secrets; print(secrets.token_urlsafe(64))'"
                )
        self.SECRET_KEY = key
        
        # Parse ALLOWED_ORIGINS from environment
        allowed_env = os.getenv("ALLOWED_ORIGINS")
        if allowed_env:
            try:
                parsed = [o.strip() for o in allowed_env.split(",") if o.strip()]
                if parsed:
                    self.ALLOWED_ORIGINS = parsed
            except Exception:
                pass
        
        # Create directories
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    @property
    def vllm_base_url(self) -> str:
        """Get vLLM API base URL."""
        return f"http://{self.VLLM_HOST}:{self.VLLM_PORT}/v1"
    
    def get_model_config(self, task: str) -> dict:
        """Get model configuration for a specific task."""
        configs = {
            "simplification": {
                "model_id": self.SIMPLIFICATION_MODEL_ID,
                "backend": self.SIMPLIFICATION_BACKEND,
                "max_length": self.SIMPLIFICATION_MAX_LENGTH,
                "temperature": self.SIMPLIFICATION_TEMPERATURE,
            },
            "translation": {
                "model_id": self.TRANSLATION_MODEL_ID,
                "backend": self.TRANSLATION_BACKEND,
                "max_length": self.TRANSLATION_MAX_LENGTH,
            },
            "embedding": {
                "model_id": self.EMBEDDING_MODEL_ID,
                "dimension": self.EMBEDDING_DIMENSION,
                "max_length": self.EMBEDDING_MAX_LENGTH,
                "batch_size": self.EMBEDDING_BATCH_SIZE,
            },
            "reranker": {
                "model_id": self.RERANKER_MODEL_ID,
                "top_k": self.RERANKER_TOP_K,
            },
            "tts": {
                "model_id": self.TTS_MODEL_ID,
                "sample_rate": self.TTS_SAMPLE_RATE,
            },
            "validation": {
                "model_id": self.VALIDATION_MODEL_ID,
                "backend": self.VALIDATION_BACKEND,
            },
            "ocr": {
                "model_id": self.OCR_MODEL_ID,
                "backend": self.OCR_BACKEND,
                "max_image_size": self.OCR_MAX_IMAGE_SIZE,
            },
        }
        return configs.get(task, {})


# Global settings instance
settings = Settings()
