"""
Unified AI Orchestrator for ShikshaSetu.

Coordinates all AI services (translation, TTS, simplification, embeddings)
with proper memory management, lazy loading, and intelligent routing.

Optimized for M4 MacBook Pro 16GB unified memory.
"""

import asyncio
import gc
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional, AsyncIterator, TypeVar, Generic
from contextlib import asynccontextmanager
import weakref

logger = logging.getLogger(__name__)

# Type variables for generic results
T = TypeVar('T')


class ServiceType(Enum):
    """Available AI service types."""
    TRANSLATION = "translation"
    TTS = "tts"
    SIMPLIFICATION = "simplification"
    EMBEDDINGS = "embeddings"
    OCR = "ocr"


class Priority(Enum):
    """Request priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class AIServiceConfig:
    """Configuration for AI services."""
    # Memory limits (in GB)
    max_memory_gb: float = 10.0  # Leave headroom for system
    
    # Model configuration
    translation_model: str = "small"  # small (600M), medium (1.3B), large (3.3B) for NLLB
    llm_model: str = "llama3.2:3b"
    embedding_model: str = "BAAI/bge-m3"
    
    # Performance settings
    enable_lazy_loading: bool = True
    enable_memory_monitoring: bool = True
    unload_after_idle_seconds: int = 300  # 5 minutes
    
    # Batch settings
    max_batch_size: int = 32
    batch_timeout_seconds: float = 0.1
    
    # Device settings
    device: str = "mps"  # Metal Performance Shaders for M4
    compute_type: str = "int8"  # Quantization for efficiency
    
    # Ollama settings
    ollama_host: str = "http://localhost:11434"
    
    @classmethod
    def from_env(cls) -> "AIServiceConfig":
        """Load configuration from environment variables."""
        return cls(
            max_memory_gb=float(os.getenv("AI_MAX_MEMORY_GB", "10.0")),
            translation_model=os.getenv("TRANSLATION_MODEL", "small"),
            llm_model=os.getenv("LLM_MODEL", "llama3.2:3b"),
            embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
            device=os.getenv("AI_DEVICE", "mps"),
            compute_type=os.getenv("AI_COMPUTE_TYPE", "int8"),
            ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            unload_after_idle_seconds=int(os.getenv("AI_IDLE_TIMEOUT", "300")),
        )


@dataclass
class ProcessingRequest:
    """Request for AI processing."""
    service: ServiceType
    input_data: Any
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: Priority = Priority.NORMAL
    request_id: Optional[str] = None
    timeout_seconds: float = 60.0


@dataclass 
class ProcessingResult(Generic[T]):
    """Result from AI processing."""
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    processing_time_ms: float = 0.0
    model_used: Optional[str] = None
    cached: bool = False


class MemoryManager:
    """Manages memory across AI services."""
    
    def __init__(self, max_memory_gb: float = 10.0):
        self.max_memory_bytes = int(max_memory_gb * 1024 * 1024 * 1024)
        self._loaded_services: Dict[ServiceType, float] = {}
        self._last_used: Dict[ServiceType, float] = {}
        self._lock = asyncio.Lock()
        
    async def can_load(self, service: ServiceType, required_bytes: int) -> bool:
        """Check if we have memory to load a service."""
        async with self._lock:
            current_usage = sum(self._loaded_services.values())
            available = self.max_memory_bytes - current_usage
            return available >= required_bytes
            
    async def register_loaded(self, service: ServiceType, memory_bytes: int) -> None:
        """Register a newly loaded service."""
        async with self._lock:
            self._loaded_services[service] = memory_bytes
            self._last_used[service] = time.time()
            logger.info(f"Registered {service.value}: {memory_bytes / 1024**2:.1f}MB")
            
    async def mark_used(self, service: ServiceType) -> None:
        """Mark a service as recently used."""
        async with self._lock:
            self._last_used[service] = time.time()
            
    async def unregister(self, service: ServiceType) -> None:
        """Unregister an unloaded service."""
        async with self._lock:
            self._loaded_services.pop(service, None)
            self._last_used.pop(service, None)
            logger.info(f"Unregistered {service.value}")
            
    async def get_idle_services(self, idle_threshold_seconds: int) -> List[ServiceType]:
        """Get services that have been idle beyond threshold."""
        async with self._lock:
            now = time.time()
            idle = []
            for service, last_used in self._last_used.items():
                if now - last_used > idle_threshold_seconds:
                    idle.append(service)
            return idle
            
    def get_memory_status(self) -> Dict[str, Any]:
        """Get current memory status."""
        total_used = sum(self._loaded_services.values())
        return {
            "max_memory_mb": self.max_memory_bytes / 1024**2,
            "used_memory_mb": total_used / 1024**2,
            "available_memory_mb": (self.max_memory_bytes - total_used) / 1024**2,
            "loaded_services": {
                s.value: m / 1024**2 
                for s, m in self._loaded_services.items()
            }
        }


class AIOrchestrator:
    """
    Unified orchestrator for all AI services.
    
    Provides:
    - Lazy loading of models
    - Memory management
    - Automatic unloading of idle models
    - Request routing
    - Error handling and retries
    """
    
    # Approximate memory requirements per service (in bytes)
    MEMORY_REQUIREMENTS = {
        ServiceType.TRANSLATION: 2.5 * 1024**3,   # 2.5GB for NLLB-200
        ServiceType.TTS: 0.1 * 1024**3,            # 100MB (Edge TTS is API-based)
        ServiceType.SIMPLIFICATION: 2.0 * 1024**3, # 2GB for Llama 3.2 3B
        ServiceType.EMBEDDINGS: 1.2 * 1024**3,     # 1.2GB for BGE-M3
        ServiceType.OCR: 0.5 * 1024**3,            # 500MB for Surya
    }
    
    def __init__(self, config: Optional[AIServiceConfig] = None):
        self.config = config or AIServiceConfig.from_env()
        self.memory_manager = MemoryManager(self.config.max_memory_gb)
        
        # Service instances (lazy loaded)
        self._translator = None
        self._tts_generator = None
        self._simplifier = None
        self._embeddings = None
        
        # Locks for thread-safe initialization
        self._service_locks: Dict[ServiceType, asyncio.Lock] = {
            service: asyncio.Lock() for service in ServiceType
        }
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info(f"AIOrchestrator initialized with config: max_memory={self.config.max_memory_gb}GB")
        
    async def start(self) -> None:
        """Start the orchestrator and background tasks."""
        if self._running:
            return
            
        self._running = True
        
        # Start memory cleanup task
        if self.config.enable_memory_monitoring:
            self._cleanup_task = asyncio.create_task(self._memory_cleanup_loop())
            
        logger.info("AIOrchestrator started")
        
    async def stop(self) -> None:
        """Stop the orchestrator and cleanup resources."""
        self._running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
                
        # Unload all services
        await self._unload_all_services()
        
        logger.info("AIOrchestrator stopped")
        
    async def _memory_cleanup_loop(self) -> None:
        """Periodically check and unload idle services."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                idle_services = await self.memory_manager.get_idle_services(
                    self.config.unload_after_idle_seconds
                )
                
                for service in idle_services:
                    await self._unload_service(service)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory cleanup loop: {e}")
                
    async def _unload_service(self, service: ServiceType) -> None:
        """Unload a specific service to free memory."""
        async with self._service_locks[service]:
            if service == ServiceType.TRANSLATION and self._translator:
                self._translator.unload()
                self._translator = None
            elif service == ServiceType.TTS and self._tts_generator:
                await self._tts_generator.close()
                self._tts_generator = None
            elif service == ServiceType.SIMPLIFICATION and self._simplifier:
                self._simplifier = None
            elif service == ServiceType.EMBEDDINGS and self._embeddings:
                self._embeddings = None
                
            await self.memory_manager.unregister(service)
            gc.collect()
            logger.info(f"Unloaded service: {service.value}")
            
    async def _unload_all_services(self) -> None:
        """Unload all services."""
        for service in ServiceType:
            try:
                await self._unload_service(service)
            except Exception as e:
                logger.error(f"Error unloading {service.value}: {e}")
                
    # ==================== Translation ====================
    
    async def _ensure_translator(self):
        """Ensure translator is loaded."""
        if self._translator is not None:
            return
            
        async with self._service_locks[ServiceType.TRANSLATION]:
            if self._translator is not None:
                return
                
            # Check memory
            required = self.MEMORY_REQUIREMENTS[ServiceType.TRANSLATION]
            if not await self.memory_manager.can_load(ServiceType.TRANSLATION, required):
                # Try to free memory
                idle = await self.memory_manager.get_idle_services(0)
                for service in idle:
                    if service != ServiceType.TRANSLATION:
                        await self._unload_service(service)
                        if await self.memory_manager.can_load(ServiceType.TRANSLATION, required):
                            break
                            
            from backend.services.translate.nllb_translator import NLLBTranslator
            
            self._translator = NLLBTranslator(
                model_size=self.config.translation_model,
                device=self.config.device,
                compute_type=self.config.compute_type,
            )
            
            await self.memory_manager.register_loaded(
                ServiceType.TRANSLATION, required
            )
            logger.info("Translator loaded")
            
    async def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
        **kwargs
    ) -> ProcessingResult[Dict[str, Any]]:
        """
        Translate text between languages.
        
        Args:
            text: Text to translate
            source_language: Source language code (ISO 639-1 or full name)
            target_language: Target language code
            
        Returns:
            ProcessingResult with translated text
        """
        start_time = time.time()
        
        try:
            await self._ensure_translator()
            await self.memory_manager.mark_used(ServiceType.TRANSLATION)
            
            result = await self._translator.translate_async(
                text=text,
                source_language=source_language,
                target_language=target_language
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            return ProcessingResult(
                success=True,
                data={
                    "translated_text": result.text,
                    "source_language": result.source_language,
                    "target_language": result.target_language,
                },
                processing_time_ms=processing_time,
                model_used=self.config.translation_model
            )
            
        except Exception as e:
            logger.error(f"Translation error: {e}")
            return ProcessingResult(
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000
            )
            
    async def translate_batch(
        self,
        texts: List[str],
        source_language: str,
        target_language: str
    ) -> ProcessingResult[List[Dict[str, Any]]]:
        """Translate multiple texts."""
        start_time = time.time()
        
        try:
            await self._ensure_translator()
            await self.memory_manager.mark_used(ServiceType.TRANSLATION)
            
            results = await self._translator.translate_batch_async(
                texts=texts,
                source_language=source_language,
                target_language=target_language
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            # Convert TranslationResult objects to dicts
            results_data = [
                {
                    "translated_text": r.text,
                    "source_language": r.source_language,
                    "target_language": r.target_language,
                }
                for r in results
            ]
            
            return ProcessingResult(
                success=True,
                data=results_data,
                processing_time_ms=processing_time,
                model_used=self.config.translation_model
            )
            
        except Exception as e:
            logger.error(f"Batch translation error: {e}")
            return ProcessingResult(
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000
            )
            
    # ==================== Text-to-Speech ====================
    
    async def _ensure_tts(self):
        """Ensure TTS generator is loaded."""
        if self._tts_generator is not None:
            return
            
        async with self._service_locks[ServiceType.TTS]:
            if self._tts_generator is not None:
                return
                
            from backend.services.speech.edge_tts_generator import EdgeTTSGenerator
            
            self._tts_generator = EdgeTTSGenerator()
            
            await self.memory_manager.register_loaded(
                ServiceType.TTS,
                self.MEMORY_REQUIREMENTS[ServiceType.TTS]
            )
            logger.info("TTS generator loaded")
            
    async def synthesize_speech(
        self,
        text: str,
        language: str = "en",
        voice: Optional[str] = None,
        rate: str = "+0%",
        output_format: str = "mp3"
    ) -> ProcessingResult[bytes]:
        """
        Convert text to speech audio.
        
        Args:
            text: Text to convert
            language: Language code
            voice: Optional specific voice name
            rate: Speech rate adjustment
            output_format: Output format (mp3, wav, ogg)
            
        Returns:
            ProcessingResult with audio bytes
        """
        start_time = time.time()
        
        try:
            await self._ensure_tts()
            await self.memory_manager.mark_used(ServiceType.TTS)
            
            result = await self._tts_generator.generate_speech(
                text=text,
                language=language,
                gender=voice if voice in ["male", "female"] else "female"
            )
            
            # Read audio bytes from file path
            if hasattr(result, 'audio_path') and result.audio_path:
                from pathlib import Path
                audio_path = Path(result.audio_path)
                if audio_path.exists():
                    audio_bytes = audio_path.read_bytes()
                else:
                    raise ValueError(f"Audio file not found: {result.audio_path}")
            else:
                audio_bytes = result
            
            processing_time = (time.time() - start_time) * 1000
            
            return ProcessingResult(
                success=True,
                data=audio_bytes,
                processing_time_ms=processing_time,
                model_used="edge-tts"
            )
            
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return ProcessingResult(
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000
            )
            
    async def synthesize_speech_stream(
        self,
        text: str,
        language: str = "en",
        **kwargs
    ) -> AsyncIterator[bytes]:
        """Stream audio bytes as they're generated."""
        await self._ensure_tts()
        await self.memory_manager.mark_used(ServiceType.TTS)
        
        async for chunk in self._tts_generator.synthesize_stream(
            text=text,
            language=language,
            **kwargs
        ):
            yield chunk
            
    # ==================== Simplification ====================
    
    async def _ensure_simplifier(self):
        """Ensure simplifier is loaded."""
        if self._simplifier is not None:
            return
            
        async with self._service_locks[ServiceType.SIMPLIFICATION]:
            if self._simplifier is not None:
                return
                
            # Check memory
            required = self.MEMORY_REQUIREMENTS[ServiceType.SIMPLIFICATION]
            if not await self.memory_manager.can_load(ServiceType.SIMPLIFICATION, required):
                idle = await self.memory_manager.get_idle_services(0)
                for service in idle:
                    if service != ServiceType.SIMPLIFICATION:
                        await self._unload_service(service)
                        if await self.memory_manager.can_load(ServiceType.SIMPLIFICATION, required):
                            break
                            
            from backend.services.simplify.ollama_simplifier import OllamaSimplifier
            
            self._simplifier = OllamaSimplifier(
                model=self.config.llm_model,
                base_url=self.config.ollama_host
            )
            
            await self.memory_manager.register_loaded(
                ServiceType.SIMPLIFICATION, required
            )
            logger.info("Simplifier loaded")
            
    async def simplify_text(
        self,
        text: str,
        target_grade: int = 8,
        subject: Optional[str] = None,
        language: str = "en",
        **kwargs
    ) -> ProcessingResult[Dict[str, Any]]:
        """
        Simplify text for a target grade level.
        
        Args:
            text: Text to simplify
            target_grade: Target grade level (5-12)
            subject: Optional subject context
            language: Language of the text
            
        Returns:
            ProcessingResult with simplified text
        """
        start_time = time.time()
        
        try:
            await self._ensure_simplifier()
            await self.memory_manager.mark_used(ServiceType.SIMPLIFICATION)
            
            result = await self._simplifier.simplify_text(
                text=text,
                grade_level=target_grade,
                subject=subject or "General"
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            return ProcessingResult(
                success=True,
                data={
                    "simplified_text": result.simplified_text,
                    "target_grade": result.grade_level,
                    "key_concepts": [],
                },
                processing_time_ms=processing_time,
                model_used=self.config.llm_model
            )
            
        except Exception as e:
            logger.error(f"Simplification error: {e}")
            return ProcessingResult(
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000
            )
            
    async def simplify_stream(
        self,
        text: str,
        target_grade: int = 8,
        **kwargs
    ) -> AsyncIterator[str]:
        """Stream simplified text as it's generated."""
        await self._ensure_simplifier()
        await self.memory_manager.mark_used(ServiceType.SIMPLIFICATION)
        
        async for token in self._simplifier.simplify_stream(
            text=text,
            target_grade=target_grade,
            **kwargs
        ):
            yield token
            
    # ==================== Embeddings ====================
    
    async def _ensure_embeddings(self):
        """Ensure embeddings model is loaded."""
        if self._embeddings is not None:
            return
            
        async with self._service_locks[ServiceType.EMBEDDINGS]:
            if self._embeddings is not None:
                return
                
            # Check memory
            required = self.MEMORY_REQUIREMENTS[ServiceType.EMBEDDINGS]
            if not await self.memory_manager.can_load(ServiceType.EMBEDDINGS, required):
                idle = await self.memory_manager.get_idle_services(0)
                for service in idle:
                    if service != ServiceType.EMBEDDINGS:
                        await self._unload_service(service)
                        if await self.memory_manager.can_load(ServiceType.EMBEDDINGS, required):
                            break
                            
            from backend.services.embeddings.bge_embeddings import BGEM3Embeddings
            
            self._embeddings = BGEM3Embeddings(
                device=self.config.device
            )
            
            await self.memory_manager.register_loaded(
                ServiceType.EMBEDDINGS, required
            )
            logger.info("Embeddings model loaded")
            
    async def generate_embeddings(
        self,
        texts: List[str],
        return_sparse: bool = False
    ) -> ProcessingResult[Dict[str, Any]]:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of texts to embed
            return_sparse: Whether to return sparse embeddings too
            
        Returns:
            ProcessingResult with embeddings
        """
        start_time = time.time()
        
        try:
            await self._ensure_embeddings()
            await self.memory_manager.mark_used(ServiceType.EMBEDDINGS)
            
            result = await self._embeddings.embed_async(
                texts=texts,
                return_dense=True
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            data = {
                "embeddings": result.embeddings if isinstance(result.embeddings[0], list) else [result.embeddings],
                "dimensions": result.dimension
            }
            
            if return_sparse:
                data["sparse_embeddings"] = [r.sparse_embedding for r in results]
                
            return ProcessingResult(
                success=True,
                data=data,
                processing_time_ms=processing_time,
                model_used=self.config.embedding_model
            )
            
        except Exception as e:
            logger.error(f"Embeddings error: {e}")
            return ProcessingResult(
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000
            )
            
    # ==================== High-level Pipeline ====================
    
    async def process_educational_content(
        self,
        content: str,
        source_language: str = "en",
        target_language: Optional[str] = None,
        target_grade: int = 8,
        generate_audio: bool = False,
        generate_embeddings: bool = True,
        subject: Optional[str] = None
    ) -> ProcessingResult[Dict[str, Any]]:
        """
        Full pipeline for processing educational content.
        
        Combines translation, simplification, TTS, and embeddings
        into a single coordinated operation.
        
        Args:
            content: Original content text
            source_language: Source language
            target_language: Optional target language for translation
            target_grade: Target grade level for simplification
            generate_audio: Whether to generate audio
            generate_embeddings: Whether to generate embeddings
            subject: Subject context for better simplification
            
        Returns:
            ProcessingResult with all generated outputs
        """
        start_time = time.time()
        results = {
            "original": content,
            "source_language": source_language,
        }
        
        try:
            # Step 1: Simplify content
            simplified = await self.simplify_text(
                text=content,
                target_grade=target_grade,
                subject=subject,
                language=source_language
            )
            
            if simplified.success:
                results["simplified"] = simplified.data["simplified_text"]
                results["key_concepts"] = simplified.data.get("key_concepts", [])
            else:
                results["simplified"] = content  # Fallback
                
            working_text = results.get("simplified", content)
            
            # Step 2: Translate if needed
            if target_language and target_language != source_language:
                translated = await self.translate(
                    text=working_text,
                    source_language=source_language,
                    target_language=target_language
                )
                
                if translated.success:
                    results["translated"] = translated.data["translated_text"]
                    results["target_language"] = target_language
                    working_text = results["translated"]
                    
            # Step 3: Generate audio if requested
            if generate_audio:
                lang_for_audio = target_language or source_language
                audio = await self.synthesize_speech(
                    text=working_text,
                    language=lang_for_audio
                )
                
                if audio.success:
                    results["audio_bytes"] = audio.data
                    results["audio_format"] = "mp3"
                    
            # Step 4: Generate embeddings if requested  
            if generate_embeddings:
                embeddings = await self.generate_embeddings(
                    texts=[working_text]
                )
                
                if embeddings.success:
                    results["embedding"] = embeddings.data["embeddings"][0]
                    
            processing_time = (time.time() - start_time) * 1000
            
            return ProcessingResult(
                success=True,
                data=results,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return ProcessingResult(
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000
            )
            
    # ==================== Status & Diagnostics ====================
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status."""
        return {
            "running": self._running,
            "memory": self.memory_manager.get_memory_status(),
            "config": {
                "max_memory_gb": self.config.max_memory_gb,
                "device": self.config.device,
                "compute_type": self.config.compute_type,
            },
            "services": {
                "translator_loaded": self._translator is not None,
                "tts_loaded": self._tts_generator is not None,
                "simplifier_loaded": self._simplifier is not None,
                "embeddings_loaded": self._embeddings is not None,
            }
        }
        
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all services."""
        health = {
            "orchestrator": "healthy",
            "services": {}
        }
        
        # Check translation
        try:
            if self._translator:
                health["services"]["translation"] = "healthy"
            else:
                health["services"]["translation"] = "not_loaded"
        except Exception as e:
            health["services"]["translation"] = f"error: {e}"
            
        # Check TTS
        try:
            if self._tts_generator:
                # Edge TTS is API-based, check connectivity
                voices = await self._tts_generator.get_voices("en")
                health["services"]["tts"] = "healthy" if voices else "degraded"
            else:
                health["services"]["tts"] = "not_loaded"
        except Exception as e:
            health["services"]["tts"] = f"error: {e}"
            
        # Check simplifier (Ollama)
        try:
            if self._simplifier:
                is_available = await self._simplifier.check_model_available()
                health["services"]["simplification"] = "healthy" if is_available else "degraded"
            else:
                health["services"]["simplification"] = "not_loaded"
        except Exception as e:
            health["services"]["simplification"] = f"error: {e}"
            
        # Check embeddings
        try:
            if self._embeddings:
                health["services"]["embeddings"] = "healthy"
            else:
                health["services"]["embeddings"] = "not_loaded"
        except Exception as e:
            health["services"]["embeddings"] = f"error: {e}"
            
        return health


# Singleton instance
_orchestrator: Optional[AIOrchestrator] = None
_orchestrator_lock = asyncio.Lock()


async def get_ai_orchestrator(
    config: Optional[AIServiceConfig] = None
) -> AIOrchestrator:
    """
    Get the singleton AI orchestrator instance.
    
    Args:
        config: Optional configuration (only used on first call)
        
    Returns:
        AIOrchestrator instance
    """
    global _orchestrator
    
    if _orchestrator is None:
        async with _orchestrator_lock:
            if _orchestrator is None:
                _orchestrator = AIOrchestrator(config)
                await _orchestrator.start()
                
    return _orchestrator


@asynccontextmanager
async def ai_orchestrator_context(
    config: Optional[AIServiceConfig] = None
):
    """
    Context manager for AI orchestrator.
    
    Usage:
        async with ai_orchestrator_context() as ai:
            result = await ai.translate(...)
    """
    orchestrator = await get_ai_orchestrator(config)
    try:
        yield orchestrator
    finally:
        pass  # Keep running for reuse


async def shutdown_orchestrator():
    """Shutdown the global orchestrator."""
    global _orchestrator
    
    if _orchestrator:
        await _orchestrator.stop()
        _orchestrator = None
