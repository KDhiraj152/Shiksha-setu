"""
Async Pipeline Service - Proper async-first implementation.

This is the NEW recommended way to run the content processing pipeline.
All operations are properly async with no sync/async mixing.

Usage:
    from backend.services.pipeline_service import PipelineService
    
    async def process():
        service = PipelineService()
        result = await service.process_content(
            text="...",
            grade_level=8,
            subject="Science",
            target_languages=["Hindi", "Tamil"],
            generate_audio=True
        )
"""
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

logger = logging.getLogger(__name__)


class AIUnifiedClient:
    """
    Unified client wrapper for the new AI orchestrator.
    Provides backward-compatible interface for pipeline operations.
    """
    
    def __init__(self, orchestrator):
        self._orchestrator = orchestrator
    
    async def simplify_text(self, text: str, grade_level: int, subject: str) -> str:
        """Simplify text using Ollama (Llama 3.2)."""
        result = await self._orchestrator.simplify_text(
            text=text,
            target_grade=grade_level,
            subject=subject
        )
        if result.success:
            return result.data.get("simplified_text", "")
        raise Exception(result.error or "Simplification failed")
    
    async def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text using NLLB-200."""
        result = await self._orchestrator.translate_text(
            text=text,
            source_language=source_lang,
            target_language=target_lang
        )
        if result.success:
            return result.data.get("translated_text", "")
        raise Exception(result.error or "Translation failed")
    
    async def validate_content(
        self,
        original_text: str,
        simplified_text: str,
        grade_level: int,
        subject: str
    ) -> float:
        """Validate content using BGE-M3 embeddings similarity."""
        try:
            # Get embeddings for both texts
            orig_result = await self._orchestrator.generate_embeddings(text=original_text)
            simp_result = await self._orchestrator.generate_embeddings(text=simplified_text)
            
            if orig_result.success and simp_result.success:
                import numpy as np
                orig_emb = np.array(orig_result.data.get("embedding", []))
                simp_emb = np.array(simp_result.data.get("embedding", []))
                
                if orig_emb.size > 0 and simp_emb.size > 0:
                    # Cosine similarity
                    similarity = np.dot(orig_emb, simp_emb) / (
                        np.linalg.norm(orig_emb) * np.linalg.norm(simp_emb)
                    )
                    return float(similarity)
        except Exception as e:
            logger.warning(f"Embedding validation failed: {e}")
        
        return 0.5  # Default score if validation fails


class PipelineStage(str, Enum):
    """Pipeline processing stages."""
    SIMPLIFICATION = "simplification"
    TRANSLATION = "translation"
    VALIDATION = "validation"
    SPEECH = "speech"


class PipelineStatus(str, Enum):
    """Pipeline status values."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class StageResult:
    """Result of a single pipeline stage."""
    stage: PipelineStage
    success: bool
    duration_ms: int
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0


@dataclass
class PipelineResult:
    """Complete pipeline result."""
    content_id: Optional[str] = None
    original_text: str = ""
    simplified_text: str = ""
    translations: Dict[str, str] = field(default_factory=dict)
    validation_score: float = 0.0
    validation_passed: bool = False
    audio_paths: Dict[str, str] = field(default_factory=dict)
    status: PipelineStatus = PipelineStatus.PENDING
    total_duration_ms: int = 0
    stage_results: List[StageResult] = field(default_factory=list)
    error: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PipelineService:
    """
    Fully async pipeline service for content processing.
    
    Features:
    - All operations are properly async
    - Proper error handling with retries
    - Parallel translation for multiple languages
    - Non-blocking speech generation
    - Metrics tracking per stage
    """
    
    # Supported languages
    SUPPORTED_LANGUAGES = ['Hindi', 'Tamil', 'Telugu', 'Bengali', 'Marathi']
    
    # Supported subjects
    SUPPORTED_SUBJECTS = ['Mathematics', 'Science', 'Social Studies', 'English', 'History', 'Geography', 'General']
    
    # Grade level range
    MIN_GRADE = 5
    MAX_GRADE = 12
    
    # Retry configuration
    MAX_RETRIES = 3
    RETRY_BACKOFF_BASE = 2.0
    
    # Quality thresholds
    NCERT_ALIGNMENT_THRESHOLD = 0.80
    
    def __init__(self):
        """Initialize the pipeline service with lazy loading of dependencies."""
        self._unified_client = None
        self._speech_generator = None
        self._ai_orchestrator = None
        self._initialized = False
    
    async def _ensure_initialized(self):
        """Lazy initialize heavy dependencies."""
        if self._initialized:
            return
        
        # Use optimized AI stack (NLLB, Ollama, Edge TTS, BGE-M3)
        from .ai import get_ai_orchestrator
        
        self._ai_orchestrator = await get_ai_orchestrator()
        
        # Create a unified interface that wraps the orchestrator
        self._unified_client = AIUnifiedClient(self._ai_orchestrator)
        
        # Use Edge TTS from orchestrator (no separate speech generator needed)
        self._speech_generator = None
        
        logger.info("PipelineService initialized with optimized AI stack (NLLB, Ollama, Edge TTS, BGE-M3)")
        
        self._initialized = True
    
    def validate_inputs(
        self,
        text: str,
        grade_level: int,
        subject: str,
        target_languages: List[str]
    ) -> List[str]:
        """Validate pipeline inputs and return list of errors."""
        errors = []
        
        if not text or not text.strip():
            errors.append("Text cannot be empty")
        
        if grade_level < self.MIN_GRADE or grade_level > self.MAX_GRADE:
            errors.append(f"Grade level must be between {self.MIN_GRADE} and {self.MAX_GRADE}")
        
        if subject not in self.SUPPORTED_SUBJECTS:
            errors.append(f"Subject must be one of: {', '.join(self.SUPPORTED_SUBJECTS)}")
        
        for lang in target_languages:
            if lang not in self.SUPPORTED_LANGUAGES:
                errors.append(f"Language '{lang}' not supported. Use: {', '.join(self.SUPPORTED_LANGUAGES)}")
        
        return errors
    
    async def process_content(
        self,
        text: str,
        grade_level: int,
        subject: str,
        target_languages: List[str],
        generate_audio: bool = False,
        store_result: bool = True,
        user_id: Optional[str] = None,
    ) -> PipelineResult:
        """
        Process content through the complete pipeline.
        
        Args:
            text: Input text to process
            grade_level: Target grade level (5-12)
            subject: Subject area
            target_languages: List of target languages for translation
            generate_audio: Whether to generate TTS audio
            store_result: Whether to store result in database
            user_id: Optional user ID for tracking
        
        Returns:
            PipelineResult with all processed content
        """
        start_time = time.time()
        result = PipelineResult(original_text=text, status=PipelineStatus.IN_PROGRESS)
        
        # Validate inputs
        errors = self.validate_inputs(text, grade_level, subject, target_languages)
        if errors:
            result.status = PipelineStatus.FAILED
            result.error = "; ".join(errors)
            return result
        
        try:
            # Ensure dependencies are initialized
            await self._ensure_initialized()
            
            # Stage 1: Simplification
            simplify_result = await self._run_stage_with_retry(
                PipelineStage.SIMPLIFICATION,
                self._simplify_text,
                text, grade_level, subject
            )
            result.stage_results.append(simplify_result)
            
            if not simplify_result.success:
                result.status = PipelineStatus.FAILED
                result.error = f"Simplification failed: {simplify_result.error}"
                return result
            
            result.simplified_text = simplify_result.result
            
            # Stage 2: Translation (parallel for all languages)
            translation_tasks = [
                self._run_stage_with_retry(
                    PipelineStage.TRANSLATION,
                    self._translate_text,
                    result.simplified_text, lang
                )
                for lang in target_languages
            ]
            
            translation_results = await asyncio.gather(*translation_tasks, return_exceptions=True)
            
            for lang, trans_result in zip(target_languages, translation_results):
                if isinstance(trans_result, Exception):
                    logger.error(f"Translation to {lang} failed: {trans_result}")
                    result.stage_results.append(StageResult(
                        stage=PipelineStage.TRANSLATION,
                        success=False,
                        duration_ms=0,
                        error=str(trans_result)
                    ))
                elif trans_result.success:
                    result.translations[lang] = trans_result.result
                    result.stage_results.append(trans_result)
                else:
                    result.stage_results.append(trans_result)
            
            # Stage 3: Validation
            validation_result = await self._run_stage_with_retry(
                PipelineStage.VALIDATION,
                self._validate_content,
                text, result.simplified_text, grade_level, subject
            )
            result.stage_results.append(validation_result)
            
            if validation_result.success:
                result.validation_score = validation_result.result
                result.validation_passed = result.validation_score >= self.NCERT_ALIGNMENT_THRESHOLD
            
            # Stage 4: Speech Generation (if requested and translations exist)
            if generate_audio and result.translations:
                audio_tasks = [
                    self._run_stage_with_retry(
                        PipelineStage.SPEECH,
                        self._generate_speech,
                        translated_text, lang, subject
                    )
                    for lang, translated_text in result.translations.items()
                ]
                
                audio_results = await asyncio.gather(*audio_tasks, return_exceptions=True)
                
                for lang, audio_result in zip(result.translations.keys(), audio_results):
                    if isinstance(audio_result, Exception):
                        logger.warning(f"Audio generation for {lang} failed: {audio_result}")
                    elif audio_result.success and audio_result.result:
                        result.audio_paths[lang] = audio_result.result
                        result.stage_results.append(audio_result)
            
            # Store result in database if requested
            if store_result:
                content_id = await self._store_result(result, grade_level, subject, user_id)
                result.content_id = content_id
            
            # Determine final status
            if result.translations and result.simplified_text:
                result.status = PipelineStatus.SUCCESS
            elif result.simplified_text:
                result.status = PipelineStatus.PARTIAL
            else:
                result.status = PipelineStatus.FAILED
            
        except Exception as e:
            logger.exception(f"Pipeline processing failed: {e}")
            result.status = PipelineStatus.FAILED
            result.error = str(e)
        
        result.total_duration_ms = int((time.time() - start_time) * 1000)
        return result
    
    async def _run_stage_with_retry(
        self,
        stage: PipelineStage,
        func,
        *args,
        **kwargs
    ) -> StageResult:
        """Run a pipeline stage with retry logic."""
        last_error = None
        
        for attempt in range(self.MAX_RETRIES + 1):
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration_ms = int((time.time() - start_time) * 1000)
                
                logger.info(f"Stage {stage.value} completed in {duration_ms}ms (attempt {attempt + 1})")
                
                return StageResult(
                    stage=stage,
                    success=True,
                    duration_ms=duration_ms,
                    result=result,
                    retry_count=attempt
                )
                
            except Exception as e:
                last_error = e
                duration_ms = int((time.time() - start_time) * 1000)
                
                logger.warning(f"Stage {stage.value} failed (attempt {attempt + 1}): {e}")
                
                if attempt < self.MAX_RETRIES:
                    backoff = self.RETRY_BACKOFF_BASE ** attempt
                    logger.info(f"Retrying in {backoff}s...")
                    await asyncio.sleep(backoff)
        
        # All retries exhausted
        return StageResult(
            stage=stage,
            success=False,
            duration_ms=duration_ms,
            error=str(last_error),
            retry_count=self.MAX_RETRIES
        )
    
    async def _simplify_text(self, text: str, grade_level: int, subject: str) -> str:
        """Simplify text for target grade level using Ollama (Llama 3.2 3B)."""
        simplified = await self._unified_client.simplify_text(text, grade_level, subject)
        
        if not simplified or not simplified.strip():
            raise ValueError("Simplification produced empty result")
        
        return simplified
    
    async def _translate_text(self, text: str, target_language: str) -> str:
        """Translate text to target language using NLLB-200."""
        translated = await self._unified_client.translate_text(text, "English", target_language)
        
        if not translated or not translated.strip():
            raise ValueError(f"Translation to {target_language} produced empty result")
        
        return translated
    
    async def _validate_content(
        self,
        original_text: str,
        simplified_text: str,
        grade_level: int,
        subject: str
    ) -> float:
        """Validate content using BGE-M3 embeddings similarity."""
        if self._unified_client:
            # Use BGE-M3 embeddings for semantic validation
            try:
                score = await self._unified_client.validate_content(
                    original_text=original_text,
                    simplified_text=simplified_text,
                    grade_level=grade_level,
                    subject=subject
                )
                return score
            except Exception as e:
                logger.warning(f"Embedding validation failed, using rule-based: {e}")
        
        # Fallback: Rule-based validation
        len_ratio = min(len(simplified_text), len(original_text)) / max(len(simplified_text), len(original_text))
        score = 0.3 * len_ratio
        
        if simplified_text.strip() and len(simplified_text) >= len(original_text) * 0.3:
            score += 0.3
        
        if len(simplified_text) <= len(original_text):
            score += 0.4
        
        return min(1.0, max(0.0, score))
    
    async def _generate_speech(
        self,
        text: str,
        language: str,
        subject: str
    ) -> Optional[str]:
        """Generate speech audio using Edge TTS (FREE & unlimited)."""
        try:
            if self._ai_orchestrator:
                # Use Edge TTS via orchestrator
                result = await self._ai_orchestrator.synthesize_speech(
                    text=text,
                    language=language,
                    output_format="mp3"
                )
                
                if result.success and result.data:
                    # Save audio to file
                    import uuid
                    from pathlib import Path
                    
                    audio_dir = Path("data/audio")
                    audio_dir.mkdir(parents=True, exist_ok=True)
                    
                    filename = f"{uuid.uuid4()}_{language}.mp3"
                    file_path = audio_dir / filename
                    
                    with open(file_path, 'wb') as f:
                        f.write(result.data)
                    
                    return str(file_path)
                    
            return None
            
        except Exception as e:
            logger.warning(f"Speech generation failed: {e}")
            return None
    
    async def _store_result(
        self,
        result: PipelineResult,
        grade_level: int,
        subject: str,
        user_id: Optional[str]
    ) -> Optional[str]:
        """Store pipeline result in database."""
        try:
            from ..core.database import get_db_session
            from ..models import ProcessedContent
            import uuid
            
            with get_db_session() as session:
                # Get primary translation
                primary_lang = list(result.translations.keys())[0] if result.translations else "English"
                primary_translation = result.translations.get(primary_lang, result.simplified_text)
                
                content = ProcessedContent(
                    original_text=result.original_text,
                    simplified_text=result.simplified_text,
                    translated_text=primary_translation,
                    language=primary_lang,
                    grade_level=grade_level,
                    subject=subject,
                    ncert_alignment_score=result.validation_score,
                    audio_file_path=result.audio_paths.get(primary_lang),
                    user_id=uuid.UUID(user_id) if user_id else None,
                    content_metadata={
                        'all_translations': result.translations,
                        'all_audio_paths': result.audio_paths,
                        'validation_passed': result.validation_passed,
                        'total_duration_ms': result.total_duration_ms,
                        'pipeline_version': '2.0'
                    }
                )
                
                session.add(content)
                session.flush()
                content_id = str(content.id)
                
                logger.info(f"Pipeline result stored with ID: {content_id}")
                return content_id
                
        except Exception as e:
            logger.error(f"Failed to store pipeline result: {e}")
            return None


# Singleton instance
_pipeline_service: Optional[PipelineService] = None


def get_pipeline_service() -> PipelineService:
    """Get or create pipeline service singleton."""
    global _pipeline_service
    if _pipeline_service is None:
        _pipeline_service = PipelineService()
    return _pipeline_service


# Convenience function for quick processing
async def process_content(
    text: str,
    grade_level: int,
    subject: str,
    target_languages: List[str],
    generate_audio: bool = False
) -> PipelineResult:
    """Quick function to process content through the pipeline."""
    service = get_pipeline_service()
    return await service.process_content(
        text=text,
        grade_level=grade_level,
        subject=subject,
        target_languages=target_languages,
        generate_audio=generate_audio
    )
