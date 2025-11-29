"""Pipeline tasks for async content processing.

IMPORTANT: All heavy ML imports are done INSIDE task functions to prevent:
1. Slow cold starts when loading Celery workers
2. Memory usage when tasks aren't running
3. Circular import issues

Pattern: Import inside the task function, not at module level.
"""
import logging
from typing import Dict, Any, Optional, List
from celery import Task, group, chain, chord
from celery.exceptions import SoftTimeLimitExceeded
import time
from datetime import datetime

from .celery_app import celery_app

# NOTE: Heavy imports (ML models, services) are done INSIDE task functions
# This is intentional for lazy loading and faster worker startup

logger = logging.getLogger(__name__)


class CallbackTask(Task):
    """Base task with callbacks for progress tracking."""
    
    def update_progress(self, task_id: str, progress: int, stage: str, message: str = ''):
        """Update task progress."""
        self.update_state(
            task_id=task_id,
            state='PROGRESS',
            meta={
                'progress': progress,
                'stage': stage,
                'message': message,
                'timestamp': time.time()
            }
        )


@celery_app.task(
    bind=True,
    base=CallbackTask,
    name='pipeline.extract_text',
    soft_time_limit=600,
    time_limit=900,
    max_retries=3
)
def extract_text_task(self, file_path: str, languages: List[str] = None) -> Dict[str, Any]:
    """
    Extract text from PDF/image file.
    
    Args:
        file_path: Path to file
        languages: OCR languages to use
        
    Returns:
        Dict with extracted text and metadata
    """
    # Lazy import - only load when task runs
    from ..services.ocr import OCRService
    
    try:
        self.update_progress(self.request.id, 10, 'extraction', 'Starting OCR...')
        
        # Initialize OCR service
        ocr_service = OCRService(languages=languages or ['English', 'Hindi'])
        
        self.update_progress(self.request.id, 30, 'extraction', 'Processing document...')
        
        # Extract text
        result = ocr_service.extract_text(file_path)
        
        # Validate
        if not ocr_service.validate_extraction(result):
            raise ValueError("Extraction validation failed")
        
        self.update_progress(self.request.id, 100, 'extraction', 'Extraction complete')
        
        return {
            'text': result.text,
            'num_pages': result.num_pages,
            'has_formulas': result.has_formulas,
            'formula_blocks': result.formula_blocks,
            'confidence': result.confidence,
            'metadata': result.metadata
        }
        
    except SoftTimeLimitExceeded:
        logger.error(f"Text extraction timed out for {file_path}")
        raise
    except Exception as e:
        logger.error(f"Text extraction failed: {e}")
        raise


@celery_app.task(
    bind=True,
    base=CallbackTask,
    name='pipeline.simplify_text',
    soft_time_limit=300,
    time_limit=600,
    max_retries=3
)
def simplify_text_task(
    self,
    text: str,
    grade_level: int,
    subject: str,
    formula_blocks: List[Dict] = None
) -> Dict[str, Any]:
    """
    Simplify text for target grade level using Ollama (Llama 3.2 3B).
    
    Args:
        text: Input text
        grade_level: Target grade (5-12)
        subject: Subject area
        formula_blocks: Preserved formulas
        
    Returns:
        Dict with simplified text
    """
    import asyncio
    
    try:
        self.update_progress(self.request.id, 10, 'simplification', 'Loading AI services...')
        
        # Use new optimized AI stack
        from ..services.ai import get_ai_orchestrator
        
        async def _simplify():
            orchestrator = await get_ai_orchestrator()
            result = await orchestrator.simplify_text(
                text=text,
                target_grade=grade_level,
                subject=subject
            )
            return result
        
        # Run async code in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(_simplify())
        finally:
            loop.close()
        
        self.update_progress(self.request.id, 70, 'simplification', 'Processing complete...')
        
        if not result.success:
            raise ValueError(f"Simplification failed: {result.error}")
        
        simplified = result.data["simplified_text"]
        
        # Restore formulas if present
        if formula_blocks:
            from ..services.ocr import MathFormulaDetector
            simplified = MathFormulaDetector.restore_formulas(simplified, formula_blocks)
        
        self.update_progress(self.request.id, 100, 'simplification', 'Simplification complete')
        
        return {
            'simplified_text': simplified,
            'grade_level': grade_level,
            'subject': subject,
            'model_used': result.model_used,
            'processing_time_ms': result.processing_time_ms
        }
        
    except SoftTimeLimitExceeded:
        logger.error("Text simplification timed out")
        raise
    except Exception as e:
        logger.error(f"Text simplification failed: {e}")
        raise


@celery_app.task(
    bind=True,
    base=CallbackTask,
    name='pipeline.translate_text',
    soft_time_limit=300,
    time_limit=600,
    max_retries=3
)
def translate_text_task(
    self,
    text: str,
    target_languages: List[str],
    formula_blocks: List[Dict] = None
) -> Dict[str, Any]:
    """
    Translate text to multiple languages using NLLB-200.
    
    Args:
        text: Input text
        target_languages: List of target languages
        formula_blocks: Preserved formulas
        
    Returns:
        Dict with translations
    """
    import asyncio
    
    try:
        self.update_progress(self.request.id, 10, 'translation', 'Loading NLLB-200 translator...')
        
        # Use new optimized AI stack with NLLB-200
        from ..services.ai import get_ai_orchestrator
        
        async def _translate_all():
            orchestrator = await get_ai_orchestrator()
            translations = {}
            
            for i, language in enumerate(target_languages):
                result = await orchestrator.translate(
                    text=text,
                    source_language="en",
                    target_language=language
                )
                if result.success:
                    translations[language] = result.data["translated_text"]
                else:
                    logger.warning(f"Translation to {language} failed: {result.error}")
                    translations[language] = f"[Translation failed] {text}"
                    
            return translations
        
        # Run async code in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            translations = loop.run_until_complete(_translate_all())
        finally:
            loop.close()
        
        self.update_progress(self.request.id, 80, 'translation', 'Post-processing...')
        
        # Restore formulas
        if formula_blocks:
            from ..services.ocr import MathFormulaDetector
            for lang, translated in translations.items():
                translations[lang] = MathFormulaDetector.restore_formulas(translated, formula_blocks)
        
        self.update_progress(self.request.id, 100, 'translation', 'Translation complete')
        
        return {
            'translations': translations,
            'languages': target_languages,
            'model': 'nllb-200-1.3B'
        }
        
    except SoftTimeLimitExceeded:
        logger.error("Translation timed out")
        raise
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        raise


@celery_app.task(
    bind=True,
    base=CallbackTask,
    name='pipeline.validate_content',
    soft_time_limit=180,
    time_limit=300,
    max_retries=2
)
def validate_content_task(
    self,
    original_text: str,
    processed_text: str,
    threshold: float = 0.80
) -> Dict[str, Any]:
    """
    Validate semantic similarity between original and processed text.
    
    Args:
        original_text: Original text
        processed_text: Processed/simplified text
        threshold: Minimum similarity score
        
    Returns:
        Dict with validation results
    """
    try:
        self.update_progress(self.request.id, 20, 'validation', 'Loading validator...')
        
        # Initialize validator
        from ..services.validate.validator import ValidationModule
        validator = ValidationModule()
        
        self.update_progress(self.request.id, 60, 'validation', 'Computing similarity...')
        
        # Validate
        validation_result = validator.validate_content(
            original_text=original_text,
            translated_text=processed_text,
            grade_level=5,
            subject='General',
            language='English'
        )
        
        # ValidationResult is a dataclass, access attributes directly
        similarity_score = validation_result.semantic_accuracy
        is_valid = similarity_score >= threshold
        
        self.update_progress(self.request.id, 100, 'validation', 'Validation complete')
        
        return {
            'similarity_score': similarity_score,
            'ncert_alignment_score': validation_result.ncert_alignment_score,
            'threshold': threshold,
            'is_valid': is_valid,
            'requires_review': not is_valid,
            'overall_status': validation_result.overall_status,
            'issues': validation_result.issues,
            'recommendations': validation_result.recommendations,
            'quality_metrics': validation_result.quality_metrics
        }
        
    except Exception as e:
        logger.error(f"Content validation failed: {e}")
        raise


@celery_app.task(
    bind=True,
    base=CallbackTask,
    name='pipeline.generate_audio',
    soft_time_limit=600,
    time_limit=900,
    max_retries=3
)
def generate_audio_task(
    self,
    text: str,
    language: str,
    content_id: str
) -> Dict[str, Any]:
    """
    Generate audio from text using Edge TTS (FREE & unlimited).
    
    Args:
        text: Input text
        language: Target language
        content_id: Content ID for file naming
        
    Returns:
        Dict with audio file path and metadata
    """
    import asyncio
    from pathlib import Path
    
    try:
        self.update_progress(self.request.id, 20, 'audio', f'Generating {language} audio with Edge TTS...')
        
        # Use new optimized AI stack with Edge TTS
        from ..services.ai import get_ai_orchestrator
        
        async def _generate_audio():
            orchestrator = await get_ai_orchestrator()
            result = await orchestrator.synthesize_speech(
                text=text,
                language=language,
                output_format="mp3"
            )
            return result
        
        # Run async code in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(_generate_audio())
        finally:
            loop.close()
        
        self.update_progress(self.request.id, 70, 'audio', 'Saving audio file...')
        
        if not result.success or not result.data:
            raise ValueError(f"Audio generation failed: {result.error}")
        
        # Save audio to file
        audio_dir = Path("data/audio")
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{content_id}_{language}.mp3"
        file_path = audio_dir / filename
        
        with open(file_path, 'wb') as f:
            f.write(result.data)
        
        # Estimate duration (rough estimate: ~150 words per minute)
        word_count = len(text.split())
        estimated_duration = word_count / 150 * 60  # seconds
        
        self.update_progress(self.request.id, 100, 'audio', 'Audio generation complete')
        
        return {
            'audio_path': str(file_path),
            'audio_url': f'/api/v1/content/audio/{content_id}',
            'duration': estimated_duration,
            'language': language,
            'model': 'edge-tts',
            'processing_time_ms': result.processing_time_ms
        }
        
    except Exception as e:
        logger.error(f"Audio generation failed: {e}")
        raise


@celery_app.task(
    bind=True,
    base=CallbackTask,
    name='pipeline.full_pipeline',
    soft_time_limit=1500,
    time_limit=1800,
    max_retries=2
)
def full_pipeline_task(
    self,
    file_path: str,
    grade_level: int,
    subject: str,
    target_languages: List[str],
    output_format: str = 'both',
    validation_threshold: float = 0.80
) -> Dict[str, Any]:
    """
    Execute full content processing pipeline.
    
    Args:
        file_path: Path to input file
        grade_level: Target grade level
        subject: Subject area
        target_languages: List of target languages
        output_format: 'text', 'audio', or 'both'
        validation_threshold: Minimum similarity threshold
        
    Returns:
        Complete processing results
    """
    try:
        task_id = self.request.id
        logger.info(f"Starting full pipeline task {task_id}")
        
        # Stage 1: Extract text (0-20%)
        self.update_progress(task_id, 5, 'pipeline', 'Starting extraction...')
        extraction_result = extract_text_task(file_path)
        
        original_text = extraction_result['text']
        formula_blocks = extraction_result.get('formula_blocks', [])
        
        # Stage 2: Simplify (20-40%)
        self.update_progress(task_id, 25, 'pipeline', 'Simplifying content...')
        simplification_result = simplify_text_task(
            text=original_text,
            grade_level=grade_level,
            subject=subject,
            formula_blocks=formula_blocks
        )
        
        simplified_text = simplification_result['simplified_text']
        
        # Stage 3: Validate (40-50%)
        self.update_progress(task_id, 45, 'pipeline', 'Validating content...')
        validation_result = validate_content_task(
            original_text=original_text,
            processed_text=simplified_text,
            threshold=validation_threshold
        )
        
        # Stage 4: Translate (50-70%)
        self.update_progress(task_id, 55, 'pipeline', 'Translating...')
        translation_result = translate_text_task(
            text=simplified_text,
            target_languages=target_languages,
            formula_blocks=formula_blocks
        )
        
        # Stage 5: Generate audio (70-100%)
        audio_results = {}
        
        if output_format in ['audio', 'both']:
            self.update_progress(task_id, 75, 'pipeline', 'Generating audio...')
            
            # Generate audio for each language
            for lang in target_languages:
                audio_result = generate_audio_task(
                    text=translation_result['translations'][lang],
                    language=lang,
                    content_id=task_id
                )
                audio_results[lang] = audio_result
        
        # Save to database
        self.update_progress(task_id, 95, 'pipeline', 'Saving results...')
        
        from ..core.database import get_db_session
        
        with get_db_session() as session:
            try:
                # Create content record for each language
                content_records = []
                
                for lang in target_languages:
                    content = ProcessedContent(
                        original_text=original_text,
                        simplified_text=simplified_text,
                        translated_text=translation_result['translations'][lang],
                        language=lang,
                        grade_level=grade_level,
                        subject=subject,
                        audio_file_path=audio_results.get(lang, {}).get('audio_path'),
                        ncert_alignment_score=validation_result['similarity_score'],
                        audio_accuracy_score=audio_results.get(lang, {}).get('accuracy_score'),
                        content_metadata={
                            'extraction': extraction_result['metadata'],
                            'formulas': len(formula_blocks),
                            'validation': validation_result,
                            'task_id': task_id
                        }
                    )
                    session.add(content)
                    content_records.append(content)
                
                session.flush()
                
                # Get content IDs
                content_ids = {
                    record.language: str(record.id)
                    for record in content_records
                }
                
            except Exception as e:
                logger.error(f"Failed to save content: {e}")
                raise
        
        self.update_progress(task_id, 100, 'pipeline', 'Pipeline complete!')
        
        # Build final result
        return {
            'status': 'completed',
            'task_id': task_id,
            'result': {
                'content_ids': content_ids,
                'original_text': original_text[:500] + '...' if len(original_text) > 500 else original_text,
                'simplified_text': simplified_text[:500] + '...' if len(simplified_text) > 500 else simplified_text,
                'translations': {
                    lang: text[:500] + '...' if len(text) > 500 else text
                    for lang, text in translation_result['translations'].items()
                },
                'audio': {
                    lang: {
                        'url': result.get('audio_url'),
                        'duration': result.get('duration')
                    }
                    for lang, result in audio_results.items()
                },
                'metadata': {
                    'grade_level': grade_level,
                    'subject': subject,
                    'num_pages': extraction_result['num_pages'],
                    'has_formulas': extraction_result['has_formulas'],
                    'similarity_score': validation_result['similarity_score'],
                    'requires_review': validation_result['requires_review'],
                    'extraction_confidence': extraction_result['confidence']
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Full pipeline failed: {e}", exc_info=True)
        raise


@celery_app.task(name='pipeline.cleanup_old_results')
def cleanup_old_results():
    """Periodic task to clean up old task results from Redis."""
    try:
        # Clean up results older than 24 hours
        # This is handled automatically by Celery's result_expires setting
        logger.info("Cleanup task executed")
        return {'status': 'cleaned'}
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        raise


# Export
__all__ = [
    'extract_text_task',
    'simplify_text_task',
    'translate_text_task',
    'validate_content_task',
    'generate_audio_task',
    'full_pipeline_task',
    'cleanup_old_results'
]
