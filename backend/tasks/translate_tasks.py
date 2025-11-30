"""
Translation Tasks (Celery)
===========================
Tasks for translation using IndicTrans2-1B.
"""

import logging
from typing import Dict, Any, Optional, List

from .celery_config import celery_app

logger = logging.getLogger(__name__)

# Lazy-loaded model
_translator = None


def get_translator():
    """Get or initialize translator (lazy loading)."""
    global _translator
    if _translator is None:
        from backend.services.translate.model import IndicTrans2Model
        _translator = IndicTrans2Model()
    return _translator


@celery_app.task(
    name="translate.text",
    bind=True,
    max_retries=2,
    default_retry_delay=5,
    soft_time_limit=120,
    time_limit=150,
)
def translate_text(
    self,
    text: str,
    source_lang: str,
    target_lang: str,
    options: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Translate text between languages.
    
    Args:
        text: Input text
        source_lang: Source language code (e.g., 'en', 'hi')
        target_lang: Target language code
        options: Additional options
        
    Returns:
        Dict with translated text
    """
    try:
        import asyncio
        
        translator = get_translator()
        options = options or {}
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                translator.translate(text, source_lang, target_lang)
            )
        finally:
            loop.close()
        
        return {
            "success": True,
            "translated_text": result,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "original_length": len(text),
            "translated_length": len(result),
        }
        
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        
        if self.request.retries < self.max_retries:
            raise self.retry(exc=e)
        
        return {
            "success": False,
            "error": str(e),
            "original_text": text,
        }


@celery_app.task(
    name="translate.batch",
    bind=True,
    max_retries=1,
    soft_time_limit=300,
    time_limit=330,
)
def translate_batch(
    self,
    texts: List[str],
    source_lang: str,
    target_lang: str,
) -> Dict[str, Any]:
    """
    Batch translate multiple texts.
    
    Args:
        texts: List of texts
        source_lang: Source language
        target_lang: Target language
        
    Returns:
        Dict with translations
    """
    try:
        import asyncio
        
        translator = get_translator()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            results = loop.run_until_complete(
                translator.translate_batch(texts, source_lang, target_lang)
            )
        finally:
            loop.close()
        
        return {
            "success": True,
            "translations": results,
            "count": len(texts),
            "source_lang": source_lang,
            "target_lang": target_lang,
        }
        
    except Exception as e:
        logger.error(f"Batch translation failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }


@celery_app.task(
    name="translate.detect_language",
    bind=True,
    soft_time_limit=30,
    time_limit=45,
)
def detect_language(self, text: str) -> Dict[str, Any]:
    """
    Detect language of text.
    
    Args:
        text: Input text
        
    Returns:
        Dict with detected language
    """
    try:
        # Simple language detection using common patterns
        # For production, use a proper language detection model
        
        # Check for Devanagari script (Hindi, Sanskrit, Marathi, etc.)
        devanagari_chars = sum(1 for c in text if '\u0900' <= c <= '\u097F')
        
        # Check for Bengali script
        bengali_chars = sum(1 for c in text if '\u0980' <= c <= '\u09FF')
        
        # Check for Tamil script
        tamil_chars = sum(1 for c in text if '\u0B80' <= c <= '\u0BFF')
        
        # Check for Telugu script
        telugu_chars = sum(1 for c in text if '\u0C00' <= c <= '\u0C7F')
        
        total_chars = len(text)
        
        if devanagari_chars / total_chars > 0.3:
            detected = "hi"  # Hindi (most common Devanagari)
        elif bengali_chars / total_chars > 0.3:
            detected = "bn"
        elif tamil_chars / total_chars > 0.3:
            detected = "ta"
        elif telugu_chars / total_chars > 0.3:
            detected = "te"
        else:
            detected = "en"  # Default to English
        
        return {
            "success": True,
            "detected_language": detected,
            "confidence": 0.8,  # Placeholder
        }
        
    except Exception as e:
        logger.error(f"Language detection failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "detected_language": "en",
        }


@celery_app.task(
    name="translate.supported_languages",
)
def get_supported_languages() -> Dict[str, Any]:
    """
    Get list of supported languages.
    
    Returns:
        Dict with supported language pairs
    """
    translator = get_translator()
    
    return {
        "success": True,
        "languages": translator.SUPPORTED_LANGUAGES if hasattr(translator, 'SUPPORTED_LANGUAGES') else [
            "en", "hi", "bn", "ta", "te", "mr", "gu", "kn", "ml", "pa", "or", "as"
        ],
    }
