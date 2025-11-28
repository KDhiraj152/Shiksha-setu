"""Translation service with async support for streaming."""
from typing import Optional
import asyncio

from ..translate.engine import TranslationEngine
from ..utils.logging import get_logger

logger = get_logger(__name__)


class TranslationService:
    """Wrapper service for translation with async streaming support."""
    
    def __init__(self):
        self.engine = TranslationEngine()
    
    async def translate_async(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> str:
        """
        Async translation method for WebSocket streaming.
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated text
        """
        # Run synchronous translation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.engine.translate,
            text,
            source_lang,
            target_lang
        )
        return result
