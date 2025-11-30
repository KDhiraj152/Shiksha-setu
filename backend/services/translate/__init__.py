"""Translation using IndicTrans2-1B - Best-in-class for Indian languages."""

from .model import IndicTrans2, TranslationResult, get_translator
from .engine import TranslationEngine, TranslatedText
from .service import TranslationService

__all__ = [
    'IndicTrans2',
    'TranslationResult',
    'get_translator',
    'TranslationEngine',
    'TranslatedText',
    'TranslationService'
]
