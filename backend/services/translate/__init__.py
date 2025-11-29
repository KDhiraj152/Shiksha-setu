"""Translation module using NLLB-200 for multi-language content translation.

NEW TECH STACK:
- NLLB-200 (No Language Left Behind) - Meta's multilingual translation model
- Supports 200+ languages including all major Indian languages
- Optimized with CTranslate2 for fast CPU/GPU inference
"""

from .nllb_translator import NLLBTranslator, get_nllb_translator

__all__ = [
    'NLLBTranslator',
    'get_nllb_translator',
]
