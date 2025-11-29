"""Service package for ShikshaSetu.

This package provides all business logic services including:
- Content processing (simplification, translation, TTS)
- OCR and document extraction
- RAG-based Q&A
- Authentication and user management
- Progress tracking and analytics
- Request caching
- Streaming responses
- Unified AI orchestration

NEW TECH STACK:
- Translation: NLLB-200 (via NLLBTranslator)
- Simplification: Ollama/Llama 3.2 (via OllamaSimplifier)
- Speech: Edge TTS (via EdgeTTSGenerator)
- Embeddings: BGE-M3 (via BGEM3Embeddings)
- Orchestration: AIOrchestrator (unified interface)
"""

from .ocr import OCRService, PDFExtractor, TesseractOCR, MathFormulaDetector, ExtractionResult
from .captions import WhisperCaptionService, CaptionResult, Caption

# New AI Stack - Translation (NLLB-200)
from .translate import NLLBTranslator, get_nllb_translator

# New AI Stack - Simplification (Ollama/Llama 3.2)
from .simplify import OllamaSimplifier, get_ollama_simplifier

# New AI Stack - Speech (Edge TTS)
from .speech import EdgeTTSGenerator, get_edge_tts_generator

# New AI Stack - Embeddings (BGE-M3)
from .embeddings import BGEM3Embeddings, get_bge_embeddings

# Lazy imports for caching and streaming
def get_request_cache():
    """Get request cache instance."""
    from .request_cache import get_request_cache as _get_cache
    return _get_cache()

def get_streaming_service():
    """Get streaming service instance."""
    from .streaming import StreamingService
    return StreamingService()

# AI orchestrator (lazy import)
async def get_ai_orchestrator():
    """Get the unified AI orchestrator instance."""
    from .ai import get_ai_orchestrator as _get_orchestrator
    return await _get_orchestrator()

__all__ = [
    # OCR Services
    'OCRService',
    'PDFExtractor',
    'TesseractOCR',
    'MathFormulaDetector',
    'ExtractionResult',
    # Caption Services
    'WhisperCaptionService',
    'CaptionResult',
    'Caption',
    # NEW AI Stack - Translation (NLLB-200)
    'NLLBTranslator',
    'get_nllb_translator',
    # NEW AI Stack - Simplification (Ollama/Llama 3.2)
    'OllamaSimplifier',
    'get_ollama_simplifier',
    # NEW AI Stack - Speech (Edge TTS)
    'EdgeTTSGenerator',
    'get_edge_tts_generator',
    # NEW AI Stack - Embeddings (BGE-M3)
    'BGEM3Embeddings',
    'get_bge_embeddings',
    # Utility Services
    'get_request_cache',
    'get_streaming_service',
    # AI Orchestrator
    'get_ai_orchestrator',
]
