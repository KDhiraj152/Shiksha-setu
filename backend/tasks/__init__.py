"""Celery tasks package.

Principle N: Each Celery worker loads (at most) one heavyweight model.

Worker Types:
- simplify: Llama-3.2-3B for text simplification
- translate: IndicTrans2-1B for translation
- ocr: GOT-OCR2 for document OCR
- embedding: BGE-M3 for embeddings
- rag: BGE-M3 + Reranker for RAG operations
"""
from .celery_config import celery_app

# Import task modules to register tasks
from . import simplify_tasks
from . import translate_tasks
from . import ocr_tasks
from . import embedding_tasks
from . import rag_tasks

__all__ = [
    'celery_app',
    'simplify_tasks',
    'translate_tasks',
    'ocr_tasks',
    'embedding_tasks',
    'rag_tasks',
]
