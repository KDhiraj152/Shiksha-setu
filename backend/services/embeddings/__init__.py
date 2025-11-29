"""
Embeddings service package.

Provides high-quality multilingual embeddings using BGE-M3.
"""

from .bge_embeddings import (
    BGEM3Embeddings,
    EmbeddingResult,
    get_bge_embeddings
)

__all__ = [
    "BGEM3Embeddings",
    "EmbeddingResult", 
    "get_bge_embeddings"
]
