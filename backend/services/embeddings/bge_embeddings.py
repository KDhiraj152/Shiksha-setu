"""
BGE-M3 Embedding Service - State-of-the-Art Multilingual Embeddings

Uses BAAI's BGE-M3 model for high-quality multilingual embeddings.
Excellent support for Indian languages.

Features:
- Dense + Sparse + ColBERT retrieval in one model
- 100+ languages including all major Indian languages
- 1024-dimensional dense embeddings
- Optimized for semantic search and RAG
- Lazy loading with memory management

Memory: ~1.2GB
"""

import asyncio
import logging
import hashlib
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

try:
    from FlagEmbedding import BGEM3FlagModel
    BGE_AVAILABLE = True
except ImportError:
    BGE_AVAILABLE = False
    logger.warning("FlagEmbedding not available. Install with: pip install FlagEmbedding")

# Fallback to sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    embeddings: np.ndarray
    dimension: int
    model_used: str
    texts_count: int
    cached: bool = False


class BGEM3Embeddings:
    """
    BGE-M3 Embedding Service
    
    State-of-the-art multilingual embeddings with excellent
    support for Indian languages.
    
    Features:
    - Dense embeddings (1024-dim)
    - Sparse embeddings (lexical)
    - ColBERT embeddings (for late interaction)
    """
    
    MODEL_ID = "BAAI/bge-m3"
    EMBEDDING_DIMENSION = 1024
    
    # Fallback models if BGE-M3 not available
    FALLBACK_MODELS = [
        "BAAI/bge-large-en-v1.5",
        "sentence-transformers/all-MiniLM-L6-v2"
    ]
    
    def __init__(
        self,
        use_fp16: bool = True,
        device: str = "auto",
        cache_dir: Optional[str] = None,
        max_length: int = 8192
    ):
        """
        Initialize BGE-M3 Embedding service.
        
        Args:
            use_fp16: Use FP16 for lower memory usage
            device: Device to use ("auto", "cpu", "cuda", "mps")
            cache_dir: Directory for model cache
            max_length: Maximum sequence length
        """
        self.use_fp16 = use_fp16
        self.max_length = max_length
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/models/embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect device
        if device == "auto":
            import torch
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Model state
        self._model = None
        self._fallback_model = None
        self._loaded = False
        self._model_name = self.MODEL_ID
        
        # Embedding cache
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_max_size = 10000
        
        logger.info(f"BGEM3Embeddings initialized: device={self.device}, fp16={use_fp16}")
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for embedding."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    def _load_model(self):
        """Load model lazily on first use."""
        if self._loaded:
            return
        
        logger.info(f"Loading embedding model: {self.MODEL_ID}")
        
        if BGE_AVAILABLE:
            try:
                self._model = BGEM3FlagModel(
                    self.MODEL_ID,
                    use_fp16=self.use_fp16,
                    device=self.device if self.device != "mps" else "cpu"  # BGE doesn't support MPS directly
                )
                self._loaded = True
                logger.info("BGE-M3 model loaded successfully")
                return
            except Exception as e:
                logger.warning(f"Failed to load BGE-M3: {e}, trying fallback")
        
        # Fallback to sentence-transformers
        if ST_AVAILABLE:
            for fallback in self.FALLBACK_MODELS:
                try:
                    self._fallback_model = SentenceTransformer(
                        fallback,
                        cache_folder=str(self.cache_dir)
                    )
                    self._model_name = fallback
                    self._loaded = True
                    logger.info(f"Loaded fallback model: {fallback}")
                    return
                except Exception as e:
                    logger.warning(f"Failed to load {fallback}: {e}")
        
        raise RuntimeError("No embedding model could be loaded")
    
    def embed(
        self,
        texts: Union[str, List[str]],
        return_dense: bool = True,
        return_sparse: bool = False,
        return_colbert: bool = False
    ) -> EmbeddingResult:
        """
        Generate embeddings for text(s).
        
        Args:
            texts: Single text or list of texts
            return_dense: Return dense embeddings
            return_sparse: Return sparse embeddings (BGE-M3 only)
            return_colbert: Return ColBERT embeddings (BGE-M3 only)
            
        Returns:
            EmbeddingResult with embeddings and metadata
        """
        # Ensure model is loaded
        self._load_model()
        
        # Normalize input
        if isinstance(texts, str):
            texts = [texts]
        
        # Check cache for single texts
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                cached_embeddings.append((i, self._cache[cache_key]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            if self._model is not None:
                # Use BGE-M3
                output = self._model.encode(
                    uncached_texts,
                    return_dense=return_dense,
                    return_sparse=return_sparse,
                    return_colbert_vecs=return_colbert,
                    max_length=self.max_length
                )
                
                if return_dense:
                    new_embeddings = output['dense_vecs']
                else:
                    new_embeddings = output.get('dense_vecs', np.array([]))
            else:
                # Use fallback sentence-transformers
                new_embeddings = self._fallback_model.encode(
                    uncached_texts,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
            
            # Cache new embeddings
            for text, embedding in zip(uncached_texts, new_embeddings):
                cache_key = self._get_cache_key(text)
                if len(self._cache) >= self._cache_max_size:
                    oldest_key = next(iter(self._cache))
                    self._cache.pop(oldest_key)
                self._cache[cache_key] = embedding
        
        # Combine cached and new embeddings in correct order
        all_embeddings = [None] * len(texts)
        
        for i, emb in cached_embeddings:
            all_embeddings[i] = emb
        
        if uncached_texts:
            for idx, emb in zip(uncached_indices, new_embeddings):
                all_embeddings[idx] = emb
        
        embeddings = np.array(all_embeddings)
        
        return EmbeddingResult(
            embeddings=embeddings,
            dimension=embeddings.shape[1] if len(embeddings.shape) > 1 else self.EMBEDDING_DIMENSION,
            model_used=self._model_name,
            texts_count=len(texts),
            cached=len(cached_embeddings) == len(texts)
        )
    
    async def embed_async(
        self,
        texts: Union[str, List[str]],
        return_dense: bool = True
    ) -> EmbeddingResult:
        """Async embedding generation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.embed(texts, return_dense)
        )
    
    def similarity(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[tuple]:
        """
        Compute similarity between query and documents.
        
        Args:
            query: Query text
            documents: List of documents to compare
            top_k: Number of top results to return
            
        Returns:
            List of (document_index, score) tuples sorted by score
        """
        # Get embeddings
        query_result = self.embed(query)
        docs_result = self.embed(documents)
        
        query_emb = query_result.embeddings
        docs_emb = docs_result.embeddings
        
        # Compute cosine similarity
        query_norm = query_emb / np.linalg.norm(query_emb)
        docs_norm = docs_emb / np.linalg.norm(docs_emb, axis=1, keepdims=True)
        
        scores = np.dot(docs_norm, query_norm.T).flatten()
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return [(int(idx), float(scores[idx])) for idx in top_indices]
    
    async def similarity_async(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[tuple]:
        """Async similarity computation."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.similarity(query, documents, top_k)
        )
    
    def unload(self):
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._fallback_model is not None:
            del self._fallback_model
            self._fallback_model = None
        
        self._loaded = False
        
        # Clear cache
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        
        import gc
        gc.collect()
        
        logger.info("Embedding model unloaded")
    
    def clear_cache(self):
        """Clear embedding cache."""
        self._cache.clear()
        logger.info("Embedding cache cleared")
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.EMBEDDING_DIMENSION
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage info."""
        import torch
        info = {
            "model_loaded": self._loaded,
            "model_name": self._model_name,
            "cache_entries": len(self._cache),
            "embedding_dimension": self.EMBEDDING_DIMENSION
        }
        
        if self.device == "cuda" and torch.cuda.is_available():
            info["gpu_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
        
        return info


# Singleton instance
_embeddings_instance: Optional[BGEM3Embeddings] = None


def get_bge_embeddings(**kwargs) -> BGEM3Embeddings:
    """Get or create singleton BGE-M3 embeddings instance."""
    global _embeddings_instance
    
    if _embeddings_instance is None:
        _embeddings_instance = BGEM3Embeddings(**kwargs)
    
    return _embeddings_instance
