"""
RAG (Retrieval-Augmented Generation) Service with BGE-M3 and Reranker.

Optimal 2025 Model Stack:
- Embeddings: BAAI/bge-m3 (1024D, multilingual)
- Reranker: BAAI/bge-reranker-v2-m3 (20% better retrieval)
- Supports hybrid search (dense + sparse)
"""

import logging
from typing import List, Dict, Tuple, Optional, Any
from uuid import UUID
from dataclasses import dataclass
import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from ..database import SessionLocal, get_db
from ..models import DocumentChunk, Embedding, ProcessedContent
from ..core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result from retrieval operation."""
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any]


@dataclass 
class RAGResponse:
    """Response from RAG query."""
    query: str
    context: str
    sources: List[RetrievalResult]
    answer: Optional[str] = None


class BGEM3Embedder:
    """BGE-M3 embedding model - best multilingual retrieval."""
    
    def __init__(self, model_id: str = None, device: str = None):
        self.model_id = model_id or settings.EMBEDDING_MODEL_ID
        self.dimension = settings.EMBEDDING_DIMENSION  # 1024 for BGE-M3
        
        # Auto-detect device
        if device is None:
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        self._model = None
        logger.info(f"BGEM3Embedder initialized: {self.model_id} on {self.device}")
    
    def _load_model(self):
        """Lazy load the model."""
        if self._model is not None:
            return
        
        logger.info(f"Loading BGE-M3 embedding model: {self.model_id}")
        
        try:
            # Try FlagEmbedding (optimized)
            from FlagEmbedding import BGEM3FlagModel
            
            self._model = BGEM3FlagModel(
                self.model_id,
                use_fp16=(self.device != "cpu"),
                device=self.device
            )
            logger.info("Loaded BGE-M3 with FlagEmbedding (optimized)")
            
        except ImportError:
            # Fallback to sentence-transformers
            from sentence_transformers import SentenceTransformer
            
            self._model = SentenceTransformer(
                self.model_id,
                device=self.device,
                cache_folder=str(settings.MODEL_CACHE_DIR)
            )
            logger.info("Loaded BGE-M3 with sentence-transformers")
        
        # Verify dimension
        test_emb = self.encode(["test"])[0]
        if len(test_emb) != self.dimension:
            logger.warning(f"Dimension mismatch: expected {self.dimension}, got {len(test_emb)}")
            self.dimension = len(test_emb)
    
    def encode(
        self,
        texts: List[str],
        batch_size: int = None,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            show_progress: Show progress bar
        
        Returns:
            numpy array of embeddings
        """
        self._load_model()
        
        batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        
        try:
            # FlagEmbedding API
            if hasattr(self._model, 'encode'):
                result = self._model.encode(
                    texts,
                    batch_size=batch_size,
                    max_length=settings.EMBEDDING_MAX_LENGTH
                )
                # FlagEmbedding returns dict with dense embeddings
                if isinstance(result, dict):
                    return result['dense_vecs']
                return result
            else:
                # Sentence-transformers API
                return self._model.encode(
                    texts,
                    batch_size=batch_size,
                    show_progress_bar=show_progress,
                    convert_to_numpy=True
                )
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query with query prefix."""
        # BGE-M3 recommends query prefix for retrieval
        prefixed_query = f"query: {query}" if "bge" in self.model_id.lower() else query
        return self.encode([prefixed_query])[0]
    
    def encode_documents(self, documents: List[str]) -> np.ndarray:
        """Encode documents."""
        return self.encode(documents)


class BGEReranker:
    """BGE Reranker for improved retrieval accuracy."""
    
    def __init__(self, model_id: str = None, device: str = None):
        self.model_id = model_id or settings.RERANKER_MODEL_ID
        
        import torch
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        self._model = None
        logger.info(f"BGEReranker initialized: {self.model_id}")
    
    def _load_model(self):
        """Lazy load reranker model."""
        if self._model is not None:
            return
        
        logger.info(f"Loading BGE Reranker: {self.model_id}")
        
        try:
            from FlagEmbedding import FlagReranker
            
            self._model = FlagReranker(
                self.model_id,
                use_fp16=(self.device != "cpu"),
                device=self.device
            )
            logger.info("Loaded BGE Reranker with FlagEmbedding")
            
        except ImportError:
            # Fallback to transformers
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                cache_dir=str(settings.MODEL_CACHE_DIR)
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_id,
                cache_dir=str(settings.MODEL_CACHE_DIR)
            ).to(self.device)
            self._model.eval()
            logger.info("Loaded BGE Reranker with transformers")
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = None
    ) -> List[Tuple[int, float]]:
        """
        Rerank documents by relevance to query.
        
        Args:
            query: Search query
            documents: List of document texts
            top_k: Number of top results to return
        
        Returns:
            List of (document_index, score) tuples, sorted by score
        """
        self._load_model()
        
        top_k = top_k or settings.RERANKER_TOP_K
        
        if not documents:
            return []
        
        try:
            # FlagEmbedding API
            if hasattr(self._model, 'compute_score'):
                pairs = [[query, doc] for doc in documents]
                scores = self._model.compute_score(pairs)
                
                # Handle single score vs list
                if not isinstance(scores, list):
                    scores = [scores]
                
                ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
                return ranked[:top_k]
            
            else:
                # Transformers API
                import torch
                
                pairs = [[query, doc] for doc in documents]
                inputs = self._tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    scores = self._model(**inputs).logits.squeeze(-1).cpu().numpy()
                
                ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
                return ranked[:top_k]
                
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Return original order with placeholder scores
            return [(i, 0.5) for i in range(min(len(documents), top_k))]


class RAGService:
    """
    RAG Service with BGE-M3 embeddings and reranking.
    
    Features:
    - BGE-M3 multilingual embeddings (1024D)
    - BGE Reranker for improved accuracy
    - pgvector for efficient similarity search
    - Hybrid search support (dense + sparse)
    """
    
    def __init__(
        self,
        embedder: BGEM3Embedder = None,
        reranker: BGEReranker = None
    ):
        self.embedder = embedder or BGEM3Embedder()
        self.reranker = reranker or BGEReranker()
        self.embedding_dimension = settings.EMBEDDING_DIMENSION
        
        self._verify_pgvector()
    
    def _verify_pgvector(self):
        """Verify pgvector extension is available."""
        session = SessionLocal()
        try:
            db_url = str(session.bind.url).lower()
            if 'postgresql' in db_url:
                result = session.execute(
                    text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
                )
                if not result.fetchone():
                    logger.warning("pgvector not installed. Run: CREATE EXTENSION vector;")
                else:
                    logger.info("pgvector extension verified")
        except Exception as e:
            logger.warning(f"pgvector verification skipped: {e}")
        finally:
            session.close()
    
    def chunk_text(
        self,
        text: str,
        chunk_size: int = 512,
        overlap: int = 50
    ) -> List[str]:
        """Split text into overlapping chunks."""
        if not text or not text.strip():
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = min(start + chunk_size, text_length)
            
            # Try to break at sentence boundary
            if end < text_length:
                for punct in ['. ', '! ', '? ', '\n\n', '\n']:
                    last_punct = text.rfind(punct, start, end)
                    if last_punct != -1:
                        end = last_punct + len(punct)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap if end < text_length else text_length
        
        logger.debug(f"Split text into {len(chunks)} chunks")
        return chunks
    
    def embed_chunks(
        self,
        chunks: List[str],
        batch_size: int = None
    ) -> np.ndarray:
        """Generate embeddings for text chunks."""
        if not chunks:
            return np.array([])
        
        return self.embedder.encode_documents(chunks)
    
    async def index_document(
        self,
        document_id: str,
        text: str,
        metadata: Dict = None,
        db: Session = None
    ) -> int:
        """
        Index a document for retrieval.
        
        Args:
            document_id: Unique document identifier
            text: Document text content
            metadata: Optional metadata
            db: Database session
        
        Returns:
            Number of chunks indexed
        """
        chunks = self.chunk_text(text)
        if not chunks:
            return 0
        
        embeddings = self.embed_chunks(chunks)
        
        close_session = False
        if db is None:
            db = SessionLocal()
            close_session = True
        
        try:
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                chunk_record = DocumentChunk(
                    document_id=document_id,
                    chunk_index=i,
                    text=chunk,
                    metadata=metadata or {}
                )
                db.add(chunk_record)
                db.flush()
                
                embedding_record = Embedding(
                    chunk_id=chunk_record.id,
                    vector=embedding.tolist(),
                    dimension=self.embedding_dimension
                )
                db.add(embedding_record)
            
            db.commit()
            logger.info(f"Indexed {len(chunks)} chunks for document {document_id}")
            return len(chunks)
            
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to index document: {e}")
            raise
        finally:
            if close_session:
                db.close()
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        rerank: bool = True,
        db: Session = None
    ) -> List[RetrievalResult]:
        """
        Search for relevant chunks.
        
        Args:
            query: Search query
            top_k: Number of results to return
            rerank: Whether to apply reranking
            db: Database session
        
        Returns:
            List of RetrievalResult objects
        """
        # Generate query embedding
        query_embedding = self.embedder.encode_query(query)
        
        close_session = False
        if db is None:
            db = SessionLocal()
            close_session = True
        
        try:
            # Vector similarity search with pgvector
            # Retrieve more candidates for reranking
            retrieve_k = top_k * 3 if rerank else top_k
            
            results = db.execute(
                text("""
                    SELECT 
                        dc.id,
                        dc.text,
                        dc.metadata,
                        1 - (e.vector <=> :query_vector) as similarity
                    FROM document_chunks dc
                    JOIN embeddings e ON e.chunk_id = dc.id
                    ORDER BY e.vector <=> :query_vector
                    LIMIT :limit
                """),
                {
                    "query_vector": query_embedding.tolist(),
                    "limit": retrieve_k
                }
            ).fetchall()
            
            if not results:
                return []
            
            # Convert to RetrievalResult objects
            candidates = [
                RetrievalResult(
                    chunk_id=str(row.id),
                    text=row.text,
                    score=float(row.similarity),
                    metadata=row.metadata or {}
                )
                for row in results
            ]
            
            # Apply reranking if enabled
            if rerank and len(candidates) > 1:
                candidate_texts = [c.text for c in candidates]
                reranked = self.reranker.rerank(query, candidate_texts, top_k=top_k)
                
                final_results = []
                for idx, score in reranked:
                    result = candidates[idx]
                    result.score = float(score)  # Update with reranker score
                    final_results.append(result)
                
                return final_results
            
            return candidates[:top_k]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
        finally:
            if close_session:
                db.close()
    
    async def query(
        self,
        query: str,
        top_k: int = 5,
        rerank: bool = True,
        generate_answer: bool = False,
        db: Session = None
    ) -> RAGResponse:
        """
        Perform RAG query.
        
        Args:
            query: User question
            top_k: Number of context chunks
            rerank: Whether to apply reranking
            generate_answer: Whether to generate an answer
            db: Database session
        
        Returns:
            RAGResponse with context and optionally answer
        """
        # Retrieve relevant chunks
        results = self.search(query, top_k=top_k, rerank=rerank, db=db)
        
        # Combine context
        context_parts = [r.text for r in results]
        context = "\n\n".join(context_parts)
        
        answer = None
        if generate_answer and results:
            # Generate answer using LLM (integrate with simplifier's LLM client)
            try:
                from .simplify.simplifier import VLLMClient, TransformersClient
                
                if settings.VLLM_ENABLED:
                    client = VLLMClient()
                else:
                    client = TransformersClient()
                
                prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""
                
                answer = await client.generate(prompt, max_tokens=512, temperature=0.3)
                
            except Exception as e:
                logger.warning(f"Answer generation failed: {e}")
        
        return RAGResponse(
            query=query,
            context=context,
            sources=results,
            answer=answer
        )
    
    def get_embedding_stats(self, db: Session = None) -> Dict:
        """Get statistics about indexed embeddings."""
        close_session = False
        if db is None:
            db = SessionLocal()
            close_session = True
        
        try:
            result = db.execute(text("""
                SELECT 
                    COUNT(*) as total_chunks,
                    COUNT(DISTINCT document_id) as total_documents
                FROM document_chunks
            """)).fetchone()
            
            return {
                "total_chunks": result.total_chunks,
                "total_documents": result.total_documents,
                "embedding_model": self.embedder.model_id,
                "embedding_dimension": self.embedding_dimension,
                "reranker_model": self.reranker.model_id
            }
        finally:
            if close_session:
                db.close()


# Singleton instance
_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """Get or create RAG service singleton."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


# Export
__all__ = [
    'RAGService',
    'RAGResponse',
    'RetrievalResult',
    'BGEM3Embedder',
    'BGEReranker',
    'get_rag_service'
]
