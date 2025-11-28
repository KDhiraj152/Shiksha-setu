"""
RAG (Retrieval-Augmented Generation) Service for Q&A functionality.

This service handles:
- Text chunking from documents
- Embedding generation using sentence-transformers
- Vector similarity search using pgvector
- Context retrieval for question answering
"""

import logging
from typing import List, Dict, Tuple, Optional
from uuid import UUID
import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from sentence_transformers import SentenceTransformer

from ..database import SessionLocal, get_db
from ..models import DocumentChunk, Embedding, ProcessedContent

logger = logging.getLogger(__name__)


class RAGService:
    """Service for Retrieval-Augmented Generation operations."""
    
    def __init__(self, embedding_model: str = "intfloat/multilingual-e5-large"):
        """
        Initialize RAG service with embedding model.
        
        Args:
            embedding_model: HuggingFace model name for embeddings (default: E5-large)
        """
        self.embedding_model_name = embedding_model
        self.embedding_model = None
        self.embedding_dimension = 1024  # E5-large dimension
        self._verify_pgvector()
        
    def _verify_pgvector(self):
        """Verify pgvector extension is available at runtime."""
        session = SessionLocal()
        try:
            # Check if using PostgreSQL
            db_url = str(session.bind.url).lower()
            if 'postgresql' in db_url:
                result = session.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector'"))
                if not result.fetchone():
                    logger.warning("pgvector extension not installed in PostgreSQL. Vector operations will be limited.")
                else:
                    logger.info("pgvector extension verified")
            else:
                logger.info(f"Using {session.bind.dialect.name} - vector operations will be in-memory only")
        except Exception as e:
            logger.warning(f"pgvector verification skipped: {e}")
        finally:
            session.close()
    
    def _load_embedding_model(self):
        """Lazy load the embedding model with optimization."""
        if self.embedding_model is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            
            # Use optimized loader if available
            try:
                from ..utils.model_loader import ModelLoader
                from ..core.config import settings
                
                loader = ModelLoader()
                self.embedding_model = loader.load_embedding_model_optimized(
                    model_id=self.embedding_model_name,
                    use_onnx=settings.EMBEDDING_USE_ONNX
                )
            except (ImportError, AttributeError, Exception) as e:
                # Fallback to standard SentenceTransformer
                logger.warning(f"Failed to load optimized model, using standard: {e}")
                self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            # Verify dimension
            test_embedding = self.embedding_model.encode("test", convert_to_numpy=True)
            detected_dimension = len(test_embedding)
            
            if detected_dimension != self.embedding_dimension:
                logger.warning(
                    f"Detected dimension {detected_dimension} != expected {self.embedding_dimension}. "
                    f"Updating to {detected_dimension}."
                )
                self.embedding_dimension = detected_dimension
            
            logger.info(f"Embedding model loaded: {self.embedding_dimension}D vectors")
            self._verify_embedding_dimension()
    
    def _find_sentence_boundary(self, text: str, start: int, end: int) -> int:
        """Find the best sentence boundary within the range."""
        for punct in ['. ', '! ', '? ', '\n\n']:
            last_punct = text.rfind(punct, start, end)
            if last_punct != -1:
                return last_punct + len(punct)
        return end
    
    def chunk_text(
        self, 
        text: str, 
        chunk_size: int = 512, 
        overlap: int = 50
    ) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            chunk_size: Maximum characters per chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
        
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < text_length:
                end = self._find_sentence_boundary(text, start, end)
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - overlap if end < text_length else text_length
        
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks
    
    def _verify_embedding_dimension(self):
        """Verify embedding dimension matches database vector column."""
        session = SessionLocal()
        try:
            # Query vector column dimension from database
            result = session.execute(text("""
                SELECT atttypmod 
                FROM pg_attribute 
                WHERE attrelid = 'embeddings'::regclass 
                  AND attname = 'embedding'
            """))
            row = result.fetchone()
            
            if row and row[0] > 0:
                db_dimension = row[0]
                if db_dimension != self.embedding_dimension:
                    raise ValueError(
                        f"Embedding dimension mismatch: model produces {self.embedding_dimension}D "
                        f"but database expects {db_dimension}D vectors. "
                        f"Run migration: ALTER TABLE embeddings ALTER COLUMN embedding TYPE vector({self.embedding_dimension});"
                    )
                logger.info(f"Embedding dimension verified: {self.embedding_dimension}D")
        except Exception as e:
            logger.warning(f"Could not verify embedding dimension: {e}")
        finally:
            session.close()
    
    def generate_embedding(self, text: str, is_query: bool = False) -> List[float]:
        """
        Generate embedding vector for text with E5 task prefix.
        
        Args:
            text: Text to embed
            is_query: Whether this is a query (True) or passage/document (False)
            
        Returns:
            Embedding vector as list of floats
        """
        self._load_embedding_model()
        
        # E5 models require task prefix for best performance
        prefix = "query: " if is_query else "passage: "
        prefixed_text = f"{prefix}{text}"
        
        # Generate embedding
        embedding = self.embedding_model.encode(prefixed_text, convert_to_numpy=True)
        return embedding.tolist()
    
    def store_document_chunks(
        self,
        content_id: UUID,
        text: str,
        chunk_size: int = 512,
        overlap: int = 50,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Chunk document and store chunks with embeddings in database.
        
        Args:
            content_id: ID of the processed content
            text: Full document text
            chunk_size: Maximum characters per chunk
            overlap: Characters to overlap between chunks
            metadata: Optional metadata for chunks (page numbers, etc.)
            
        Returns:
            Number of chunks created
        """
        logger.info(f"Storing chunks for content {content_id}")
        
        # Chunk the text
        chunks = self.chunk_text(text, chunk_size, overlap)
        
        if not chunks:
            logger.warning(f"No chunks generated for content {content_id}")
            return 0
        
        # Generate embeddings for all chunks
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = [self.generate_embedding(chunk) for chunk in chunks]
        
        # Store in database
        session = SessionLocal()
        try:
            for idx, (chunk_text, embedding_vector) in enumerate(zip(chunks, embeddings)):
                # Create chunk record
                chunk = DocumentChunk(
                    content_id=content_id,
                    chunk_index=idx,
                    chunk_text=chunk_text,
                    chunk_size=len(chunk_text),
                    chunk_metadata=metadata or {}
                )
                session.add(chunk)
                session.flush()  # Get chunk ID
                
                # Create embedding record with vector
                embedding_record = Embedding(
                    chunk_id=chunk.id,
                    content_id=content_id,
                    embedding_model=self.embedding_model_name
                )
                session.add(embedding_record)
                session.flush()
                
                # Update vector column using raw SQL
                session.execute(
                    text("UPDATE embeddings SET embedding = :vector::vector WHERE id = :id"),
                    {"vector": str(embedding_vector), "id": str(embedding_record.id)}
                )
            
            session.commit()
            logger.info(f"Stored {len(chunks)} chunks with embeddings for content {content_id}")
            return len(chunks)
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing chunks: {e}")
            raise
        finally:
            session.close()
    
    def search_similar_chunks(
        self,
        query: str,
        content_id: UUID,
        top_k: int = 3,
        similarity_threshold: float = 0.3
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for chunks similar to query using vector similarity.
        
        Args:
            query: Question or search query
            content_id: ID of content to search within
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of (chunk, similarity_score) tuples
        """
        logger.info(f"Searching for chunks similar to: {query[:50]}...")
        
        # Generate query embedding with is_query=True for E5
        query_embedding = self.generate_embedding(query, is_query=True)
        
        # Search using pgvector cosine similarity
        session = SessionLocal()
        try:
            # Using raw SQL for vector similarity search with parameterized vector
            # CRITICAL FIX: Use proper parameter binding for vector to prevent SQL injection
            sql = text("""
                SELECT 
                    dc.id, 
                    dc.content_id,
                    dc.chunk_index,
                    dc.chunk_text,
                    dc.chunk_size,
                    dc.chunk_metadata,
                    dc.created_at,
                    1 - (e.embedding <=> CAST(:query_vector AS vector)) as similarity
                FROM document_chunks dc
                JOIN embeddings e ON dc.id = e.chunk_id
                WHERE dc.content_id = :content_id
                  AND 1 - (e.embedding <=> CAST(:query_vector AS vector)) >= :threshold
                ORDER BY e.embedding <=> CAST(:query_vector AS vector)
                LIMIT :limit
            """)
            
            result = session.execute(
                sql,
                {
                    "query_vector": str(query_embedding),
                    "content_id": str(content_id),
                    "threshold": similarity_threshold,
                    "limit": top_k
                }
            )
            
            results = []
            for row in result:
                chunk = DocumentChunk(
                    id=row.id,
                    content_id=row.content_id,
                    chunk_index=row.chunk_index,
                    chunk_text=row.chunk_text,
                    chunk_size=row.chunk_size,
                    chunk_metadata=row.chunk_metadata,
                    created_at=row.created_at
                )
                similarity = float(row.similarity)
                results.append((chunk, similarity))
            
            logger.info(f"Found {len(results)} similar chunks (similarity >= {similarity_threshold})")
            return results
            
        except Exception as e:
            logger.error(f"Error searching chunks: {e}")
            raise
        finally:
            session.close()
    
    def retrieve_context(
        self,
        question: str,
        content_id: UUID,
        top_k: int = 3
    ) -> Dict[str, any]:
        """
        Retrieve relevant context for answering a question.
        
        Args:
            question: User's question
            content_id: ID of content to search
            top_k: Number of chunks to retrieve
            
        Returns:
            Dict with context_text, chunk_ids, and scores
        """
        # Search for similar chunks
        similar_chunks = self.search_similar_chunks(question, content_id, top_k)
        
        if not similar_chunks:
            return {
                "context_text": "",
                "chunk_ids": [],
                "scores": [],
                "has_context": False
            }
        
        # Combine chunks into context
        context_parts = []
        chunk_ids = []
        scores = []
        
        for chunk, score in similar_chunks:
            context_parts.append(chunk.chunk_text)
            chunk_ids.append(chunk.id)
            scores.append(score)
        
        context_text = "\n\n".join(context_parts)
        
        return {
            "context_text": context_text,
            "chunk_ids": chunk_ids,
            "scores": scores,
            "has_context": True,
            "avg_score": sum(scores) / len(scores) if scores else 0.0
        }
    
    def delete_document_chunks(self, content_id: UUID) -> int:
        """
        Delete all chunks and embeddings for a document.
        
        Args:
            content_id: ID of content to delete chunks for
            
        Returns:
            Number of chunks deleted
        """
        session = SessionLocal()
        try:
            # Delete embeddings (will cascade to chunks due to foreign key)
            result = session.query(Embedding).filter(
                Embedding.content_id == content_id
            ).delete()
            
            session.commit()
            logger.info(f"Deleted {result} chunks/embeddings for content {content_id}")
            return result
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error deleting chunks: {e}")
            raise
        finally:
            session.close()


# Global RAG service instance
_rag_service = None


def get_rag_service() -> RAGService:
    """Get or create global RAG service instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service
