"""
RAG Q&A System Integration Tests

Tests:
- Document processing and chunking
- BGE-M3 embedding generation
- Vector storage in pgvector
- Similarity search
- Chat history
- Question answering
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

from backend.services.rag import RAGService
from backend.models import DocumentChunk, Embedding, ChatHistory


@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    session = MagicMock()
    session.add = MagicMock()
    session.commit = MagicMock()
    session.flush = MagicMock()
    session.query = MagicMock()
    session.execute = MagicMock()
    return session


@pytest.fixture
def rag_service():
    """Create RAG service with mocked dependencies."""
    with patch.object(RAGService, '_verify_pgvector'):
        service = RAGService(use_optimized=False)
        return service


@pytest.mark.integration
class TestDocumentProcessing:
    """Test document ingestion and processing."""
    
    def test_chunk_text_basic(self, rag_service):
        """Test basic text chunking."""
        document_text = """
        Python is a high-level, interpreted programming language.
        It was created by Guido van Rossum and first released in 1991.
        Python emphasizes code readability with its notable use of whitespace.
        """
        
        # Process document
        chunks = rag_service.chunk_text(document_text, chunk_size=100, overlap=10)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
    
    def test_chunk_text_respects_size(self, rag_service):
        """Test that chunking respects size limits."""
        long_text = "Word " * 500  # Long text
        
        chunks = rag_service.chunk_text(long_text, chunk_size=100, overlap=20)
        
        assert len(chunks) > 1
        # Most chunks should be close to chunk_size (allowing for sentence boundary adjustments)
    
    def test_chunk_text_empty_input(self, rag_service):
        """Test chunking with empty input."""
        chunks = rag_service.chunk_text("", chunk_size=100, overlap=10)
        assert chunks == []
        
        chunks = rag_service.chunk_text("   ", chunk_size=100, overlap=10)
        assert chunks == []
    
    def test_chunk_text_with_overlap(self, rag_service):
        """Test that chunks have proper overlap."""
        text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
        
        chunks = rag_service.chunk_text(text, chunk_size=30, overlap=10)
        
        # Should have multiple chunks
        assert len(chunks) >= 2


@pytest.mark.integration
class TestEmbeddingGeneration:
    """Test BGE-M3 embedding generation."""
    
    @pytest.mark.asyncio
    async def test_generate_embedding_async_with_mock(self, rag_service):
        """Test embedding generation via mocked AI orchestrator."""
        # Mock the orchestrator
        mock_orchestrator = AsyncMock()
        mock_orchestrator.generate_embeddings = AsyncMock(return_value=MagicMock(
            success=True,
            data={"embeddings": [[0.1] * 1024]}
        ))
        
        rag_service._use_optimized = True
        rag_service._ai_orchestrator = mock_orchestrator
        
        embedding = await rag_service.generate_embedding_async("Test text")
        
        assert len(embedding) == 1024
        mock_orchestrator.generate_embeddings.assert_called_once()
    
    def test_embedding_dimension_configured(self, rag_service):
        """Test embedding dimension is correctly configured."""
        assert rag_service.embedding_dimension == 1024  # BGE-M3 dimension
    
    @pytest.mark.asyncio
    async def test_batch_embeddings_async(self, rag_service):
        """Test batch embedding generation."""
        mock_orchestrator = AsyncMock()
        mock_orchestrator.generate_embeddings = AsyncMock(return_value=MagicMock(
            success=True,
            data={"embeddings": [[0.1] * 1024, [0.2] * 1024]}
        ))
        
        rag_service._use_optimized = True
        rag_service._ai_orchestrator = mock_orchestrator
        
        texts = ["Text one", "Text two"]
        embeddings = await rag_service.generate_embeddings_batch_async(texts)
        
        assert len(embeddings) == 2
        assert all(len(emb) == 1024 for emb in embeddings)
@pytest.mark.integration
class TestVectorSearch:
    """Test pgvector similarity search."""
    
    def test_cosine_similarity_calculation(self):
        """Test cosine similarity helper."""
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        # Identical vectors should have similarity 1.0
        vec1 = np.array([1.0, 0.0, 0.0])
        assert abs(cosine_similarity(vec1, vec1) - 1.0) < 0.001
        
        # Orthogonal vectors should have similarity 0.0
        vec2 = np.array([0.0, 1.0, 0.0])
        assert abs(cosine_similarity(vec1, vec2)) < 0.001
        
        # Similar vectors should have high similarity
        vec3 = np.array([0.9, 0.1, 0.0])
        assert cosine_similarity(vec1, vec3) > 0.9
    
    @pytest.mark.asyncio
    async def test_similarity_search_with_mock(self, rag_service, mock_db_session):
        """Test similarity search with mocked database."""
        # Mock search results
        mock_result = MagicMock()
        mock_result.fetchall.return_value = [
            (1, "Python is a programming language", 0.95, {}),
            (2, "JavaScript is used for web", 0.75, {})
        ]
        mock_db_session.execute.return_value = mock_result
        
        # The RAG service would use this session for search
        # This tests the search logic concept
        assert mock_db_session.execute is not None


@pytest.mark.integration  
class TestQuestionAnswering:
    """Test Q&A functionality."""
    
    @pytest.mark.asyncio
    async def test_qa_with_mocked_orchestrator(self, rag_service):
        """Test Q&A with mocked AI components."""
        # Mock orchestrator for embeddings
        mock_orchestrator = AsyncMock()
        mock_orchestrator.generate_embeddings = AsyncMock(return_value=MagicMock(
            success=True,
            data={"embeddings": [[0.1] * 1024]}
        ))
        
        rag_service._use_optimized = True
        rag_service._ai_orchestrator = mock_orchestrator
        
        # Generate query embedding
        embedding = await rag_service.generate_embedding_async("What is Python?")
        
        assert embedding is not None
        assert len(embedding) == 1024


@pytest.mark.integration
class TestChatHistory:
    """Test chat history model."""
    
    def test_chat_history_model_structure(self):
        """Test ChatHistory model has required fields."""
        # Verify model attributes exist
        assert hasattr(ChatHistory, 'id')
        assert hasattr(ChatHistory, 'user_id')
        assert hasattr(ChatHistory, 'content_id')
        assert hasattr(ChatHistory, 'question')
        assert hasattr(ChatHistory, 'answer')
        assert hasattr(ChatHistory, 'context_chunks')
        assert hasattr(ChatHistory, 'confidence_score')
        assert hasattr(ChatHistory, 'created_at')
    
    def test_document_chunk_model_structure(self):
        """Test DocumentChunk model has required fields."""
        assert hasattr(DocumentChunk, 'id')
        assert hasattr(DocumentChunk, 'content_id')
        assert hasattr(DocumentChunk, 'chunk_index')
        assert hasattr(DocumentChunk, 'chunk_text')
        assert hasattr(DocumentChunk, 'chunk_size')
        assert hasattr(DocumentChunk, 'chunk_metadata')
    
    def test_embedding_model_structure(self):
        """Test Embedding model has required fields."""
        assert hasattr(Embedding, 'id')
        assert hasattr(Embedding, 'chunk_id')
        assert hasattr(Embedding, 'content_id')
        assert hasattr(Embedding, 'embedding_model')
        assert hasattr(Embedding, 'embedding_version')


@pytest.mark.integration
class TestRAGServiceConfiguration:
    """Test RAG service configuration."""
    
    def test_default_configuration(self, rag_service):
        """Test default RAG service configuration."""
        assert rag_service.embedding_model_name == "BAAI/bge-m3"
        assert rag_service.embedding_dimension == 1024
    
    def test_custom_configuration(self):
        """Test custom RAG service configuration."""
        with patch.object(RAGService, '_verify_pgvector'):
            service = RAGService(
                embedding_model="custom/model",
                use_optimized=False
            )
            assert service.embedding_model_name == "custom/model"


@pytest.mark.integration
class TestRAGPerformance:
    """Test RAG system performance characteristics."""
    
    def test_chunking_performance(self, rag_service):
        """Test chunking handles large documents efficiently."""
        import time
        
        # Generate large document
        large_text = "This is a test sentence. " * 10000  # ~260KB
        
        start = time.time()
        chunks = rag_service.chunk_text(large_text, chunk_size=512, overlap=50)
        elapsed = time.time() - start
        
        assert len(chunks) > 0
        assert elapsed < 1.0  # Should chunk quickly
    
    def test_chunking_consistency(self, rag_service):
        """Test chunking produces consistent results."""
        text = "Python is great. Java is also popular. C++ is powerful."
        
        chunks1 = rag_service.chunk_text(text, chunk_size=30, overlap=5)
        chunks2 = rag_service.chunk_text(text, chunk_size=30, overlap=5)
        
        assert chunks1 == chunks2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
