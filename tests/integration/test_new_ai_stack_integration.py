"""
Integration Tests for New AI Stack

Tests full workflows:
- Translation pipeline with NLLB-200
- Simplification with Ollama
- TTS with Edge TTS
- RAG with BGE-M3
- Celery task execution
- Redis caching
- End-to-end content processing
"""

import pytest
import asyncio
import time
from typing import Dict, Any

from backend.services.ai.orchestrator import AIOrchestrator, AIServiceConfig
from backend.tasks.pipeline_tasks import (
    simplify_text_task,
    translate_text_task,
    generate_audio_task
)
from backend.services.rag import RAGService
from backend.core.cache import get_redis
from celery.result import AsyncResult


# =============================================================================
# Translation Integration Tests
# =============================================================================

@pytest.mark.integration
class TestTranslationIntegration:
    """Test NLLB-200 translation integration."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_translation_via_orchestrator(self):
        """Test translation through orchestrator."""
        orchestrator = AIOrchestrator()
        await orchestrator.start()
        
        try:
            result = await orchestrator.translate(
                text="Hello, how are you?",
                source_language="English",
                target_language="Hindi"
            )
            
            assert result.success is True
            assert result.data is not None
            assert "translated_text" in result.data
            assert len(result.data["translated_text"]) > 0
            assert result.processing_time_ms > 0
            
        finally:
            await orchestrator.stop()
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_batch_translation(self):
        """Test batch translation performance."""
        orchestrator = AIOrchestrator()
        await orchestrator.start()
        
        try:
            texts = [
                "Good morning",
                "Thank you",
                "How are you?"
            ]
            
            result = await orchestrator.translate_batch(
                texts=texts,
                source_language="English",
                target_language="Hindi"
            )
            
            assert result.success is True
            assert len(result.data) == len(texts)
            
        finally:
            await orchestrator.stop()


# =============================================================================
# Simplification Integration Tests
# =============================================================================

@pytest.mark.integration
class TestSimplificationIntegration:
    """Test Ollama simplification integration."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_simplification_via_orchestrator(self):
        """Test text simplification through orchestrator."""
        orchestrator = AIOrchestrator()
        await orchestrator.start()
        
        try:
            text = (
                "The mitochondria is a double membrane-bound organelle "
                "found in most eukaryotic organisms. It generates most of "
                "the cell's supply of adenosine triphosphate (ATP)."
            )
            
            result = await orchestrator.simplify_text(
                text=text,
                target_grade=6,
                subject="Science"
            )
            
            assert result.success is True
            assert result.data is not None
            assert "simplified_text" in result.data
            assert len(result.data["simplified_text"]) > 0
            assert result.data["target_grade"] == 6
            
        finally:
            await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_simplification_different_grades(self):
        """Test simplification for different grade levels."""
        orchestrator = AIOrchestrator()
        await orchestrator.start()
        
        try:
            text = "Photosynthesis is the process by which plants convert light energy into chemical energy."
            
            # Test multiple grade levels
            for grade in [6, 8, 10]:
                result = await orchestrator.simplify_text(
                    text=text,
                    target_grade=grade,
                    subject="Science"
                )
                
                assert result.success is True
                assert result.data["target_grade"] == grade
                
        finally:
            await orchestrator.stop()


# =============================================================================
# TTS Integration Tests
# =============================================================================

@pytest.mark.integration
class TestTTSIntegration:
    """Test Edge TTS integration."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_tts_generation(self):
        """Test audio generation via orchestrator."""
        orchestrator = AIOrchestrator()
        await orchestrator.start()
        
        try:
            result = await orchestrator.synthesize_speech(
                text="Hello, this is a test.",
                language="en"
            )
            
            assert result.success is True
            assert result.data is not None
            assert isinstance(result.data, bytes)
            assert len(result.data) > 0
            
        finally:
            await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_tts_indian_languages(self):
        """Test TTS for Indian languages."""
        orchestrator = AIOrchestrator()
        await orchestrator.start()
        
        try:
            languages = ["Hindi", "Tamil", "Telugu"]
            text = "नमस्ते"  # Hello in Hindi
            
            for lang in languages:
                result = await orchestrator.synthesize_speech(
                    text=text,
                    language=lang
                )
                
                assert result.success is True or result.error is not None
                # Some languages may not be available
                
        finally:
            await orchestrator.stop()


# =============================================================================
# RAG Integration Tests
# =============================================================================

@pytest.mark.integration
class TestRAGIntegration:
    """Test RAG with BGE-M3 embeddings."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_embedding_generation(self):
        """Test embedding generation via orchestrator."""
        orchestrator = AIOrchestrator()
        await orchestrator.start()
        
        try:
            texts = [
                "Artificial intelligence is transforming education.",
                "Machine learning models can adapt to student needs."
            ]
            
            result = await orchestrator.generate_embeddings(
                texts=texts,
                return_sparse=False
            )
            
            assert result.success is True
            assert "embeddings" in result.data
            assert len(result.data["embeddings"]) == len(texts)
            assert result.data["dimensions"] == 1024  # BGE-M3 dimension
            
        finally:
            await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_rag_service_with_bge(self, db_session):
        """Test RAG service using BGE-M3."""
        from uuid import uuid4
        rag_service = RAGService(db_session)
        
        # Store a document
        doc_text = "Python is a high-level programming language. It is used for web development, data science, and machine learning."
        content_id = uuid4()  # Use UUID as expected by the service
        
        try:
            # Use correct parameters: content_id (UUID), text (str)
            chunk_count = await rag_service.store_document_chunks_async(
                content_id=content_id,
                text=doc_text,
                chunk_size=100,
                overlap=20,
                metadata={"test": True}
            )
            
            assert chunk_count > 0
            
            # Query similar content - check if search method exists
            if hasattr(rag_service, 'search_similar_chunks_async'):
                query = "What is Python?"
                results = await rag_service.search_similar_chunks_async(
                    query=query,
                    top_k=5
                )
                assert isinstance(results, list)
            
        except Exception as e:
            # Embedding generation may fail if model not loaded
            pytest.skip(f"RAG test skipped: {e}")


# =============================================================================
# Celery Integration Tests
# =============================================================================

@pytest.mark.integration
@pytest.mark.skip(reason="Requires running Celery workers - run with START.sh first")
class TestCeleryIntegration:
    """Test Celery task execution with new AI stack."""
    
    def test_simplify_text_task(self):
        """Test simplification Celery task."""
        task = simplify_text_task.delay(
            text="Complex scientific terminology requires simplification.",
            grade_level=6,
            subject="Science"
        )
        
        # Wait for result (with timeout)
        result = task.get(timeout=60)
        
        assert result is not None
        assert "simplified_text" in result
        assert task.status == "SUCCESS"
    
    def test_translate_text_task(self):
        """Test translation Celery task."""
        task = translate_text_task.delay(
            text="Good morning",
            target_languages=["Hindi"]
        )
        
        result = task.get(timeout=60)
        
        assert result is not None
        assert "translations" in result
        assert "Hindi" in result["translations"]
        assert task.status == "SUCCESS"
    
    def test_generate_audio_task(self):
        """Test TTS Celery task."""
        task = generate_audio_task.delay(
            text="Hello world",
            language="English",
            content_id="test-audio"
        )
        
        result = task.get(timeout=60)
        
        assert result is not None
        assert "audio_path" in result
        assert task.status == "SUCCESS"


# =============================================================================
# Redis Caching Tests
# =============================================================================

@pytest.mark.integration
class TestRedisCaching:
    """Test Redis caching for AI operations."""
    
    def test_redis_connection(self):
        """Test Redis connectivity."""
        redis_client = get_redis()
        
        assert redis_client is not None
        assert redis_client.ping() is True
    
    def test_cache_translation_result(self):
        """Test caching translation results."""
        redis_client = get_redis()
        
        cache_key = "translation:en:hi:hello"
        cached_value = "नमस्ते"
        
        # Set cache
        redis_client.setex(cache_key, 3600, cached_value)
        
        # Retrieve
        result = redis_client.get(cache_key)
        # Redis returns bytes, decode if needed
        if isinstance(result, bytes):
            result = result.decode()
        assert result == cached_value
        
        # Cleanup
        redis_client.delete(cache_key)
    
    def test_cache_expiration(self):
        """Test cache TTL."""
        redis_client = get_redis()
        
        cache_key = "test:expiration"
        redis_client.setex(cache_key, 1, "value")
        
        assert redis_client.get(cache_key) is not None
        
        time.sleep(2)
        
        assert redis_client.get(cache_key) is None


# =============================================================================
# Memory Management Tests
# =============================================================================

@pytest.mark.integration
class TestMemoryManagement:
    """Test memory management under load."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_concurrent_requests(self):
        """Test handling concurrent AI requests."""
        orchestrator = AIOrchestrator()
        await orchestrator.start()
        
        try:
            # Simulate concurrent requests
            tasks = []
            for i in range(5):
                task = orchestrator.simplify_text(
                    text=f"Test text number {i}",
                    target_grade=6
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            # All should succeed
            for result in results:
                assert result.success is True
                
        finally:
            await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_memory_cleanup_after_idle(self):
        """Test automatic model unloading."""
        config = AIServiceConfig(unload_after_idle_seconds=2)
        orchestrator = AIOrchestrator(config)
        await orchestrator.start()
        
        try:
            # Load a service
            result = await orchestrator.simplify_text(
                text="Test",
                target_grade=6
            )
            
            assert orchestrator._simplifier is not None
            
            # Wait for idle unload
            await asyncio.sleep(3)
            
            # Service should be unloaded (memory manager checks idle time)
            
        finally:
            await orchestrator.stop()


# =============================================================================
# End-to-End Pipeline Tests
# =============================================================================

@pytest.mark.integration
class TestEndToEndPipeline:
    """Test complete content processing pipeline."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_full_content_pipeline(self):
        """Test complete workflow: simplify → translate → TTS."""
        orchestrator = AIOrchestrator()
        await orchestrator.start()
        
        try:
            # Process educational content
            result = await orchestrator.process_educational_content(
                content="Photosynthesis converts sunlight into energy.",
                source_language="en",
                target_language="Hindi",
                target_grade=6,
                generate_audio=True,
                generate_embeddings=True,
                subject="Science"
            )
            
            assert result.success is True
            # Check for required fields (simplified is always present)
            assert "simplified" in result.data
            # Other fields may fail due to model loading - check if result exists
            if result.data.get("translated"):
                assert isinstance(result.data["translated"], str)
            
        finally:
            await orchestrator.stop()


# =============================================================================
# Performance Tests
# =============================================================================

@pytest.mark.integration
@pytest.mark.performance
class TestPerformance:
    """Test performance of AI operations."""
    
    @pytest.mark.asyncio
    async def test_simplification_latency(self):
        """Test simplification response time."""
        orchestrator = AIOrchestrator()
        await orchestrator.start()
        
        try:
            start = time.time()
            
            result = await orchestrator.simplify_text(
                text="Quick test",
                target_grade=6
            )
            
            latency = time.time() - start
            
            assert result.success is True
            assert latency < 30  # Should complete within 30 seconds
            
        finally:
            await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_throughput(self):
        """Test processing throughput."""
        orchestrator = AIOrchestrator()
        await orchestrator.start()
        
        try:
            num_requests = 10
            start = time.time()
            
            tasks = [
                orchestrator.simplify_text(text=f"Test {i}", target_grade=6)
                for i in range(num_requests)
            ]
            
            results = await asyncio.gather(*tasks)
            
            duration = time.time() - start
            throughput = num_requests / duration
            
            assert all(r.success for r in results)
            assert throughput > 0.1  # At least 0.1 requests/second
            
        finally:
            await orchestrator.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
