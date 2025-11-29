"""
Unit Tests for New AI Stack Components

Tests for:
- NLLB-200 Translator
- Ollama Simplifier (Llama 3.2 3B)
- Edge TTS Generator
- BGE-M3 Embeddings
- AI Orchestrator
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import numpy as np

from backend.services.translate.nllb_translator import NLLBTranslator
from backend.services.simplify.ollama_simplifier import OllamaSimplifier
from backend.services.speech.edge_tts_generator import EdgeTTSGenerator
from backend.services.embeddings.bge_embeddings import BGEM3Embeddings
from backend.services.ai.orchestrator import (
    AIOrchestrator,
    AIServiceConfig,
    ServiceType,
    MemoryManager,
    ProcessingResult
)


# =============================================================================
# NLLB Translator Tests
# =============================================================================

class TestNLLBTranslator:
    """Test NLLB-200 translation service."""
    
    @pytest.mark.unit
    def test_language_code_mapping(self):
        """Test language code conversion."""
        translator = NLLBTranslator(model_size="small")
        
        # Test common mappings
        assert "Hindi" in translator.LANGUAGE_CODES
        assert "Tamil" in translator.LANGUAGE_CODES
        assert "English" in translator.LANGUAGE_CODES
        
        # Verify format
        assert translator.LANGUAGE_CODES["Hindi"] == "hin_Deva"
        assert translator.LANGUAGE_CODES["Tamil"] == "tam_Taml"
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_translator_initialization(self):
        """Test translator lazy loading."""
        translator = NLLBTranslator(model_size="small")
        
        assert translator.model_size == "small"
        assert not translator._loaded
        assert translator._model is None
    
    @pytest.mark.unit
    def test_cache_key_generation(self):
        """Test translation cache key generation."""
        translator = NLLBTranslator(model_size="small")
        
        key1 = translator._get_cache_key("Hello", "en", "hi")
        key2 = translator._get_cache_key("Hello", "en", "hi")
        key3 = translator._get_cache_key("Hello", "en", "ta")
        
        assert key1 == key2
        assert key1 != key3
        assert len(key1) == 16  # SHA256 hash truncated


# =============================================================================
# Ollama Simplifier Tests
# =============================================================================

class TestOllamaSimplifier:
    """Test Ollama-based text simplification."""
    
    @pytest.mark.unit
    def test_simplifier_initialization(self):
        """Test simplifier configuration."""
        simplifier = OllamaSimplifier(model="llama3.2:3b")
        
        assert simplifier.model == "llama3.2:3b"
        assert simplifier.base_url == "http://localhost:11434"
        assert simplifier.cache_enabled is True
    
    @pytest.mark.unit
    def test_grade_level_guidance(self):
        """Test grade-level guidance configuration."""
        simplifier = OllamaSimplifier()
        
        # Check guidance exists for all grades
        for grade in range(5, 13):
            assert grade in simplifier.GRADE_GUIDANCE
            assert len(simplifier.GRADE_GUIDANCE[grade]) > 0
    
    @pytest.mark.unit
    def test_subject_instructions(self):
        """Test subject-specific instructions."""
        simplifier = OllamaSimplifier()
        
        assert "Mathematics" in simplifier.SUBJECT_INSTRUCTIONS
        assert "Science" in simplifier.SUBJECT_INSTRUCTIONS
        assert "preserve" in simplifier.SUBJECT_INSTRUCTIONS["Mathematics"].lower()
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_check_model_available(self):
        """Test Ollama health check method exists and is callable."""
        simplifier = OllamaSimplifier()
        
        # Verify the health_check method exists and is async
        assert hasattr(simplifier, 'health_check')
        assert asyncio.iscoroutinefunction(simplifier.health_check)
        
        # Test actual health check (works if Ollama is running)
        # This will return True if Ollama is running, False otherwise
        result = await simplifier.health_check()
        assert isinstance(result, bool)


# =============================================================================
# Edge TTS Tests
# =============================================================================

class TestEdgeTTSGenerator:
    """Test Edge TTS speech generation."""
    
    @pytest.mark.unit
    def test_tts_initialization(self):
        """Test TTS generator initialization."""
        tts = EdgeTTSGenerator()
        
        assert tts.cache_enabled is True
        assert tts.output_dir.exists()
    
    @pytest.mark.unit
    def test_voice_mappings(self):
        """Test voice configuration for Indian languages."""
        tts = EdgeTTSGenerator()
        
        # Check key languages
        assert "Hindi" in tts.VOICES
        assert "Tamil" in tts.VOICES
        assert "English" in tts.VOICES
        
        # Verify structure
        hindi_voices = tts.VOICES["Hindi"]
        assert "male" in hindi_voices
        assert "female" in hindi_voices
        assert hindi_voices["female"] == "hi-IN-SwaraNeural"
    
    @pytest.mark.unit
    def test_get_voice_for_language(self):
        """Test voice selection logic."""
        tts = EdgeTTSGenerator()
        
        # Use get_voices_for_language (actual method name)
        voices = tts.get_voices_for_language("Hindi")
        assert "female" in voices
        assert "male" in voices
        assert voices["female"] == "hi-IN-SwaraNeural"
        assert voices["male"] == "hi-IN-MadhurNeural"


# =============================================================================
# BGE-M3 Embeddings Tests
# =============================================================================

class TestBGEM3Embeddings:
    """Test BGE-M3 embedding generation."""
    
    @pytest.mark.unit
    def test_embeddings_initialization(self):
        """Test embeddings model configuration."""
        embeddings = BGEM3Embeddings(use_fp16=True)
        
        assert embeddings.use_fp16 is True
        assert embeddings.max_length == 8192
        assert embeddings.MODEL_ID == "BAAI/bge-m3"
    
    @pytest.mark.unit
    def test_embedding_dimension(self):
        """Test embedding dimension."""
        embeddings = BGEM3Embeddings()
        
        assert embeddings.EMBEDDING_DIMENSION == 1024


# =============================================================================
# Memory Manager Tests
# =============================================================================

class TestMemoryManager:
    """Test AI orchestrator memory management."""
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_memory_tracking(self):
        """Test memory usage tracking."""
        manager = MemoryManager(max_memory_gb=10.0)
        
        # Initially empty
        status = manager.get_memory_status()
        assert status["used_memory_mb"] == 0.0
        assert status["max_memory_mb"] == 10240.0
        
        # Register a service
        await manager.register_loaded(ServiceType.TRANSLATION, 2.5 * 1024**3)
        
        status = manager.get_memory_status()
        assert status["used_memory_mb"] > 0
        assert ServiceType.TRANSLATION.value in status["loaded_services"]
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_can_load_check(self):
        """Test memory availability check."""
        manager = MemoryManager(max_memory_gb=5.0)
        
        # Should be able to load 2GB
        can_load = await manager.can_load(ServiceType.TRANSLATION, 2 * 1024**3)
        assert can_load is True
        
        # Load it
        await manager.register_loaded(ServiceType.TRANSLATION, 2 * 1024**3)
        
        # Should not be able to load another 4GB
        can_load = await manager.can_load(ServiceType.SIMPLIFICATION, 4 * 1024**3)
        assert can_load is False
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_idle_service_detection(self):
        """Test idle service detection."""
        manager = MemoryManager(max_memory_gb=10.0)
        
        # Register and mark as used
        await manager.register_loaded(ServiceType.TRANSLATION, 2 * 1024**3)
        await manager.mark_used(ServiceType.TRANSLATION)
        
        # Should not be idle immediately
        idle = await manager.get_idle_services(idle_threshold_seconds=300)
        assert ServiceType.TRANSLATION not in idle
        
        # Should be idle with 0 threshold
        idle = await manager.get_idle_services(idle_threshold_seconds=0)
        assert ServiceType.TRANSLATION in idle


# =============================================================================
# AI Orchestrator Tests
# =============================================================================

class TestAIOrchestrator:
    """Test AI orchestrator integration."""
    
    @pytest.mark.unit
    def test_orchestrator_initialization(self):
        """Test orchestrator setup."""
        config = AIServiceConfig(
            max_memory_gb=10.0,
            device="mps",
            compute_type="int8"
        )
        
        orchestrator = AIOrchestrator(config)
        
        assert orchestrator.config.max_memory_gb == 10.0
        assert orchestrator.config.device == "mps"
        assert orchestrator.config.compute_type == "int8"
        assert orchestrator.memory_manager is not None
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_orchestrator_lifecycle(self):
        """Test start/stop lifecycle."""
        orchestrator = AIOrchestrator()
        
        await orchestrator.start()
        assert orchestrator._running is True
        
        await orchestrator.stop()
        assert orchestrator._running is False
    
    @pytest.mark.unit
    def test_memory_requirements(self):
        """Test model memory estimates."""
        orchestrator = AIOrchestrator()
        
        assert ServiceType.TRANSLATION in orchestrator.MEMORY_REQUIREMENTS
        assert ServiceType.TTS in orchestrator.MEMORY_REQUIREMENTS
        assert ServiceType.SIMPLIFICATION in orchestrator.MEMORY_REQUIREMENTS
        assert ServiceType.EMBEDDINGS in orchestrator.MEMORY_REQUIREMENTS
        
        # Check reasonable sizes
        translation_mem = orchestrator.MEMORY_REQUIREMENTS[ServiceType.TRANSLATION]
        assert 2 * 1024**3 < translation_mem < 4 * 1024**3  # 2-4GB
    
    @pytest.mark.unit
    def test_get_status(self):
        """Test status reporting."""
        orchestrator = AIOrchestrator()
        
        status = orchestrator.get_status()
        
        assert "memory" in status
        assert "config" in status
        assert "services" in status
        assert status["config"]["device"] == "mps"


# =============================================================================
# Configuration Tests
# =============================================================================

class TestAIServiceConfig:
    """Test AI service configuration."""
    
    @pytest.mark.unit
    def test_default_configuration(self):
        """Test default config values."""
        config = AIServiceConfig()
        
        assert config.max_memory_gb == 10.0
        assert config.translation_model == "medium"  # small, medium, large for NLLB
        assert config.llm_model == "llama3.2:3b"
        assert config.embedding_model == "BAAI/bge-m3"
        assert config.device == "mps"
        assert config.compute_type == "int8"
    
    @pytest.mark.unit
    def test_config_from_env(self):
        """Test loading config from environment."""
        import os
        
        os.environ["AI_MAX_MEMORY_GB"] = "8.0"
        os.environ["AI_DEVICE"] = "cpu"
        os.environ["AI_COMPUTE_TYPE"] = "float16"
        
        config = AIServiceConfig.from_env()
        
        assert config.max_memory_gb == 8.0
        assert config.device == "cpu"
        assert config.compute_type == "float16"
        
        # Cleanup
        del os.environ["AI_MAX_MEMORY_GB"]
        del os.environ["AI_DEVICE"]
        del os.environ["AI_COMPUTE_TYPE"]


# =============================================================================
# Integration Tests
# =============================================================================

@pytest.mark.integration
class TestAIStackIntegration:
    """Integration tests for AI stack components."""
    
    @pytest.mark.asyncio
    async def test_orchestrator_service_loading(self):
        """Test lazy loading of services."""
        orchestrator = AIOrchestrator()
        await orchestrator.start()
        
        try:
            # Initially no services loaded
            status = orchestrator.get_status()
            assert not status["services"]["translator_loaded"]
            
            # Services should load on first use (mocked)
            # In real integration, this would call actual services
            
        finally:
            await orchestrator.stop()
    
    @pytest.mark.asyncio
    async def test_memory_cleanup(self):
        """Test automatic memory cleanup."""
        config = AIServiceConfig(unload_after_idle_seconds=1)
        orchestrator = AIOrchestrator(config)
        
        await orchestrator.start()
        
        try:
            # Simulate service usage and idle detection
            # Memory should be freed after idle period
            await asyncio.sleep(2)
            
        finally:
            await orchestrator.stop()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "unit"])
