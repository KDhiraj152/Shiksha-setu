"""
Edge TTS Generator - High Quality Neural Text-to-Speech

Uses Microsoft Edge TTS (FREE, unlimited, high quality) for primary TTS.
Supports all major Indian languages with male/female voice options.

Features:
- Zero cost, unlimited usage
- Neural voice quality
- All Indian languages supported
- Streaming support for real-time playback
- No API key required

Note: Requires internet connection. Use Coqui XTTS for offline fallback.
"""

import asyncio
import logging
import uuid
import hashlib
from pathlib import Path
from typing import Optional, Dict, Any, AsyncGenerator, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    logger.warning("edge-tts not available. Install with: pip install edge-tts")


@dataclass
class TTSResult:
    """Result of TTS generation."""
    audio_path: str
    language: str
    voice: str
    duration_estimate: float  # Estimated duration in seconds
    format: str
    cached: bool = False


class EdgeTTSGenerator:
    """
    Microsoft Edge TTS Generator
    
    FREE, high-quality neural TTS for all Indian languages.
    No API key required, unlimited usage.
    """
    
    # Voice mapping for Indian languages (Neural voices)
    VOICES = {
        "Hindi": {
            "female": "hi-IN-SwaraNeural",
            "male": "hi-IN-MadhurNeural"
        },
        "Tamil": {
            "female": "ta-IN-PallaviNeural",
            "male": "ta-IN-ValluvarNeural"
        },
        "Telugu": {
            "female": "te-IN-ShrutiNeural",
            "male": "te-IN-MohanNeural"
        },
        "Bengali": {
            "female": "bn-IN-TanishaaNeural",
            "male": "bn-IN-BashkarNeural"
        },
        "Marathi": {
            "female": "mr-IN-AarohiNeural",
            "male": "mr-IN-ManoharNeural"
        },
        "Gujarati": {
            "female": "gu-IN-DhwaniNeural",
            "male": "gu-IN-NiranjanNeural"
        },
        "Kannada": {
            "female": "kn-IN-SapnaNeural",
            "male": "kn-IN-GaganNeural"
        },
        "Malayalam": {
            "female": "ml-IN-SobhanaNeural",
            "male": "ml-IN-MidhunNeural"
        },
        "English": {
            "female": "en-IN-NeerjaNeural",
            "male": "en-IN-PrabhatNeural"
        },
        "Punjabi": {
            "female": "pa-IN-Neerja",  # Limited support
            "male": "pa-IN-Neerja"
        },
        "Odia": {
            "female": "or-IN-SubhasiniNeural",
            "male": "or-IN-SukantNeural"
        },
        "Urdu": {
            "female": "ur-IN-GulNeural",
            "male": "ur-IN-SalmanNeural"
        }
    }
    
    # Rate and pitch defaults
    DEFAULT_RATE = "+0%"
    DEFAULT_PITCH = "+0Hz"
    DEFAULT_VOLUME = "+0%"
    
    def __init__(
        self,
        output_dir: str = "data/audio",
        cache_enabled: bool = True,
        default_format: str = "mp3"
    ):
        """
        Initialize Edge TTS Generator.
        
        Args:
            output_dir: Directory for generated audio files
            cache_enabled: Whether to cache generated audio
            default_format: Default audio format (mp3, wav)
        """
        if not EDGE_TTS_AVAILABLE:
            raise ImportError("edge-tts not installed. Run: pip install edge-tts")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_enabled = cache_enabled
        self.default_format = default_format
        
        # Cache for generated audio (hash -> file path)
        self._cache: Dict[str, str] = {}
        self._cache_max_size = 500
        
        logger.info(f"EdgeTTSGenerator initialized: output_dir={output_dir}")
    
    def _get_cache_key(
        self,
        text: str,
        language: str,
        gender: str,
        rate: str,
        pitch: str
    ) -> str:
        """Generate cache key for TTS output."""
        content = f"{text}|{language}|{gender}|{rate}|{pitch}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _get_voice(self, language: str, gender: str = "female") -> str:
        """Get voice ID for language and gender."""
        if language not in self.VOICES:
            logger.warning(f"Language {language} not found, defaulting to Hindi")
            language = "Hindi"
        
        voices = self.VOICES[language]
        return voices.get(gender, voices["female"])
    
    async def generate_speech(
        self,
        text: str,
        language: str = "Hindi",
        gender: str = "female",
        rate: str = "+0%",
        pitch: str = "+0Hz",
        volume: str = "+0%",
        output_format: Optional[str] = None
    ) -> TTSResult:
        """
        Generate speech from text.
        
        Args:
            text: Text to convert to speech
            language: Target language (Hindi, Tamil, Telugu, etc.)
            gender: Voice gender ("male" or "female")
            rate: Speech rate (-50% to +100%)
            pitch: Pitch adjustment (-50Hz to +50Hz)
            volume: Volume adjustment (-50% to +50%)
            output_format: Output format (mp3, wav)
            
        Returns:
            TTSResult with audio file path and metadata
        """
        output_format = output_format or self.default_format
        
        # Check cache
        if self.cache_enabled:
            cache_key = self._get_cache_key(text, language, gender, rate, pitch)
            if cache_key in self._cache:
                cached_path = self._cache[cache_key]
                if Path(cached_path).exists():
                    return TTSResult(
                        audio_path=cached_path,
                        language=language,
                        voice=self._get_voice(language, gender),
                        duration_estimate=len(text) * 0.06,  # ~60ms per character
                        format=output_format,
                        cached=True
                    )
        
        # Get voice
        voice = self._get_voice(language, gender)
        
        # Generate unique filename
        filename = f"{uuid.uuid4().hex}.{output_format}"
        output_path = self.output_dir / filename
        
        # Create TTS communicator
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=rate,
            pitch=pitch,
            volume=volume
        )
        
        # Generate and save audio
        await communicate.save(str(output_path))
        
        # Cache result
        if self.cache_enabled:
            if len(self._cache) >= self._cache_max_size:
                # Remove oldest entry
                oldest_key = next(iter(self._cache))
                self._cache.pop(oldest_key)
            self._cache[cache_key] = str(output_path)
        
        # Estimate duration (rough estimate based on text length)
        duration_estimate = len(text) * 0.06  # ~60ms per character
        
        logger.info(f"Generated speech: {language}/{gender}, {len(text)} chars -> {output_path}")
        
        return TTSResult(
            audio_path=str(output_path),
            language=language,
            voice=voice,
            duration_estimate=duration_estimate,
            format=output_format,
            cached=False
        )
    
    async def generate_speech_stream(
        self,
        text: str,
        language: str = "Hindi",
        gender: str = "female",
        rate: str = "+0%",
        pitch: str = "+0Hz"
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream audio chunks for real-time playback.
        
        Yields audio chunks as they're generated.
        Useful for streaming to clients without waiting for full generation.
        """
        voice = self._get_voice(language, gender)
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=rate,
            pitch=pitch
        )
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]
    
    async def generate_batch(
        self,
        items: List[Dict[str, Any]]
    ) -> List[TTSResult]:
        """
        Generate speech for multiple items in parallel.
        
        Args:
            items: List of dicts with text, language, gender, etc.
            
        Returns:
            List of TTSResult objects
        """
        tasks = []
        for item in items:
            task = self.generate_speech(
                text=item.get("text", ""),
                language=item.get("language", "Hindi"),
                gender=item.get("gender", "female"),
                rate=item.get("rate", self.DEFAULT_RATE),
                pitch=item.get("pitch", self.DEFAULT_PITCH)
            )
            tasks.append(task)
        
        return await asyncio.gather(*tasks)
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return list(self.VOICES.keys())
    
    def get_voices_for_language(self, language: str) -> Dict[str, str]:
        """Get available voices for a language."""
        return self.VOICES.get(language, {})
    
    @staticmethod
    async def list_all_voices() -> List[Dict[str, Any]]:
        """List all available Edge TTS voices."""
        voices = await edge_tts.list_voices()
        return voices
    
    @staticmethod
    async def list_indian_voices() -> List[Dict[str, Any]]:
        """List all Indian language voices."""
        voices = await edge_tts.list_voices()
        return [v for v in voices if v.get("Locale", "").endswith("-IN")]
    
    def cleanup_cache(self, max_age_hours: int = 24):
        """Remove old cached audio files."""
        import time
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        removed = 0
        for cache_key, file_path in list(self._cache.items()):
            path = Path(file_path)
            if path.exists():
                file_age = current_time - path.stat().st_mtime
                if file_age > max_age_seconds:
                    path.unlink()
                    del self._cache[cache_key]
                    removed += 1
        
        logger.info(f"Cleaned up {removed} cached audio files")
        return removed
    
    def clear_cache(self):
        """Clear all cached audio files."""
        for file_path in self._cache.values():
            path = Path(file_path)
            if path.exists():
                path.unlink()
        self._cache.clear()
        logger.info("TTS cache cleared")


# Singleton instance
_edge_tts_instance: Optional[EdgeTTSGenerator] = None


def get_edge_tts() -> EdgeTTSGenerator:
    """Get or create singleton Edge TTS instance."""
    global _edge_tts_instance
    
    if _edge_tts_instance is None:
        _edge_tts_instance = EdgeTTSGenerator()
    
    return _edge_tts_instance
