"""Speech generator module using Edge TTS for text-to-speech conversion.

NEW TECH STACK:
- Edge TTS - Microsoft's free, high-quality TTS service
- Supports all major Indian languages (Hindi, Tamil, Telugu, etc.)
- Natural neural voices with prosody control
- No API costs, unlimited usage
"""

from .edge_tts_generator import EdgeTTSGenerator, get_edge_tts

# Alias for backward compatibility
get_edge_tts_generator = get_edge_tts

__all__ = [
    'EdgeTTSGenerator',
    'get_edge_tts',
    'get_edge_tts_generator',
]
