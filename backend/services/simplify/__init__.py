"""Text Simplifier module using Ollama/Llama 3.2 for grade-level content adaptation.

NEW TECH STACK:
- Ollama with Llama 3.2 3B - Local LLM for text simplification
- Optimized for educational content with grade-level prompts
- No API costs, runs locally
"""

from .ollama_simplifier import OllamaSimplifier, get_ollama_simplifier

__all__ = [
    'OllamaSimplifier',
    'get_ollama_simplifier',
]
