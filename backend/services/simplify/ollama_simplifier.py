"""
Ollama Simplifier - Text Simplification via Local LLM

Uses Ollama to run Llama 3.2 (3B) locally for text simplification.
Optimized for educational content adaptation based on grade level.

Features:
- Runs fully locally via Ollama
- No API keys required
- Grade-level aware simplification (5-12)
- Subject-specific terminology preservation
- Streaming support for real-time output

Memory: ~2GB for Llama 3.2 3B quantized
Prerequisites: Ollama installed and running (ollama pull llama3.2:3b)
"""

import asyncio
import logging
import json
import hashlib
from typing import Optional, Dict, Any, AsyncGenerator, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("ollama package not available. Install with: pip install ollama")


@dataclass
class SimplificationResult:
    """Result of text simplification."""
    original_text: str
    simplified_text: str
    grade_level: int
    subject: str
    model_used: str
    complexity_reduction: float
    cached: bool = False


class OllamaSimplifier:
    """
    Text Simplification using Ollama (Local LLM)
    
    Uses Llama 3.2 3B for high-quality text simplification.
    Runs fully locally, no API keys required.
    
    Prerequisites:
    1. Install Ollama: brew install ollama (Mac) or curl -fsSL https://ollama.ai/install.sh | sh
    2. Pull model: ollama pull llama3.2:3b
    """
    
    # Grade-level vocabulary guidance
    GRADE_GUIDANCE = {
        5: "Use very simple words. Short sentences of 5-8 words. Avoid all technical terms. Use examples from daily life.",
        6: "Use simple words. Sentences of 8-10 words. Explain any technical term in parentheses.",
        7: "Use common vocabulary. Sentences of 10-12 words. Introduce basic technical terms with simple explanations.",
        8: "Use standard vocabulary. Sentences of 12-15 words. Use technical terms with brief explanations.",
        9: "Use appropriate academic vocabulary. Moderate sentence complexity. Technical terms can be used.",
        10: "Standard academic language. Include subject-specific terminology with context.",
        11: "Advanced academic language. Technical terminology expected. Complex concepts can be discussed.",
        12: "College-prep language. Full technical vocabulary. Sophisticated explanations allowed."
    }
    
    # Subject-specific instructions
    SUBJECT_INSTRUCTIONS = {
        "Mathematics": "Preserve all mathematical formulas, equations, and numerical expressions exactly. Use Indian numbering system examples where appropriate.",
        "Science": "Keep scientific terms accurate. Explain concepts using familiar examples. Preserve chemical formulas and scientific notation.",
        "Social Studies": "Use examples from Indian context. Preserve names of historical figures, places, and events accurately.",
        "History": "Maintain chronological accuracy. Keep dates and historical events precise. Use Indian historical context when relevant.",
        "Geography": "Keep geographical terms and place names accurate. Use Indian geography examples where possible.",
        "English": "Focus on clarity and readability. Use appropriate vocabulary for the grade level.",
        "General": "Focus on clarity while maintaining factual accuracy."
    }
    
    # Default model configurations
    DEFAULT_MODELS = {
        "small": "llama3.2:1b",    # 1B parameters, fastest
        "medium": "llama3.2:3b",   # 3B parameters, balanced
        "large": "qwen2.5:7b"      # 7B parameters, best quality
    }
    
    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        cache_enabled: bool = True
    ):
        """
        Initialize Ollama Simplifier.
        
        Args:
            model: Ollama model name (e.g., "llama3.2:3b", "qwen2.5:7b")
            base_url: Ollama server URL
            timeout: Request timeout in seconds
            cache_enabled: Whether to cache results
        """
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.cache_enabled = cache_enabled
        
        # HTTP client for API calls
        self._client: Optional[httpx.AsyncClient] = None
        
        # Cache for simplification results
        self._cache: Dict[str, str] = {}
        self._cache_max_size = 500
        
        logger.info(f"OllamaSimplifier initialized: model={model}, url={base_url}")
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout
            )
        return self._client
    
    def _get_cache_key(self, text: str, grade_level: int, subject: str) -> str:
        """Generate cache key for simplification."""
        content = f"{text}|{grade_level}|{subject}|{self.model}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _build_prompt(self, text: str, grade_level: int, subject: str) -> str:
        """Build simplification prompt for the model."""
        grade_instruction = self.GRADE_GUIDANCE.get(grade_level, self.GRADE_GUIDANCE[8])
        subject_instruction = self.SUBJECT_INSTRUCTIONS.get(subject, self.SUBJECT_INSTRUCTIONS["General"])
        
        prompt = f"""You are an expert educational content simplifier for Indian schools following NCERT curriculum.

TASK: Simplify the following {subject} text for Grade {grade_level} students in India.

GRADE LEVEL REQUIREMENTS:
{grade_instruction}

SUBJECT-SPECIFIC RULES:
{subject_instruction}

GENERAL RULES:
1. Preserve ALL factual accuracy - do not change facts, dates, or numbers
2. Keep mathematical formulas and scientific notation intact
3. Maintain logical flow and key concepts
4. Use examples relevant to Indian students when helpful
5. Do not add information not present in the original
6. Output ONLY the simplified text, no explanations or meta-commentary

ORIGINAL TEXT:
{text}

SIMPLIFIED TEXT:"""
        
        return prompt
    
    async def health_check(self) -> bool:
        """Check if Ollama server is running and model is available."""
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            
            if response.status_code == 200:
                data = response.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                
                # Check if our model is available
                model_base = self.model.split(":")[0]
                for m in models:
                    if m.startswith(model_base):
                        return True
                
                logger.warning(f"Model {self.model} not found. Available: {models}")
                return False
            
            return False
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
    
    async def simplify_text(
        self,
        text: str,
        grade_level: int,
        subject: str = "General",
        temperature: float = 0.3,
        max_tokens: int = 2048
    ) -> SimplificationResult:
        """
        Simplify text for target grade level.
        
        Args:
            text: Original text to simplify
            grade_level: Target grade level (5-12)
            subject: Subject area (Mathematics, Science, etc.)
            temperature: Model temperature (lower = more consistent)
            max_tokens: Maximum output tokens
            
        Returns:
            SimplificationResult with simplified text and metadata
        """
        # Validate inputs
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        if grade_level < 5 or grade_level > 12:
            raise ValueError(f"Grade level must be between 5 and 12, got {grade_level}")
        
        # Check cache
        if self.cache_enabled:
            cache_key = self._get_cache_key(text, grade_level, subject)
            if cache_key in self._cache:
                return SimplificationResult(
                    original_text=text,
                    simplified_text=self._cache[cache_key],
                    grade_level=grade_level,
                    subject=subject,
                    model_used=self.model,
                    complexity_reduction=0.0,
                    cached=True
                )
        
        # Build prompt
        prompt = self._build_prompt(text, grade_level, subject)
        
        # Call Ollama API
        client = await self._get_client()
        
        try:
            response = await client.post(
                "/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": temperature,
                        "top_p": 0.9,
                        "num_predict": max_tokens
                    }
                }
            )
            
            if response.status_code != 200:
                raise RuntimeError(f"Ollama API error: {response.status_code} - {response.text}")
            
            data = response.json()
            simplified_text = data.get("response", "").strip()
            
        except httpx.TimeoutException:
            raise TimeoutError(f"Simplification timed out after {self.timeout}s")
        except httpx.ConnectError:
            raise ConnectionError("Cannot connect to Ollama. Is it running? (ollama serve)")
        
        # Calculate complexity reduction (rough estimate)
        original_words = len(text.split())
        simplified_words = len(simplified_text.split())
        complexity_reduction = 1.0 - (simplified_words / original_words) if original_words > 0 else 0.0
        
        # Cache result
        if self.cache_enabled:
            if len(self._cache) >= self._cache_max_size:
                oldest_key = next(iter(self._cache))
                self._cache.pop(oldest_key)
            self._cache[cache_key] = simplified_text
        
        logger.info(
            f"Simplified text: grade={grade_level}, subject={subject}, "
            f"words={original_words}->{simplified_words}"
        )
        
        return SimplificationResult(
            original_text=text,
            simplified_text=simplified_text,
            grade_level=grade_level,
            subject=subject,
            model_used=self.model,
            complexity_reduction=complexity_reduction,
            cached=False
        )
    
    async def simplify_stream(
        self,
        text: str,
        grade_level: int,
        subject: str = "General",
        temperature: float = 0.3
    ) -> AsyncGenerator[str, None]:
        """
        Stream simplified text tokens for real-time display.
        
        Yields tokens as they're generated.
        """
        prompt = self._build_prompt(text, grade_level, subject)
        client = await self._get_client()
        
        async with client.stream(
            "POST",
            "/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "top_p": 0.9
                }
            }
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if token := data.get("response"):
                            yield token
                    except json.JSONDecodeError:
                        continue
    
    async def batch_simplify(
        self,
        items: List[Dict[str, Any]]
    ) -> List[SimplificationResult]:
        """
        Simplify multiple texts.
        
        Args:
            items: List of dicts with text, grade_level, subject
            
        Returns:
            List of SimplificationResult objects
        """
        # Process sequentially to avoid overloading Ollama
        results = []
        for item in items:
            result = await self.simplify_text(
                text=item.get("text", ""),
                grade_level=item.get("grade_level", 8),
                subject=item.get("subject", "General")
            )
            results.append(result)
        
        return results
    
    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
    
    def clear_cache(self):
        """Clear the simplification cache."""
        self._cache.clear()
        logger.info("Simplification cache cleared")
    
    @staticmethod
    async def pull_model(model: str, base_url: str = "http://localhost:11434") -> bool:
        """
        Pull a model from Ollama registry.
        
        Args:
            model: Model name (e.g., "llama3.2:3b")
            base_url: Ollama server URL
            
        Returns:
            True if successful
        """
        async with httpx.AsyncClient(base_url=base_url, timeout=600.0) as client:
            try:
                response = await client.post(
                    "/api/pull",
                    json={"name": model, "stream": False}
                )
                return response.status_code == 200
            except Exception as e:
                logger.error(f"Failed to pull model {model}: {e}")
                return False


# Singleton instance
_simplifier_instance: Optional[OllamaSimplifier] = None


def get_ollama_simplifier(model: str = "llama3.2:3b") -> OllamaSimplifier:
    """Get or create singleton Ollama simplifier instance."""
    global _simplifier_instance
    
    if _simplifier_instance is None:
        _simplifier_instance = OllamaSimplifier(model=model)
    
    return _simplifier_instance
