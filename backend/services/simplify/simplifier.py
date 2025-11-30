"""Text Simplifier using Llama-3.2-3B-Instruct for grade-level content adaptation.

Optimal 2025 Model Stack: Llama-3.2-3B-Instruct
- Best instruction-following for educational prompts
- 3GB with INT4 quantization
- Supports vLLM serving for production
"""
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import re
import httpx
from abc import ABC, abstractmethod

from ...core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class SimplifiedText:
    """Result of text simplification."""
    text: str
    complexity_score: float
    grade_level: int
    subject: str
    metadata: Dict[str, Any]


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass


class VLLMClient(BaseLLMClient):
    """vLLM OpenAI-compatible API client."""
    
    def __init__(
        self,
        base_url: str = None,
        model: str = None,
        api_key: str = None
    ):
        self.base_url = base_url or settings.vllm_base_url
        self.model = model or settings.SIMPLIFICATION_MODEL_ID
        self.api_key = api_key or settings.VLLM_API_KEY
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text using vLLM OpenAI-compatible API."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an expert educational content simplifier for Indian students."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        
        try:
            response = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"vLLM generation failed: {e}")
            raise
    
    async def close(self):
        await self.client.aclose()


class TransformersClient(BaseLLMClient):
    """Direct transformers inference client (fallback)."""
    
    def __init__(self, model_id: str = None):
        self.model_id = model_id or settings.SIMPLIFICATION_MODEL_ID
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            logger.info(f"Loading model: {self.model_id}")
            
            # Determine device
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
            
            # Load with quantization if enabled
            load_kwargs = {
                "device_map": "auto" if device == "cuda" else None,
                "torch_dtype": torch.float16 if device != "cpu" else torch.float32,
            }
            
            if settings.USE_QUANTIZATION and device == "cuda":
                try:
                    from transformers import BitsAndBytesConfig
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                except ImportError:
                    logger.warning("bitsandbytes not available, loading without quantization")
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **load_kwargs
            )
            
            if device in ("mps", "cpu") and load_kwargs.get("device_map") is None:
                self._model = self._model.to(device)
            
            logger.info(f"Model loaded on {device}")
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text using transformers."""
        self._load_model()
        
        # Format as chat template for instruction-tuned model
        messages = [
            {"role": "system", "content": "You are an expert educational content simplifier for Indian students."},
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template
        if hasattr(self._tokenizer, 'apply_chat_template'):
            formatted_prompt = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            formatted_prompt = f"System: You are an expert educational content simplifier.\n\nUser: {prompt}\n\nAssistant:"
        
        inputs = self._tokenizer(formatted_prompt, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        
        import torch
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        
        # Decode only the new tokens
        generated_text = self._tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return generated_text.strip()


class TextSimplifier:
    """
    Text simplification using Llama-3.2-3B-Instruct.
    
    Adapts educational content complexity based on grade level (5-12)
    using state-of-the-art instruction-following model.
    """
    
    # Grade level ranges
    ELEMENTARY_GRADES = range(5, 7)   # Grades 5-6
    MIDDLE_GRADES = range(7, 9)       # Grades 7-8
    SECONDARY_GRADES = range(9, 13)   # Grades 9-12
    
    # Complexity thresholds
    COMPLEXITY_THRESHOLDS = {
        'elementary': (0.0, 0.4),
        'middle': (0.4, 0.7),
        'secondary': (0.7, 1.0)
    }
    
    # Subject terminology to preserve
    SUBJECT_TERMINOLOGY = {
        'Mathematics': [
            'equation', 'variable', 'coefficient', 'polynomial', 'theorem',
            'proof', 'derivative', 'integral', 'function', 'graph'
        ],
        'Science': [
            'photosynthesis', 'molecule', 'atom', 'cell', 'organism',
            'energy', 'force', 'velocity', 'acceleration', 'chemical'
        ],
        'Social Studies': [
            'democracy', 'constitution', 'government', 'economy', 'culture'
        ],
        'History': [
            'ancient', 'medieval', 'modern', 'empire', 'dynasty'
        ],
        'Geography': [
            'latitude', 'longitude', 'climate', 'topography', 'ecosystem'
        ]
    }
    
    def __init__(self, client: BaseLLMClient = None):
        """
        Initialize Text Simplifier.
        
        Args:
            client: LLM client (vLLM or transformers). Auto-selects based on config.
        """
        if client:
            self.client = client
        elif settings.VLLM_ENABLED and settings.SIMPLIFICATION_BACKEND == "vllm":
            self.client = VLLMClient()
        else:
            self.client = TransformersClient()
        
        logger.info(f"TextSimplifier initialized with {type(self.client).__name__}")
    
    async def simplify_text(
        self,
        content: str,
        grade_level: int,
        subject: str
    ) -> SimplifiedText:
        """
        Simplify text content for a specific grade level.
        
        Args:
            content: Original text to simplify
            grade_level: Target grade level (5-12)
            subject: Subject area for context
        
        Returns:
            SimplifiedText with simplified content and metadata
        """
        if not content or len(content.strip()) == 0:
            raise ValueError("Content cannot be empty")
        
        if grade_level < 5 or grade_level > 12:
            raise ValueError(f"Grade level must be between 5 and 12, got {grade_level}")
        
        logger.info(f"Simplifying text for grade {grade_level}, subject {subject}")
        
        # Calculate original complexity
        original_complexity = self.get_complexity_score(content)
        target_range = self._get_target_complexity_range(grade_level)
        
        # Create optimized prompt for Llama-3.2
        prompt = self._create_llama_prompt(content, grade_level, subject)
        
        try:
            # Generate simplified text
            simplified_content = await self.client.generate(
                prompt,
                max_tokens=settings.SIMPLIFICATION_MAX_LENGTH,
                temperature=settings.SIMPLIFICATION_TEMPERATURE
            )
            method = "llama-3.2"
        except Exception as e:
            logger.warning(f"LLM generation failed: {e}, using rule-based fallback")
            simplified_content = self._rule_based_simplification(content, grade_level)
            method = "rule_based"
        
        final_complexity = self.get_complexity_score(simplified_content)
        
        return SimplifiedText(
            text=simplified_content,
            complexity_score=final_complexity,
            grade_level=grade_level,
            subject=subject,
            metadata={
                'original_complexity': original_complexity,
                'target_complexity_range': target_range,
                'simplification_method': method,
                'model': settings.SIMPLIFICATION_MODEL_ID
            }
        )
    
    def _create_llama_prompt(
        self,
        content: str,
        grade_level: int,
        subject: str
    ) -> str:
        """Create optimized prompt for Llama-3.2-3B-Instruct."""
        
        # Grade-specific instructions
        if grade_level in self.ELEMENTARY_GRADES:
            level_desc = "elementary school students (grades 5-6)"
            style = """
- Use very simple sentences (10-15 words max)
- Replace difficult words with everyday alternatives
- Add helpful examples from daily life
- Break complex ideas into small, digestible parts"""
        elif grade_level in self.MIDDLE_GRADES:
            level_desc = "middle school students (grades 7-8)"
            style = """
- Use clear, straightforward language
- Explain technical terms when first used
- Provide relevant examples
- Maintain logical flow between ideas"""
        else:
            level_desc = "high school students (grades 9-12)"
            style = """
- Maintain academic rigor while ensuring clarity
- Use appropriate terminology with brief explanations
- Present concepts with supporting evidence
- Keep sophisticated structure but improve readability"""
        
        # Subject-specific terms to preserve
        preserve_terms = self.SUBJECT_TERMINOLOGY.get(subject, [])
        terms_note = ""
        if preserve_terms:
            terms_note = f"\n\nIMPORTANT: Preserve these key {subject} terms (explain if needed): {', '.join(preserve_terms[:5])}"
        
        prompt = f"""Simplify the following {subject} educational content for {level_desc}.

REQUIREMENTS:
{style}{terms_note}

ORIGINAL TEXT:
{content}

SIMPLIFIED VERSION:"""
        
        return prompt
    
    def _get_target_complexity_range(self, grade_level: int) -> tuple:
        """Get target complexity range for grade level."""
        if grade_level in self.ELEMENTARY_GRADES:
            return self.COMPLEXITY_THRESHOLDS['elementary']
        elif grade_level in self.MIDDLE_GRADES:
            return self.COMPLEXITY_THRESHOLDS['middle']
        return self.COMPLEXITY_THRESHOLDS['secondary']
    
    def get_complexity_score(self, text: str) -> float:
        """
        Calculate text complexity score (0-1).
        
        Uses readability metrics:
        - Average sentence length
        - Average word length
        - Syllable count
        """
        if not text or len(text.strip()) == 0:
            return 0.0
        
        sentences = self._split_sentences(text)
        words = self._split_words(text)
        
        if not sentences or not words:
            return 0.0
        
        avg_sentence_len = len(words) / len(sentences)
        avg_word_len = sum(len(w) for w in words) / len(words)
        avg_syllables = sum(self._count_syllables(w) for w in words) / len(words)
        
        # Normalize to 0-1 scale
        sentence_score = min(max((avg_sentence_len - 5) / 25, 0), 1)
        word_score = min(max((avg_word_len - 3) / 7, 0), 1)
        syllable_score = min(max((avg_syllables - 1) / 3, 0), 1)
        
        return 0.4 * sentence_score + 0.3 * word_score + 0.3 * syllable_score
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_words(self, text: str) -> List[str]:
        """Split text into words."""
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return words
    
    def _count_syllables(self, word: str) -> int:
        """Count syllables in a word (approximate)."""
        word = word.lower()
        if len(word) <= 3:
            return 1
        
        vowels = "aeiou"
        count = 0
        prev_vowel = False
        
        for char in word:
            is_vowel = char in vowels
            if is_vowel and not prev_vowel:
                count += 1
            prev_vowel = is_vowel
        
        # Adjust for silent 'e'
        if word.endswith('e'):
            count -= 1
        
        return max(count, 1)
    
    def _rule_based_simplification(self, text: str, grade_level: int) -> str:
        """Fallback rule-based simplification."""
        # Simple transformations
        result = text
        
        # Replace complex words (basic dictionary)
        replacements = {
            'utilize': 'use',
            'approximately': 'about',
            'demonstrate': 'show',
            'consequently': 'so',
            'furthermore': 'also',
            'nevertheless': 'but',
            'subsequently': 'then',
            'therefore': 'so',
            'regarding': 'about',
            'numerous': 'many',
            'sufficient': 'enough',
            'commence': 'start',
            'terminate': 'end',
            'endeavor': 'try',
            'facilitate': 'help',
        }
        
        for complex_word, simple_word in replacements.items():
            result = re.sub(rf'\b{complex_word}\b', simple_word, result, flags=re.IGNORECASE)
        
        # For lower grades, break long sentences
        if grade_level <= 7:
            result = self._break_long_sentences(result)
        
        return result
    
    def _break_long_sentences(self, text: str) -> str:
        """Break long sentences into shorter ones."""
        sentences = self._split_sentences(text)
        result = []
        
        for sentence in sentences:
            words = sentence.split()
            if len(words) > 25:
                # Try to split at conjunctions
                parts = re.split(r'\b(and|but|or|because|so|while|when)\b', sentence, flags=re.IGNORECASE)
                for i, part in enumerate(parts):
                    part = part.strip()
                    if part and part.lower() not in ('and', 'but', 'or', 'because', 'so', 'while', 'when'):
                        result.append(part.capitalize() if not part[0].isupper() else part)
            else:
                result.append(sentence)
        
        return '. '.join(result) + '.'


# Synchronous wrapper for backward compatibility
def simplify_text_sync(
    content: str,
    grade_level: int,
    subject: str,
    simplifier: TextSimplifier = None
) -> SimplifiedText:
    """Synchronous wrapper for text simplification."""
    import asyncio
    
    if simplifier is None:
        simplifier = TextSimplifier()
    
    return asyncio.run(simplifier.simplify_text(content, grade_level, subject))


# Export
__all__ = [
    'TextSimplifier',
    'SimplifiedText',
    'VLLMClient',
    'TransformersClient',
    'simplify_text_sync'
]
