"""
NLLB-200 Translation Engine - High Quality Multilingual Translation

Uses Meta's No Language Left Behind (NLLB-200) model for translation.
Optimized for Apple Silicon M4 with 16GB RAM.

Features:
- 200+ language support including all major Indian languages
- ctranslate2 backend for optimized inference
- Lazy loading with memory management
- Async support for non-blocking operations

Memory: ~2.5GB for 1.3B model in FP16, ~1.5GB with INT8 quantization
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)

# Try to import ctranslate2 for optimized inference
try:
    import ctranslate2
    CTRANSLATE2_AVAILABLE = True
except ImportError:
    CTRANSLATE2_AVAILABLE = False
    logger.warning("ctranslate2 not available, falling back to transformers")

# Fallback to transformers
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class TranslationResult:
    """Result of a translation operation."""
    text: str
    source_language: str
    target_language: str
    confidence: float
    model_used: str
    cached: bool = False


class NLLBTranslator:
    """
    NLLB-200 Translation Engine
    
    Supports all major Indian languages with high quality translation.
    Uses ctranslate2 for optimized inference on Apple Silicon.
    """
    
    # NLLB-200 language codes for Indian languages
    LANGUAGE_CODES = {
        # Indian Languages - Full names
        "English": "eng_Latn",
        "Hindi": "hin_Deva",
        "Tamil": "tam_Taml",
        "Telugu": "tel_Telu",
        "Bengali": "ben_Beng",
        "Marathi": "mar_Deva",
        "Gujarati": "guj_Gujr",
        "Kannada": "kan_Knda",
        "Malayalam": "mal_Mlym",
        "Punjabi": "pan_Guru",
        "Odia": "ory_Orya",
        "Assamese": "asm_Beng",
        "Urdu": "urd_Arab",
        "Sanskrit": "san_Deva",
        "Nepali": "npi_Deva",
        "Sindhi": "snd_Arab",
        "Konkani": "gom_Deva",
        "Maithili": "mai_Deva",
        "Santali": "sat_Olck",
        "Kashmiri": "kas_Arab",
        "Manipuri": "mni_Beng",
        "Bodo": "brx_Deva",
        "Dogri": "doi_Deva",
        # Short codes (for frontend compatibility)
        "en": "eng_Latn",
        "hi": "hin_Deva",
        "ta": "tam_Taml",
        "te": "tel_Telu",
        "bn": "ben_Beng",
        "mr": "mar_Deva",
        "gu": "guj_Gujr",
        "kn": "kan_Knda",
        "ml": "mal_Mlym",
        "pa": "pan_Guru",
        "or": "ory_Orya",
        "as": "asm_Beng",
        "ur": "urd_Arab",
        "sa": "san_Deva",
        "ne": "npi_Deva",
    }
    
    # Reverse mapping for script detection
    SCRIPT_TO_LANGUAGE = {
        "Devanagari": ["Hindi", "Marathi", "Sanskrit", "Nepali", "Konkani", "Bodo", "Dogri"],
        "Tamil": ["Tamil"],
        "Telugu": ["Telugu"],
        "Bengali": ["Bengali", "Assamese", "Manipuri"],
        "Gujarati": ["Gujarati"],
        "Kannada": ["Kannada"],
        "Malayalam": ["Malayalam"],
        "Gurmukhi": ["Punjabi"],
        "Odia": ["Odia"],
        "Arabic": ["Urdu", "Sindhi", "Kashmiri"],
    }
    
    # Model configurations
    MODEL_CONFIGS = {
        "small": {
            "model_id": "facebook/nllb-200-distilled-600M",
            "memory_gb": 1.2,
            "quality": 3,
        },
        "medium": {
            "model_id": "facebook/nllb-200-distilled-1.3B",
            "memory_gb": 2.5,
            "quality": 4,
        },
        "large": {
            "model_id": "facebook/nllb-200-3.3B",
            "memory_gb": 6.5,
            "quality": 5,
        }
    }
    
    def __init__(
        self,
        model_size: str = "small",
        use_ct2: bool = True,
        device: str = "auto",
        compute_type: str = "float16",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize NLLB Translator.
        
        Args:
            model_size: "small" (600M), "medium" (1.3B), or "large" (3.3B)
            use_ct2: Use ctranslate2 for optimized inference
            device: "auto", "cpu", "cuda", or "mps"
            compute_type: "float16", "int8", or "float32"
            cache_dir: Directory for model cache
        """
        self.model_size = model_size
        # Disable ct2 on macOS due to compatibility issues
        self.use_ct2 = False  # use_ct2 and CTRANSLATE2_AVAILABLE
        self.compute_type = compute_type
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/models/nllb")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Detect device
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Model state
        self._model = None
        self._tokenizer = None
        self._ct2_translator = None
        self._loaded = False
        
        # Translation cache
        self._cache: Dict[str, str] = {}
        self._cache_max_size = 1000
        
        config = self.MODEL_CONFIGS[model_size]
        self.model_id = config["model_id"]
        
        logger.info(
            f"NLLBTranslator initialized: size={model_size}, "
            f"device={self.device}, ct2={self.use_ct2}"
        )
    
    def _get_cache_key(self, text: str, src_lang: str, tgt_lang: str) -> str:
        """Generate cache key for translation."""
        content = f"{text}|{src_lang}|{tgt_lang}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def _load_model(self):
        """Load model lazily on first use."""
        if self._loaded:
            return
        
        logger.info(f"Loading NLLB model: {self.model_id}")
        
        if self.use_ct2:
            self._load_ct2_model()
        else:
            self._load_transformers_model()
        
        self._loaded = True
        logger.info("NLLB model loaded successfully")
    
    def _load_ct2_model(self):
        """Load ctranslate2 optimized model."""
        # ctranslate2 model path
        ct2_model_path = self.cache_dir / f"nllb-{self.model_size}-ct2"
        
        if not ct2_model_path.exists():
            logger.info("Converting model to ctranslate2 format (first time only)...")
            # First load with transformers, then convert
            self._load_transformers_model()
            
            # Convert to CT2 format
            try:
                import ctranslate2
                converter = ctranslate2.converters.TransformersConverter(self.model_id)
                converter.convert(
                    str(ct2_model_path),
                    quantization=self.compute_type if self.compute_type != "float16" else None
                )
                logger.info(f"Model converted to CT2 format at {ct2_model_path}")
            except Exception as e:
                logger.warning(f"CT2 conversion failed, using transformers: {e}")
                return
        
        # Load CT2 translator
        self._ct2_translator = ctranslate2.Translator(
            str(ct2_model_path),
            device="cpu" if self.device == "mps" else self.device,  # CT2 doesn't support MPS directly
            compute_type=self.compute_type
        )
        
        # Still need tokenizer from transformers
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            cache_dir=str(self.cache_dir)
        )
    
    def _load_transformers_model(self):
        """Load standard transformers model."""
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            cache_dir=str(self.cache_dir)
        )
        
        # Determine dtype
        if self.compute_type == "float16":
            dtype = torch.float16
        elif self.compute_type == "int8":
            dtype = torch.float16  # Will quantize separately
        else:
            dtype = torch.float32
        
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            cache_dir=str(self.cache_dir)
        )
        
        # Move to device
        if self.device != "cpu":
            self._model.to(self.device)
        
        self._model.eval()
    
    def translate(
        self,
        text: str,
        source_language: str = "English",
        target_language: str = "Hindi",
        max_length: int = 512
    ) -> TranslationResult:
        """
        Translate text between languages.
        
        Args:
            text: Text to translate
            source_language: Source language name (e.g., "English", "Hindi")
            target_language: Target language name
            max_length: Maximum output length
            
        Returns:
            TranslationResult with translated text and metadata
        """
        # Validate languages
        if source_language not in self.LANGUAGE_CODES:
            raise ValueError(f"Unsupported source language: {source_language}")
        if target_language not in self.LANGUAGE_CODES:
            raise ValueError(f"Unsupported target language: {target_language}")
        
        # Check cache
        cache_key = self._get_cache_key(text, source_language, target_language)
        if cache_key in self._cache:
            return TranslationResult(
                text=self._cache[cache_key],
                source_language=source_language,
                target_language=target_language,
                confidence=1.0,
                model_used=self.model_id,
                cached=True
            )
        
        # Load model if needed
        self._load_model()
        
        # Get language codes
        src_code = self.LANGUAGE_CODES[source_language]
        tgt_code = self.LANGUAGE_CODES[target_language]
        
        # Translate
        if self._ct2_translator:
            translated = self._translate_ct2(text, src_code, tgt_code, max_length)
        else:
            translated = self._translate_transformers(text, src_code, tgt_code, max_length)
        
        # Cache result
        if len(self._cache) >= self._cache_max_size:
            # Remove oldest entry
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = translated
        
        return TranslationResult(
            text=translated,
            source_language=source_language,
            target_language=target_language,
            confidence=0.9,  # Estimated
            model_used=self.model_id,
            cached=False
        )
    
    def _translate_ct2(
        self,
        text: str,
        src_code: str,
        tgt_code: str,
        max_length: int
    ) -> str:
        """Translate using ctranslate2."""
        # Tokenize
        self._tokenizer.src_lang = src_code
        tokens = self._tokenizer.tokenize(text)
        
        # Add language tokens
        source_tokens = [src_code] + tokens
        
        # Translate
        results = self._ct2_translator.translate_batch(
            [source_tokens],
            target_prefix=[[tgt_code]],
            max_decoding_length=max_length,
            beam_size=4
        )
        
        # Decode
        output_tokens = results[0].hypotheses[0][1:]  # Remove language token
        translated = self._tokenizer.convert_tokens_to_string(output_tokens)
        
        return translated
    
    def _translate_transformers(
        self,
        text: str,
        src_code: str,
        tgt_code: str,
        max_length: int
    ) -> str:
        """Translate using transformers."""
        import torch
        
        # Set source language
        self._tokenizer.src_lang = src_code
        
        # Tokenize
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        )
        
        # Move to device
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get target language token ID
        forced_bos_token_id = self._tokenizer.convert_tokens_to_ids(tgt_code)
        
        # Generate
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_token_id,
                max_length=max_length,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode
        translated = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return translated
    
    async def translate_async(
        self,
        text: str,
        source_language: str = "English",
        target_language: str = "Hindi",
        max_length: int = 512
    ) -> TranslationResult:
        """Async translation for non-blocking operations."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.translate(text, source_language, target_language, max_length)
        )
    
    async def translate_batch_async(
        self,
        texts: List[str],
        source_language: str = "English",
        target_language: str = "Hindi"
    ) -> List[TranslationResult]:
        """Translate multiple texts asynchronously."""
        tasks = [
            self.translate_async(text, source_language, target_language)
            for text in texts
        ]
        return await asyncio.gather(*tasks)
    
    def unload(self):
        """Unload model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        if self._ct2_translator is not None:
            del self._ct2_translator
            self._ct2_translator = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        
        self._loaded = False
        
        # Clear CUDA/MPS cache
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
        
        import gc
        gc.collect()
        
        logger.info("NLLB model unloaded")
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language names."""
        return list(self.LANGUAGE_CODES.keys())
    
    def is_loaded(self) -> bool:
        """Check if model is currently loaded."""
        return self._loaded
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage info."""
        import torch
        info = {
            "model_loaded": self._loaded,
            "model_size": self.model_size,
            "cache_entries": len(self._cache),
        }
        
        if self.device == "cuda" and torch.cuda.is_available():
            info["gpu_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            info["gpu_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
        elif self.device == "mps" and hasattr(torch.mps, 'current_allocated_memory'):
            try:
                info["mps_allocated_gb"] = torch.mps.current_allocated_memory() / 1e9
            except:
                pass
        
        return info


# Singleton instance
_translator_instance: Optional[NLLBTranslator] = None


def get_nllb_translator(
    model_size: str = "small",
    **kwargs
) -> NLLBTranslator:
    """Get or create singleton NLLB translator instance."""
    global _translator_instance
    
    if _translator_instance is None:
        _translator_instance = NLLBTranslator(model_size=model_size, **kwargs)
    
    return _translator_instance
