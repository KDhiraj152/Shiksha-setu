"""IndicTrans2 Translation Model - Best-in-class for Indian Languages.

Optimal 2025 Model Stack: ai4bharat/indictrans2-en-indic-1B
- Purpose-built for Indian languages
- 2GB with INT4 quantization
- Supports 22 Indian languages
- Superior to mBART/NLLB for Indic languages
"""
import logging
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
import torch

from ...core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class TranslationResult:
    """Result of translation operation."""
    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    confidence: float
    metadata: Dict


class IndicTrans2:
    """
    IndicTrans2 translation model wrapper.
    
    Best-in-class for English ↔ Indian language translation.
    Supports: Hindi, Tamil, Telugu, Bengali, Marathi, Gujarati, 
              Kannada, Malayalam, Punjabi, Odia, Assamese, Urdu, etc.
    """
    
    # IndicTrans2 language codes (BCP-47 format)
    LANG_CODES = {
        # Indian languages
        'Hindi': 'hin_Deva',
        'Tamil': 'tam_Taml',
        'Telugu': 'tel_Telu',
        'Bengali': 'ben_Beng',
        'Marathi': 'mar_Deva',
        'Gujarati': 'guj_Gujr',
        'Kannada': 'kan_Knda',
        'Malayalam': 'mal_Mlym',
        'Punjabi': 'pan_Guru',
        'Odia': 'ory_Orya',
        'Assamese': 'asm_Beng',
        'Urdu': 'urd_Arab',
        'Konkani': 'kok_Deva',
        'Maithili': 'mai_Deva',
        'Nepali': 'npi_Deva',
        'Sanskrit': 'san_Deva',
        'Sindhi': 'snd_Arab',
        'Santali': 'sat_Olck',
        'Kashmiri': 'kas_Arab',
        'Dogri': 'doi_Deva',
        'Manipuri': 'mni_Beng',
        'Bodo': 'brx_Deva',
        # English
        'English': 'eng_Latn',
    }
    
    # Reverse mapping
    CODE_TO_LANG = {v: k for k, v in LANG_CODES.items()}
    
    def __init__(self, model_id: str = None, device: str = None):
        """
        Initialize IndicTrans2 model.
        
        Args:
            model_id: Model identifier (default from config)
            device: Device to use (auto-detected if not specified)
        """
        self.model_id = model_id or settings.TRANSLATION_MODEL_ID
        
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        self._model = None
        self._tokenizer = None
        self._processor = None
        
        logger.info(f"IndicTrans2 initialized: {self.model_id} on {self.device}")
    
    def _load_model(self):
        """Lazy load the model."""
        if self._model is not None:
            return
        
        logger.info(f"Loading IndicTrans2 model: {self.model_id}")
        
        try:
            # Try loading with AI4Bharat's IndicTrans2 library
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
            
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                cache_dir=str(settings.MODEL_CACHE_DIR)
            )
            
            # Load model with quantization if available
            load_kwargs = {
                "trust_remote_code": True,
                "cache_dir": str(settings.MODEL_CACHE_DIR),
            }
            
            if self.device == "cuda" and settings.USE_QUANTIZATION:
                try:
                    from transformers import BitsAndBytesConfig
                    load_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                    )
                    load_kwargs["device_map"] = "auto"
                except ImportError:
                    load_kwargs["torch_dtype"] = torch.float16
                    load_kwargs["device_map"] = "auto"
            elif self.device == "cuda":
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["device_map"] = "auto"
            
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_id,
                **load_kwargs
            )
            
            # Move to device if not using device_map
            if "device_map" not in load_kwargs:
                self._model = self._model.to(self.device)
            
            self._model.eval()
            logger.info(f"IndicTrans2 model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load IndicTrans2: {e}")
            logger.info("Attempting fallback to NLLB...")
            self._load_nllb_fallback()
    
    def _load_nllb_fallback(self):
        """Load NLLB as fallback if IndicTrans2 fails."""
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        
        fallback_model = "facebook/nllb-200-distilled-600M"
        logger.info(f"Loading fallback model: {fallback_model}")
        
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                fallback_model,
                cache_dir=str(settings.MODEL_CACHE_DIR)
            )
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                fallback_model,
                cache_dir=str(settings.MODEL_CACHE_DIR)
            ).to(self.device)
            
            self.model_id = fallback_model
            self._model.eval()
            logger.info("NLLB fallback model loaded")
        except Exception as e:
            logger.error(f"Fallback model loading also failed: {e}")
            raise
    
    def get_lang_code(self, language: str) -> str:
        """Get IndicTrans2/NLLB language code."""
        return self.LANG_CODES.get(language, language)
    
    def translate(
        self,
        text: str,
        source_lang: str = "English",
        target_lang: str = "Hindi",
        max_length: int = None
    ) -> TranslationResult:
        """
        Translate text between languages.
        
        Args:
            text: Source text to translate
            source_lang: Source language name
            target_lang: Target language name
            max_length: Maximum output length
        
        Returns:
            TranslationResult with translated text
        """
        self._load_model()
        
        if not text or not text.strip():
            return TranslationResult(
                source_text=text,
                translated_text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                confidence=1.0,
                metadata={"error": "empty_input"}
            )
        
        max_length = max_length or settings.TRANSLATION_MAX_LENGTH
        
        src_code = self.get_lang_code(source_lang)
        tgt_code = self.get_lang_code(target_lang)
        
        try:
            # Tokenize with source language
            if hasattr(self._tokenizer, 'src_lang'):
                self._tokenizer.src_lang = src_code
            
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            
            # Move to device
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            
            # Generate translation
            with torch.no_grad():
                # Get forced BOS token for target language
                if hasattr(self._tokenizer, 'lang_code_to_id'):
                    forced_bos_token_id = self._tokenizer.lang_code_to_id.get(tgt_code)
                else:
                    forced_bos_token_id = self._tokenizer.convert_tokens_to_ids(tgt_code)
                
                outputs = self._model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=max_length,
                    num_beams=5,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                )
            
            # Decode
            translated_text = self._tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            return TranslationResult(
                source_text=text,
                translated_text=translated_text.strip(),
                source_lang=source_lang,
                target_lang=target_lang,
                confidence=0.95,  # IndicTrans2 is highly accurate
                metadata={
                    "model": self.model_id,
                    "src_code": src_code,
                    "tgt_code": tgt_code
                }
            )
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return TranslationResult(
                source_text=text,
                translated_text=text,
                source_lang=source_lang,
                target_lang=target_lang,
                confidence=0.0,
                metadata={"error": str(e)}
            )
    
    async def translate_async(
        self,
        text: str,
        source_lang: str = "English",
        target_lang: str = "Hindi",
        max_length: int = None
    ) -> TranslationResult:
        """Async wrapper for translation."""
        import asyncio
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.translate(text, source_lang, target_lang, max_length)
        )
    
    def translate_batch(
        self,
        texts: List[str],
        source_lang: str = "English",
        target_lang: str = "Hindi",
        max_length: int = None
    ) -> List[TranslationResult]:
        """
        Translate multiple texts efficiently.
        
        Args:
            texts: List of texts to translate
            source_lang: Source language
            target_lang: Target language
            max_length: Maximum output length per text
        
        Returns:
            List of TranslationResults
        """
        self._load_model()
        
        if not texts:
            return []
        
        max_length = max_length or settings.TRANSLATION_MAX_LENGTH
        src_code = self.get_lang_code(source_lang)
        tgt_code = self.get_lang_code(target_lang)
        
        results = []
        
        try:
            if hasattr(self._tokenizer, 'src_lang'):
                self._tokenizer.src_lang = src_code
            
            # Batch tokenize
            inputs = self._tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length
            )
            inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                if hasattr(self._tokenizer, 'lang_code_to_id'):
                    forced_bos_token_id = self._tokenizer.lang_code_to_id.get(tgt_code)
                else:
                    forced_bos_token_id = self._tokenizer.convert_tokens_to_ids(tgt_code)
                
                outputs = self._model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    max_length=max_length,
                    num_beams=5,
                    early_stopping=True,
                )
            
            # Decode all
            translated_texts = self._tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True
            )
            
            for src_text, tgt_text in zip(texts, translated_texts):
                results.append(TranslationResult(
                    source_text=src_text,
                    translated_text=tgt_text.strip(),
                    source_lang=source_lang,
                    target_lang=target_lang,
                    confidence=0.95,
                    metadata={"model": self.model_id}
                ))
            
        except Exception as e:
            logger.error(f"Batch translation failed: {e}")
            # Return individual failures
            for text in texts:
                results.append(TranslationResult(
                    source_text=text,
                    translated_text=text,
                    source_lang=source_lang,
                    target_lang=target_lang,
                    confidence=0.0,
                    metadata={"error": str(e)}
                ))
        
        return results
    
    @property
    def supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return list(self.LANG_CODES.keys())
    
    def is_language_supported(self, language: str) -> bool:
        """Check if language is supported."""
        return language in self.LANG_CODES


# Singleton instance
_translator_instance: Optional[IndicTrans2] = None


def get_translator() -> IndicTrans2:
    """Get or create translator singleton."""
    global _translator_instance
    if _translator_instance is None:
        _translator_instance = IndicTrans2()
    return _translator_instance


# Export
__all__ = [
    'IndicTrans2',
    'TranslationResult',
    'get_translator'
]
