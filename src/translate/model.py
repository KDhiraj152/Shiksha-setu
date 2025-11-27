"""Simple IndicTrans2/NLLB/mBART model wrapper for local translation."""
import logging
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MBartForConditionalGeneration, MBart50TokenizerFast
import torch

logger = logging.getLogger(__name__)


class IndicTrans2Model:
    """Lightweight wrapper for mBART/NLLB translation model supporting Indian languages."""
    
    # Language code mappings for mBART-50 model
    MBART_LANG_CODES = {
        'Hindi': 'hi_IN',
        'Tamil': 'ta_IN',
        'Telugu': 'te_IN',
        'Bengali': 'bn_IN',
        'Marathi': 'mr_IN',
        'Gujarati': 'gu_IN',
        'Kannada': 'kn_IN',
        'Malayalam': 'ml_IN',
        'Punjabi': 'pa_IN',
        'Odia': 'or_IN'
    }
    
    # Fallback NLLB codes
    NLLB_LANG_CODES = {
        'Hindi': 'hin_Deva',
        'Tamil': 'tam_Taml',
        'Telugu': 'tel_Telu',
        'Bengali': 'ben_Beng',
        'Marathi': 'mar_Deva',
        'Gujarati': 'guj_Gujr',
        'Kannada': 'kan_Knda',
        'Malayalam': 'mal_Mlym',
        'Punjabi': 'pan_Guru',
        'Urdu': 'urd_Arab'
    }
    
    def __init__(self, model_name: str = None):
        """
        Initialize translation model.
        
        Args:
            model_name: HuggingFace model identifier
                       Default: facebook/mbart-large-50-many-to-many-mmt (best for Indian languages)
        """
        if model_name is None:
            # Use mBART-50 as default (excellent for Indian languages, already downloaded)
            model_name = "facebook/mbart-large-50-many-to-many-mmt"
        
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.is_mbart = "mbart" in model_name.lower()
        
        logger.info(f"IndicTrans2Model initialized with: {model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model type: {'mBART' if self.is_mbart else 'NLLB'}")
        logger.info("Model will be loaded on first translation request")
    
    def _load_model(self):
        """Lazy load the model on first use."""
        if self.model is None:
            logger.info(f"Loading translation model: {self.model_name}")
            logger.info("Loading from cache (already downloaded)...")
            
            try:
                if self.is_mbart:
                    # Load mBART model
                    self.tokenizer = MBart50TokenizerFast.from_pretrained(self.model_name)
                    self.model = MBartForConditionalGeneration.from_pretrained(
                        self.model_name
                    ).to(self.device)
                    logger.info("mBART-50 model loaded successfully")
                else:
                    # Load NLLB model
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        src_lang="eng_Latn"
                    )
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        self.model_name
                    ).to(self.device)
                    logger.info("NLLB model loaded successfully")
                
            except Exception as e:
                logger.error(f"Failed to load translation model: {e}")
                raise
    
    def translate(self, text: str, target_language_code: str) -> str:
        """
        Translate text to target language.
        
        Args:
            text: Source text in English
            target_language_code: Language code (e.g., 'hi_IN' for mBART, 'hin_Deva' for NLLB)
        
        Returns:
            Translated text
        """
        # Load model if not already loaded
        if self.model is None:
            self._load_model()
        
        try:
            if self.is_mbart:
                # mBART translation
                self.tokenizer.src_lang = "en_XX"
                
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                # Get target language token ID
                forced_bos_token_id = self.tokenizer.lang_code_to_id[target_language_code]
                
                # Generate translation
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        forced_bos_token_id=forced_bos_token_id,
                        max_length=512,
                        num_beams=5,
                        early_stopping=True
                    )
            else:
                # NLLB translation
                self.tokenizer.src_lang = "eng_Latn"
                
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                
                forced_bos_token_id = self.tokenizer.convert_tokens_to_ids(target_language_code)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        forced_bos_token_id=forced_bos_token_id,
                        max_length=512,
                        num_beams=5,
                        early_stopping=True
                    )
            
            # Decode output
            translated = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            
            logger.info(f"Translation successful: {len(text)} chars -> {len(translated)} chars")
            return translated
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            # Return fallback
            logger.warning("Using fallback for translation")
            return text  # Return original text as fallback
    
    def process(self, text: str, target_language: str) -> str:
        """
        Process method compatible with existing model client interface.
        
        Args:
            text: Source text
            target_language: Language name (Hindi, Tamil, etc.)
        
        Returns:
            Translated text
        """
        # Use mBART codes if mBART model, else NLLB codes
        if self.is_mbart:
            target_code = self.MBART_LANG_CODES.get(target_language, 'hi_IN')
        else:
            target_code = self.NLLB_LANG_CODES.get(target_language, 'hin_Deva')
        
        return self.translate(text, target_code)
