"""Hugging Face model client wrappers with authentication, rate limiting, and Bhashini fallback."""
import os
import time
import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# Content-Type constant
CONTENT_TYPE_JSON = "application/json"


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, max_calls: int = 100, time_window: int = 60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
    
    def wait_if_needed(self):
        """Wait if rate limit is exceeded."""
        now = time.time()
        # Remove calls outside the time window
        self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
        
        if len(self.calls) >= self.max_calls:
            sleep_time = self.time_window - (now - self.calls[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
                self.calls = []
        
        self.calls.append(now)


class BaseModelClient(ABC):
    """Base class for Hugging Face model clients."""
    
    def __init__(self, model_id: str, api_key: Optional[str] = None):
        self.model_id = model_id
        self.api_key = api_key or os.getenv('HUGGINGFACE_API_KEY')
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        self.rate_limiter = RateLimiter(max_calls=100, time_window=60)
        
        # Configure session with retry logic
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication."""
        headers = {"Content-Type": CONTENT_TYPE_JSON}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make API request with rate limiting and error handling."""
        self.rate_limiter.wait_if_needed()
        
        try:
            response = self.session.post(
                self.api_url,
                headers=self._get_headers(),
                json=payload,
                timeout=30
            )
            
            if response.status_code == 503:
                # Model is loading, wait and retry
                time.sleep(20)
                response = self.session.post(
                    self.api_url,
                    headers=self._get_headers(),
                    json=payload,
                    timeout=30
                )
            
            # Check for deprecated/moved models (410)
            if response.status_code == 410:
                raise RuntimeError(f"Model {self.model_id} is deprecated or moved. Using fallback.")
            
            response.raise_for_status()
            return response.json()
        except (requests.exceptions.HTTPError, RuntimeError) as e:
            # Log the error and raise for fallback handling
            raise RuntimeError(f"API request failed: {e}")
    
    @abstractmethod
    def process(self, *args, **kwargs):
        """Process input through the model."""
        pass


class FlanT5Client(BaseModelClient):
    """Client for Flan-T5 text simplification model."""
    
    def __init__(self, api_key: Optional[str] = None):
        model_id = os.getenv('FLANT5_MODEL_ID', 'google/flan-t5-base')
        super().__init__(model_id, api_key)
    
    def process(self, text: str, grade_level: int, subject: str) -> str:
        """Simplify text for the specified grade level."""
        try:
            prompt = self._create_prompt(text, grade_level, subject)
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_length": 512,
                    "temperature": 0.7,
                    "do_sample": True
                }
            }
            
            result = self._make_request(payload)
            return result[0]['generated_text'] if isinstance(result, list) else result.get('generated_text', '')
        except RuntimeError:
            # Fallback to rule-based simplification
            return self._fallback_simplification(text, grade_level)
    
    def _fallback_simplification(self, text: str, grade_level: int) -> str:
        """Simple rule-based text simplification as fallback."""
        # Basic simplification: shorter sentences, simpler words
        import re
        sentences = re.split(r'[.!?]+', text)
        simplified_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            # Break long sentences at conjunctions for lower grades
            if grade_level <= 8 and len(sentence.split()) > 20:
                parts = re.split(r'\s+(and|but|because|so|which)\s+', sentence, maxsplit=1)
                if len(parts) >= 3:
                    simplified_sentences.append(parts[0].strip() + '.')
                    simplified_sentences.append(parts[2].strip())
                else:
                    simplified_sentences.append(sentence)
            else:
                simplified_sentences.append(sentence)
        
        return '. '.join(simplified_sentences) + '.'
    
    def _create_prompt(self, text: str, grade_level: int, subject: str) -> str:
        """Create grade-appropriate simplification prompt."""
        return f"Simplify the following {subject} text for grade {grade_level} students: {text}"


class IndicTrans2Client(BaseModelClient):
    """Client for IndicTrans2 translation model with Bhashini fallback."""
    
    def __init__(self, api_key: Optional[str] = None):
        # Use working IndicTrans2 model
        model_id = os.getenv('INDICTRANS2_MODEL_ID', 'ai4bharat/indictrans2-en-indic-dist-200M')
        super().__init__(model_id, api_key)
        
        # Initialize Bhashini client for fallback
        self.bhashini_available = bool(os.getenv('BHASHINI_API_KEY'))
        self.bhashini_client = None
    
    def _get_bhashini_client(self):
        """Lazy load Bhashini client."""
        if self.bhashini_client is None and self.bhashini_available:
            try:
                from ..services.bhashini import get_bhashini_client
                self.bhashini_client = get_bhashini_client()
            except Exception as e:
                logger.warning(f"Failed to initialize Bhashini client: {e}")
                self.bhashini_available = False
        return self.bhashini_client
    
    def process(self, text: str, target_language: str, source_language: str = "English") -> str:
        """
        Translate text to target Indian language with Bhashini fallback.
        
        Priority:
        1. HuggingFace IndicTrans2 API (fastest for single requests)
        2. Bhashini API (government service, reliable)
        3. Marked text fallback
        """
        # Try HuggingFace API first
        try:
            payload = {
                "inputs": text,
                "parameters": {
                    "src_lang": "eng_Latn" if source_language == "English" else self._get_language_code(source_language),
                    "tgt_lang": self._get_language_code(target_language)
                }
            }
            
            result = self._make_request(payload)
            translated = result[0]['translation_text'] if isinstance(result, list) else result.get('translation_text', '')
            
            if translated:
                return translated
                
        except RuntimeError as e:
            logger.warning(f"HuggingFace translation failed: {e}, trying Bhashini...")
        
        # Fallback to Bhashini API
        if self.bhashini_available:
            try:
                bhashini = self._get_bhashini_client()
                if bhashini:
                    translated = bhashini.translate(text, source_language, target_language)
                    if translated:
                        logger.info("Translation successful via Bhashini API")
                        return translated
            except Exception as e:
                logger.warning(f"Bhashini translation failed: {e}")
        
        # Final fallback: marked text
        logger.warning("All translation methods failed, using marked text fallback")
        return f"[Translation to {target_language}] {text}"
    
    def _get_language_code(self, language: str) -> str:
        """Map language names to IndicTrans2 codes."""
        language_map = {
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
        return language_map.get(language, 'hin_Deva')
    
    def translate(self, text: str, source_lang: str = "English", target_lang: str = "Hindi") -> str:
        """Wrapper method for translation (calls process())."""
        return self.process(text, target_lang, source_lang)


class BERTClient(BaseModelClient):
    """Client for BERT validation model."""
    
    def __init__(self, api_key: Optional[str] = None):
        model_id = os.getenv('BERT_MODEL_ID', 'bert-base-multilingual-cased')
        super().__init__(model_id, api_key)
    
    def process(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        try:
            payload = {
                "inputs": {
                    "source_sentence": text1,
                    "sentences": [text2]
                }
            }
            
            result = self._make_request(payload)
            # Return similarity score (0-1)
            if isinstance(result, list) and len(result) > 0:
                return result[0]
            return 0.0
        except RuntimeError:
            # Fallback heuristic based on word overlap
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.0
            intersection = words1.intersection(words2)
            union = words1.union(words2)
            return len(intersection) / len(union) if union else 0.0


class VITSClient(BaseModelClient):
    """Client for VITS text-to-speech model with Bhashini fallback."""
    
    DEFAULT_TTS_MODEL = 'facebook/mms-tts-hin-script_devanagari'
    
    def __init__(self, api_key: Optional[str] = None):
        model_id = os.getenv('VITS_MODEL_ID', self.DEFAULT_TTS_MODEL)
        super().__init__(model_id, api_key)
        
        # Initialize Bhashini client for fallback
        self.bhashini_available = bool(os.getenv('BHASHINI_API_KEY'))
        self.bhashini_client = None
    
    def _get_bhashini_client(self):
        """Lazy load Bhashini client."""
        if self.bhashini_client is None and self.bhashini_available:
            try:
                from ..services.bhashini import get_bhashini_client
                self.bhashini_client = get_bhashini_client()
            except Exception as e:
                logger.warning(f"Failed to initialize Bhashini client: {e}")
                self.bhashini_available = False
        return self.bhashini_client
    
    def process(self, text: str, language: str) -> bytes:
        """
        Generate speech audio from text with Bhashini fallback.
        
        Priority:
        1. HuggingFace VITS/MMS-TTS API
        2. Bhashini TTS API (government service, reliable)
        3. Empty bytes fallback
        """
        # Try HuggingFace API first
        try:
            # Update model ID based on language
            self.model_id = self._get_model_for_language(language)
            self.api_url = f"https://api-inference.huggingface.co/models/{self.model_id}"
            
            payload = {"inputs": text}
            
            response = self.session.post(
                self.api_url,
                headers=self._get_headers(),
                json=payload,
                timeout=30  # Reduced timeout
            )
            
            # If model is loading or deprecated, don't retry
            if response.status_code in [503, 404, 401]:
                logger.warning(f"HuggingFace TTS model unavailable (status {response.status_code}), using fallback")
                raise RuntimeError("Model unavailable")
            
            # Check for deprecated models
            if response.status_code == 410:
                raise RuntimeError(f"TTS model {self.model_id} is deprecated")
            
            response.raise_for_status()
            
            if response.content and len(response.content) > 0:
                return response.content
                
        except RuntimeError as e:
            logger.warning(f"HuggingFace TTS failed: {e}, trying Bhashini...")
        
        # Fallback to Bhashini API
        if self.bhashini_available:
            try:
                bhashini = self._get_bhashini_client()
                if bhashini:
                    audio_bytes = bhashini.text_to_speech(text, language)
                    if audio_bytes and len(audio_bytes) > 0:
                        logger.info("TTS generation successful via Bhashini API")
                        return audio_bytes
            except Exception as e:
                logger.warning(f"Bhashini TTS failed: {e}")
        
        # Final fallback: Generate minimal WAV header (for testing only)
        logger.warning("All TTS methods failed, returning minimal WAV header")
        # Return a minimal valid WAV file header (44 bytes)
        return bytes([
            0x52, 0x49, 0x46, 0x46,  # "RIFF"
            0x24, 0x00, 0x00, 0x00,  # File size - 8
            0x57, 0x41, 0x56, 0x45,  # "WAVE"
            0x66, 0x6D, 0x74, 0x20,  # "fmt "
            0x10, 0x00, 0x00, 0x00,  # Subchunk1Size (16)
            0x01, 0x00,              # AudioFormat (PCM)
            0x01, 0x00,              # NumChannels (Mono)
            0x44, 0xAC, 0x00, 0x00,  # SampleRate (44100)
            0x88, 0x58, 0x01, 0x00,  # ByteRate
            0x02, 0x00,              # BlockAlign
            0x10, 0x00,              # BitsPerSample (16)
            0x64, 0x61, 0x74, 0x61,  # "data"
            0x00, 0x00, 0x00, 0x00   # Subchunk2Size (0 - empty audio)
        ])
    
    def _get_model_for_language(self, language: str) -> str:
        """Get appropriate TTS model for language."""
        model_map = {
            'Hindi': self.DEFAULT_TTS_MODEL,
            'Tamil': 'facebook/mms-tts-tam',
            'Telugu': 'facebook/mms-tts-tel',
            'Bengali': 'facebook/mms-tts-ben',
            'Marathi': 'facebook/mms-tts-mar'
        }
        return model_map.get(language, self.DEFAULT_TTS_MODEL)
    
    def synthesize(self, text: str, language: str = "Hindi") -> bytes:
        """Wrapper method for speech synthesis (calls process())."""
        return self.process(text, language)


class BhashiniTTSClient:
    """Client for Bhashini TTS API (preferred for local development)."""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv('BHASHINI_API_KEY')
        self.api_url = os.getenv(
            'BHASHINI_API_URL',
            'https://dhruva-api.bhashini.gov.in/services/inference/pipeline'
        )
        self.rate_limiter = RateLimiter(max_calls=100, time_window=60)
        self.session = requests.Session()
        
        # Configure session with retry
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
    
    def process(self, text: str, language: str) -> bytes:
        """Generate speech using Bhashini TTS with error handling."""
        if not self.api_key:
            raise ValueError("BHASHINI_API_KEY not configured")
        
        self.rate_limiter.wait_if_needed()
        
        try:
            payload = {
                "pipelineTasks": [
                    {
                        "taskType": "tts",
                        "config": {
                            "language": {
                                "sourceLanguage": self._get_language_code(language)
                            },
                            "gender": "female",
                            "samplingRate": 16000
                        }
                    }
                ],
                "inputData": {
                    "input": [{"source": text}]
                }
            }
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json"
            }
            
            response = self.session.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            # Extract audio from response
            result = response.json()
            pipeline_response = result.get('pipelineResponse', [])
            
            if not pipeline_response:
                raise ValueError("Empty response from Bhashini API")
            
            audio_data = pipeline_response[0].get('audio', [])
            if not audio_data:
                raise ValueError("No audio data in Bhashini response")
            
            audio_content = audio_data[0].get('audioContent', '')
            
            if not audio_content:
                raise ValueError("Empty audio content from Bhashini")
            
            # Decode base64 audio
            import base64
            return base64.b64decode(audio_content)
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Bhashini API request failed: {e}")
        except (KeyError, IndexError, ValueError) as e:
            raise RuntimeError(f"Bhashini API response parsing failed: {e}")
    
    def _get_language_code(self, language: str) -> str:
        """Map language names to Bhashini codes."""
        language_map = {
            'Hindi': 'hi',
            'Tamil': 'ta',
            'Telugu': 'te',
            'Bengali': 'bn',
            'Marathi': 'mr'
        }
        return language_map.get(language, 'hi')


class QwenClient(BaseModelClient):
    """Client for Qwen2.5-7B-Instruct content generation."""
    
    def __init__(self, api_key: Optional[str] = None, use_vllm: bool = False):
        model_id = os.getenv('CONTENT_GEN_MODEL_ID', 'Qwen/Qwen2.5-7B-Instruct')
        super().__init__(model_id, api_key)
        self.use_vllm = use_vllm
        self.local_model = None
        self.local_tokenizer = None
    
    def _load_local_model(self):
        """Load local model for offline inference."""
        if self.local_model is None:
            from ..utils.model_loader import ModelLoader, BITSANDBYTES_AVAILABLE
            from ..core.config import settings
            
            loader = ModelLoader()
            
            # If quantization requested but unavailable, use FP16/FP32
            quantization = settings.CONTENT_GEN_QUANTIZATION
            if not BITSANDBYTES_AVAILABLE and quantization in ['4bit', '8bit']:
                logger.warning(f"Quantization not available (bitsandbytes missing). Loading in FP16 instead.")
                quantization = 'none'  # Load in FP16/FP32
            
            self.local_model, self.local_tokenizer = loader.load_causal_lm_model(
                model_id=self.model_id,
                quantization=quantization,
                use_vllm=self.use_vllm
            )
            
            # If model loading returned None (API mode), keep as None
            if self.local_model is None:
                logger.info(f"Model {self.model_id} will use API mode")
    
    def generate(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        use_local: bool = False
    ) -> str:
        """
        Generate content using Qwen2.5.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            use_local: Use local model instead of API (default: False, use API)
            
        Returns:
            Generated text
        """
        # Try API first (faster and no local resource requirements)
        if not use_local and self.api_key:
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_length,
                    "temperature": temperature,
                    "return_full_text": False
                }
            }
            try:
                response = self._make_request(payload)
                if isinstance(response, list) and len(response) > 0:
                    return response[0].get("generated_text", "")
                return str(response)
            except RuntimeError as e:
                logger.warning(f"API request failed: {e}, falling back to local model")
        
        # Fallback to local model
        if use_local or not self.api_key:
            self._load_local_model()
            
            # Check if model loaded successfully
            if self.local_model is None:
                raise RuntimeError("Local model unavailable and API failed. Cannot generate content.")
            
            # Format prompt for Qwen2.5
            formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            if self.use_vllm:
                # vLLM inference
                try:
                    from vllm import SamplingParams
                    sampling_params = SamplingParams(
                        temperature=temperature,
                        max_tokens=max_length,
                        top_p=0.9
                    )
                    outputs = self.local_model.generate([formatted_prompt], sampling_params)
                    return outputs[0].outputs[0].text
                except ImportError:
                    pass  # Fall through to standard inference
            
            # Standard inference
            import torch
            inputs = self.local_tokenizer(formatted_prompt, return_tensors="pt").to(self.local_model.device)
            outputs = self.local_model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9
            )
            response = self.local_tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract assistant response
            return response.split("<|im_start|>assistant\n")[-1].strip()
    
    def process(self, text: str, grade_level: int, subject: str) -> str:
        """Simplify text for the specified grade level using Qwen."""
        prompt = f"Simplify the following {subject} text for grade {grade_level} students (ages {grade_level + 5}-{grade_level + 6}). Make it clear and age-appropriate:\n\n{text}"
        return self.generate(prompt, max_length=512, temperature=0.7, use_local=True)


class E5EmbeddingClient:
    """Client for Multilingual-E5-Large embeddings."""
    
    def __init__(self, use_onnx: bool = True):
        self.model_id = os.getenv('EMBEDDING_MODEL_ID', 'intfloat/multilingual-e5-large')
        self.use_onnx = use_onnx
        self.model = None
    
    def _load_model(self):
        """Load embedding model with ONNX optimization."""
        if self.model is None:
            from ..utils.model_loader import ModelLoader
            
            loader = ModelLoader()
            self.model = loader.load_embedding_model_optimized(
                model_id=self.model_id,
                use_onnx=self.use_onnx
            )
    
    def encode(
        self,
        texts: list,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        query_mode: bool = False
    ) -> list:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for encoding
            normalize_embeddings: Normalize to unit length
            query_mode: Use "query:" prefix instead of "passage:"
            
        Returns:
            List of embedding vectors (1024-dim)
        """
        self._load_model()
        
        # E5 models require task prefix
        prefix = "query: " if query_mode else "passage: "
        texts_with_prefix = [f"{prefix}{text}" for text in texts]
        
        embeddings = self.model.encode(
            texts_with_prefix,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=False
        )
        
        return embeddings.tolist()

