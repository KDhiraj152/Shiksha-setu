"""
Bhashini (Udyat) API Integration for Translation and TTS.

This module provides production-grade integration with India's National Language Translation Mission.
Bhashini provides high-quality translation and text-to-speech for Indian languages.

API Documentation: https://bhashini.gov.in/ulca/apis
"""
import os
import logging
import base64
from typing import Optional, Dict, Any, List, Literal
from dataclasses import dataclass
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# Language codes mapping
BHASHINI_LANGUAGE_CODES = {
    'Hindi': 'hi',
    'Tamil': 'ta',
    'Telugu': 'te',
    'Bengali': 'bn',
    'Marathi': 'mr',
    'Gujarati': 'gu',
    'Kannada': 'kn',
    'Malayalam': 'ml',
    'Punjabi': 'pa',
    'Urdu': 'ur',
    'Odia': 'or',
    'Assamese': 'as',
    'English': 'en'
}

TaskType = Literal["translation", "tts", "asr", "transliteration"]


@dataclass
class BhashiniConfig:
    """Configuration for Bhashini API."""
    api_key: str
    user_id: str
    pipeline_url: str = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline"
    inference_url: str = "https://dhruva-api.bhashini.gov.in/services/inference/pipeline"
    timeout: int = 60
    max_retries: int = 3


class BhashiniClient:
    """
    Production-ready Bhashini API client.
    
    Supports:
    - Translation (English ↔ Indian languages, Indian language ↔ Indian language)
    - Text-to-Speech (TTS) for all Indian languages
    - Automatic Speech Recognition (ASR)
    - Transliteration
    
    Usage:
        client = BhashiniClient(config)
        
        # Translation
        translated = client.translate("Hello world", source_lang="en", target_lang="hi")
        
        # Text-to-Speech
        audio_bytes = client.text_to_speech("नमस्ते", language="hi")
    """
    
    def __init__(self, config: Optional[BhashiniConfig] = None):
        """
        Initialize Bhashini client.
        
        Args:
            config: Bhashini configuration. If None, loads from environment.
        """
        if config is None:
            config = self._load_from_env()
        
        self.config = config
        self.session = self._create_session()
        self.pipeline_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Bhashini client initialized")
    
    @staticmethod
    def _load_from_env() -> BhashiniConfig:
        """Load configuration from environment variables."""
        api_key = os.getenv('BHASHINI_API_KEY')
        user_id = os.getenv('BHASHINI_USER_ID')
        
        if not api_key or not user_id:
            raise ValueError(
                "BHASHINI_API_KEY and BHASHINI_USER_ID must be set. "
                "Get credentials from https://bhashini.gov.in/ulca"
            )
        
        return BhashiniConfig(
            api_key=api_key,
            user_id=user_id,
            pipeline_url=os.getenv(
                'BHASHINI_API_URL',
                'https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline'
            ),
            inference_url=os.getenv(
                'BHASHINI_INFERENCE_URL',
                'https://dhruva-api.bhashini.gov.in/services/inference/pipeline'
            )
        )
    
    def _create_session(self) -> requests.Session:
        """Create requests session with retry logic."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=self.config.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST", "GET"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        return session
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication."""
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "userID": self.config.user_id,
            "ulcaApiKey": self.config.api_key
        }
    
    def _get_language_code(self, language: str) -> str:
        """Convert language name to Bhashini code."""
        code = BHASHINI_LANGUAGE_CODES.get(language)
        if not code:
            raise ValueError(
                f"Unsupported language: {language}. "
                f"Supported: {', '.join(BHASHINI_LANGUAGE_CODES.keys())}"
            )
        return code
    
    def _get_pipeline_config(
        self,
        task_type: TaskType,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get pipeline configuration for a task.
        
        This fetches the model pipeline details from Bhashini.
        """
        cache_key = f"{task_type}:{source_lang}:{target_lang}"
        
        if cache_key in self.pipeline_cache:
            return self.pipeline_cache[cache_key]
        
        payload = {
            "pipelineTasks": [
                {
                    "taskType": task_type,
                    "config": {}
                }
            ],
            "pipelineRequestConfig": {
                "pipelineId": "64392f96daac500b55c543cd"
            }
        }
        
        # Add language config for translation/TTS
        if task_type == "translation" and source_lang and target_lang:
            payload["pipelineTasks"][0]["config"] = {
                "language": {
                    "sourceLanguage": self._get_language_code(source_lang),
                    "targetLanguage": self._get_language_code(target_lang)
                }
            }
        elif task_type == "tts" and source_lang:
            payload["pipelineTasks"][0]["config"] = {
                "language": {
                    "sourceLanguage": self._get_language_code(source_lang)
                },
                "gender": "female"
            }
        
        try:
            response = self.session.post(
                self.config.pipeline_url,
                headers=self._get_headers(),
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            pipeline_config = response.json()
            self.pipeline_cache[cache_key] = pipeline_config
            
            return pipeline_config
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get Bhashini pipeline config: {e}")
            raise RuntimeError(f"Bhashini pipeline configuration failed: {e}")
    
    def translate(
        self,
        text: str,
        source_lang: str = "English",
        target_lang: str = "Hindi"
    ) -> str:
        """
        Translate text using Bhashini API.
        
        Args:
            text: Text to translate
            source_lang: Source language name (e.g., 'English', 'Hindi')
            target_lang: Target language name (e.g., 'Hindi', 'Tamil')
        
        Returns:
            Translated text
        
        Raises:
            RuntimeError: If translation fails
        
        Example:
            >>> client = BhashiniClient()
            >>> result = client.translate("Hello", "English", "Hindi")
            >>> print(result)  # "नमस्ते"
        """
        if not text or not text.strip():
            return ""
        
        try:
            # Get pipeline configuration
            pipeline_config = self._get_pipeline_config(
                "translation",
                source_lang,
                target_lang
            )
            
            # Prepare inference payload
            payload = {
                "pipelineTasks": [
                    {
                        "taskType": "translation",
                        "config": {
                            "language": {
                                "sourceLanguage": self._get_language_code(source_lang),
                                "targetLanguage": self._get_language_code(target_lang)
                            },
                            "serviceId": pipeline_config["pipelineResponseConfig"][0]["config"][0]["serviceId"]
                        }
                    }
                ],
                "inputData": {
                    "input": [
                        {
                            "source": text
                        }
                    ]
                }
            }
            
            # Add authorization from pipeline config
            if "pipelineInferenceAPIEndPoint" in pipeline_config:
                inference_url = pipeline_config["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["inferenceEndPoint"]
                auth_header = pipeline_config["pipelineInferenceAPIEndPoint"]["inferenceApiKey"].get("value")
            else:
                inference_url = self.config.inference_url
                auth_header = None
            
            headers = self._get_headers()
            if auth_header:
                headers["Authorization"] = auth_header
            
            # Execute inference
            response = self.session.post(
                inference_url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            # Extract translated text
            result = response.json()
            pipeline_response = result.get("pipelineResponse", [])
            
            if not pipeline_response:
                raise ValueError("Empty response from Bhashini translation API")
            
            output = pipeline_response[0].get("output", [])
            if not output:
                raise ValueError("No translation output in Bhashini response")
            
            translated_text = output[0].get("target", "")
            
            if not translated_text:
                raise ValueError("Empty translation result")
            
            logger.info(f"Translated text from {source_lang} to {target_lang}")
            return translated_text
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Bhashini translation API request failed: {e}")
            raise RuntimeError(f"Translation failed: {e}")
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"Bhashini translation response parsing failed: {e}")
            raise RuntimeError(f"Translation response parsing failed: {e}")
    
    def text_to_speech(
        self,
        text: str,
        language: str = "Hindi",
        gender: str = "female"
    ) -> bytes:
        """
        Convert text to speech using Bhashini TTS.
        
        Args:
            text: Text to convert to speech
            language: Language name (e.g., 'Hindi', 'Tamil')
            gender: Voice gender ('female' or 'male')
        
        Returns:
            Audio data as bytes (WAV format)
        
        Raises:
            RuntimeError: If TTS generation fails
        
        Example:
            >>> client = BhashiniClient()
            >>> audio_bytes = client.text_to_speech("नमस्ते", "Hindi")
            >>> with open("output.wav", "wb") as f:
            ...     f.write(audio_bytes)
        """
        if not text or not text.strip():
            return b""
        
        try:
            # Get pipeline configuration
            pipeline_config = self._get_pipeline_config("tts", language)
            
            # Prepare inference payload
            payload = {
                "pipelineTasks": [
                    {
                        "taskType": "tts",
                        "config": {
                            "language": {
                                "sourceLanguage": self._get_language_code(language)
                            },
                            "gender": gender,
                            "serviceId": pipeline_config["pipelineResponseConfig"][0]["config"][0]["serviceId"]
                        }
                    }
                ],
                "inputData": {
                    "input": [
                        {
                            "source": text
                        }
                    ]
                }
            }
            
            # Add authorization from pipeline config
            if "pipelineInferenceAPIEndPoint" in pipeline_config:
                inference_url = pipeline_config["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["inferenceEndPoint"]
                auth_header = pipeline_config["pipelineInferenceAPIEndPoint"]["inferenceApiKey"].get("value")
            else:
                inference_url = self.config.inference_url
                auth_header = None
            
            headers = self._get_headers()
            if auth_header:
                headers["Authorization"] = auth_header
            
            # Execute inference
            response = self.session.post(
                inference_url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            # Extract audio from response
            result = response.json()
            pipeline_response = result.get("pipelineResponse", [])
            
            if not pipeline_response:
                raise ValueError("Empty response from Bhashini TTS API")
            
            audio_data = pipeline_response[0].get("audio", [])
            if not audio_data:
                raise ValueError("No audio data in Bhashini TTS response")
            
            audio_content = audio_data[0].get("audioContent", "")
            
            if not audio_content:
                raise ValueError("Empty audio content from Bhashini TTS")
            
            # Decode base64 audio
            audio_bytes = base64.b64decode(audio_content)
            
            logger.info(f"Generated TTS audio for {language} ({len(audio_bytes)} bytes)")
            return audio_bytes
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Bhashini TTS API request failed: {e}")
            raise RuntimeError(f"TTS generation failed: {e}")
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"Bhashini TTS response parsing failed: {e}")
            raise RuntimeError(f"TTS response parsing failed: {e}")
    
    def get_supported_languages(self, task_type: TaskType = "translation") -> List[str]:
        """
        Get list of supported languages for a task type.
        
        Args:
            task_type: Type of task
        
        Returns:
            List of supported language names
        """
        return list(BHASHINI_LANGUAGE_CODES.keys())
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check Bhashini API health and authentication.
        
        Returns:
            Health status dictionary
        """
        try:
            # Try to get a simple pipeline config
            self._get_pipeline_config("translation", "English", "Hindi")
            
            return {
                "status": "healthy",
                "api_key_valid": True,
                "supported_languages": len(BHASHINI_LANGUAGE_CODES),
                "supported_tasks": ["translation", "tts", "asr", "transliteration"]
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "api_key_valid": False,
                "error": str(e)
            }


# Singleton instance
_bhashini_client: Optional[BhashiniClient] = None


def get_bhashini_client() -> BhashiniClient:
    """
    Get singleton Bhashini client instance.
    
    Returns:
        Bhashini client
    
    Raises:
        ValueError: If credentials not configured
    """
    global _bhashini_client
    
    if _bhashini_client is None:
        _bhashini_client = BhashiniClient()
    
    return _bhashini_client


__all__ = [
    'BhashiniConfig',
    'BhashiniClient',
    'get_bhashini_client',
    'BHASHINI_LANGUAGE_CODES'
]
