"""Enhanced ML model manager with Apple Silicon MPS + Production GPU support."""
import os
import logging
from typing import Optional, Dict, Any, List, Literal
import torch
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelManager:
    """Centralized ML model management with device-aware optimization."""
    
    def __init__(self, cache_dir: str = "data/models"):
        """Initialize model manager with Apple Silicon MPS detection."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Device detection with Apple Silicon MPS support
        if torch.cuda.is_available():
            self.device = "cuda"
            self.device_type = "gpu"
            logger.info(f"CUDA device detected: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.device_type = "mps"
            logger.info("Apple Silicon MPS device detected - M4 optimization enabled")
        else:
            self.device = "cpu"
            self.device_type = "cpu"
            logger.info("Running on CPU - consider using Bhashini API for production")
        
        logger.info(f"Model Manager initialized on device: {self.device}")
        
        self._models = {}
        self.use_quantization = os.getenv("USE_QUANTIZATION", "true").lower() == "true"
        self.use_flash_attention = os.getenv("USE_FLASH_ATTENTION", "false").lower() == "true"
        
        # MLX support for Apple Silicon (optional)
        self.mlx_available = False
        if self.device == "mps":
            try:
                import mlx.core as mx
                self.mlx_available = True
                logger.info("MLX framework available for Apple Silicon acceleration")
            except ImportError:
                logger.warning("MLX not installed - install with: pip install mlx mlx-lm")
    
    def _get_torch_dtype(self) -> torch.dtype:
        """Get optimal torch dtype for current device."""
        if self.device == "cuda":
            return torch.float16  # GPU uses half precision
        elif self.device == "mps":
            return torch.float32  # MPS currently works best with float32
        else:
            return torch.float32  # CPU uses full precision
    
    def _get_quantization_config(self) -> Optional[Any]:
        """Get quantization config for current device."""
        if not self.use_quantization:
            return None
        
        try:
            from transformers import BitsAndBytesConfig
            
            if self.device == "cuda":
                # 8-bit quantization for GPU
                return BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False
                )
            elif self.device == "mps":
                # MPS doesn't support bitsandbytes yet
                logger.info("Quantization not available on MPS - using float32")
                return None
            else:
                # CPU 8-bit quantization
                return BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )
        except ImportError:
            logger.warning("bitsandbytes not installed - running without quantization")
            return None
    
    def load_flan_t5(self, model_size: str = "base") -> Any:
        """Load Flan-T5 for text simplification with device optimization."""
        key = f"flan-t5-{model_size}"
        
        if key in self._models:
            return self._models[key]
        
        logger.info(f"Loading Flan-T5 {model_size} optimized for {self.device}...")
        
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        model_name = f"google/flan-t5-{model_size}"
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=str(self.cache_dir),
            local_files_only=False
        )
        
        # Device-specific loading
        quantization_config = self._get_quantization_config()
        
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=str(self.cache_dir),
            torch_dtype=self._get_torch_dtype(),
            quantization_config=quantization_config,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        
        if self.device != "cuda":
            model = model.to(self.device)
        
        self._models[key] = {"model": model, "tokenizer": tokenizer}
        logger.info(f"Flan-T5 {model_size} loaded on {self.device}")
        
        return self._models[key]
    
    def load_indictrans2(self) -> Any:
        """Load IndicTrans2 for Indian language translation with device optimization."""
        key = "indictrans2"
        
        if key in self._models:
            return self._models[key]
        
        logger.info(f"Loading IndicTrans2 optimized for {self.device}...")
        
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        model_name = "ai4bharat/indictrans2-en-indic-1B"
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=str(self.cache_dir),
            local_files_only=False
        )
        
        quantization_config = self._get_quantization_config()
        
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            cache_dir=str(self.cache_dir),
            torch_dtype=self._get_torch_dtype(),
            quantization_config=quantization_config,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True
        )
        
        if self.device != "cuda":
            model = model.to(self.device)
        
        self._models[key] = {"model": model, "tokenizer": tokenizer}
        logger.info(f"IndicTrans2 loaded on {self.device}")
        
        return self._models[key]
    
    def load_bert_multilingual(self) -> Any:
        """Load BERT for semantic validation."""
        key = "bert-multilingual"
        
        if key in self._models:
            return self._models[key]
        
        logger.info("Loading BERT multilingual...")
        
        from transformers import AutoTokenizer, AutoModel
        
        model_name = "bert-base-multilingual-cased"
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(self.cache_dir))
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=str(self.cache_dir)
        ).to(self.device)
        
        self._models[key] = {"model": model, "tokenizer": tokenizer}
        logger.info("BERT multilingual loaded")
        
        return self._models[key]
    
    def load_sentence_bert(self) -> Any:
        """Load Sentence-BERT for semantic similarity."""
        key = "sentence-bert"
        
        if key in self._models:
            return self._models[key]
        
        logger.info("Loading Sentence-BERT...")
        
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer(
            'paraphrase-multilingual-mpnet-base-v2',
            cache_folder=str(self.cache_dir),
            device=self.device
        )
        
        self._models[key] = model
        logger.info("Sentence-BERT loaded")
        
        return self._models[key]
    
    def load_vits_tts(self, language: str = "hin") -> Any:
        """Load VITS TTS model."""
        key = f"vits-{language}"
        
        if key in self._models:
            return self._models[key]
        
        logger.info(f"Loading VITS TTS for {language}...")
        
        from transformers import VitsModel, AutoTokenizer
        
        model_name = f"facebook/mms-tts-{language}"
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=str(self.cache_dir))
        model = VitsModel.from_pretrained(
            model_name,
            cache_dir=str(self.cache_dir)
        ).to(self.device)
        
        self._models[key] = {"model": model, "tokenizer": tokenizer}
        logger.info(f"VITS TTS {language} loaded")
        
        return self._models[key]
    
    def load_coqui_tts(self) -> Any:
        """Load Coqui TTS for high-quality speech synthesis."""
        key = "coqui-tts"
        
        if key in self._models:
            return self._models[key]
        
        logger.info("Loading Coqui TTS...")
        
        from TTS.api import TTS
        
        # Use multilingual model
        tts = TTS(
            model_name="tts_models/multilingual/multi-dataset/xtts_v2",
            progress_bar=False
        ).to(self.device)
        
        self._models[key] = tts
        logger.info("Coqui TTS loaded")
        
        return self._models[key]
    
    def load_whisper(self, model_size: str = "base") -> Any:
        """Load Whisper ASR model."""
        key = f"whisper-{model_size}"
        
        if key in self._models:
            return self._models[key]
        
        logger.info(f"Loading Whisper {model_size}...")
        
        import whisper
        
        model = whisper.load_model(model_size, device=self.device)
        
        self._models[key] = model
        logger.info(f"Whisper {model_size} loaded")
        
        return self._models[key]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models and device status."""
        info = {
            "loaded_models": list(self._models.keys()),
            "device": self.device,
            "device_type": self.device_type,
            "cuda_available": torch.cuda.is_available(),
            "mps_available": torch.backends.mps.is_available(),
            "cache_dir": str(self.cache_dir),
            "quantization_enabled": self.use_quantization,
            "mlx_available": self.mlx_available
        }
        
        # Add device-specific info
        if self.device == "cuda":
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_memory_allocated"] = f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB"
            info["cuda_memory_reserved"] = f"{torch.cuda.memory_reserved() / 1024**3:.2f} GB"
        elif self.device == "mps":
            info["apple_silicon"] = True
            info["mps_optimization"] = "enabled"
        
        return info
    
    def unload_model(self, key: str) -> bool:
        """Unload a specific model to free memory."""
        if key in self._models:
            del self._models[key]
            
            # Clear device cache
            if self.device == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            elif self.device == "mps":
                torch.mps.empty_cache()
            
            logger.info(f"Model {key} unloaded")
            return True
        return False
    
    def unload_all(self):
        """Unload all models and clear cache."""
        self._models.clear()
        
        if self.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.empty_cache()
        
        logger.info("All models unloaded")


# Singleton instance
_model_manager = None


def get_model_manager() -> ModelManager:
    """Get singleton model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager


# Export
__all__ = ['ModelManager', 'get_model_manager']
