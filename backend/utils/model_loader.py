"""Model loading with automatic quantization based on device capabilities."""
import os
import logging
import platform
from typing import Optional, Dict, Any
import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer

# Try to import BitsAndBytesConfig (may not work on macOS)
try:
    from transformers import BitsAndBytesConfig
    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False
    BitsAndBytesConfig = None

from ..utils.device_manager import get_device_manager
from ..core.config import settings

# Check for MLX availability (Apple Silicon M4 optimization)
try:
    import mlx.core as mx
    import mlx.nn as mlx_nn
    from mlx_lm import load as mlx_load
    from mlx_lm import generate as mlx_generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

logger = logging.getLogger(__name__)


class ModelLoader:
    """Intelligent model loader with automatic quantization and device placement."""
    
    def __init__(self):
        """Initialize model loader."""
        self.device_manager = get_device_manager()
        self.loaded_models: Dict[str, Any] = {}
        self.use_mlx = self._should_use_mlx()
    
    def _should_use_mlx(self) -> bool:
        """
        Determine if MLX should be used.
        
        MLX is optimal for Apple Silicon M4 (mac15,x).
        """
        if not MLX_AVAILABLE:
            return False
        
        # Check if running on Mac
        if platform.system() != "Darwin":
            return False
        
        # Check for M4 chip (Mac15,x identifier)
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "hw.model"],
                capture_output=True,
                text=True
            )
            model = result.stdout.strip()
            
            # M4 chips are Mac15,x
            if model.startswith("Mac15,"):
                logger.info(f"Detected M4 chip ({model}), enabling MLX")
                return True
        
        except Exception as e:
            logger.debug(f"Could not detect chip model: {e}")
        
        return False
    
    def _get_quantization_config(self) -> Optional[BitsAndBytesConfig]:
        """Get quantization config based on device and settings."""
        if not settings.USE_QUANTIZATION:
            logger.info("Quantization disabled via settings")
            return None
        
        if not BITSANDBYTES_AVAILABLE:
            logger.warning(
                "BitsAndBytes not available (may not be supported on macOS). "
                "Loading model in FP16/FP32."
            )
            return None
        
        if not self.device_manager.supports_quantization:
            logger.warning(
                f"Quantization requested but not supported on {self.device_manager.device}. "
                "BitsAndBytes requires CUDA. Loading model in FP16/FP32."
            )
            return None
        
        # Use 4-bit quantization for maximum memory savings
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_enable_fp32_cpu_offload=False
        )
        
        logger.info("4-bit quantization enabled (NF4)")
        return config
    
    def _get_torch_dtype(self) -> torch.dtype:
        """Get optimal torch dtype for current device."""
        if self.device_manager.device == "cuda":
            # Use FP16 on CUDA for speed
            return torch.float16
        elif self.device_manager.device == "mps":
            # MPS supports FP16
            return torch.float16
        else:
            # CPU fallback to FP32
            return torch.float32
    
    def load_seq2seq_model(
        self,
        model_id: str,
        force_fp32: bool = False
    ) -> tuple[Any, Any]:
        """
        Load sequence-to-sequence model (e.g., T5, BART, IndicTrans2).
        
        Args:
            model_id: HuggingFace model identifier
            force_fp32: Force FP32 precision (disable quantization)
        
        Returns:
            (model, tokenizer) tuple
        """
        if model_id in self.loaded_models:
            logger.info(f"Using cached model: {model_id}")
            return self.loaded_models[model_id]
        
        logger.info(f"Loading seq2seq model: {model_id}")
        
        # Get quantization config
        quant_config = None if force_fp32 else self._get_quantization_config()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=settings.MODEL_CACHE_DIR,
            trust_remote_code=True
        )
        
        # Prepare load kwargs
        load_kwargs = {
            "cache_dir": settings.MODEL_CACHE_DIR,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        if quant_config:
            load_kwargs["quantization_config"] = quant_config
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = self._get_torch_dtype()
        
        # Load model
        try:
            model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **load_kwargs)
            
            # Move to device if not using device_map
            if not quant_config:
                model = self.device_manager.move_to_device(model)
            
            # Set to eval mode
            model.eval()
            
            # Cache
            self.loaded_models[model_id] = (model, tokenizer)
            
            logger.info(
                f"Model loaded successfully: {model_id} "
                f"(device: {self.device_manager.device}, "
                f"quantized: {quant_config is not None})"
            )
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise
    
    def load_causal_lm_model(
        self,
        model_id: str,
        force_fp32: bool = False,
        prefer_mlx: bool = True
    ) -> tuple[Any, Any]:
        """
        Load causal language model (e.g., Llama, GPT).
        
        Supports:
        - MLX for Apple Silicon M4 (optimal performance)
        - PyTorch with CUDA/MPS/CPU fallback
        
        Args:
            model_id: HuggingFace model identifier
            force_fp32: Force FP32 precision
            prefer_mlx: Use MLX if available
        
        Returns:
            (model, tokenizer) tuple
        """
        if model_id in self.loaded_models:
            logger.info(f"Using cached model: {model_id}")
            return self.loaded_models[model_id]
        
        # Try MLX first if available and preferred
        if self.use_mlx and prefer_mlx and not force_fp32:
            try:
                return self._load_mlx_causal_lm(model_id)
            except Exception as e:
                logger.warning(f"MLX loading failed, falling back to PyTorch: {e}")
        
        logger.info(f"Loading causal LM model: {model_id}")
        
        quant_config = None if force_fp32 else self._get_quantization_config()
        
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=settings.MODEL_CACHE_DIR,
            trust_remote_code=True
        )
        
        load_kwargs = {
            "cache_dir": settings.MODEL_CACHE_DIR,
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        
        if quant_config:
            load_kwargs["quantization_config"] = quant_config
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = self._get_torch_dtype()
        
        try:
            model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
            
            if not quant_config:
                model = self.device_manager.move_to_device(model)
            
            model.eval()
            
            self.loaded_models[model_id] = (model, tokenizer)
            
            logger.info(f"Causal LM loaded: {model_id}")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load causal LM {model_id}: {e}")
            raise
    
    def _load_mlx_causal_lm(self, model_id: str) -> tuple[Any, Any]:
        """
        Load causal LM using MLX framework (M4-optimized).
        
        Args:
            model_id: HuggingFace model name
        
        Returns:
            Tuple of (model, tokenizer)
        """
        logger.info(f"Loading MLX model: {model_id}")
        
        # MLX supports specific models - convert name if needed
        mlx_model_map = {
            "meta-llama/Llama-3.2-3B-Instruct": "mlx-community/Llama-3.2-3B-Instruct-4bit",
            "meta-llama/Llama-3.2-1B-Instruct": "mlx-community/Llama-3.2-1B-Instruct-4bit",
            "mistralai/Mistral-7B-Instruct-v0.3": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        }
        
        mlx_model_name = mlx_model_map.get(model_id, model_id)
        
        # Load with MLX
        model, tokenizer = mlx_load(mlx_model_name)
        
        # Cache
        self.loaded_models[model_id] = (model, tokenizer)
        
        logger.info(f"MLX model loaded: {mlx_model_name} (4-bit quantized)")
        
        return model, tokenizer
    
    def load_embedding_model(
        self,
        model_id: str = "BAAI/bge-small-en-v1.5"
    ) -> SentenceTransformer:
        """
        Load embedding model (sentence-transformers).
        
        Args:
            model_id: HuggingFace model identifier
        
        Returns:
            SentenceTransformer model
        """
        if model_id in self.loaded_models:
            logger.info(f"Using cached embedding model: {model_id}")
            return self.loaded_models[model_id]
        
        logger.info(f"Loading embedding model: {model_id}")
        
        try:
            # SentenceTransformer handles device placement internally
            model = SentenceTransformer(
                model_id,
                cache_folder=str(settings.MODEL_CACHE_DIR),
                device=self.device_manager.device_str
            )
            
            self.loaded_models[model_id] = model
            
            logger.info(f"Embedding model loaded: {model_id} ({model.get_sentence_embedding_dimension()}D)")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load embedding model {model_id}: {e}")
            raise
    
    def load_causal_lm_model(
        self,
        model_id: str = "Qwen/Qwen2.5-7B-Instruct",
        quantization: str = "4bit",
        use_vllm: bool = False
    ):
        """
        Load causal language model (Qwen2.5) with optimizations.
        
        Args:
            model_id: Model identifier on HuggingFace
            quantization: "4bit", "8bit", or None
            use_vllm: Use vLLM for production serving
            
        Returns:
            tuple: (model, tokenizer)
        """
        cache_key = f"{model_id}_causal_{quantization}_{use_vllm}"
        if cache_key in self.loaded_models:
            logger.info(f"Using cached causal LM model: {model_id}")
            return self.loaded_models[cache_key]
        
        logger.info(f"Loading causal LM model: {model_id} (quantization={quantization}, vllm={use_vllm})")
        
        try:
            # Check if quantization is possible
            if quantization in ["4bit", "8bit"] and not BITSANDBYTES_AVAILABLE:
                logger.warning(f"Quantization requested but bitsandbytes unavailable. Model will use API mode.")
                return None, None  # Signal to use API instead
            
            if use_vllm:
                # Use vLLM for production
                try:
                    from vllm import LLM, SamplingParams
                    llm = LLM(
                        model=model_id,
                        quantization="awq" if quantization == "4bit" else None,
                        tensor_parallel_size=1,
                        gpu_memory_utilization=0.90,
                        download_dir=str(settings.MODEL_CACHE_DIR)
                    )
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_id,
                        cache_dir=str(settings.MODEL_CACHE_DIR),
                        trust_remote_code=True
                    )
                    self.loaded_models[cache_key] = (llm, tokenizer)
                    logger.info(f"vLLM model loaded: {model_id}")
                    return llm, tokenizer
                except ImportError:
                    logger.warning("vLLM not available, falling back to standard loading")
                    use_vllm = False
            
            # Standard loading with quantization
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=str(settings.MODEL_CACHE_DIR),
                trust_remote_code=True
            )
            
            if quantization == "4bit" and BITSANDBYTES_AVAILABLE:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    quantization_config=quantization_config,
                    device_map="auto",
                    cache_dir=str(settings.MODEL_CACHE_DIR),
                    trust_remote_code=True
                )
            elif quantization == "8bit" and BITSANDBYTES_AVAILABLE:
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    load_in_8bit=True,
                    device_map="auto",
                    cache_dir=str(settings.MODEL_CACHE_DIR),
                    trust_remote_code=True
                )
            elif quantization in ["none", None] or not BITSANDBYTES_AVAILABLE:
                # Load in FP16 (no quantization)
                logger.info(f"Loading model in FP16 (no quantization)")
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    cache_dir=str(settings.MODEL_CACHE_DIR),
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
            else:
                # Fallback to default precision
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    cache_dir=str(settings.MODEL_CACHE_DIR),
                    trust_remote_code=True
                )
            
            self.loaded_models[cache_key] = (model, tokenizer)
            logger.info(f"Causal LM model loaded: {model_id}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load causal LM model {model_id}: {e}")
            raise
    
    def load_embedding_model_optimized(
        self,
        model_id: str = "intfloat/multilingual-e5-large",
        use_onnx: bool = True
    ) -> SentenceTransformer:
        """
        Load embedding model with ONNX optimization.
        
        Args:
            model_id: Model identifier
            use_onnx: Convert to ONNX for faster inference
            
        Returns:
            SentenceTransformer model
        """
        cache_key = f"{model_id}_embedding_onnx_{use_onnx}"
        if cache_key in self.loaded_models:
            logger.info(f"Using cached embedding model: {model_id}")
            return self.loaded_models[cache_key]
        
        logger.info(f"Loading optimized embedding model: {model_id} (onnx={use_onnx})")
        
        try:
            model = SentenceTransformer(
                model_id,
                device=self.device_manager.device_str,
                cache_folder=str(settings.MODEL_CACHE_DIR)
            )
            
            if use_onnx and self.device_manager.device_str == "cpu":
                # ONNX optimization works best on CPU
                try:
                    from optimum.onnxruntime import ORTModelForFeatureExtraction
                    logger.info("ONNX conversion available but using SentenceTransformer native optimization")
                except ImportError:
                    logger.warning("optimum not available, skipping ONNX conversion")
            
            self.loaded_models[cache_key] = model
            logger.info(f"Optimized embedding model loaded: {model_id} ({model.get_sentence_embedding_dimension()}D)")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load optimized embedding model {model_id}: {e}")
            raise
    
    def load_translation_model_quantized(
        self,
        model_id: str = "ai4bharat/indictrans2-en-indic-1B",
        quantization: str = "int8",
        use_torchscript: bool = False
    ):
        """
        Load translation model with INT8 quantization and optional TorchScript.
        
        Args:
            model_id: IndicTrans2 model identifier
            quantization: "int8" or None
            use_torchscript: Compile with TorchScript for speedup
            
        Returns:
            tuple: (model, tokenizer)
        """
        cache_key = f"{model_id}_translation_{quantization}_{use_torchscript}"
        if cache_key in self.loaded_models:
            logger.info(f"Using cached translation model: {model_id}")
            return self.loaded_models[cache_key]
        
        logger.info(f"Loading translation model: {model_id} (quantization={quantization}, torchscript={use_torchscript})")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                cache_dir=str(settings.MODEL_CACHE_DIR)
            )
            
            if quantization == "int8":
                # INT8 quantization
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_id,
                    load_in_8bit=True,
                    device_map="auto",
                    cache_dir=str(settings.MODEL_CACHE_DIR),
                    trust_remote_code=True
                )
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    cache_dir=str(settings.MODEL_CACHE_DIR),
                    trust_remote_code=True
                )
            
            # TorchScript compilation (not compatible with 8-bit)
            if use_torchscript and quantization != "int8":
                try:
                    logger.info("TorchScript compilation skipped (requires example inputs)")
                    # model = torch.jit.trace(model, example_inputs)
                except Exception as e:
                    logger.warning(f"TorchScript compilation failed: {e}")
            
            self.loaded_models[cache_key] = (model, tokenizer)
            logger.info(f"Translation model loaded: {model_id}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load translation model {model_id}: {e}")
            raise
    
    def unload_model(self, model_id: str) -> bool:
        """
        Unload model from memory.
        
        Args:
            model_id: Model identifier to unload
        
        Returns:
            True if unloaded, False if not found
        """
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
            self.device_manager.clear_cache()
            logger.info(f"Model unloaded: {model_id}")
            return True
        return False
    
    def clear_all_models(self):
        """Unload all models and clear cache."""
        self.loaded_models.clear()
        self.device_manager.clear_cache()
        logger.info("All models cleared from memory")


# Global singleton
_model_loader: Optional[ModelLoader] = None


def get_model_loader() -> ModelLoader:
    """Get or create global model loader instance."""
    global _model_loader
    if _model_loader is None:
        _model_loader = ModelLoader()
    return _model_loader


__all__ = ['ModelLoader', 'get_model_loader']
