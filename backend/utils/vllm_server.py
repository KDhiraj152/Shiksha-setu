"""vLLM Production Inference Server for GPU Deployment."""
import os
import logging
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from dataclasses import dataclass

# Use TYPE_CHECKING to avoid runtime import errors when vLLM not installed
if TYPE_CHECKING:
    from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)


@dataclass
class VLLMConfig:
    """Configuration for vLLM inference server."""
    model_name: str
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.90
    max_model_len: int = 2048
    dtype: str = "float16"
    enforce_eager: bool = False
    enable_prefix_caching: bool = True


class VLLMInferenceServer:
    """
    Production-grade vLLM inference server for scalable GPU deployment.
    
    Use this for production environments with GPU clusters.
    Local development should use ModelManager with MPS/CPU.
    """
    
    def __init__(self, config: VLLMConfig):
        """Initialize vLLM server."""
        self.config = config
        self.llm: Optional['LLM'] = None
        self.sampling_params: Optional['SamplingParams'] = None
        
        logger.info(f"Initializing vLLM server for {config.model_name}")
    
    def start(self):
        """Start vLLM inference server."""
        try:
            # Import at runtime to avoid hard dependency
            from vllm import LLM, SamplingParams
            
            self.llm = LLM(
                model=self.config.model_name,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_model_len=self.config.max_model_len,
                dtype=self.config.dtype,
                enforce_eager=self.config.enforce_eager,
                enable_prefix_caching=self.config.enable_prefix_caching,
                trust_remote_code=True
            )
            
            self.sampling_params = SamplingParams(
                temperature=0.7,
                top_p=0.9,
                max_tokens=512,
                stop=["\n\n"]
            )
            
            logger.info("vLLM server started successfully")
            
        except ImportError:
            logger.error("vLLM not installed. Install with: pip install vllm==0.8.5")
            raise RuntimeError(
                "vLLM is not installed. This is expected for local M4 development. "
                "For production GPU deployment, install: pip install vllm==0.8.5"
            )
        except Exception as e:
            logger.error(f"Failed to start vLLM server: {e}")
            raise
    
    def generate(self, prompts: List[str]) -> List[str]:
        """Generate text using vLLM."""
        if self.llm is None:
            raise RuntimeError("vLLM server not started. Call start() first.")
        
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]
    
    def generate_single(self, prompt: str) -> str:
        """Generate single text output."""
        results = self.generate([prompt])
        return results[0] if results else ""


class VLLMServerManager:
    """Manager for multiple vLLM inference servers."""
    
    def __init__(self):
        """Initialize server manager."""
        self.servers: Dict[str, VLLMInferenceServer] = {}
        logger.info("vLLM Server Manager initialized")
    
    def add_server(self, name: str, config: VLLMConfig) -> VLLMInferenceServer:
        """Add and start a new vLLM server."""
        server = VLLMInferenceServer(config)
        server.start()
        self.servers[name] = server
        logger.info(f"Added vLLM server: {name}")
        return server
    
    def get_server(self, name: str) -> Optional[VLLMInferenceServer]:
        """Get a vLLM server by name."""
        return self.servers.get(name)
    
    def remove_server(self, name: str) -> bool:
        """Remove a vLLM server."""
        if name in self.servers:
            del self.servers[name]
            logger.info(f"Removed vLLM server: {name}")
            return True
        return False


# Production configuration examples
FLAN_T5_VLLM_CONFIG = VLLMConfig(
    model_name="google/flan-t5-xl",  # Larger model for production
    tensor_parallel_size=2,  # 2 GPUs
    gpu_memory_utilization=0.90,
    max_model_len=2048
)

INDICTRANS2_VLLM_CONFIG = VLLMConfig(
    model_name="ai4bharat/indictrans2-en-indic-1B",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.85,
    max_model_len=1024
)


def create_production_servers() -> VLLMServerManager:
    """
    Create production vLLM servers for GPU deployment.
    
    This should only be used in production environments with GPU clusters.
    """
    manager = VLLMServerManager()
    
    try:
        # Add simplification server
        manager.add_server("simplification", FLAN_T5_VLLM_CONFIG)
        
        # Add translation server
        manager.add_server("translation", INDICTRANS2_VLLM_CONFIG)
        
        logger.info("Production vLLM servers initialized")
        
    except Exception as e:
        logger.error(f"Failed to create production servers: {e}")
        logger.info("Falling back to standard model loading")
    
    return manager


# Singleton instance for production
_vllm_manager: Optional[VLLMServerManager] = None


def get_vllm_manager() -> VLLMServerManager:
    """Get singleton vLLM manager instance."""
    global _vllm_manager
    
    if _vllm_manager is None:
        _vllm_manager = VLLMServerManager()
    
    return _vllm_manager


__all__ = [
    'VLLMConfig',
    'VLLMInferenceServer',
    'VLLMServerManager',
    'get_vllm_manager',
    'create_production_servers'
]
