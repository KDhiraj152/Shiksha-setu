"""
vLLM + Ray Serve Production Inference

High-throughput inference server with autoscaling, model parallelism,
and continuous batching.
"""
import os
from typing import Optional, List, Dict, Any
import asyncio

try:
    import ray
    from ray import serve
    from vllm import AsyncLLMEngine, SamplingParams, AsyncEngineArgs
    from vllm.engine.arg_utils import AsyncEngineArgs
except ImportError:
    ray = None
    serve = None
    AsyncLLMEngine = None
    SamplingParams = None

from backend.utils.logging import get_logger
from backend.core.config import settings

logger = get_logger(__name__)


class vLLMConfig:
    """vLLM configuration."""
    
    # Model settings
    MODEL_NAME = os.getenv("VLLM_MODEL", "meta-llama/Llama-3.2-3B-Instruct")
    TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
    
    # Performance settings
    MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "2048"))
    GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.9"))
    MAX_NUM_SEQS = int(os.getenv("MAX_NUM_SEQS", "256"))
    
    # Ray Serve settings
    NUM_REPLICAS = int(os.getenv("VLLM_REPLICAS", "2"))
    MAX_CONCURRENT_QUERIES = int(os.getenv("MAX_CONCURRENT_QUERIES", "100"))


if ray and serve and AsyncLLMEngine:
    @serve.deployment(
        num_replicas=vLLMConfig.NUM_REPLICAS,
        ray_actor_options={
            "num_gpus": vLLMConfig.TENSOR_PARALLEL_SIZE,
            "num_cpus": 4
        },
        autoscaling_config={
            "min_replicas": 1,
            "max_replicas": 8,
            "target_num_ongoing_requests_per_replica": 10
        },
        max_concurrent_queries=vLLMConfig.MAX_CONCURRENT_QUERIES
    )
    class VLLMDeployment:
        """
        Ray Serve deployment for vLLM inference.
        
        Features:
        - Continuous batching for high throughput
        - Model parallelism for large models
        - Autoscaling based on load
        - Async request handling
        """
        
        def __init__(self):
            """Initialize vLLM engine."""
            logger.info(f"Initializing vLLM with {vLLMConfig.MODEL_NAME}")
            
            engine_args = AsyncEngineArgs(
                model=vLLMConfig.MODEL_NAME,
                tensor_parallel_size=vLLMConfig.TENSOR_PARALLEL_SIZE,
                max_model_len=vLLMConfig.MAX_MODEL_LEN,
                gpu_memory_utilization=vLLMConfig.GPU_MEMORY_UTILIZATION,
                max_num_seqs=vLLMConfig.MAX_NUM_SEQS,
                disable_log_stats=False,
                trust_remote_code=True
            )
            
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            logger.info("vLLM engine initialized successfully")
        
        async def generate(
            self,
            prompt: str,
            max_tokens: int = 512,
            temperature: float = 0.7,
            top_p: float = 0.9,
            stop: Optional[List[str]] = None
        ) -> str:
            """
            Generate text from prompt.
            
            Args:
                prompt: Input prompt
                max_tokens: Max tokens to generate
                temperature: Sampling temperature
                top_p: Nucleus sampling threshold
                stop: Stop sequences
            
            Returns:
                Generated text
            """
            sampling_params = SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop or []
            )
            
            # Generate
            request_id = f"req_{id(prompt)}"
            results_generator = self.engine.generate(
                prompt, sampling_params, request_id
            )
            
            # Collect results
            final_output = ""
            async for request_output in results_generator:
                if request_output.finished:
                    final_output = request_output.outputs[0].text
            
            return final_output
        
        async def batch_generate(
            self,
            prompts: List[str],
            max_tokens: int = 512,
            temperature: float = 0.7,
            top_p: float = 0.9
        ) -> List[str]:
            """
            Batch generation for multiple prompts.
            
            Args:
                prompts: List of input prompts
                max_tokens: Max tokens per prompt
                temperature: Sampling temperature
                top_p: Nucleus sampling threshold
            
            Returns:
                List of generated texts
            """
            tasks = [
                self.generate(prompt, max_tokens, temperature, top_p)
                for prompt in prompts
            ]
            
            return await asyncio.gather(*tasks)


class VLLMClient:
    """
    Client for vLLM Ray Serve deployment.
    
    Usage:
        client = VLLMClient()
        await client.connect()
        result = await client.generate("Translate to Hindi: Hello")
    """
    
    def __init__(self, ray_address: Optional[str] = None):
        """
        Initialize client.
        
        Args:
            ray_address: Ray cluster address (default: auto)
        """
        self.ray_address = ray_address or "auto"
        self.handle = None
    
    async def connect(self):
        """Connect to Ray Serve deployment."""
        if not ray:
            raise RuntimeError("Ray not installed. Install with: pip install ray[serve] vllm")
        
        try:
            ray.init(address=self.ray_address, ignore_reinit_error=True)
            logger.info(f"Connected to Ray cluster: {self.ray_address}")
            
            # Get deployment handle
            self.handle = serve.get_deployment("VLLMDeployment").get_handle()
            logger.info("Connected to vLLM deployment")
        
        except Exception as e:
            logger.error(f"Failed to connect to vLLM: {e}")
            raise
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        Generate text.
        
        Args:
            prompt: Input prompt
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            stop: Stop sequences
        
        Returns:
            Generated text
        """
        if not self.handle:
            await self.connect()
        
        result = await self.handle.generate.remote(
            prompt, max_tokens, temperature, top_p, stop
        )
        
        return await result
    
    async def batch_generate(
        self,
        prompts: List[str],
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> List[str]:
        """
        Batch generate.
        
        Args:
            prompts: List of input prompts
            max_tokens: Max tokens per prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
        
        Returns:
            List of generated texts
        """
        if not self.handle:
            await self.connect()
        
        result = await self.handle.batch_generate.remote(
            prompts, max_tokens, temperature, top_p
        )
        
        return await result
    
    def disconnect(self):
        """Disconnect from Ray."""
        if ray:
            ray.shutdown()
            logger.info("Disconnected from Ray")


async def deploy_vllm():
    """
    Deploy vLLM to Ray Serve.
    
    Usage:
        python -c "import asyncio; from backend.services.vllm_serve import deploy_vllm; asyncio.run(deploy_vllm())"
    """
    if not ray or not serve:
        raise RuntimeError("Ray Serve not installed")
    
    # Initialize Ray
    ray.init(address="auto", ignore_reinit_error=True)
    
    # Start Serve
    serve.start(detached=True)
    
    # Deploy
    VLLMDeployment.deploy()
    
    logger.info("vLLM deployment started")
    logger.info(f"Model: {vLLMConfig.MODEL_NAME}")
    logger.info(f"Replicas: {vLLMConfig.NUM_REPLICAS}")
    logger.info(f"Tensor parallel size: {vLLMConfig.TENSOR_PARALLEL_SIZE}")


if __name__ == "__main__":
    # Deploy vLLM
    asyncio.run(deploy_vllm())
