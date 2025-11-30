"""
vLLM Production Serving - High-throughput LLM Inference.

Optimal 2025 Model Stack:
- Llama-3.2-3B-Instruct for simplification
- AWQ/GPTQ quantization support
- OpenAI-compatible API
- Continuous batching for high throughput
"""
import os
import logging
from typing import Optional, List, Dict, Any, AsyncGenerator
from dataclasses import dataclass
import asyncio
import httpx

from ..core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    stop: Optional[List[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0


class VLLMClient:
    """
    vLLM OpenAI-compatible API client.
    
    For production serving of Llama-3.2-3B-Instruct and other models.
    """
    
    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        model: str = None,
        timeout: float = 120.0
    ):
        """
        Initialize vLLM client.
        
        Args:
            base_url: vLLM server URL (default from config)
            api_key: API key for authentication
            model: Model name to use
            timeout: Request timeout in seconds
        """
        self.base_url = base_url or settings.vllm_base_url
        self.api_key = api_key or settings.VLLM_API_KEY
        self.model = model or settings.SIMPLIFICATION_MODEL_ID
        self.timeout = timeout
        
        self._client = httpx.AsyncClient(timeout=timeout)
        
        logger.info(f"VLLMClient initialized: {self.base_url}, model: {self.model}")
    
    @property
    def headers(self) -> Dict[str, str]:
        """Get request headers."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    async def generate(
        self,
        prompt: str,
        config: GenerationConfig = None,
        system_prompt: str = None
    ) -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: User prompt
            config: Generation configuration
            system_prompt: System message
        
        Returns:
            Generated text
        """
        config = config or GenerationConfig()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "presence_penalty": config.presence_penalty,
            "frequency_penalty": config.frequency_penalty,
        }
        
        if config.stop:
            payload["stop"] = config.stop
        
        try:
            response = await self._client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self.headers
            )
            response.raise_for_status()
            
            data = response.json()
            return data["choices"][0]["message"]["content"]
            
        except httpx.HTTPStatusError as e:
            logger.error(f"vLLM HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"vLLM generation failed: {e}")
            raise
    
    async def generate_stream(
        self,
        prompt: str,
        config: GenerationConfig = None,
        system_prompt: str = None
    ) -> AsyncGenerator[str, None]:
        """
        Stream generated text.
        
        Args:
            prompt: User prompt
            config: Generation configuration
            system_prompt: System message
        
        Yields:
            Text chunks as they are generated
        """
        config = config or GenerationConfig()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "stream": True,
        }
        
        try:
            async with self._client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self.headers
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        
                        import json
                        try:
                            chunk = json.loads(data)
                            content = chunk["choices"][0].get("delta", {}).get("content", "")
                            if content:
                                yield content
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            logger.error(f"vLLM streaming failed: {e}")
            raise
    
    async def embeddings(
        self,
        texts: List[str],
        model: str = None
    ) -> List[List[float]]:
        """
        Generate embeddings (if vLLM supports embedding model).
        
        Args:
            texts: List of texts to embed
            model: Embedding model to use
        
        Returns:
            List of embedding vectors
        """
        model = model or settings.EMBEDDING_MODEL_ID
        
        payload = {
            "model": model,
            "input": texts,
        }
        
        try:
            response = await self._client.post(
                f"{self.base_url}/embeddings",
                json=payload,
                headers=self.headers
            )
            response.raise_for_status()
            
            data = response.json()
            return [item["embedding"] for item in data["data"]]
            
        except Exception as e:
            logger.error(f"vLLM embeddings failed: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if vLLM server is healthy."""
        try:
            response = await self._client.get(
                f"{self.base_url.rsplit('/v1', 1)[0]}/health"
            )
            return response.status_code == 200
        except Exception:
            return False
    
    async def list_models(self) -> List[str]:
        """List available models on vLLM server."""
        try:
            response = await self._client.get(
                f"{self.base_url}/models",
                headers=self.headers
            )
            response.raise_for_status()
            
            data = response.json()
            return [model["id"] for model in data.get("data", [])]
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()


class VLLMServer:
    """
    vLLM server launcher for local deployment.
    
    Wraps vLLM CLI commands for easy startup.
    """
    
    @staticmethod
    def get_launch_command(
        model: str = None,
        host: str = None,
        port: int = None,
        tensor_parallel: int = None,
        pipeline_parallel: int = None,
        gpu_memory_utilization: float = None,
        quantization: str = None,
        max_model_len: int = None,
        swap_space_gb: int = None,
        cpu_offload_gb: float = None,
        enable_prefix_caching: bool = None,
        block_size: int = None,
    ) -> str:
        """
        Get vLLM server launch command.
        
        Implements:
        - Principle C: Generous swap space
        - Principle D: Tensor parallelism for multi-GPU
        - Principle H: KV cache configuration
        
        Args:
            model: Model to serve
            host: Host to bind to
            port: Port to bind to
            tensor_parallel: Number of GPUs for tensor parallelism (Principle D)
            pipeline_parallel: Pipeline parallel size (Principle D)
            gpu_memory_utilization: GPU memory fraction to use
            quantization: Quantization method (awq, gptq, etc.)
            max_model_len: Maximum model context length
            swap_space_gb: CPU swap space in GB (Principle C)
            cpu_offload_gb: CPU offload memory in GB (Principle C)
            enable_prefix_caching: Enable prefix caching (Principle H)
            block_size: KV cache block size (Principle H)
        
        Returns:
            Command string to launch vLLM
        """
        model = model or settings.SIMPLIFICATION_MODEL_ID
        host = host or settings.VLLM_HOST
        port = port or settings.VLLM_PORT
        tensor_parallel = tensor_parallel or settings.VLLM_TENSOR_PARALLEL_SIZE
        pipeline_parallel = pipeline_parallel or settings.VLLM_PIPELINE_PARALLEL_SIZE
        gpu_memory_utilization = gpu_memory_utilization or settings.VLLM_GPU_MEMORY_UTILIZATION
        quantization = quantization or settings.VLLM_QUANTIZATION
        max_model_len = max_model_len or settings.VLLM_MAX_MODEL_LEN
        swap_space_gb = swap_space_gb if swap_space_gb is not None else settings.VLLM_SWAP_SPACE_GB
        cpu_offload_gb = cpu_offload_gb if cpu_offload_gb is not None else settings.VLLM_CPU_OFFLOAD_GB
        enable_prefix_caching = enable_prefix_caching if enable_prefix_caching is not None else settings.VLLM_ENABLE_PREFIX_CACHING
        block_size = block_size or settings.VLLM_BLOCK_SIZE
        
        cmd_parts = [
            "vllm serve",
            f'"{model}"',
            f"--host {host}",
            f"--port {port}",
            f"--tensor-parallel-size {tensor_parallel}",
            f"--gpu-memory-utilization {gpu_memory_utilization}",
            f"--max-model-len {max_model_len}",
            "--trust-remote-code",
        ]
        
        # Principle C: Generous swap space for KV cache overflow
        if swap_space_gb > 0:
            cmd_parts.append(f"--swap-space {swap_space_gb}")
        
        # Principle C: CPU offload (optional, for memory-constrained systems)
        if cpu_offload_gb > 0:
            cmd_parts.append(f"--cpu-offload-gb {cpu_offload_gb}")
        
        # Principle D: Pipeline parallelism for multi-node
        if pipeline_parallel > 1:
            cmd_parts.append(f"--pipeline-parallel-size {pipeline_parallel}")
        
        # Principle H: KV cache block size
        if block_size != 16:  # Default is 16
            cmd_parts.append(f"--block-size {block_size}")
        
        # Principle H: Prefix caching for KV cache reuse
        if enable_prefix_caching:
            cmd_parts.append("--enable-prefix-caching")
        
        # Enforce eager mode if configured (disables CUDA graph)
        if settings.VLLM_ENFORCE_EAGER:
            cmd_parts.append("--enforce-eager")
        
        if quantization and quantization != "none":
            cmd_parts.append(f"--quantization {quantization}")
        
        return " ".join(cmd_parts)
    
    @staticmethod
    def get_docker_command(
        model: str = None,
        port: int = None,
        quantization: str = None
    ) -> str:
        """
        Get Docker command to run vLLM.
        
        Returns:
            Docker run command string
        """
        model = model or settings.SIMPLIFICATION_MODEL_ID
        port = port or settings.VLLM_PORT
        quantization = quantization or settings.VLLM_QUANTIZATION
        
        quant_arg = f"--quantization {quantization}" if quantization != "none" else ""
        
        return f"""docker run --gpus all \\
    -v ~/.cache/huggingface:/root/.cache/huggingface \\
    -p {port}:8000 \\
    --ipc=host \\
    vllm/vllm-openai:latest \\
    --model {model} \\
    --trust-remote-code \\
    {quant_arg}"""


# Celery task integration for async processing
def create_celery_tasks():
    """Create Celery tasks for vLLM inference."""
    try:
        from celery import shared_task
        
        @shared_task(name="vllm.generate")
        def generate_task(
            prompt: str,
            max_tokens: int = 2048,
            temperature: float = 0.7,
            model: str = None
        ) -> str:
            """Celery task for text generation."""
            import asyncio
            
            async def _generate():
                client = VLLMClient(model=model)
                config = GenerationConfig(
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                result = await client.generate(prompt, config)
                await client.close()
                return result
            
            return asyncio.run(_generate())
        
        @shared_task(name="vllm.batch_generate")
        def batch_generate_task(
            prompts: List[str],
            max_tokens: int = 2048,
            temperature: float = 0.7
        ) -> List[str]:
            """Celery task for batch generation."""
            import asyncio
            
            async def _batch_generate():
                client = VLLMClient()
                config = GenerationConfig(
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                results = []
                for prompt in prompts:
                    result = await client.generate(prompt, config)
                    results.append(result)
                
                await client.close()
                return results
            
            return asyncio.run(_batch_generate())
        
        logger.info("Celery tasks for vLLM created")
        return generate_task, batch_generate_task
        
    except ImportError:
        logger.warning("Celery not available, skipping task creation")
        return None, None


# Singleton client instance
_vllm_client: Optional[VLLMClient] = None


def get_vllm_client() -> VLLMClient:
    """Get or create vLLM client singleton."""
    global _vllm_client
    if _vllm_client is None:
        _vllm_client = VLLMClient()
    return _vllm_client


# Export
__all__ = [
    'VLLMClient',
    'VLLMServer',
    'GenerationConfig',
    'get_vllm_client',
    'create_celery_tasks'
]
