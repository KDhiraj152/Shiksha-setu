"""
Model Serving Client

Unified client interface for all model serving backends.
Handles request routing, load balancing, retries, and fallbacks.

Issue: CODE-REVIEW-GPT #3 (CRITICAL)
"""

import httpx
import asyncio
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import logging
from abc import ABC, abstractmethod

from .model_serving import (
    ModelConfig,
    ModelBackend,
    DeploymentConfig,
    DEPLOYMENT_CONFIG,
    get_model_config,
)

logger = logging.getLogger(__name__)


class ModelServingError(Exception):
    """Base exception for model serving errors."""
    pass


class ModelUnavailableError(ModelServingError):
    """Raised when model endpoint is unavailable."""
    pass


class InferenceTimeoutError(ModelServingError):
    """Raised when inference times out."""
    pass


# =============================================================================
# BASE MODEL CLIENT
# =============================================================================

class BaseModelClient(ABC):
    """Abstract base class for model clients."""
    
    def __init__(self, config: ModelConfig, deployment_config: DeploymentConfig):
        self.config = config
        self.deployment_config = deployment_config
        self.endpoint = self._get_endpoint()
        self.client = httpx.AsyncClient(timeout=deployment_config.request_timeout)
    
    def _get_endpoint(self) -> str:
        """Get endpoint URL based on backend."""
        if self.config.backend == ModelBackend.VLLM:
            return self.deployment_config.vllm_endpoint
        elif self.config.backend == ModelBackend.TRITON:
            return self.deployment_config.triton_endpoint
        elif self.config.backend == ModelBackend.OLLAMA:
            return self.deployment_config.ollama_endpoint
        else:
            raise ValueError(f"Unsupported backend: {self.config.backend}")
    
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        pass
    
    async def health_check(self) -> bool:
        """Check if model endpoint is healthy."""
        try:
            response = await self.client.get(
                f"{self.endpoint}{self.config.health_check_path}",
                timeout=5.0
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed for {self.config.model_id}: {e}")
            return False
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


# =============================================================================
# vLLM CLIENT
# =============================================================================

class VLLMClient(BaseModelClient):
    """Client for vLLM serving backend."""
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ) -> str:
        """Generate text using vLLM."""
        request_data = {
            "model": self.config.model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            **kwargs
        }
        
        for attempt in range(self.deployment_config.max_retries + 1):
            try:
                response = await self.client.post(
                    f"{self.endpoint}/v1/completions",
                    json=request_data
                )
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["text"]
            
            except httpx.TimeoutException:
                if attempt == self.deployment_config.max_retries:
                    raise InferenceTimeoutError(f"Request timed out after {attempt + 1} attempts")
                await asyncio.sleep(self.deployment_config.retry_delay * (attempt + 1))
            
            except httpx.HTTPStatusError as e:
                if attempt == self.deployment_config.max_retries:
                    raise ModelServingError(f"HTTP error: {e.response.status_code}")
                await asyncio.sleep(self.deployment_config.retry_delay * (attempt + 1))
        
        raise ModelServingError("Maximum retries exceeded")
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """vLLM doesn't support embeddings."""
        raise NotImplementedError("Use TritonClient for embeddings")


# =============================================================================
# TRITON CLIENT
# =============================================================================

class TritonClient(BaseModelClient):
    """Client for NVIDIA Triton serving backend."""
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Triton optimized for embeddings/classification, not generation."""
        raise NotImplementedError("Use VLLMClient for text generation")
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Triton."""
        request_data = {
            "inputs": [
                {
                    "name": "input_text",
                    "shape": [len(texts)],
                    "datatype": "BYTES",
                    "data": texts
                }
            ]
        }
        
        for attempt in range(self.deployment_config.max_retries + 1):
            try:
                response = await self.client.post(
                    f"{self.endpoint}/v2/models/{self.config.model_id}/infer",
                    json=request_data
                )
                response.raise_for_status()
                result = response.json()
                
                # Extract embeddings from Triton response
                embeddings = result["outputs"][0]["data"]
                return embeddings
            
            except httpx.TimeoutException:
                if attempt == self.deployment_config.max_retries:
                    raise InferenceTimeoutError(f"Request timed out after {attempt + 1} attempts")
                await asyncio.sleep(self.deployment_config.retry_delay * (attempt + 1))
            
            except Exception as e:
                if attempt == self.deployment_config.max_retries:
                    raise ModelServingError(f"Embedding failed: {str(e)}")
                await asyncio.sleep(self.deployment_config.retry_delay * (attempt + 1))
        
        raise ModelServingError("Maximum retries exceeded")


# =============================================================================
# OLLAMA CLIENT (Development/Offline)
# =============================================================================

class OllamaClient(BaseModelClient):
    """Client for Ollama serving backend (local development)."""
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Ollama."""
        request_data = {
            "model": self.config.model_path,  # Ollama model name
            "prompt": prompt,
            "stream": False,
            **kwargs
        }
        
        try:
            response = await self.client.post(
                f"{self.endpoint}/api/generate",
                json=request_data
            )
            response.raise_for_status()
            result = response.json()
            return result["response"]
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise ModelServingError(f"Ollama error: {str(e)}")
    
    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using Ollama."""
        embeddings = []
        for text in texts:
            request_data = {
                "model": self.config.model_path,
                "prompt": text
            }
            try:
                response = await self.client.post(
                    f"{self.endpoint}/api/embeddings",
                    json=request_data
                )
                response.raise_for_status()
                result = response.json()
                embeddings.append(result["embedding"])
            except Exception as e:
                logger.error(f"Ollama embedding failed: {e}")
                raise ModelServingError(f"Ollama error: {str(e)}")
        
        return embeddings


# =============================================================================
# UNIFIED MODEL CLIENT
# =============================================================================

class ModelServingClient:
    """Unified client for all model serving backends."""
    
    def __init__(self):
        self.clients: Dict[str, BaseModelClient] = {}
        self.deployment_config = DEPLOYMENT_CONFIG
        self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize clients for all configured models."""
        # LLM client
        llm_config = get_model_config("llm")
        if llm_config.backend == ModelBackend.VLLM:
            self.clients["llm"] = VLLMClient(llm_config, self.deployment_config)
        elif llm_config.backend == ModelBackend.OLLAMA:
            self.clients["llm"] = OllamaClient(llm_config, self.deployment_config)
        
        # Embedding client
        embedding_config = get_model_config("embedding")
        if embedding_config.backend == ModelBackend.TRITON:
            self.clients["embedding"] = TritonClient(embedding_config, self.deployment_config)
        elif embedding_config.backend == ModelBackend.OLLAMA:
            self.clients["embedding"] = OllamaClient(embedding_config, self.deployment_config)
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using LLM."""
        client = self.clients.get("llm")
        if not client:
            raise ModelUnavailableError("LLM client not initialized")
        
        return await client.generate(prompt, **kwargs)
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings."""
        client = self.clients.get("embedding")
        if not client:
            raise ModelUnavailableError("Embedding client not initialized")
        
        return await client.embed(texts)
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Check health of all model endpoints."""
        health_status = {}
        for name, client in self.clients.items():
            health_status[name] = await client.health_check()
        return health_status
    
    async def close_all(self):
        """Close all clients."""
        for client in self.clients.values():
            await client.close()


# =============================================================================
# GLOBAL CLIENT INSTANCE
# =============================================================================

_global_client: Optional[ModelServingClient] = None


def get_model_client() -> ModelServingClient:
    """Get global model serving client (singleton)."""
    global _global_client
    if _global_client is None:
        _global_client = ModelServingClient()
    return _global_client


async def close_model_client():
    """Close global model client."""
    global _global_client
    if _global_client:
        await _global_client.close_all()
        _global_client = None
