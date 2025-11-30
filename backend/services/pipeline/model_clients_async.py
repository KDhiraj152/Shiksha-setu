"""Async HuggingFace model client wrappers with httpx for non-blocking inference."""
import os
import time
import asyncio
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import httpx
import logging

from ..utils.device_manager import get_device_manager

logger = logging.getLogger(__name__)

CONTENT_TYPE_JSON = "application/json"


class RateLimiter:
    """Async rate limiter for API calls."""
    
    def __init__(self, max_calls: int = 100, time_window: int = 60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        self._lock = asyncio.Lock()
    
    async def wait_if_needed(self):
        """Wait if rate limit is exceeded."""
        async with self._lock:
            now = time.time()
            # Remove calls outside the time window
            self.calls = [call_time for call_time in self.calls if now - call_time < self.time_window]
            
            if len(self.calls) >= self.max_calls:
                sleep_time = self.time_window - (now - self.calls[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                    self.calls = []
            
            self.calls.append(now)


class BaseAsyncModelClient(ABC):
    """Base class for async HuggingFace model clients."""
    
    def __init__(self, model_id: str, api_key: Optional[str] = None):
        self.model_id = model_id
        self.api_key = api_key or os.getenv('HUGGINGFACE_API_KEY')
        self.api_url = f"https://api-inference.huggingface.co/models/{model_id}"
        self.rate_limiter = RateLimiter(max_calls=100, time_window=60)
        
        # Async HTTP client with retry
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            transport=httpx.AsyncHTTPTransport(retries=3)
        )
        
        # Device manager for local inference fallback
        self.device_manager = get_device_manager()
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with authentication."""
        headers = {"Content-Type": CONTENT_TYPE_JSON}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers
    
    async def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make async API request with rate limiting and error handling."""
        await self.rate_limiter.wait_if_needed()
        
        try:
            response = await self.client.post(
                self.api_url,
                headers=self._get_headers(),
                json=payload
            )
            
            if response.status_code == 503:
                # Model is loading, wait and retry
                await asyncio.sleep(20)
                response = await self.client.post(
                    self.api_url,
                    headers=self._get_headers(),
                    json=payload
                )
            
            # Check for deprecated/moved models
            if response.status_code == 410:
                raise RuntimeError(f"Model {self.model_id} is deprecated or moved. Using fallback.")
            
            response.raise_for_status()
            return response.json()
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP request failed: {e}")
            raise RuntimeError(f"API request failed: {e}")
    
    @abstractmethod
    async def process(self, *args, **kwargs):
        """Process input through the model."""
        pass
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


__all__ = ['BaseAsyncModelClient', 'RateLimiter']
