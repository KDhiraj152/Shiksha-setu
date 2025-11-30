"""
Threadpool Executor for Non-Model I/O (Principle O)
====================================================
Use threadpool for file I/O, network calls, and other blocking operations.

Strategy:
- Dedicated threadpool for blocking I/O operations
- Keep async event loop responsive
- Prevent model inference from blocking on I/O
- Configurable pool size based on workload

Reference: "Threadpool for file-I/O, external calls"
"""

import asyncio
import logging
import functools
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Any, Callable, TypeVar, Optional
from contextlib import contextmanager
import time

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ThreadPoolManager:
    """
    Manages threadpools for non-model operations (Principle O).
    
    Features:
    - Separate pools for I/O and CPU-bound tasks
    - Automatic async integration
    - Performance tracking
    """
    
    def __init__(self, max_workers: int = 4, cpu_workers: int = 2):
        """
        Initialize thread pools.
        
        Args:
            max_workers: Max threads for I/O operations
            cpu_workers: Max processes for CPU-bound tasks
        """
        self.max_workers = max_workers
        self.cpu_workers = cpu_workers
        
        self._io_executor: Optional[ThreadPoolExecutor] = None
        self._cpu_executor: Optional[ProcessPoolExecutor] = None
        
        self._stats = {
            "io_tasks_submitted": 0,
            "io_tasks_completed": 0,
            "cpu_tasks_submitted": 0,
            "cpu_tasks_completed": 0,
            "total_io_time_ms": 0.0,
            "total_cpu_time_ms": 0.0,
        }
    
    @property
    def io_executor(self) -> ThreadPoolExecutor:
        """Get or create I/O thread pool."""
        if self._io_executor is None:
            self._io_executor = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="ssetu_io"
            )
            logger.info(f"Created I/O thread pool with {self.max_workers} workers")
        return self._io_executor
    
    @property
    def cpu_executor(self) -> ProcessPoolExecutor:
        """Get or create CPU process pool."""
        if self._cpu_executor is None:
            self._cpu_executor = ProcessPoolExecutor(
                max_workers=self.cpu_workers
            )
            logger.info(f"Created CPU process pool with {self.cpu_workers} workers")
        return self._cpu_executor
    
    async def run_in_thread(self, func: Callable[..., T], *args, **kwargs) -> T:
        """
        Run blocking function in thread pool.
        
        Use for:
        - File I/O operations
        - Network calls (non-async libraries)
        - Database operations (sync drivers)
        - External API calls
        
        Args:
            func: Blocking function to run
            *args, **kwargs: Arguments to pass to function
            
        Returns:
            Function result
        """
        loop = asyncio.get_event_loop()
        self._stats["io_tasks_submitted"] += 1
        
        start_time = time.time()
        
        try:
            # Wrap function with kwargs
            if kwargs:
                wrapped = functools.partial(func, *args, **kwargs)
                result = await loop.run_in_executor(self.io_executor, wrapped)
            else:
                result = await loop.run_in_executor(self.io_executor, func, *args)
            
            return result
            
        finally:
            elapsed_ms = (time.time() - start_time) * 1000
            self._stats["io_tasks_completed"] += 1
            self._stats["total_io_time_ms"] += elapsed_ms
    
    async def run_in_process(self, func: Callable[..., T], *args) -> T:
        """
        Run CPU-bound function in process pool.
        
        Use for:
        - Heavy computation
        - Image processing
        - Data transformation
        
        Note: Function and args must be picklable.
        
        Args:
            func: CPU-bound function to run
            *args: Arguments to pass to function
            
        Returns:
            Function result
        """
        loop = asyncio.get_event_loop()
        self._stats["cpu_tasks_submitted"] += 1
        
        start_time = time.time()
        
        try:
            result = await loop.run_in_executor(self.cpu_executor, func, *args)
            return result
            
        finally:
            elapsed_ms = (time.time() - start_time) * 1000
            self._stats["cpu_tasks_completed"] += 1
            self._stats["total_cpu_time_ms"] += elapsed_ms
    
    def submit_io(self, func: Callable[..., T], *args, **kwargs):
        """
        Submit blocking function to thread pool (non-async).
        
        Returns a Future that can be waited on.
        """
        self._stats["io_tasks_submitted"] += 1
        
        if kwargs:
            wrapped = functools.partial(func, *args, **kwargs)
            return self.io_executor.submit(wrapped)
        return self.io_executor.submit(func, *args)
    
    def get_stats(self) -> dict:
        """Get pool statistics."""
        avg_io_time = (
            self._stats["total_io_time_ms"] / max(self._stats["io_tasks_completed"], 1)
        )
        avg_cpu_time = (
            self._stats["total_cpu_time_ms"] / max(self._stats["cpu_tasks_completed"], 1)
        )
        
        return {
            **self._stats,
            "avg_io_time_ms": avg_io_time,
            "avg_cpu_time_ms": avg_cpu_time,
            "io_pool_workers": self.max_workers,
            "cpu_pool_workers": self.cpu_workers,
        }
    
    def shutdown(self, wait: bool = True):
        """Shutdown all pools."""
        if self._io_executor:
            self._io_executor.shutdown(wait=wait)
            self._io_executor = None
        
        if self._cpu_executor:
            self._cpu_executor.shutdown(wait=wait)
            self._cpu_executor = None
        
        logger.info("Thread pools shut down")


# Global pool manager
_pool_manager: Optional[ThreadPoolManager] = None


def get_pool_manager() -> ThreadPoolManager:
    """Get or create global pool manager."""
    global _pool_manager
    if _pool_manager is None:
        from backend.core.config import settings
        _pool_manager = ThreadPoolManager(
            max_workers=settings.THREADPOOL_MAX_WORKERS,
            cpu_workers=max(1, settings.THREADPOOL_MAX_WORKERS // 2)
        )
    return _pool_manager


# =============================================================================
# Convenience Functions
# =============================================================================

async def run_blocking(func: Callable[..., T], *args, **kwargs) -> T:
    """
    Run blocking function in thread pool.
    
    Convenience wrapper for pool_manager.run_in_thread().
    
    Args:
        func: Blocking function
        *args, **kwargs: Arguments
        
    Returns:
        Function result
    """
    manager = get_pool_manager()
    return await manager.run_in_thread(func, *args, **kwargs)


async def run_cpu_bound(func: Callable[..., T], *args) -> T:
    """
    Run CPU-bound function in process pool.
    
    Args:
        func: CPU-bound function (must be picklable)
        *args: Arguments (must be picklable)
        
    Returns:
        Function result
    """
    manager = get_pool_manager()
    return await manager.run_in_process(func, *args)


# =============================================================================
# Decorators
# =============================================================================

def run_in_threadpool(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to run sync function in threadpool.
    
    Converts a blocking function to an async function that runs in threadpool.
    
    Usage:
        @run_in_threadpool
        def read_large_file(path):
            with open(path) as f:
                return f.read()
        
        # Now can be awaited
        content = await read_large_file("large.txt")
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        return await run_blocking(func, *args, **kwargs)
    
    return wrapper


def blocking_io(func: Callable[..., T]) -> Callable[..., T]:
    """Alias for run_in_threadpool decorator."""
    return run_in_threadpool(func)


# =============================================================================
# Common I/O Operations
# =============================================================================

@run_in_threadpool
def read_file_blocking(path: str, mode: str = "r") -> str:
    """Read file in threadpool."""
    with open(path, mode) as f:
        return f.read()


@run_in_threadpool
def write_file_blocking(path: str, content: str, mode: str = "w") -> int:
    """Write file in threadpool."""
    with open(path, mode) as f:
        return f.write(content)


@run_in_threadpool
def read_json_blocking(path: str) -> dict:
    """Read JSON file in threadpool."""
    import json
    with open(path, "r") as f:
        return json.load(f)


@run_in_threadpool
def write_json_blocking(path: str, data: dict) -> None:
    """Write JSON file in threadpool."""
    import json
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


async def fetch_url(url: str, timeout: float = 30.0) -> str:
    """
    Fetch URL content (uses async HTTP client).
    
    For sync HTTP libraries, use run_blocking() instead.
    """
    import aiohttp
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
            return await resp.text()


# =============================================================================
# Context Manager for Temporary Threadpool
# =============================================================================

@contextmanager
def temporary_pool(max_workers: int = 4):
    """
    Context manager for temporary thread pool.
    
    Usage:
        with temporary_pool(max_workers=8) as pool:
            futures = [pool.submit(task, arg) for arg in args]
            results = [f.result() for f in futures]
    """
    pool = ThreadPoolExecutor(max_workers=max_workers)
    try:
        yield pool
    finally:
        pool.shutdown(wait=True)
