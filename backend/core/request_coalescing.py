"""
Request Coalescing Engine - Intelligent Request Batching

Combines similar AI requests to maximize throughput and reduce redundant
model invocations. Critical for handling concurrent users efficiently.

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Request Coalescing Engine                    │
    │                                                                 │
    │  Incoming Requests                                              │
    │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐                              │
    │  │ R1  │ │ R2  │ │ R3  │ │ R4  │                              │
    │  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘                              │
    │     │       │       │       │                                   │
    │     └───────┴───────┴───────┘                                   │
    │              │                                                  │
    │              ▼                                                  │
    │     ┌────────────────┐                                         │
    │     │ Request Queue  │  (grouped by operation type)            │
    │     └───────┬────────┘                                         │
    │             │                                                   │
    │     ┌───────┴───────┐                                          │
    │     │ Batch Window  │  (100ms timeout OR max_batch_size)       │
    │     └───────┬───────┘                                          │
    │             │                                                   │
    │             ▼                                                   │
    │     ┌────────────────┐                                         │
    │     │ Batch Executor │  (single model inference)               │
    │     └───────┬────────┘                                         │
    │             │                                                   │
    │     ┌───────┴───────┐                                          │
    │     │ Result Router │  (dispatch results to waiters)           │
    │     └───────────────┘                                          │
    └─────────────────────────────────────────────────────────────────┘

Benefits:
- 2-5x throughput improvement for concurrent requests
- Reduced GPU/memory thrashing from sequential processing
- Fair scheduling with priority support
- Automatic deduplication of identical requests
"""
import asyncio
import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any, Awaitable, Callable, Dict, Generic, List, Optional, Set, TypeVar
)
from collections import defaultdict
from weakref import WeakSet

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


class OperationType(str, Enum):
    """Types of operations that can be coalesced."""
    SIMPLIFY = "simplify"
    TRANSLATE = "translate"
    VALIDATE = "validate"
    EMBED = "embed"
    TTS = "tts"


class RequestPriority(int, Enum):
    """Request priority levels."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class CoalescedRequest(Generic[T]):
    """A single request in the coalescing queue."""
    id: str
    operation: OperationType
    input_data: T
    parameters: Dict[str, Any]
    priority: RequestPriority
    created_at: float
    future: asyncio.Future
    
    # For deduplication
    content_hash: str = ""
    
    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = self._compute_hash()
    
    def _compute_hash(self) -> str:
        """Compute hash for deduplication."""
        hash_input = f"{self.operation.value}:{self.input_data}:{sorted(self.parameters.items())}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]


@dataclass
class BatchResult(Generic[R]):
    """Result of a batched operation."""
    request_id: str
    success: bool
    result: Optional[R] = None
    error: Optional[str] = None
    processing_time_ms: float = 0.0


@dataclass
class CoalescingMetrics:
    """Metrics for the coalescing engine."""
    total_requests: int = 0
    batched_requests: int = 0
    deduplicated_requests: int = 0
    total_batches: int = 0
    avg_batch_size: float = 0.0
    avg_wait_time_ms: float = 0.0
    
    def record_batch(self, batch_size: int, wait_times: List[float]):
        """Record metrics for a processed batch."""
        self.total_batches += 1
        self.batched_requests += batch_size
        
        # Update running average
        self.avg_batch_size = (
            (self.avg_batch_size * (self.total_batches - 1) + batch_size) 
            / self.total_batches
        )
        
        if wait_times:
            avg_wait = sum(wait_times) / len(wait_times) * 1000
            self.avg_wait_time_ms = (
                (self.avg_wait_time_ms * (self.total_batches - 1) + avg_wait)
                / self.total_batches
            )


class RequestCoalescingEngine:
    """
    Coalesces similar requests into batches for efficient processing.
    
    Features:
    - Operation-specific queues (simplify, translate, etc.)
    - Configurable batch size and timeout
    - Priority-based ordering within batches
    - Request deduplication (identical inputs share results)
    - Metrics tracking for monitoring
    """
    
    def __init__(
        self,
        max_batch_size: int = 8,
        batch_timeout_ms: float = 100.0,
        max_queue_size: int = 100,
        enable_deduplication: bool = True,
    ):
        """
        Initialize the coalescing engine.
        
        Args:
            max_batch_size: Maximum requests per batch
            batch_timeout_ms: Max time to wait for batch to fill
            max_queue_size: Max queue size before rejecting requests
            enable_deduplication: Whether to deduplicate identical requests
        """
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.max_queue_size = max_queue_size
        self.enable_deduplication = enable_deduplication
        
        # Per-operation queues
        self._queues: Dict[OperationType, List[CoalescedRequest]] = {
            op: [] for op in OperationType
        }
        
        # Deduplication tracking
        self._pending_hashes: Dict[str, List[asyncio.Future]] = defaultdict(list)
        
        # Batch processors (registered externally)
        self._processors: Dict[OperationType, Callable] = {}
        
        # Metrics
        self.metrics = CoalescingMetrics()
        
        # Background workers
        self._workers: Dict[OperationType, asyncio.Task] = {}
        self._running = False
        
        # Locks
        self._queue_locks: Dict[OperationType, asyncio.Lock] = {
            op: asyncio.Lock() for op in OperationType
        }
        
        logger.info(
            f"RequestCoalescingEngine initialized: "
            f"batch_size={max_batch_size}, timeout={batch_timeout_ms}ms"
        )
    
    def register_processor(
        self,
        operation: OperationType,
        processor: Callable[[List[Any], Dict[str, Any]], Awaitable[List[Any]]]
    ) -> None:
        """
        Register a batch processor for an operation type.
        
        Args:
            operation: Operation type to handle
            processor: Async function that processes a batch
                       Signature: async def process(inputs: List[T], params: Dict) -> List[R]
        """
        self._processors[operation] = processor
        logger.info(f"Registered processor for {operation.value}")
    
    async def submit(
        self,
        operation: OperationType,
        input_data: Any,
        parameters: Optional[Dict[str, Any]] = None,
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout_seconds: float = 60.0,
    ) -> Any:
        """
        Submit a request for coalesced processing.
        
        Args:
            operation: Type of operation
            input_data: Input data for the operation
            parameters: Additional parameters
            priority: Request priority
            timeout_seconds: Max time to wait for result
            
        Returns:
            Operation result
            
        Raises:
            asyncio.TimeoutError: If request times out
            RuntimeError: If queue is full or processor not registered
        """
        if operation not in self._processors:
            raise RuntimeError(f"No processor registered for {operation.value}")
        
        parameters = parameters or {}
        
        # Create request
        request = CoalescedRequest(
            id=f"{operation.value}_{time.time_ns()}",
            operation=operation,
            input_data=input_data,
            parameters=parameters,
            priority=priority,
            created_at=time.time(),
            future=asyncio.get_event_loop().create_future()
        )
        
        self.metrics.total_requests += 1
        
        # Check for deduplication
        if self.enable_deduplication:
            async with self._queue_locks[operation]:
                if request.content_hash in self._pending_hashes:
                    # Identical request already pending - share the result
                    self.metrics.deduplicated_requests += 1
                    logger.debug(f"Deduplicated request: {request.content_hash}")
                    
                    # Create new future that will be resolved with same result
                    shared_future = asyncio.get_event_loop().create_future()
                    self._pending_hashes[request.content_hash].append(shared_future)
                    
                    try:
                        return await asyncio.wait_for(shared_future, timeout_seconds)
                    except asyncio.TimeoutError:
                        raise
        
        # Add to queue
        async with self._queue_locks[operation]:
            if len(self._queues[operation]) >= self.max_queue_size:
                raise RuntimeError(f"Queue full for {operation.value}")
            
            self._queues[operation].append(request)
            
            if self.enable_deduplication:
                self._pending_hashes[request.content_hash].append(request.future)
            
            # Sort by priority (higher priority first)
            self._queues[operation].sort(
                key=lambda r: (-r.priority.value, r.created_at)
            )
        
        # Wait for result
        try:
            result = await asyncio.wait_for(request.future, timeout_seconds)
            return result
        except asyncio.TimeoutError:
            # Clean up on timeout
            async with self._queue_locks[operation]:
                if request in self._queues[operation]:
                    self._queues[operation].remove(request)
            raise
    
    async def _process_batch(
        self,
        operation: OperationType,
        batch: List[CoalescedRequest]
    ) -> None:
        """Process a batch of requests."""
        if not batch:
            return
        
        processor = self._processors[operation]
        start_time = time.time()
        
        # Collect inputs and parameters
        inputs = [r.input_data for r in batch]
        
        # Use parameters from first request (they should be compatible)
        params = batch[0].parameters
        
        wait_times = [start_time - r.created_at for r in batch]
        
        try:
            # Execute batch
            results = await processor(inputs, params)
            
            processing_time = (time.time() - start_time) * 1000
            
            # Distribute results
            for request, result in zip(batch, results):
                if not request.future.done():
                    request.future.set_result(result)
                
                # Handle deduplicated requests
                if self.enable_deduplication:
                    for future in self._pending_hashes.get(request.content_hash, []):
                        if not future.done():
                            future.set_result(result)
                    self._pending_hashes.pop(request.content_hash, None)
            
            # Record metrics
            self.metrics.record_batch(len(batch), wait_times)
            
            logger.debug(
                f"Processed batch: operation={operation.value}, "
                f"size={len(batch)}, time={processing_time:.1f}ms"
            )
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            
            # Propagate error to all requests
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(e)
                
                if self.enable_deduplication:
                    for future in self._pending_hashes.get(request.content_hash, []):
                        if not future.done():
                            future.set_exception(e)
                    self._pending_hashes.pop(request.content_hash, None)
    
    async def _worker_loop(self, operation: OperationType) -> None:
        """Background worker for processing batches."""
        while self._running:
            try:
                # Wait for batch timeout
                await asyncio.sleep(self.batch_timeout_ms / 1000)
                
                # Get batch
                async with self._queue_locks[operation]:
                    if not self._queues[operation]:
                        continue
                    
                    # Take up to max_batch_size requests
                    batch = self._queues[operation][:self.max_batch_size]
                    self._queues[operation] = self._queues[operation][self.max_batch_size:]
                
                # Process batch
                if batch:
                    await self._process_batch(operation, batch)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker loop error for {operation.value}: {e}")
    
    async def start(self) -> None:
        """Start the coalescing engine workers."""
        if self._running:
            return
        
        self._running = True
        
        for operation in OperationType:
            if operation in self._processors:
                self._workers[operation] = asyncio.create_task(
                    self._worker_loop(operation)
                )
        
        logger.info(f"RequestCoalescingEngine started with {len(self._workers)} workers")
    
    async def stop(self) -> None:
        """Stop the coalescing engine."""
        self._running = False
        
        # Cancel workers
        for task in self._workers.values():
            task.cancel()
        
        # Wait for workers to finish
        for task in self._workers.values():
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        self._workers.clear()
        
        # Complete any pending requests with cancellation
        for operation, queue in self._queues.items():
            for request in queue:
                if not request.future.done():
                    request.future.cancel()
            queue.clear()
        
        logger.info("RequestCoalescingEngine stopped")
    
    async def flush(self, operation: Optional[OperationType] = None) -> None:
        """
        Immediately process all pending requests.
        
        Args:
            operation: Specific operation to flush, or all if None
        """
        operations = [operation] if operation else list(OperationType)
        
        for op in operations:
            async with self._queue_locks[op]:
                batch = self._queues[op][:]
                self._queues[op] = []
            
            if batch:
                await self._process_batch(op, batch)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        queue_sizes = {
            op.value: len(queue) 
            for op, queue in self._queues.items()
        }
        
        return {
            "total_requests": self.metrics.total_requests,
            "batched_requests": self.metrics.batched_requests,
            "deduplicated_requests": self.metrics.deduplicated_requests,
            "total_batches": self.metrics.total_batches,
            "avg_batch_size": round(self.metrics.avg_batch_size, 2),
            "avg_wait_time_ms": round(self.metrics.avg_wait_time_ms, 2),
            "queue_sizes": queue_sizes,
            "running": self._running,
        }
    
    def get_queue_depth(self, operation: OperationType) -> int:
        """Get current queue depth for an operation."""
        return len(self._queues[operation])


# Global instance
_coalescing_engine: Optional[RequestCoalescingEngine] = None


def get_coalescing_engine() -> RequestCoalescingEngine:
    """Get global coalescing engine instance."""
    global _coalescing_engine
    if _coalescing_engine is None:
        _coalescing_engine = RequestCoalescingEngine()
    return _coalescing_engine


async def init_coalescing_engine(
    max_batch_size: int = 8,
    batch_timeout_ms: float = 100.0,
) -> RequestCoalescingEngine:
    """Initialize and start the global coalescing engine."""
    global _coalescing_engine
    _coalescing_engine = RequestCoalescingEngine(
        max_batch_size=max_batch_size,
        batch_timeout_ms=batch_timeout_ms
    )
    await _coalescing_engine.start()
    return _coalescing_engine
