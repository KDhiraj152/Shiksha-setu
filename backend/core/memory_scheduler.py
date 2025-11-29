"""
Predictive Memory Scheduler - Workload-Aware Model Preloading

Uses time-series analysis to predict upcoming service demand and preload
models before they're needed, eliminating cold-start latency.

Key Features:
- EWMA-based workload prediction (lightweight alternative to ARIMA)
- Service demand forecasting with confidence intervals
- Memory-aware preloading decisions
- Idle-time optimization for model switching

Architecture:
    Request History → Demand Prediction → Memory Budget → Preload Decision
                                                ↓
                                        Model Orchestrator
"""
import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable, Any
from statistics import mean, stdev

logger = logging.getLogger(__name__)


class ServiceType(str, Enum):
    """AI service types (mirrors orchestrator)."""
    TRANSLATION = "translation"
    TTS = "tts"
    SIMPLIFICATION = "simplification"
    EMBEDDINGS = "embeddings"
    VALIDATION = "validation"


@dataclass
class ServiceRequest:
    """Record of a service request."""
    service: ServiceType
    timestamp: datetime
    duration_ms: float
    memory_mb: float
    success: bool


@dataclass
class DemandForecast:
    """Forecast for a service's demand."""
    service: ServiceType
    predicted_requests_per_minute: float
    confidence: float  # 0.0 - 1.0
    trend: str  # "increasing", "stable", "decreasing"
    predicted_next_request_seconds: float
    priority_score: float  # Higher = should preload


@dataclass
class MemoryBudget:
    """Current memory allocation status."""
    total_gb: float
    used_gb: float
    available_gb: float
    loaded_services: Dict[ServiceType, float]  # Service -> memory in GB
    

@dataclass
class PreloadDecision:
    """Decision about what to preload/unload."""
    action: str  # "preload", "unload", "keep"
    service: ServiceType
    reason: str
    estimated_memory_gb: float
    priority: int  # 1-10, higher is more urgent


class ExponentialMovingAverage:
    """EWMA calculator for demand forecasting."""
    
    def __init__(self, alpha: float = 0.3, span: int = 10):
        """
        Initialize EWMA.
        
        Args:
            alpha: Smoothing factor (0-1, higher = more weight to recent)
            span: Window size for computing alpha if not provided
        """
        self.alpha = alpha if alpha else 2 / (span + 1)
        self._value: Optional[float] = None
        self._variance: float = 0.0
        self._count: int = 0
    
    def update(self, value: float) -> float:
        """Update with new observation."""
        if self._value is None:
            self._value = value
        else:
            diff = value - self._value
            self._value = self._value + self.alpha * diff
            self._variance = (1 - self.alpha) * (self._variance + self.alpha * diff ** 2)
        
        self._count += 1
        return self._value
    
    @property
    def value(self) -> float:
        """Current smoothed value."""
        return self._value or 0.0
    
    @property
    def std(self) -> float:
        """Current standard deviation estimate."""
        return self._variance ** 0.5
    
    @property
    def confidence_interval(self) -> Tuple[float, float]:
        """95% confidence interval."""
        margin = 1.96 * self.std
        return (self.value - margin, self.value + margin)


class PredictiveMemoryScheduler:
    """
    Predicts service demand and optimizes model preloading.
    
    Strategy:
    1. Track request patterns over time windows
    2. Use EWMA to smooth demand signals
    3. Predict next-likely services based on:
       - Recent request frequency
       - Time-of-day patterns (optional)
       - Request sequences (e.g., simplify → translate → tts)
    4. Preload during idle periods
    5. Evict based on predicted demand, not just LRU
    """
    
    # Common request sequences (pipeline patterns)
    COMMON_SEQUENCES = [
        [ServiceType.SIMPLIFICATION, ServiceType.TRANSLATION, ServiceType.TTS],
        [ServiceType.SIMPLIFICATION, ServiceType.VALIDATION],
        [ServiceType.TRANSLATION, ServiceType.TTS],
        [ServiceType.EMBEDDINGS, ServiceType.VALIDATION],
    ]
    
    # Memory requirements per service (in GB)
    SERVICE_MEMORY: Dict[ServiceType, float] = {
        ServiceType.TRANSLATION: 2.5,    # NLLB-200
        ServiceType.TTS: 0.1,             # Edge TTS (API-based)
        ServiceType.SIMPLIFICATION: 2.0,  # Ollama/Qwen
        ServiceType.EMBEDDINGS: 1.2,      # BGE-M3
        ServiceType.VALIDATION: 0.5,      # BERT validation
    }
    
    def __init__(
        self,
        max_memory_gb: float = 10.0,
        history_size: int = 500,
        prediction_window_minutes: int = 15,
        preload_threshold: float = 0.6,
    ):
        """
        Initialize predictive scheduler.
        
        Args:
            max_memory_gb: Maximum memory budget
            history_size: Number of requests to keep in history
            prediction_window_minutes: Time window for predictions
            preload_threshold: Minimum probability to trigger preload
        """
        self.max_memory_gb = max_memory_gb
        self.history_size = history_size
        self.prediction_window = timedelta(minutes=prediction_window_minutes)
        self.preload_threshold = preload_threshold
        
        # Request history
        self._history: deque[ServiceRequest] = deque(maxlen=history_size)
        
        # EWMA trackers per service
        self._demand_trackers: Dict[ServiceType, ExponentialMovingAverage] = {
            service: ExponentialMovingAverage(alpha=0.3)
            for service in ServiceType
        }
        
        # Last request time per service
        self._last_request: Dict[ServiceType, datetime] = {}
        
        # Sequence detection
        self._recent_sequence: deque[ServiceType] = deque(maxlen=5)
        
        # Current memory state
        self._loaded_services: Dict[ServiceType, float] = {}
        
        # Background task handle
        self._optimization_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Callbacks for preload/unload actions
        self._preload_callback: Optional[Callable[[ServiceType], Any]] = None
        self._unload_callback: Optional[Callable[[ServiceType], Any]] = None
        
        logger.info(
            f"PredictiveMemoryScheduler initialized: "
            f"max_memory={max_memory_gb}GB, window={prediction_window_minutes}min"
        )
    
    def record_request(
        self,
        service: ServiceType,
        duration_ms: float,
        success: bool = True
    ) -> None:
        """
        Record a service request for demand tracking.
        
        Args:
            service: Service type that was called
            duration_ms: Request duration in milliseconds
            success: Whether request succeeded
        """
        now = datetime.now()
        
        # Calculate requests per minute for this service
        recent_requests = [
            r for r in self._history
            if r.service == service and now - r.timestamp < timedelta(minutes=1)
        ]
        requests_per_minute = len(recent_requests) + 1
        
        # Update EWMA tracker
        self._demand_trackers[service].update(requests_per_minute)
        
        # Record request
        request = ServiceRequest(
            service=service,
            timestamp=now,
            duration_ms=duration_ms,
            memory_mb=self.SERVICE_MEMORY.get(service, 0.5) * 1024,
            success=success
        )
        self._history.append(request)
        
        # Update last request time
        self._last_request[service] = now
        
        # Update sequence tracking
        self._recent_sequence.append(service)
        
        logger.debug(
            f"Recorded {service.value}: {duration_ms:.0f}ms, "
            f"demand={requests_per_minute:.1f} req/min"
        )
    
    def register_loaded_service(self, service: ServiceType, memory_gb: float) -> None:
        """Register a service as loaded in memory."""
        self._loaded_services[service] = memory_gb
        logger.debug(f"Registered loaded: {service.value} ({memory_gb:.1f}GB)")
    
    def unregister_service(self, service: ServiceType) -> None:
        """Unregister a service from loaded state."""
        self._loaded_services.pop(service, None)
        logger.debug(f"Unregistered: {service.value}")
    
    def get_memory_budget(self) -> MemoryBudget:
        """Get current memory allocation status."""
        used = sum(self._loaded_services.values())
        return MemoryBudget(
            total_gb=self.max_memory_gb,
            used_gb=used,
            available_gb=self.max_memory_gb - used,
            loaded_services=dict(self._loaded_services)
        )
    
    def forecast_demand(self, service: ServiceType) -> DemandForecast:
        """
        Forecast demand for a specific service.
        
        Args:
            service: Service to forecast
            
        Returns:
            DemandForecast with predictions
        """
        tracker = self._demand_trackers[service]
        
        # Calculate predicted requests per minute
        predicted_rpm = tracker.value
        
        # Calculate confidence based on sample size and variance
        sample_count = len([r for r in self._history if r.service == service])
        variance_factor = 1.0 / (1.0 + tracker.std) if tracker.std > 0 else 1.0
        sample_factor = min(1.0, sample_count / 50)  # Full confidence at 50 samples
        confidence = variance_factor * sample_factor
        
        # Determine trend
        if tracker._count < 3:
            trend = "unknown"
        else:
            # Compare recent to historical
            recent_requests = [
                r for r in list(self._history)[-20:]
                if r.service == service
            ]
            older_requests = [
                r for r in list(self._history)[:-20]
                if r.service == service
            ]
            
            recent_rate = len(recent_requests)
            older_rate = len(older_requests) / max(1, len(list(self._history)[:-20]) / 20)
            
            if recent_rate > older_rate * 1.2:
                trend = "increasing"
            elif recent_rate < older_rate * 0.8:
                trend = "decreasing"
            else:
                trend = "stable"
        
        # Predict next request timing
        last_request = self._last_request.get(service)
        if last_request and predicted_rpm > 0:
            avg_interval = 60.0 / predicted_rpm
            seconds_since_last = (datetime.now() - last_request).total_seconds()
            predicted_next = max(0, avg_interval - seconds_since_last)
        else:
            predicted_next = float('inf')
        
        # Calculate priority score (for preloading decisions)
        priority_score = self._calculate_priority_score(
            service, predicted_rpm, confidence, predicted_next
        )
        
        return DemandForecast(
            service=service,
            predicted_requests_per_minute=predicted_rpm,
            confidence=confidence,
            trend=trend,
            predicted_next_request_seconds=predicted_next,
            priority_score=priority_score
        )
    
    def _calculate_priority_score(
        self,
        service: ServiceType,
        predicted_rpm: float,
        confidence: float,
        predicted_next_seconds: float
    ) -> float:
        """
        Calculate priority score for preloading.
        
        Higher score = more important to have loaded.
        """
        score = 0.0
        
        # Demand factor (0-40 points)
        score += min(40, predicted_rpm * 10)
        
        # Urgency factor (0-30 points) - closer next request = higher
        if predicted_next_seconds < 5:
            score += 30
        elif predicted_next_seconds < 30:
            score += 20
        elif predicted_next_seconds < 60:
            score += 10
        
        # Sequence factor (0-20 points) - part of common pipeline?
        if len(self._recent_sequence) >= 2:
            for seq in self.COMMON_SEQUENCES:
                recent = list(self._recent_sequence)[-2:]
                if recent == seq[:2] and service in seq[2:]:
                    score += 20
                    break
        
        # Confidence factor (scale by confidence)
        score *= confidence
        
        # Memory efficiency bonus (smaller services get slight boost)
        memory = self.SERVICE_MEMORY.get(service, 1.0)
        if memory < 1.0:
            score *= 1.1
        
        return score
    
    def get_preload_decisions(self) -> List[PreloadDecision]:
        """
        Get recommended preload/unload actions.
        
        Returns:
            List of PreloadDecisions sorted by priority
        """
        decisions: List[PreloadDecision] = []
        budget = self.get_memory_budget()
        
        # Get forecasts for all services
        forecasts = {
            service: self.forecast_demand(service)
            for service in ServiceType
        }
        
        # Sort by priority score
        sorted_services = sorted(
            forecasts.items(),
            key=lambda x: x[1].priority_score,
            reverse=True
        )
        
        # Check what should be preloaded
        for service, forecast in sorted_services:
            memory_needed = self.SERVICE_MEMORY.get(service, 1.0)
            
            if service in self._loaded_services:
                # Already loaded - should we keep it?
                if forecast.priority_score < 10 and forecast.trend == "decreasing":
                    decisions.append(PreloadDecision(
                        action="unload",
                        service=service,
                        reason=f"Low demand (score={forecast.priority_score:.1f}, trend={forecast.trend})",
                        estimated_memory_gb=memory_needed,
                        priority=int(100 - forecast.priority_score)
                    ))
                else:
                    decisions.append(PreloadDecision(
                        action="keep",
                        service=service,
                        reason=f"Active demand (score={forecast.priority_score:.1f})",
                        estimated_memory_gb=memory_needed,
                        priority=int(forecast.priority_score)
                    ))
            else:
                # Not loaded - should we preload?
                if forecast.priority_score > self.preload_threshold * 100:
                    if budget.available_gb >= memory_needed:
                        decisions.append(PreloadDecision(
                            action="preload",
                            service=service,
                            reason=f"High predicted demand (score={forecast.priority_score:.1f})",
                            estimated_memory_gb=memory_needed,
                            priority=int(forecast.priority_score)
                        ))
                    else:
                        decisions.append(PreloadDecision(
                            action="preload",
                            service=service,
                            reason=f"Would preload but insufficient memory ({budget.available_gb:.1f}GB < {memory_needed:.1f}GB)",
                            estimated_memory_gb=memory_needed,
                            priority=int(forecast.priority_score * 0.5)  # Lower priority
                        ))
        
        # Sort by priority
        decisions.sort(key=lambda d: d.priority, reverse=True)
        
        return decisions
    
    async def _optimization_loop(self) -> None:
        """Background loop for memory optimization."""
        while self._running:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                decisions = self.get_preload_decisions()
                
                for decision in decisions:
                    if decision.action == "preload" and self._preload_callback:
                        logger.info(f"Preloading {decision.service.value}: {decision.reason}")
                        try:
                            await self._preload_callback(decision.service)
                            self.register_loaded_service(
                                decision.service,
                                decision.estimated_memory_gb
                            )
                        except Exception as e:
                            logger.warning(f"Preload failed: {e}")
                    
                    elif decision.action == "unload" and self._unload_callback:
                        logger.info(f"Unloading {decision.service.value}: {decision.reason}")
                        try:
                            await self._unload_callback(decision.service)
                            self.unregister_service(decision.service)
                        except Exception as e:
                            logger.warning(f"Unload failed: {e}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
    
    async def start(
        self,
        preload_callback: Optional[Callable] = None,
        unload_callback: Optional[Callable] = None
    ) -> None:
        """Start the optimization loop."""
        self._running = True
        self._preload_callback = preload_callback
        self._unload_callback = unload_callback
        self._optimization_task = asyncio.create_task(self._optimization_loop())
        logger.info("PredictiveMemoryScheduler started")
    
    async def stop(self) -> None:
        """Stop the optimization loop."""
        self._running = False
        if self._optimization_task:
            self._optimization_task.cancel()
            try:
                await self._optimization_task
            except asyncio.CancelledError:
                pass
        logger.info("PredictiveMemoryScheduler stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current scheduler status."""
        budget = self.get_memory_budget()
        forecasts = {
            service.value: {
                "predicted_rpm": f.predicted_requests_per_minute,
                "confidence": f.confidence,
                "trend": f.trend,
                "priority_score": f.priority_score
            }
            for service, f in [
                (s, self.forecast_demand(s)) for s in ServiceType
            ]
        }
        
        return {
            "memory": {
                "total_gb": budget.total_gb,
                "used_gb": budget.used_gb,
                "available_gb": budget.available_gb,
                "loaded_services": list(budget.loaded_services.keys())
            },
            "forecasts": forecasts,
            "history_size": len(self._history),
            "running": self._running
        }


# Global scheduler instance
_scheduler: Optional[PredictiveMemoryScheduler] = None


def get_memory_scheduler() -> PredictiveMemoryScheduler:
    """Get global scheduler instance (lazy init)."""
    global _scheduler
    if _scheduler is None:
        from ..core.config import settings
        max_memory = getattr(settings, 'MAX_MODEL_MEMORY_GB', 10.0)
        _scheduler = PredictiveMemoryScheduler(max_memory_gb=max_memory)
    return _scheduler


async def init_memory_scheduler(
    max_memory_gb: float = 10.0,
    preload_callback: Optional[Callable] = None,
    unload_callback: Optional[Callable] = None
) -> PredictiveMemoryScheduler:
    """Initialize and start global scheduler."""
    global _scheduler
    _scheduler = PredictiveMemoryScheduler(max_memory_gb=max_memory_gb)
    await _scheduler.start(preload_callback, unload_callback)
    return _scheduler
