"""
Streaming Service

Provides streaming functionality for various backend operations.
Re-exports from api/routes/streaming for backward compatibility.
"""
from typing import AsyncGenerator, Any, Callable, Optional
import asyncio
import json
import logging

logger = logging.getLogger(__name__)


class StreamingService:
    """
    Service for managing streaming responses.
    
    Provides a unified interface for streaming data from various sources
    like pipeline processing, Q&A, and TTS generation.
    """
    
    def __init__(self):
        self._active_streams: dict = {}
    
    async def create_stream(
        self,
        stream_id: str,
        generator_fn: Callable[..., AsyncGenerator[Any, None]],
        *args,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """
        Create a new SSE stream.
        
        Args:
            stream_id: Unique identifier for this stream
            generator_fn: Async generator function to produce events
            *args, **kwargs: Arguments to pass to generator function
            
        Yields:
            SSE formatted event strings
        """
        self._active_streams[stream_id] = {
            "status": "active",
            "started_at": asyncio.get_event_loop().time(),
        }
        
        try:
            async for event in generator_fn(*args, **kwargs):
                if isinstance(event, dict):
                    yield self._format_sse(event)
                else:
                    yield self._format_sse({"data": event})
                    
            # Send completion event
            yield self._format_sse({
                "event": "complete",
                "data": {"stream_id": stream_id}
            })
            
        except asyncio.CancelledError:
            yield self._format_sse({
                "event": "cancelled",
                "data": {"stream_id": stream_id}
            })
        except Exception as e:
            logger.error(f"Stream {stream_id} error: {e}")
            yield self._format_sse({
                "event": "error",
                "data": {"error": str(e)}
            })
        finally:
            self._active_streams.pop(stream_id, None)
    
    def _format_sse(self, data: dict) -> str:
        """Format data as SSE event."""
        event_type = data.pop("event", "message")
        json_data = json.dumps(data, default=str)
        return f"event: {event_type}\ndata: {json_data}\n\n"
    
    def cancel_stream(self, stream_id: str) -> bool:
        """
        Cancel an active stream.
        
        Args:
            stream_id: Stream to cancel
            
        Returns:
            True if stream was cancelled
        """
        if stream_id in self._active_streams:
            self._active_streams[stream_id]["status"] = "cancelled"
            return True
        return False
    
    def get_active_streams(self) -> dict:
        """Get all active streams."""
        return dict(self._active_streams)
    
    async def heartbeat_stream(
        self,
        interval: float = 30.0
    ) -> AsyncGenerator[str, None]:
        """
        Generate heartbeat events to keep connection alive.
        
        Args:
            interval: Seconds between heartbeats
            
        Yields:
            SSE heartbeat events
        """
        while True:
            yield self._format_sse({
                "event": "heartbeat",
                "data": {"timestamp": asyncio.get_event_loop().time()}
            })
            await asyncio.sleep(interval)


# Singleton instance
_streaming_service: Optional[StreamingService] = None


def get_streaming_service() -> StreamingService:
    """Get global streaming service instance."""
    global _streaming_service
    if _streaming_service is None:
        _streaming_service = StreamingService()
    return _streaming_service
