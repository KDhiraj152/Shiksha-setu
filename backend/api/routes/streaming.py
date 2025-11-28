"""WebSocket streaming endpoints for real-time translation and processing."""
import asyncio
import json
import time
from typing import Optional, Dict, Any
from datetime import datetime, timezone

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query, Depends, status
from fastapi.exceptions import WebSocketException
import langdetect

from ...core.config import settings
from ...utils.logging import get_logger
from ...utils.auth import get_current_user_ws, TokenData
from ...utils.request_context import set_request_id, get_request_id
from ...services.translate import TranslationService
from ...monitoring import track_websocket_connection, track_translation_latency

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/streaming", tags=["streaming"])

# Connection management
active_connections: Dict[str, WebSocket] = {}
MAX_CONNECTIONS = 1000
TRANSLATION_CHUNK_SIZE = 500  # Characters per chunk
PARTIAL_RESULT_INTERVAL = 0.5  # 500ms


class ConnectionManager:
    """Manage WebSocket connections with rate limiting and monitoring."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_start_times: Dict[str, float] = {}
        self.translation_buffers: Dict[str, str] = {}
        self.redis_client = None
        self.pubsub = None
        self._listener_task = None
        self._redis_initialized = False
        
    async def _init_redis(self):
        """Initialize Redis connection asynchronously (called on first connect)."""
        if self._redis_initialized:
            return
        self._redis_initialized = True
        
        try:
            from ...cache import get_redis
            self.redis_client = get_redis()
            if self.redis_client:
                self.pubsub = self.redis_client.pubsub()
                # Subscribe to broadcast channel
                self.pubsub.subscribe('websocket_broadcast')
                # Start listener task
                self._listener_task = asyncio.create_task(self._redis_listener())
                logger.info("Redis Pub/Sub initialized for WebSockets")
        except Exception as e:
            logger.warning(f"Redis unavailable for WebSockets, falling back to in-memory: {e}")

    async def _redis_listener(self):
        """Listen for Redis messages and broadcast to local clients."""
        if not self.pubsub:
            return
            
        try:
            while True:
                message = self.pubsub.get_message(ignore_subscribe_messages=True)
                if message and message['type'] == 'message':
                    data = json.loads(message['data'])
                    # Broadcast to local connections
                    await self._local_broadcast(data)
                await asyncio.sleep(0.01)
        except Exception as e:
            logger.error(f"Redis listener error: {e}")

    async def _local_broadcast(self, message: dict):
        """Broadcast to locally connected clients only."""
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Failed to broadcast to {client_id}: {e}")
    
    async def connect(self, websocket: WebSocket, client_id: str) -> bool:
        """
        Accept new WebSocket connection if under limit.
        
        Args:
            websocket: WebSocket instance
            client_id: Unique client identifier
            
        Returns:
            True if connected, False if rejected
        """
        # Initialize Redis on first connection (now in async context)
        await self._init_redis()
        
        if len(self.active_connections) >= MAX_CONNECTIONS:
            await websocket.close(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="Maximum concurrent connections reached"
            )
            return False
        
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.connection_start_times[client_id] = time.time()
        self.translation_buffers[client_id] = ""
        
        logger.info(
            f"WebSocket connected: {client_id}",
            extra={
                "client_id": client_id,
                "total_connections": len(self.active_connections)
            }
        )
        
        return True
    
    def disconnect(self, client_id: str):
        """Remove connection and cleanup."""
        if client_id in self.active_connections:
            connection_duration = time.time() - self.connection_start_times.get(client_id, time.time())
            del self.active_connections[client_id]
            self.connection_start_times.pop(client_id, None)
            self.translation_buffers.pop(client_id, None)
            
            logger.info(
                f"WebSocket disconnected: {client_id}",
                extra={
                    "client_id": client_id,
                    "duration_seconds": connection_duration,
                    "remaining_connections": len(self.active_connections)
                }
            )
    
    async def send_json(self, client_id: str, data: dict):
        """Send JSON data to client."""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_json(data)
            except Exception as e:
                logger.error(f"Failed to send to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients (distributed)."""
        # 1. Send to local clients
        await self._local_broadcast(message)
        
        # 2. Publish to Redis for other workers
        if self.redis_client:
            try:
                self.redis_client.publish('websocket_broadcast', json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to publish broadcast to Redis: {e}")
            try:
                await websocket.send_json(message)
            except Exception:
                disconnected.append(client_id)
        
        for client_id in disconnected:
            self.disconnect(client_id)


manager = ConnectionManager()


@router.websocket("/translate")
async def websocket_translate_endpoint(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
    target_lang: str = Query("hi", description="Target language code")
):
    """
    WebSocket endpoint for real-time streaming translation.
    
    Protocol:
    - Client sends: {"text": "...", "source_lang": "en", "request_id": "..."}
    - Server responds: {"type": "partial|final|error", "text": "...", "timestamp": "..."}
    
    Features:
    - Translation begins within 200ms of first keypress
    - Partial results every 500ms
    - Handles 1000 concurrent connections
    - Auto-detects source language if not provided
    
    Query Parameters:
        token: Optional JWT token for authenticated users
        target_lang: Target language code (default: hi)
    """
    # Authenticate user (optional)
    user = None
    if token:
        try:
            user = await get_current_user_ws(token)
        except Exception as e:
            logger.warning(f"WebSocket auth failed: {e}")
    
    # Generate client ID
    client_id = f"{user.id if user else 'anon'}_{int(time.time() * 1000)}"
    request_id = f"ws_{client_id}"
    set_request_id(request_id)
    
    # Connect
    if not await manager.connect(websocket, client_id):
        return
    
    try:
        # Initialize translation service
        translation_service = TranslationService()
        
        # Send connection confirmation
        await manager.send_json(client_id, {
            "type": "connected",
            "client_id": client_id,
            "max_connections": MAX_CONNECTIONS,
            "current_connections": len(manager.active_connections),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        # Main message loop
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            text = data.get("text", "").strip()
            source_lang = data.get("source_lang")
            msg_request_id = data.get("request_id", request_id)
            set_request_id(msg_request_id)
            
            if not text:
                await manager.send_json(client_id, {
                    "type": "error",
                    "error": "Empty text received",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                continue
            
            # Start translation timer
            start_time = time.time()
            
            # Auto-detect source language if not provided
            if not source_lang:
                try:
                    source_lang = langdetect.detect(text)
                except Exception as e:
                    logger.warning(f"Language detection failed: {e}")
                    source_lang = "en"
            
            # Send acknowledgment (within 200ms requirement)
            await manager.send_json(client_id, {
                "type": "ack",
                "text_length": len(text),
                "source_lang": source_lang,
                "target_lang": target_lang,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            try:
                # Stream translation in chunks
                partial_result = ""
                last_partial_time = time.time()
                
                # Process text in chunks for incremental translation
                chunks = [text[i:i + TRANSLATION_CHUNK_SIZE] 
                         for i in range(0, len(text), TRANSLATION_CHUNK_SIZE)]
                
                for idx, chunk in enumerate(chunks):
                    # Translate chunk
                    chunk_translation = await translation_service.translate_async(
                        text=chunk,
                        source_lang=source_lang,
                        target_lang=target_lang
                    )
                    
                    partial_result += chunk_translation + " "
                    
                    # Send partial result every 500ms or on last chunk
                    current_time = time.time()
                    is_last_chunk = idx == len(chunks) - 1
                    
                    if (current_time - last_partial_time >= PARTIAL_RESULT_INTERVAL) or is_last_chunk:
                        result_type = "final" if is_last_chunk else "partial"
                        
                        await manager.send_json(client_id, {
                            "type": result_type,
                            "text": partial_result.strip(),
                            "source_lang": source_lang,
                            "target_lang": target_lang,
                            "progress": (idx + 1) / len(chunks),
                            "latency_ms": int((current_time - start_time) * 1000),
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        })
                        
                        last_partial_time = current_time
                
                # Track metrics
                total_latency = time.time() - start_time
                track_translation_latency(source_lang, target_lang, total_latency)
                
                logger.info(
                    f"Translation completed: {len(text)} chars in {total_latency:.2f}s",
                    extra={
                        "client_id": client_id,
                        "request_id": msg_request_id,
                        "source_lang": source_lang,
                        "target_lang": target_lang,
                        "text_length": len(text),
                        "latency_seconds": total_latency
                    }
                )
            
            except Exception as e:
                logger.error(f"Translation error: {e}", exc_info=True)
                await manager.send_json(client_id, {
                    "type": "error",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
    
    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}", exc_info=True)
    finally:
        manager.disconnect(client_id)


@router.websocket("/heartbeat")
async def websocket_heartbeat(websocket: WebSocket):
    """
    Simple heartbeat endpoint for connection health checks.
    
    Client sends: {"ping": timestamp}
    Server responds: {"pong": timestamp, "server_time": timestamp}
    """
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_json()
            
            if "ping" in data:
                await websocket.send_json({
                    "pong": data["ping"],
                    "server_time": datetime.now(timezone.utc).isoformat(),
                    "active_connections": len(manager.active_connections)
                })
    
    except WebSocketDisconnect:
        # Normal disconnect, no action needed
        logger.debug("WebSocket disconnected during heartbeat")
    except Exception as e:
        logger.error(f"Heartbeat error: {e}")


@router.get("/connections")
async def get_connection_stats():
    """Get current WebSocket connection statistics."""
    return {
        "active_connections": len(manager.active_connections),
        "max_connections": MAX_CONNECTIONS,
        "utilization": len(manager.active_connections) / MAX_CONNECTIONS,
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
