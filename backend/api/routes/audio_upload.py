"""Audio upload and speech-to-text transcription endpoints."""
import os
import uuid
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

from fastapi import APIRouter, File, UploadFile, HTTPException, Form, Depends
from fastapi.responses import JSONResponse
import magic

from ...core.config import settings
from ...utils.logging import get_logger
from ...utils.auth import get_current_user, TokenData
from ...tasks.celery_app import celery_app
from ...database import get_db_session
from ...models import ProcessedContent

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/audio", tags=["audio"])

# Audio configuration
AUDIO_DIR = Path("data/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
MAX_AUDIO_DURATION = 600  # 10 minutes in seconds
SUPPORTED_FORMATS = ["audio/mpeg", "audio/wav", "audio/ogg", "audio/webm", "audio/mp4"]


@router.post("/upload")
async def upload_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form("hi", description="Audio language (hi, en, ta)"),
    task_type: str = Form("transcribe", description="transcribe or translate"),
    current_user: TokenData = Depends(get_current_user)
):
    """
    Upload audio file for speech-to-text transcription.
    
    Supports:
    - Audio formats: MP3, WAV, OGG, WebM, MP4
    - Max duration: 10 minutes
    - Languages: Hindi (hi), English (en), Tamil (ta)
    - Transcription time: <60 seconds for 10 min audio
    
    Args:
        file: Audio file upload
        language: Source language code
        task_type: 'transcribe' or 'translate' (to English)
        
    Returns:
        Task ID for tracking transcription progress
    """
    try:
        # Read file content
        content = await file.read()
        
        # Validate file type
        mime_type = magic.from_buffer(content, mime=True)
        if mime_type not in SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported audio format: {mime_type}. "
                       f"Supported: {', '.join(SUPPORTED_FORMATS)}"
            )
        
        # Check file size (max 100MB)
        max_size = 100 * 1024 * 1024  # 100MB
        if len(content) > max_size:
            raise HTTPException(
                status_code=413,
                detail=f"Audio file too large. Max size: {max_size / (1024*1024)}MB"
            )
        
        # Save audio file
        audio_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix or ".mp3"
        audio_path = AUDIO_DIR / f"{audio_id}{file_extension}"
        
        with open(audio_path, "wb") as f:
            f.write(content)
        
        logger.info(
            f"Audio uploaded: {audio_path} ({len(content)} bytes)",
            extra={
                "audio_id": audio_id,
                "user_id": current_user.id,
                "language": language,
                "mime_type": mime_type
            }
        )
        
        # Start transcription task
        from ...tasks.audio_tasks import transcribe_audio_task
        task = transcribe_audio_task.delay(
            audio_path=str(audio_path),
            language=language,
            task_type=task_type,
            user_id=current_user.id
        )
        
        return {
            "status": "processing",
            "audio_id": audio_id,
            "task_id": task.id,
            "file_size": len(content),
            "language": language,
            "estimated_time": "30-60 seconds",
            "check_status_url": f"/api/v1/audio/status/{task.id}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio upload failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{task_id}")
async def get_transcription_status(task_id: str):
    """
    Check transcription task status.
    
    Args:
        task_id: Celery task ID
        
    Returns:
        Task status and transcription result if complete
    """
    try:
        task = celery_app.AsyncResult(task_id)
        
        if task.state == "PENDING":
            return {
                "status": "pending",
                "progress": 0,
                "message": "Transcription queued"
            }
        
        elif task.state == "STARTED":
            return {
                "status": "processing",
                "progress": 50,
                "message": "Transcribing audio..."
            }
        
        elif task.state == "SUCCESS":
            result = task.result
            return {
                "status": "completed",
                "progress": 100,
                "transcription": result.get("transcription"),
                "language": result.get("language"),
                "duration": result.get("duration"),
                "word_count": result.get("word_count"),
                "confidence": result.get("confidence", 0.0),
                "timestamp": result.get("timestamp")
            }
        
        elif task.state == "FAILURE":
            return {
                "status": "failed",
                "progress": 0,
                "error": str(task.info),
                "message": "Transcription failed. Please try again."
            }
        
        else:
            return {
                "status": task.state.lower(),
                "progress": 25,
                "message": f"Task state: {task.state}"
            }
    
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{audio_id}")
async def delete_audio(
    audio_id: str,
    current_user: TokenData = Depends(get_current_user)
):
    """
    Delete uploaded audio file.
    
    Args:
        audio_id: Audio file UUID
        
    Returns:
        Deletion confirmation
    """
    try:
        # Find audio file
        audio_files = list(AUDIO_DIR.glob(f"{audio_id}.*"))
        
        if not audio_files:
            raise HTTPException(status_code=404, detail="Audio file not found")
        
        # Delete file
        for audio_file in audio_files:
            audio_file.unlink()
            logger.info(f"Deleted audio: {audio_file}")
        
        return {
            "status": "deleted",
            "audio_id": audio_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
