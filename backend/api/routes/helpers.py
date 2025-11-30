"""Helper functions for content processing routes to reduce complexity."""
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import aiofiles
import magic
from fastapi import HTTPException, UploadFile

from ...utils.sanitizer import InputSanitizer

logger = logging.getLogger(__name__)
sanitizer = InputSanitizer()

# Constants
WAIT_SYNC_DESCRIPTION = "Wait for result synchronously"
TASK_PROCESSING_MESSAGE = "Task is still processing"
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100MB


def parse_chunk_metadata(
    metadata: Optional[str],
    upload_id: Optional[str],
    chunk_index: Optional[int],
    total_chunks: Optional[int],
    checksum: Optional[str],
    filename: str
) -> Dict[str, Any]:
    """Parse chunked upload metadata from JSON or form fields."""
    if metadata:
        return json.loads(metadata)
    elif None not in (upload_id, chunk_index, total_chunks):
        parsed = {
            "filename": filename or "chunk.bin",
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "upload_id": upload_id
        }
        if checksum:
            parsed["checksum"] = checksum
        return parsed
    else:
        raise HTTPException(status_code=400, detail="Chunk metadata missing")


async def save_file_chunk(
    content: bytes,
    upload_path: Path,
    chunk_index: int
) -> None:
    """Save a file chunk to disk."""
    chunk_path = upload_path / f"chunk_{chunk_index}"
    async with aiofiles.open(chunk_path, 'wb') as f:
        await f.write(content)


async def reassemble_chunks(
    upload_path: Path,
    final_path: Path,
    total_chunks: int,
    filename: str
) -> None:
    """Reassemble file chunks into final file."""
    async with aiofiles.open(final_path, 'wb') as outfile:
        for i in range(total_chunks):
            chunk_path = upload_path / f"chunk_{i}"
            async with aiofiles.open(chunk_path, 'rb') as infile:
                await outfile.write(await infile.read())
    
    # Validate the reconstructed file
    async with aiofiles.open(final_path, 'rb') as validated:
        final_bytes = await validated.read()
        sanitizer.validate_file_upload(filename, final_bytes, max_size_mb=100)
    
    # Cleanup chunks
    shutil.rmtree(upload_path)


def validate_file_type(
    content: bytes,
    filename: str
) -> Tuple[str, str]:
    """
    Validate file type and return (mime_type, extension).
    
    Raises HTTPException if invalid.
    """
    file_extension = Path(filename).suffix.lower()
    allowed_extensions = ['.pdf', '.txt', '.jpeg', '.jpg', '.png']
    allowed_types = ['application/pdf', 'image/jpeg', 'image/png', 'image/jpg', 'text/plain']
    
    try:
        mime_type = magic.from_buffer(content, mime=True)
        if mime_type not in allowed_types and file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {mime_type}. Allowed: PDF, TXT, JPEG, PNG"
            )
    except Exception as e:
        logger.warning(f"File type detection issue: {e}, using extension check")
        mime_type = "unknown"
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file extension: {file_extension}. Allowed: {', '.join(allowed_extensions)}"
            )
    
    return mime_type, file_extension


def extract_text_from_content(
    content: bytes,
    file_extension: str
) -> str:
    """Extract text from file content based on file type."""
    extracted_text = ""
    
    if file_extension == '.txt':
        try:
            extracted_text = content.decode('utf-8')
        except Exception as e:
            logger.warning(f"Failed to decode text file: {e}")
            extracted_text = content.decode('utf-8', errors='ignore')
    elif file_extension == '.pdf':
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(stream=content, filetype="pdf")
            for page in doc:
                extracted_text += page.get_text()
            doc.close()
        except Exception as e:
            logger.warning(f"Failed to extract PDF text: {e}")
    
    return extracted_text


def check_task_completion(task_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check if task is complete and return appropriate response.
    
    Returns either completed result or processing status.
    """
    if task_info['state'] == 'SUCCESS':
        return {
            'task_id': task_info['task_id'],
            'state': 'SUCCESS',
            'result': task_info.get('result')
        }
    elif task_info['state'] in ['PENDING', 'STARTED', 'RETRY']:
        return {
            'task_id': task_info['task_id'],
            'state': task_info['state'],
            'progress': task_info.get('progress', 0),
            'stage': task_info.get('stage', 'initializing'),
            'message': TASK_PROCESSING_MESSAGE
        }
    elif task_info['state'] == 'FAILURE':
        error_msg = task_info.get('error', 'Task failed')
        raise HTTPException(status_code=500, detail=error_msg)
    else:
        return {
            'task_id': task_info['task_id'],
            'state': task_info['state'],
            'message': f"Task in {task_info['state']} state"
        }


def wait_for_task_result(task, timeout_seconds: int = 60) -> Dict[str, Any]:
    """
    Wait for task completion and return result.
    
    Raises HTTPException on timeout or failure.
    """
    try:
        result = task.get(timeout=timeout_seconds)
        return {
            'task_id': task.id,
            'state': 'SUCCESS',
            'result': result
        }
    except Exception as e:
        task_info = {
            'task_id': task.id,
            'state': task.state,
            'progress': getattr(task, 'info', {}).get('progress', 0) if hasattr(task, 'info') else 0,
            'stage': getattr(task, 'info', {}).get('stage', '') if hasattr(task, 'info') else '',
            'error': str(e) if task.state == 'FAILURE' else None
        }
        return check_task_completion(task_info)


async def async_wait_for_task(task, max_wait: int = 60) -> Dict[str, Any]:
    """
    Asynchronously wait for task completion.
    
    Polls task status without blocking event loop.
    """
    import asyncio
    
    poll_interval = 0.5
    elapsed = 0
    
    while elapsed < max_wait:
        task_status = task.state
        
        if task_status == 'SUCCESS':
            return {
                'result': task.result,
                'task_id': task.id,
                'status': 'completed',
                'state': 'SUCCESS'
            }
        elif task_status == 'FAILURE':
            error = str(task.info) if hasattr(task, 'info') else "Task failed"
            return {
                'error': error,
                'task_id': task.id,
                'status': 'failed',
                'state': 'FAILURE'
            }
        
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval
    
    # Timeout reached
    return {
        'task_id': task.id,
        'state': 'PENDING',
        'message': TASK_PROCESSING_MESSAGE
    }
