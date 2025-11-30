"""
Streaming utilities for memory-efficient file handling.

Optimized for Apple Silicon M4 with limited unified memory.
"""

import asyncio
import hashlib
import tempfile
from pathlib import Path
from typing import AsyncIterator, Optional, Tuple, BinaryIO
import aiofiles
from fastapi import UploadFile


# Chunk size for streaming (64KB optimal for M4 memory bandwidth)
STREAM_CHUNK_SIZE = 64 * 1024

# Maximum file size to keep in memory (5MB)
MEMORY_THRESHOLD = 5 * 1024 * 1024


async def stream_to_tempfile(
    file: UploadFile,
    max_size: int = 100 * 1024 * 1024
) -> Tuple[Path, int, str]:
    """
    Stream uploaded file to a temporary file.
    
    Returns:
        Tuple of (temp_path, file_size, content_hash)
        
    Raises:
        ValueError: If file exceeds max_size
    """
    hasher = hashlib.sha256()
    total_size = 0
    
    # Create temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "upload").suffix) as tmp:
        tmp_path = Path(tmp.name)
        
        async for chunk in stream_upload(file):
            total_size += len(chunk)
            if total_size > max_size:
                # Clean up and raise
                tmp_path.unlink(missing_ok=True)
                raise ValueError(f"File exceeds maximum size of {max_size / 1024 / 1024:.0f}MB")
            
            hasher.update(chunk)
            tmp.write(chunk)
    
    return tmp_path, total_size, hasher.hexdigest()


async def stream_upload(file: UploadFile) -> AsyncIterator[bytes]:
    """
    Async generator to stream file upload in chunks.
    
    Yields:
        Chunks of bytes from the uploaded file
    """
    while True:
        chunk = await file.read(STREAM_CHUNK_SIZE)
        if not chunk:
            break
        yield chunk
    
    # Reset file position for potential re-read
    await file.seek(0)


async def stream_file_to_storage(
    file: UploadFile,
    dest_path: Path,
    max_size: int = 100 * 1024 * 1024
) -> Tuple[int, str]:
    """
    Stream uploaded file directly to storage destination.
    
    Returns:
        Tuple of (file_size, content_hash)
        
    Raises:
        ValueError: If file exceeds max_size
    """
    hasher = hashlib.sha256()
    total_size = 0
    
    # Ensure parent directory exists
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    async with aiofiles.open(dest_path, 'wb') as f:
        async for chunk in stream_upload(file):
            total_size += len(chunk)
            if total_size > max_size:
                # Clean up and raise
                dest_path.unlink(missing_ok=True)
                raise ValueError(f"File exceeds maximum size of {max_size / 1024 / 1024:.0f}MB")
            
            hasher.update(chunk)
            await f.write(chunk)
    
    return total_size, hasher.hexdigest()


async def read_file_in_chunks(
    file_path: Path,
    chunk_size: int = STREAM_CHUNK_SIZE
) -> AsyncIterator[bytes]:
    """
    Read a file in chunks asynchronously.
    
    Yields:
        Chunks of bytes from the file
    """
    async with aiofiles.open(file_path, 'rb') as f:
        while True:
            chunk = await f.read(chunk_size)
            if not chunk:
                break
            yield chunk


async def compute_file_hash(file_path: Path) -> str:
    """
    Compute SHA256 hash of a file without loading it fully into memory.
    """
    hasher = hashlib.sha256()
    async for chunk in read_file_in_chunks(file_path):
        hasher.update(chunk)
    return hasher.hexdigest()


class SmartFileBuffer:
    """
    Smart file buffer that uses memory for small files, disk for large ones.
    
    Automatically switches to disk-based storage when content exceeds
    MEMORY_THRESHOLD.
    """
    
    def __init__(self, threshold: int = MEMORY_THRESHOLD):
        self.threshold = threshold
        self._buffer = bytearray()
        self._temp_file: Optional[BinaryIO] = None
        self._temp_path: Optional[Path] = None
        self._size = 0
        self._hash = hashlib.sha256()
        self._in_memory = True
    
    async def write(self, data: bytes) -> None:
        """Write data to buffer, spilling to disk if needed."""
        self._size += len(data)
        self._hash.update(data)
        
        if self._in_memory:
            if self._size <= self.threshold:
                self._buffer.extend(data)
            else:
                # Spill to disk
                self._temp_file = tempfile.NamedTemporaryFile(delete=False)
                self._temp_path = Path(self._temp_file.name)
                self._temp_file.write(bytes(self._buffer))
                self._temp_file.write(data)
                self._buffer.clear()
                self._in_memory = False
        else:
            self._temp_file.write(data)
    
    async def read_all(self) -> bytes:
        """Read entire content (only safe for small files)."""
        if self._in_memory:
            return bytes(self._buffer)
        else:
            self._temp_file.seek(0)
            return self._temp_file.read()
    
    def get_path(self) -> Optional[Path]:
        """Get path to temp file if spilled to disk."""
        return self._temp_path
    
    @property
    def size(self) -> int:
        return self._size
    
    @property
    def hash(self) -> str:
        return self._hash.hexdigest()
    
    @property
    def is_in_memory(self) -> bool:
        return self._in_memory
    
    def cleanup(self) -> None:
        """Clean up temporary file if created."""
        if self._temp_file:
            self._temp_file.close()
        if self._temp_path and self._temp_path.exists():
            self._temp_path.unlink()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.cleanup()


async def smart_upload_buffer(
    file: UploadFile,
    max_size: int = 100 * 1024 * 1024
) -> SmartFileBuffer:
    """
    Stream upload into a SmartFileBuffer.
    
    Small files stay in memory, large files spill to disk.
    
    Returns:
        SmartFileBuffer with the uploaded content
        
    Raises:
        ValueError: If file exceeds max_size
    """
    buffer = SmartFileBuffer()
    
    async for chunk in stream_upload(file):
        if buffer.size + len(chunk) > max_size:
            buffer.cleanup()
            raise ValueError(f"File exceeds maximum size of {max_size / 1024 / 1024:.0f}MB")
        await buffer.write(chunk)
    
    return buffer
