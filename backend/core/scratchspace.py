"""
Scratchspace Manager (Principle R)
===================================
Manage temporary files with /tmp/ssetu/<task_id>/ structure.

Strategy:
- Isolated scratchspace per task
- Automatic cleanup on task completion
- Memory-mapped file support for large data
- Secure temporary file handling

Reference: "/tmp/ssetu/<task_id>/"
"""

import asyncio
import logging
import os
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import mmap

logger = logging.getLogger(__name__)


@dataclass
class ScratchspaceConfig:
    """Configuration for scratchspace management."""
    # Base directory (Principle R: /tmp/ssetu/)
    base_dir: str = "/tmp/ssetu"
    
    # Cleanup settings
    auto_cleanup: bool = True
    cleanup_on_error: bool = True
    max_age_seconds: int = 3600  # 1 hour max lifetime
    
    # Size limits
    max_task_size_mb: int = 1024  # 1GB per task
    max_total_size_mb: int = 4096  # 4GB total
    
    # Permissions
    dir_mode: int = 0o700
    file_mode: int = 0o600


@dataclass
class TaskScratchspace:
    """Scratchspace for a single task."""
    task_id: str
    path: Path
    created_at: float = field(default_factory=time.time)
    files: Dict[str, Path] = field(default_factory=dict)
    size_bytes: int = 0
    
    def get_file_path(self, name: str) -> Path:
        """Get path for a named file in scratchspace."""
        return self.path / name
    
    def add_file(self, name: str, path: Path, size: int = 0):
        """Register a file in the scratchspace."""
        self.files[name] = path
        self.size_bytes += size


class ScratchspaceManager:
    """
    Manages temporary scratchspace directories for tasks.
    
    Features:
    - Isolated directory per task (/tmp/ssetu/<task_id>/)
    - Automatic cleanup on task completion
    - Memory-mapped file support
    - Size tracking and limits
    """
    
    def __init__(self, config: Optional[ScratchspaceConfig] = None):
        self.config = config or ScratchspaceConfig()
        self._tasks: Dict[str, TaskScratchspace] = {}
        self._lock = asyncio.Lock()
        self._total_size_bytes = 0
        
        # Ensure base directory exists
        self._ensure_base_dir()
    
    def _ensure_base_dir(self):
        """Create base scratchspace directory."""
        base = Path(self.config.base_dir)
        base.mkdir(parents=True, exist_ok=True, mode=self.config.dir_mode)
        logger.info(f"Scratchspace base directory: {base}")
    
    def _generate_task_id(self) -> str:
        """Generate unique task ID."""
        return f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
    
    async def create(self, task_id: Optional[str] = None) -> TaskScratchspace:
        """
        Create scratchspace for a task.
        
        Args:
            task_id: Optional custom task ID
            
        Returns:
            TaskScratchspace instance
        """
        async with self._lock:
            # Generate ID if not provided
            if task_id is None:
                task_id = self._generate_task_id()
            
            # Check if already exists
            if task_id in self._tasks:
                return self._tasks[task_id]
            
            # Create directory
            task_path = Path(self.config.base_dir) / task_id
            task_path.mkdir(parents=True, exist_ok=True, mode=self.config.dir_mode)
            
            # Create scratchspace record
            scratch = TaskScratchspace(
                task_id=task_id,
                path=task_path
            )
            
            self._tasks[task_id] = scratch
            logger.debug(f"Created scratchspace: {task_path}")
            
            return scratch
    
    async def cleanup(self, task_id: str) -> bool:
        """
        Clean up scratchspace for a task.
        
        Args:
            task_id: Task ID to clean up
            
        Returns:
            True if cleanup successful
        """
        async with self._lock:
            if task_id not in self._tasks:
                return False
            
            scratch = self._tasks[task_id]
            
            try:
                # Remove directory and contents
                if scratch.path.exists():
                    shutil.rmtree(scratch.path)
                
                # Update tracking
                self._total_size_bytes -= scratch.size_bytes
                del self._tasks[task_id]
                
                logger.debug(f"Cleaned up scratchspace: {scratch.path}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to cleanup scratchspace {task_id}: {e}")
                return False
    
    async def cleanup_expired(self) -> int:
        """
        Clean up expired scratchspaces.
        
        Returns:
            Number of scratchspaces cleaned up
        """
        cleaned = 0
        now = time.time()
        max_age = self.config.max_age_seconds
        
        # Find expired tasks
        expired = [
            task_id for task_id, scratch in self._tasks.items()
            if (now - scratch.created_at) > max_age
        ]
        
        # Clean them up
        for task_id in expired:
            if await self.cleanup(task_id):
                cleaned += 1
        
        if cleaned > 0:
            logger.info(f"Cleaned up {cleaned} expired scratchspaces")
        
        return cleaned
    
    def get(self, task_id: str) -> Optional[TaskScratchspace]:
        """Get scratchspace for task ID."""
        return self._tasks.get(task_id)
    
    async def write_file(
        self,
        task_id: str,
        name: str,
        data: Union[bytes, str],
        mode: str = "wb"
    ) -> Optional[Path]:
        """
        Write data to a file in task's scratchspace.
        
        Args:
            task_id: Task ID
            name: Filename
            data: Data to write
            mode: File mode ('wb' for bytes, 'w' for text)
            
        Returns:
            Path to written file or None on error
        """
        scratch = self.get(task_id)
        if not scratch:
            scratch = await self.create(task_id)
        
        file_path = scratch.get_file_path(name)
        
        try:
            # Convert string to bytes if needed
            if isinstance(data, str) and 'b' in mode:
                data = data.encode('utf-8')
            
            # Write with secure permissions
            fd = os.open(
                str(file_path),
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
                self.config.file_mode
            )
            
            with os.fdopen(fd, mode) as f:
                f.write(data)
            
            # Update tracking
            size = file_path.stat().st_size
            scratch.add_file(name, file_path, size)
            self._total_size_bytes += size
            
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to write file {name} in {task_id}: {e}")
            return None
    
    async def read_file(
        self,
        task_id: str,
        name: str,
        mode: str = "rb"
    ) -> Optional[Union[bytes, str]]:
        """
        Read file from task's scratchspace.
        
        Args:
            task_id: Task ID
            name: Filename
            mode: File mode
            
        Returns:
            File contents or None on error
        """
        scratch = self.get(task_id)
        if not scratch:
            return None
        
        file_path = scratch.get_file_path(name)
        
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, mode) as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read file {name} in {task_id}: {e}")
            return None
    
    @contextmanager
    def memory_mapped_file(
        self,
        task_id: str,
        name: str,
        size: int = 0,
        access: int = mmap.ACCESS_WRITE
    ):
        """
        Create or access a memory-mapped file.
        
        Useful for large data processing without loading into memory.
        
        Args:
            task_id: Task ID
            name: Filename
            size: Size for new file (0 for existing)
            access: mmap access mode
            
        Yields:
            mmap object
        """
        scratch = self.get(task_id)
        if not scratch:
            raise ValueError(f"Unknown task ID: {task_id}")
        
        file_path = scratch.get_file_path(name)
        
        # Create file if needed
        if size > 0 and not file_path.exists():
            with open(file_path, 'wb') as f:
                f.seek(size - 1)
                f.write(b'\0')
            scratch.add_file(name, file_path, size)
        
        # Memory map the file
        with open(file_path, 'r+b') as f:
            mm = mmap.mmap(f.fileno(), 0, access=access)
            try:
                yield mm
            finally:
                mm.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get scratchspace statistics."""
        return {
            "active_tasks": len(self._tasks),
            "total_size_mb": self._total_size_bytes / (1024 * 1024),
            "max_size_mb": self.config.max_total_size_mb,
            "base_dir": self.config.base_dir,
            "tasks": {
                task_id: {
                    "path": str(scratch.path),
                    "files": len(scratch.files),
                    "size_mb": scratch.size_bytes / (1024 * 1024),
                    "age_seconds": time.time() - scratch.created_at
                }
                for task_id, scratch in self._tasks.items()
            }
        }


# Global manager instance
_manager: Optional[ScratchspaceManager] = None


def get_scratchspace_manager() -> ScratchspaceManager:
    """Get or create global scratchspace manager."""
    global _manager
    if _manager is None:
        _manager = ScratchspaceManager()
    return _manager


@asynccontextmanager
async def task_scratchspace(task_id: Optional[str] = None):
    """
    Context manager for task scratchspace with automatic cleanup.
    
    Usage:
        async with task_scratchspace() as scratch:
            await scratch.write_file("data.json", json.dumps(data))
            # ... do work ...
        # Automatic cleanup on exit
    
    Args:
        task_id: Optional custom task ID
        
    Yields:
        TaskScratchspaceHelper with convenience methods
    """
    manager = get_scratchspace_manager()
    scratch = await manager.create(task_id)
    
    helper = TaskScratchspaceHelper(scratch, manager)
    
    try:
        yield helper
    finally:
        if manager.config.auto_cleanup:
            await manager.cleanup(scratch.task_id)


class TaskScratchspaceHelper:
    """Helper class for task scratchspace operations."""
    
    def __init__(self, scratch: TaskScratchspace, manager: ScratchspaceManager):
        self.scratch = scratch
        self.manager = manager
        self.task_id = scratch.task_id
        self.path = scratch.path
    
    async def write(self, name: str, data: Union[bytes, str]) -> Optional[Path]:
        """Write file to scratchspace."""
        mode = "wb" if isinstance(data, bytes) else "w"
        return await self.manager.write_file(self.task_id, name, data, mode)
    
    async def read(self, name: str, binary: bool = True) -> Optional[Union[bytes, str]]:
        """Read file from scratchspace."""
        mode = "rb" if binary else "r"
        return await self.manager.read_file(self.task_id, name, mode)
    
    def get_path(self, name: str) -> Path:
        """Get path to file in scratchspace."""
        return self.scratch.get_file_path(name)
    
    def memory_map(self, name: str, size: int = 0):
        """Create memory-mapped file."""
        return self.manager.memory_mapped_file(self.task_id, name, size)


# Cleanup task for periodic expiration
async def cleanup_task(interval_seconds: int = 300):
    """
    Background task to clean up expired scratchspaces.
    
    Args:
        interval_seconds: How often to run cleanup
    """
    manager = get_scratchspace_manager()
    
    while True:
        try:
            await manager.cleanup_expired()
        except Exception as e:
            logger.error(f"Scratchspace cleanup error: {e}")
        
        await asyncio.sleep(interval_seconds)
