"""
Storage service for file uploads and downloads.
Supports both Local filesystem and AWS S3.
"""
import io
import os
import shutil
import mimetypes
from abc import ABC, abstractmethod
from typing import Optional, BinaryIO
from pathlib import Path
import logging

import boto3
from botocore.exceptions import ClientError, BotoCoreError

from ..core.config import settings
from ..utils.logging import get_logger

logger = get_logger(__name__)


class StorageService(ABC):
    """Abstract base class for storage services."""
    
    @abstractmethod
    def upload_file(self, file_obj: BinaryIO, key: str, content_type: Optional[str] = None) -> str:
        """Upload file and return URL/Path."""
        pass
    
    @abstractmethod
    def download_file(self, key: str, file_obj: BinaryIO) -> None:
        """Download file to file object."""
        pass
    
    @abstractmethod
    def generate_presigned_url(self, key: str, operation: str = 'get_object', expiration: int = 3600) -> str:
        """Generate access URL."""
        pass
    
    @abstractmethod
    def delete_file(self, key: str) -> bool:
        """Delete file."""
        pass


class LocalStorageService(StorageService):
    """Local filesystem storage implementation."""
    
    def __init__(self):
        self.base_path = settings.UPLOAD_DIR
        self.base_path.mkdir(parents=True, exist_ok=True)
        # Base URL for serving files (assuming served via Nginx/FastAPI static)
        self.base_url = f"{settings.API_V1_PREFIX}/static/uploads"

    def upload_file(self, file_obj: BinaryIO, key: str, content_type: Optional[str] = None) -> str:
        file_path = self.base_path / key
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Ensure file_obj is at start
        if file_obj.seekable():
            file_obj.seek(0)
            
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file_obj, f)
            
        logger.info(f"Uploaded file to local storage: {file_path}")
        return str(file_path)

    def download_file(self, key: str, file_obj: BinaryIO) -> None:
        file_path = self.base_path / key
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {key}")
            
        with open(file_path, "rb") as f:
            shutil.copyfileobj(f, file_obj)
            
    def generate_presigned_url(self, key: str, operation: str = 'get_object', expiration: int = 3600) -> str:
        # For local storage, return a direct path or a static URL
        # In production, this would be served by Nginx
        return f"{self.base_url}/{key}"

    def delete_file(self, key: str) -> bool:
        file_path = self.base_path / key
        if file_path.exists():
            os.remove(file_path)
            return True
        return False
    
    def save_temp_file(self, temp_path: Path, key: str) -> str:
        """
        Move a temporary file to permanent storage.
        
        Args:
            temp_path: Path to temporary file
            key: Storage key/filename
            
        Returns:
            Final file path
        """
        final_path = self.base_path / key
        final_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Move file (more efficient than copy)
        shutil.move(str(temp_path), str(final_path))
        
        logger.info(f"Moved temp file to storage: {final_path}")
        return str(final_path)


class S3StorageService(StorageService):
    """AWS S3 storage service."""
    
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
        self.bucket_name = settings.S3_BUCKET_NAME
    
    def upload_file(self, file_obj: BinaryIO, key: str, content_type: Optional[str] = None) -> str:
        if content_type is None:
            content_type = mimetypes.guess_type(key)[0] or 'application/octet-stream'
        
        # Ensure file_obj is at start
        if file_obj.seekable():
            file_obj.seek(0)
            
        try:
            self.s3_client.upload_fileobj(
                file_obj,
                self.bucket_name,
                key,
                ExtraArgs={'ContentType': content_type}
            )
            return f"s3://{self.bucket_name}/{key}"
        except (ClientError, BotoCoreError) as e:
            logger.error(f"S3 upload failed: {e}")
            raise
    
    def download_file(self, key: str, file_obj: BinaryIO) -> None:
        try:
            self.s3_client.download_fileobj(self.bucket_name, key, file_obj)
        except (ClientError, BotoCoreError) as e:
            logger.error(f"S3 download failed: {e}")
            raise
    
    def generate_presigned_url(self, key: str, operation: str = 'get_object', expiration: int = 3600) -> str:
        try:
            return self.s3_client.generate_presigned_url(
                operation,
                Params={'Bucket': self.bucket_name, 'Key': key},
                ExpiresIn=expiration
            )
        except (ClientError, BotoCoreError) as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            return ""

    def delete_file(self, key: str) -> bool:
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=key)
            return True
        except (ClientError, BotoCoreError) as e:
            logger.error(f"S3 delete failed: {e}")
            return False


def get_storage_service() -> StorageService:
    """Factory to get storage service based on config."""
    if settings.STORAGE_TYPE == "s3":
        return S3StorageService()
    return LocalStorageService()
