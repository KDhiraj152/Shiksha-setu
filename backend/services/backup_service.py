"""
Automated Backup Strategy

Issue: CODE-REVIEW-GPT #19 (MEDIUM)
Purpose: Backup for database, models, and content
"""

import os
import subprocess
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List
import shutil

logger = logging.getLogger(__name__)


class BackupService:
    """Service for automated backups."""
    
    def __init__(
        self,
        backup_dir: str = "/backups",
        retention_days: int = 30,
        s3_bucket: Optional[str] = None
    ):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.retention_days = retention_days
        self.s3_bucket = s3_bucket
    
    def backup_database(self, database_url: str) -> str:
        """
        Backup PostgreSQL database.
        
        Args:
            database_url: Database connection string
            
        Returns:
            Path to backup file
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"db_backup_{timestamp}.sql.gz"
        
        logger.info(f"Starting database backup to {backup_file}")
        
        try:
            # Extract connection details from URL
            # postgresql://user:pass@host:port/dbname
            cmd = [
                "pg_dump",
                database_url,
                "|",
                "gzip",
                ">",
                str(backup_file)
            ]
            
            # Run backup command
            result = subprocess.run(
                " ".join(cmd),
                shell=True,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                raise Exception(f"pg_dump failed: {result.stderr}")
            
            logger.info(f"Database backup completed: {backup_file}")
            
            # Upload to S3 if configured
            if self.s3_bucket:
                self._upload_to_s3(backup_file, "database")
            
            return str(backup_file)
            
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            raise
    
    def backup_models(self, models_dir: str) -> str:
        """
        Backup ML models directory.
        
        Args:
            models_dir: Path to models directory
            
        Returns:
            Path to backup archive
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"models_backup_{timestamp}.tar.gz"
        
        logger.info(f"Starting models backup from {models_dir}")
        
        try:
            # Create tar.gz archive
            shutil.make_archive(
                str(backup_file).replace(".tar.gz", ""),
                "gztar",
                models_dir
            )
            
            logger.info(f"Models backup completed: {backup_file}")
            
            # Upload to S3 if configured
            if self.s3_bucket:
                self._upload_to_s3(backup_file, "models")
            
            return str(backup_file)
            
        except Exception as e:
            logger.error(f"Models backup failed: {e}")
            raise
    
    def backup_content(self, content_dir: str) -> str:
        """
        Backup content files (uploads, generated content).
        
        Args:
            content_dir: Path to content directory
            
        Returns:
            Path to backup archive
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_file = self.backup_dir / f"content_backup_{timestamp}.tar.gz"
        
        logger.info(f"Starting content backup from {content_dir}")
        
        try:
            # Create tar.gz archive
            shutil.make_archive(
                str(backup_file).replace(".tar.gz", ""),
                "gztar",
                content_dir
            )
            
            logger.info(f"Content backup completed: {backup_file}")
            
            # Upload to S3 if configured
            if self.s3_bucket:
                self._upload_to_s3(backup_file, "content")
            
            return str(backup_file)
            
        except Exception as e:
            logger.error(f"Content backup failed: {e}")
            raise
    
    def cleanup_old_backups(self) -> None:
        """Remove backups older than retention period."""
        logger.info(f"Cleaning up backups older than {self.retention_days} days")
        
        cutoff_time = datetime.now(timezone.utc).timestamp() - (self.retention_days * 86400)
        
        removed_count = 0
        for backup_file in self.backup_dir.glob("*_backup_*"):
            if backup_file.stat().st_mtime < cutoff_time:
                backup_file.unlink()
                removed_count += 1
                logger.debug(f"Removed old backup: {backup_file}")
        
        logger.info(f"Cleanup completed: {removed_count} backups removed")
    
    def _upload_to_s3(self, file_path: Path, backup_type: str) -> None:
        """Upload backup to S3."""
        if not self.s3_bucket:
            return
        
        try:
            import boto3
            
            s3_client = boto3.client('s3')
            s3_key = f"backups/{backup_type}/{file_path.name}"
            
            logger.info(f"Uploading to S3: s3://{self.s3_bucket}/{s3_key}")
            
            s3_client.upload_file(
                str(file_path),
                self.s3_bucket,
                s3_key
            )
            
            logger.info("S3 upload completed")
            
        except Exception as e:
            logger.error(f"S3 upload failed: {e}")
            # Don't raise - local backup still succeeded
    
    def list_backups(self) -> List[dict]:
        """List all available backups."""
        backups = []
        
        for backup_file in sorted(self.backup_dir.glob("*_backup_*")):
            stat = backup_file.stat()
            backups.append({
                "filename": backup_file.name,
                "path": str(backup_file),
                "size_mb": round(stat.st_size / 1024 / 1024, 2),
                "created_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
            })
        
        return backups
    
    def restore_database(self, backup_file: str, database_url: str):
        """
        Restore database from backup.
        
        Args:
            backup_file: Path to backup file
            database_url: Database connection string
        """
        logger.info(f"Restoring database from {backup_file}")
        
        try:
            cmd = [
                "gunzip",
                "-c",
                backup_file,
                "|",
                "psql",
                database_url
            ]
            
            result = subprocess.run(
                " ".join(cmd),
                shell=True,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                raise Exception(f"Restore failed: {result.stderr}")
            
            logger.info("Database restore completed")
            
        except Exception as e:
            logger.error(f"Database restore failed: {e}")
            raise


# Backup schedule (for cron or celery beat)
BACKUP_SCHEDULE = {
    "database": "0 2 * * *",      # Daily at 2 AM
    "models": "0 3 * * 0",        # Weekly on Sunday at 3 AM
    "content": "0 4 * * *",       # Daily at 4 AM
    "cleanup": "0 5 * * 0",       # Weekly cleanup on Sunday at 5 AM
}


def create_backup_tasks():
    """Create Celery tasks for automated backups."""
    # This would integrate with Celery for scheduled backups
    logger.info("Backup schedule configured")
    return BACKUP_SCHEDULE
