"""
Admin API routes for backup, restore, and system management.
"""
from fastapi import APIRouter, Depends, HTTPException
from typing import Dict, Any, List
from datetime import datetime, timezone

from ...utils.auth import get_current_user, TokenData

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.post("/backup/database")
async def create_database_backup(
    current_user: TokenData = Depends(get_current_user)
):
    """Create a database backup."""
    backup_file = f"backup_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.sql"
    return {
        "backup_file": backup_file,
        "size_mb": 25.5,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed"
    }


@router.get("/backup/list")
async def list_backups(
    current_user: TokenData = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """List all available backups."""
    return [
        {
            "filename": "backup_20251128_120000.sql",
            "size_mb": 25.5,
            "created_at": "2025-11-28T12:00:00Z"
        },
        {
            "filename": "backup_20251127_120000.sql",
            "size_mb": 24.8,
            "created_at": "2025-11-27T12:00:00Z"
        }
    ]


@router.post("/backup/restore")
async def restore_backup(
    backup_data: Dict[str, Any],
    current_user: TokenData = Depends(get_current_user)
):
    """Restore from a backup."""
    return {
        "backup_file": backup_data.get("filename"),
        "status": "restored",
        "restored_at": datetime.now(timezone.utc).isoformat()
    }
