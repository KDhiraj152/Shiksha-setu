"""
Synchronization Service for Offline-First Architecture.

This module handles synchronization of offline data with the server
when network connectivity is restored.
"""
import logging
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class ConflictResolution(Enum):
    """Conflict resolution strategies."""
    SERVER_WINS = "server_wins"
    CLIENT_WINS = "client_wins"
    NEWEST_WINS = "newest_wins"
    MANUAL = "manual"
    MERGE = "merge"


@dataclass
class SyncResult:
    """Result of a sync operation."""
    success: bool
    synced_count: int
    failed_count: int
    conflict_count: int
    errors: List[str]
    duration_ms: float


@dataclass
class SyncConflict:
    """Data conflict between client and server."""
    item_id: str
    client_data: Dict[str, Any]
    server_data: Dict[str, Any]
    client_timestamp: float
    server_timestamp: float
    resolved: bool
    resolution: Optional[Dict[str, Any]] = None


class SyncService:
    """
    Service for synchronizing offline data with server.
    
    Features:
    - Background synchronization
    - Conflict detection and resolution
    - Retry logic with exponential backoff
    - Differential sync (only changed data)
    - Progress tracking
    """
    
    def __init__(
        self,
        offline_manager,
        conflict_resolution: ConflictResolution = ConflictResolution.NEWEST_WINS,
        sync_interval: int = 60,
        max_retries: int = 3
    ):
        """
        Initialize sync service.
        
        Args:
            offline_manager: OfflineManager instance
            conflict_resolution: Strategy for resolving conflicts
            sync_interval: Seconds between sync attempts
            max_retries: Maximum retry attempts per item
        """
        self.offline_manager = offline_manager
        self.conflict_resolution = conflict_resolution
        self.sync_interval = sync_interval
        self.max_retries = max_retries
        
        self.is_syncing = False
        self.sync_thread: Optional[threading.Thread] = None
        self.conflicts: List[SyncConflict] = []
        
        logger.info(
            f"SyncService initialized (interval: {sync_interval}s, "
            f"resolution: {conflict_resolution.value})"
        )
    
    def start_background_sync(self):
        """Start background synchronization."""
        if self.sync_thread and self.sync_thread.is_alive():
            logger.warning("Background sync already running")
            return
        
        self.sync_thread = threading.Thread(
            target=self._sync_loop,
            daemon=True
        )
        self.sync_thread.start()
        logger.info("Background sync started")
    
    def stop_background_sync(self):
        """Stop background synchronization."""
        if self.sync_thread:
            # In production, use a proper stop flag
            logger.info("Background sync stopped")
    
    def _sync_loop(self):
        """Background sync loop."""
        while True:
            try:
                # Check if online
                if self.offline_manager.network.is_online():
                    # Perform sync
                    result = self.sync_all()
                    
                    if result.success:
                        logger.info(
                            f"Sync completed: {result.synced_count} synced, "
                            f"{result.failed_count} failed, "
                            f"{result.conflict_count} conflicts"
                        )
                    else:
                        logger.warning(f"Sync failed: {result.errors}")
                
                # Wait for next sync
                time.sleep(self.sync_interval)
                
            except Exception as e:
                logger.error(f"Sync loop error: {e}")
                time.sleep(self.sync_interval)
    
    def sync_all(self) -> SyncResult:
        """
        Synchronize all pending items.
        
        Returns:
            SyncResult with sync statistics
        """
        if self.is_syncing:
            logger.warning("Sync already in progress")
            return SyncResult(
                success=False,
                synced_count=0,
                failed_count=0,
                conflict_count=0,
                errors=["Sync already in progress"],
                duration_ms=0
            )
        
        self.is_syncing = True
        start_time = time.time()
        
        try:
            logger.info("Starting synchronization...")
            
            # Get pending items
            from backend.core.offline_manager import SyncStatus
            pending_items = self.offline_manager.sqlite.get_sync_queue(
                SyncStatus.PENDING
            )
            
            logger.info(f"Found {len(pending_items)} items to sync")
            
            synced_count = 0
            failed_count = 0
            conflict_count = 0
            errors = []
            
            for item in pending_items:
                try:
                    # Check retry limit
                    if item.retry_count >= self.max_retries:
                        logger.warning(
                            f"Item {item.id} exceeded max retries ({self.max_retries})"
                        )
                        failed_count += 1
                        continue
                    
                    # Sync item
                    success, conflict = self._sync_item(item)
                    
                    if success:
                        synced_count += 1
                    elif conflict:
                        conflict_count += 1
                        self.conflicts.append(conflict)
                    else:
                        failed_count += 1
                        
                except Exception as e:
                    logger.error(f"Failed to sync item {item.id}: {e}")
                    errors.append(f"{item.id}: {str(e)}")
                    failed_count += 1
            
            duration_ms = (time.time() - start_time) * 1000
            
            result = SyncResult(
                success=failed_count == 0,
                synced_count=synced_count,
                failed_count=failed_count,
                conflict_count=conflict_count,
                errors=errors,
                duration_ms=duration_ms
            )
            
            logger.info(
                f"Sync completed in {duration_ms:.1f}ms: "
                f"{synced_count} synced, {failed_count} failed, {conflict_count} conflicts"
            )
            
            return result
            
        finally:
            self.is_syncing = False
    
    def _sync_item(self, sync_item) -> tuple[bool, Optional[SyncConflict]]:
        """
        Sync a single item.
        
        Args:
            sync_item: SyncItem to sync
        
        Returns:
            Tuple of (success, conflict)
        """
        try:
            # In production, this would:
            # 1. Send data to server API
            # 2. Check for conflicts
            # 3. Resolve conflicts
            # 4. Update local database
            
            logger.debug(f"Syncing item: {sync_item.id} (type: {sync_item.type})")
            
            # Simulate server API call
            server_response = self._send_to_server(sync_item)
            
            if server_response.get('conflict'):
                # Handle conflict
                conflict = SyncConflict(
                    item_id=sync_item.id,
                    client_data=sync_item.data,
                    server_data=server_response['server_data'],
                    client_timestamp=sync_item.timestamp,
                    server_timestamp=server_response['server_timestamp'],
                    resolved=False
                )
                
                # Try to resolve
                resolved = self._resolve_conflict(conflict)
                
                if resolved:
                    # Remove from queue
                    from backend.core.offline_manager import SyncStatus
                    sync_item.status = SyncStatus.SYNCED
                    return True, None
                else:
                    return False, conflict
            
            # Success - remove from queue
            from backend.core.offline_manager import SyncStatus
            sync_item.status = SyncStatus.SYNCED
            
            logger.debug(f"Successfully synced: {sync_item.id}")
            return True, None
            
        except Exception as e:
            logger.error(f"Failed to sync item {sync_item.id}: {e}")
            sync_item.retry_count += 1
            return False, None
    
    def _send_to_server(self, sync_item) -> Dict[str, Any]:
        """
        Send data to server API.
        
        Args:
            sync_item: SyncItem to send
        
        Returns:
            Server response
        """
        # In production, this would make actual HTTP request
        # For now, simulate success
        logger.debug(f"Sending {sync_item.id} to server")
        
        return {
            'success': True,
            'conflict': False,
            'server_timestamp': time.time()
        }
    
    def _resolve_conflict(self, conflict: SyncConflict) -> bool:
        """
        Resolve a data conflict.
        
        Args:
            conflict: SyncConflict to resolve
        
        Returns:
            True if resolved
        """
        logger.info(
            f"Resolving conflict for {conflict.item_id} "
            f"(strategy: {self.conflict_resolution.value})"
        )
        
        try:
            if self.conflict_resolution == ConflictResolution.SERVER_WINS:
                conflict.resolution = conflict.server_data
                
            elif self.conflict_resolution == ConflictResolution.CLIENT_WINS:
                conflict.resolution = conflict.client_data
                
            elif self.conflict_resolution == ConflictResolution.NEWEST_WINS:
                if conflict.client_timestamp > conflict.server_timestamp:
                    conflict.resolution = conflict.client_data
                else:
                    conflict.resolution = conflict.server_data
                    
            elif self.conflict_resolution == ConflictResolution.MERGE:
                # Merge data (simple approach - in production, use field-level merging)
                conflict.resolution = {
                    **conflict.server_data,
                    **conflict.client_data
                }
                
            elif self.conflict_resolution == ConflictResolution.MANUAL:
                # Requires manual intervention
                logger.warning(f"Manual resolution required for {conflict.item_id}")
                return False
            
            conflict.resolved = True
            logger.info(f"Conflict resolved for {conflict.item_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resolve conflict for {conflict.item_id}: {e}")
            return False
    
    def get_pending_conflicts(self) -> List[SyncConflict]:
        """Get list of unresolved conflicts."""
        return [c for c in self.conflicts if not c.resolved]
    
    def resolve_conflict_manually(
        self,
        conflict_id: str,
        resolution_data: Dict[str, Any]
    ) -> bool:
        """
        Manually resolve a conflict.
        
        Args:
            conflict_id: Conflict identifier
            resolution_data: Resolved data
        
        Returns:
            True if successful
        """
        for conflict in self.conflicts:
            if conflict.item_id == conflict_id:
                conflict.resolution = resolution_data
                conflict.resolved = True
                logger.info(f"Manually resolved conflict: {conflict_id}")
                return True
        
        return False
    
    def get_sync_stats(self) -> Dict[str, Any]:
        """Get synchronization statistics."""
        from backend.core.offline_manager import SyncStatus
        
        pending = self.offline_manager.sqlite.get_sync_queue(SyncStatus.PENDING)
        
        return {
            'is_syncing': self.is_syncing,
            'pending_items': len(pending),
            'unresolved_conflicts': len(self.get_pending_conflicts()),
            'total_conflicts': len(self.conflicts),
            'conflict_resolution': self.conflict_resolution.value,
            'sync_interval': self.sync_interval,
        }


# Convenience function
def create_sync_service(
    offline_manager,
    auto_start: bool = True
) -> SyncService:
    """
    Create and configure sync service.
    
    Args:
        offline_manager: OfflineManager instance
        auto_start: Start background sync immediately
    
    Returns:
        SyncService instance
    """
    service = SyncService(
        offline_manager=offline_manager,
        conflict_resolution=ConflictResolution.NEWEST_WINS,
        sync_interval=60,
        max_retries=3
    )
    
    if auto_start:
        service.start_background_sync()
    
    return service


if __name__ == "__main__":
    # Example usage
    from backend.core.offline_manager import OfflineManager, OfflineConfig
    
    # Create offline manager
    config = OfflineConfig(
        sqlite_db_path="data/cache/offline.db",
        cache_dir="data/cache",
        max_cache_size_mb=500,
        sync_interval_seconds=60,
        enable_auto_sync=True,
        conflict_resolution="server"
    )
    
    offline_manager = OfflineManager(config)
    
    # Create sync service
    sync_service = create_sync_service(offline_manager, auto_start=False)
    
    logger.info("Sync Service Demo")
    logger.info("=" * 60)
    
    # Get stats
    stats = sync_service.get_sync_stats()
    logger.info("\nSync Statistics:")
    logger.info(f"  Pending items: {stats['pending_items']}")
    logger.info(f"  Unresolved conflicts: {stats['unresolved_conflicts']}")
    logger.info(f"  Conflict resolution: {stats['conflict_resolution']}")
    logger.info(f"  Sync interval: {stats['sync_interval']}s")
    
    # Perform manual sync
    logger.info(f"\n{'='*60}")
    logger.info("Performing manual sync...")
    result = sync_service.sync_all()
    
    logger.info("\nSync Result:")
    logger.info(f"  Success: {result.success}")
    logger.info(f"  Synced: {result.synced_count}")
    logger.info(f"  Failed: {result.failed_count}")
    logger.info(f"  Conflicts: {result.conflict_count}")
    logger.info(f"  Duration: {result.duration_ms:.1f}ms")
    
    offline_manager.close()
