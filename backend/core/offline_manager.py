"""
Offline Manager for Offline-First Architecture.

This module provides offline-first capabilities with SQLite fallback,
local caching, and network detection.
"""
import logging
import sqlite3
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class NetworkStatus(Enum):
    """Network connectivity status."""
    ONLINE = "online"
    OFFLINE = "offline"
    SLOW = "slow"
    UNKNOWN = "unknown"


class SyncStatus(Enum):
    """Data synchronization status."""
    SYNCED = "synced"
    PENDING = "pending"
    SYNCING = "syncing"
    CONFLICT = "conflict"
    ERROR = "error"


@dataclass
class OfflineConfig:
    """Configuration for offline manager."""
    sqlite_db_path: str
    cache_dir: str
    max_cache_size_mb: int
    sync_interval_seconds: int
    enable_auto_sync: bool
    conflict_resolution: str  # "server", "client", "manual"


@dataclass
class SyncItem:
    """Item to be synchronized."""
    id: str
    type: str  # "content", "progress", "settings", etc.
    data: Dict[str, Any]
    timestamp: float
    status: SyncStatus
    retry_count: int
    error_message: Optional[str] = None


class SQLiteFallback:
    """
    SQLite database fallback for offline operation.
    
    Provides local storage when PostgreSQL is unavailable.
    """
    
    def __init__(self, db_path: str = "data/cache/offline.db"):
        """
        Initialize SQLite fallback.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.conn: Optional[sqlite3.Connection] = None
        self._initialize_db()
        
        logger.info(f"SQLiteFallback initialized: {db_path}")
    
    def _initialize_db(self):
        """Initialize database schema."""
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
        # Create tables
        self.conn.executescript('''
            -- Content cache table
            CREATE TABLE IF NOT EXISTS content_cache (
                id TEXT PRIMARY KEY,
                content_type TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                accessed_at REAL NOT NULL,
                access_count INTEGER DEFAULT 0,
                size_bytes INTEGER NOT NULL,
                sync_status TEXT DEFAULT 'synced'
            );
            
            -- User progress table
            CREATE TABLE IF NOT EXISTS user_progress (
                user_id TEXT NOT NULL,
                content_id TEXT NOT NULL,
                progress_data TEXT NOT NULL,
                completed BOOLEAN DEFAULT 0,
                score REAL,
                time_spent_seconds INTEGER,
                last_updated REAL NOT NULL,
                sync_status TEXT DEFAULT 'pending',
                PRIMARY KEY (user_id, content_id)
            );
            
            -- Sync queue table
            CREATE TABLE IF NOT EXISTS sync_queue (
                id TEXT PRIMARY KEY,
                item_type TEXT NOT NULL,
                item_data TEXT NOT NULL,
                timestamp REAL NOT NULL,
                retry_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending',
                error_message TEXT
            );
            
            -- Settings table
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id TEXT PRIMARY KEY,
                settings_data TEXT NOT NULL,
                updated_at REAL NOT NULL,
                sync_status TEXT DEFAULT 'synced'
            );
            
            -- Downloaded models table
            CREATE TABLE IF NOT EXISTS downloaded_models (
                model_name TEXT PRIMARY KEY,
                model_path TEXT NOT NULL,
                size_mb REAL NOT NULL,
                version TEXT,
                downloaded_at REAL NOT NULL,
                last_used REAL
            );
            
            -- Create indexes
            CREATE INDEX IF NOT EXISTS idx_content_type ON content_cache(content_type);
            CREATE INDEX IF NOT EXISTS idx_sync_status ON content_cache(sync_status);
            CREATE INDEX IF NOT EXISTS idx_progress_sync ON user_progress(sync_status);
            CREATE INDEX IF NOT EXISTS idx_queue_status ON sync_queue(status);
        ''')
        
        self.conn.commit()
        logger.info("SQLite database initialized")
    
    def cache_content(
        self,
        content_id: str,
        content_type: str,
        data: Dict[str, Any]
    ) -> bool:
        """
        Cache content in SQLite.
        
        Args:
            content_id: Unique content identifier
            content_type: Type of content
            data: Content data
        
        Returns:
            True if successful
        """
        try:
            data_json = json.dumps(data)
            size_bytes = len(data_json.encode('utf-8'))
            now = time.time()
            
            self.conn.execute('''
                INSERT OR REPLACE INTO content_cache 
                (id, content_type, data, created_at, updated_at, accessed_at, size_bytes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (content_id, content_type, data_json, now, now, now, size_bytes))
            
            self.conn.commit()
            
            logger.debug(f"Cached content: {content_id} ({size_bytes} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cache content {content_id}: {e}")
            return False
    
    def get_cached_content(self, content_id: str) -> Optional[Dict[str, Any]]:
        """
        Get cached content from SQLite.
        
        Args:
            content_id: Content identifier
        
        Returns:
            Content data or None
        """
        try:
            cursor = self.conn.execute(
                'SELECT data, access_count FROM content_cache WHERE id = ?',
                (content_id,)
            )
            
            row = cursor.fetchone()
            
            if row:
                # Update access statistics
                self.conn.execute('''
                    UPDATE content_cache 
                    SET accessed_at = ?, access_count = ?
                    WHERE id = ?
                ''', (time.time(), row['access_count'] + 1, content_id))
                self.conn.commit()
                
                logger.debug(f"Retrieved cached content: {content_id}")
                return json.loads(row['data'])
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get cached content {content_id}: {e}")
            return None
    
    def save_progress(
        self,
        user_id: str,
        content_id: str,
        progress_data: Dict[str, Any],
        completed: bool = False,
        score: Optional[float] = None,
        time_spent: Optional[int] = None
    ) -> bool:
        """
        Save user progress.
        
        Args:
            user_id: User identifier
            content_id: Content identifier
            progress_data: Progress data
            completed: Whether content is completed
            score: Score achieved
            time_spent: Time spent in seconds
        
        Returns:
            True if successful
        """
        try:
            data_json = json.dumps(progress_data)
            
            self.conn.execute('''
                INSERT OR REPLACE INTO user_progress
                (user_id, content_id, progress_data, completed, score, 
                 time_spent_seconds, last_updated, sync_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')
            ''', (user_id, content_id, data_json, completed, score, time_spent, time.time()))
            
            self.conn.commit()
            
            logger.debug(f"Saved progress for user {user_id}, content {content_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")
            return False
    
    def get_progress(self, user_id: str, content_id: str) -> Optional[Dict[str, Any]]:
        """Get user progress."""
        try:
            cursor = self.conn.execute('''
                SELECT progress_data, completed, score, time_spent_seconds
                FROM user_progress
                WHERE user_id = ? AND content_id = ?
            ''', (user_id, content_id))
            
            row = cursor.fetchone()
            
            if row:
                return {
                    'progress_data': json.loads(row['progress_data']),
                    'completed': bool(row['completed']),
                    'score': row['score'],
                    'time_spent_seconds': row['time_spent_seconds']
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get progress: {e}")
            return None
    
    def add_to_sync_queue(self, sync_item: SyncItem) -> bool:
        """Add item to sync queue."""
        try:
            data_json = json.dumps(sync_item.data)
            
            self.conn.execute('''
                INSERT OR REPLACE INTO sync_queue
                (id, item_type, item_data, timestamp, status, retry_count)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (sync_item.id, sync_item.type, data_json, sync_item.timestamp,
                  sync_item.status.value, sync_item.retry_count))
            
            self.conn.commit()
            
            logger.debug(f"Added to sync queue: {sync_item.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add to sync queue: {e}")
            return False
    
    def get_sync_queue(self, status: Optional[SyncStatus] = None) -> List[SyncItem]:
        """Get items from sync queue."""
        try:
            if status:
                cursor = self.conn.execute(
                    'SELECT * FROM sync_queue WHERE status = ? ORDER BY timestamp',
                    (status.value,)
                )
            else:
                cursor = self.conn.execute(
                    'SELECT * FROM sync_queue ORDER BY timestamp'
                )
            
            items = []
            for row in cursor.fetchall():
                items.append(SyncItem(
                    id=row['id'],
                    type=row['item_type'],
                    data=json.loads(row['item_data']),
                    timestamp=row['timestamp'],
                    status=SyncStatus(row['status']),
                    retry_count=row['retry_count'],
                    error_message=row['error_message']
                ))
            
            return items
            
        except Exception as e:
            logger.error(f"Failed to get sync queue: {e}")
            return []
    
    def clear_sync_queue(self, status: Optional[SyncStatus] = None):
        """Clear sync queue."""
        try:
            if status:
                self.conn.execute(
                    'DELETE FROM sync_queue WHERE status = ?',
                    (status.value,)
                )
            else:
                self.conn.execute('DELETE FROM sync_queue')
            
            self.conn.commit()
            logger.info(f"Cleared sync queue (status: {status})")
            
        except Exception as e:
            logger.error(f"Failed to clear sync queue: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            cursor = self.conn.execute('''
                SELECT 
                    COUNT(*) as total_items,
                    SUM(size_bytes) as total_size_bytes,
                    COUNT(DISTINCT content_type) as content_types
                FROM content_cache
            ''')
            
            row = cursor.fetchone()
            
            return {
                'total_items': row['total_items'],
                'total_size_mb': row['total_size_bytes'] / (1024 * 1024) if row['total_size_bytes'] else 0,
                'content_types': row['content_types']
            }
            
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {}
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("SQLite connection closed")


class NetworkDetector:
    """
    Detects network connectivity and quality.
    """
    
    def __init__(self):
        """Initialize network detector."""
        self.current_status = NetworkStatus.UNKNOWN
        self.last_check_time = 0
        self.check_interval = 5  # seconds
        logger.info("NetworkDetector initialized")
    
    def check_connectivity(self) -> NetworkStatus:
        """
        Check network connectivity.
        
        Returns:
            NetworkStatus
        """
        now = time.time()
        
        # Rate limit checks
        if now - self.last_check_time < self.check_interval:
            return self.current_status
        
        self.last_check_time = now
        
        try:
            # In production, this would:
            # 1. Try to ping a known server
            # 2. Measure latency
            # 3. Test bandwidth
            
            # For now, simulate online status
            self.current_status = NetworkStatus.ONLINE
            logger.debug(f"Network status: {self.current_status.value}")
            
        except Exception as e:
            logger.warning(f"Network check failed: {e}")
            self.current_status = NetworkStatus.OFFLINE
        
        return self.current_status
    
    def is_online(self) -> bool:
        """Check if online."""
        return self.check_connectivity() == NetworkStatus.ONLINE
    
    def is_offline(self) -> bool:
        """Check if offline."""
        return self.check_connectivity() == NetworkStatus.OFFLINE


class OfflineManager:
    """
    Main offline manager coordinating all offline capabilities.
    """
    
    def __init__(self, config: OfflineConfig):
        """
        Initialize offline manager.
        
        Args:
            config: Offline configuration
        """
        self.config = config
        
        # Initialize components
        self.sqlite = SQLiteFallback(config.sqlite_db_path)
        self.network = NetworkDetector()
        
        logger.info("OfflineManager initialized")
    
    def get_content(
        self,
        content_id: str,
        fetch_func: Optional[callable] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get content with offline fallback.
        
        Args:
            content_id: Content identifier
            fetch_func: Function to fetch from server (called if online)
        
        Returns:
            Content data
        """
        # Try cache first (always)
        cached = self.sqlite.get_cached_content(content_id)
        
        if self.network.is_online() and fetch_func:
            try:
                # Fetch from server
                logger.debug(f"Fetching {content_id} from server")
                content = fetch_func(content_id)
                
                # Update cache
                if content:
                    self.sqlite.cache_content(content_id, "content", content)
                
                return content
                
            except Exception as e:
                logger.warning(f"Server fetch failed, using cache: {e}")
                return cached
        else:
            # Offline mode
            logger.debug(f"Offline mode: using cached {content_id}")
            return cached
    
    def save_progress(
        self,
        user_id: str,
        content_id: str,
        progress_data: Dict[str, Any],
        **kwargs
    ) -> bool:
        """
        Save progress with offline support.
        
        Args:
            user_id: User identifier
            content_id: Content identifier
            progress_data: Progress data
            **kwargs: Additional progress data
        
        Returns:
            True if successful
        """
        # Save to local SQLite
        success = self.sqlite.save_progress(
            user_id, content_id, progress_data, **kwargs
        )
        
        if success:
            # Add to sync queue if offline
            if self.network.is_offline():
                sync_item = SyncItem(
                    id=f"progress_{user_id}_{content_id}_{int(time.time())}",
                    type="progress",
                    data={
                        'user_id': user_id,
                        'content_id': content_id,
                        'progress_data': progress_data,
                        **kwargs
                    },
                    timestamp=time.time(),
                    status=SyncStatus.PENDING,
                    retry_count=0
                )
                self.sqlite.add_to_sync_queue(sync_item)
                logger.info(f"Progress queued for sync: {sync_item.id}")
        
        return success
    
    def get_network_status(self) -> NetworkStatus:
        """Get current network status."""
        return self.network.check_connectivity()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get offline manager statistics."""
        cache_stats = self.sqlite.get_cache_stats()
        sync_queue = self.sqlite.get_sync_queue(SyncStatus.PENDING)
        
        return {
            'network_status': self.network.current_status.value,
            'cache': cache_stats,
            'sync_queue_size': len(sync_queue),
            'database_path': str(self.sqlite.db_path)
        }
    
    def close(self):
        """Close offline manager."""
        self.sqlite.close()


if __name__ == "__main__":
    # Example usage
    config = OfflineConfig(
        sqlite_db_path="data/cache/offline.db",
        cache_dir="data/cache",
        max_cache_size_mb=500,
        sync_interval_seconds=60,
        enable_auto_sync=True,
        conflict_resolution="server"
    )
    
    manager = OfflineManager(config)
    
    print("Offline Manager Demo")
    print("=" * 60)
    
    # Check network status
    status = manager.get_network_status()
    print(f"\nNetwork Status: {status.value}")
    
    # Cache some content
    print("\n1. Caching content...")
    content = {
        'title': 'Photosynthesis',
        'subject': 'Science',
        'grade': 8,
        'content': 'Plants convert light energy into chemical energy...'
    }
    manager.sqlite.cache_content("science_8_photo", "content", content)
    
    # Retrieve cached content
    print("\n2. Retrieving cached content...")
    cached = manager.sqlite.get_cached_content("science_8_photo")
    print(f"   Title: {cached['title']}")
    
    # Save progress
    print("\n3. Saving progress...")
    manager.save_progress(
        user_id="user123",
        content_id="science_8_photo",
        progress_data={'completed_sections': 2, 'total_sections': 5},
        completed=False,
        score=75.0,
        time_spent=300
    )
    
    # Get stats
    print("\n4. Statistics:")
    print("=" * 60)
    stats = manager.get_stats()
    print(f"Network: {stats['network_status']}")
    print(f"Cache: {stats['cache']['total_items']} items "
          f"({stats['cache']['total_size_mb']:.2f}MB)")
    print(f"Sync queue: {stats['sync_queue_size']} items pending")
    
    manager.close()
