/**
 * Offline Sync Manager
 * 
 * Handles synchronization of offline data when network is restored.
 */

import { storageManager, type ProgressItem } from './storage-manager';

interface SyncConfig {
  apiBaseUrl: string;
  syncInterval: number; // milliseconds
  maxRetries: number;
  retryDelay: number; // milliseconds
}

interface SyncStatus {
  isOnline: boolean;
  isSyncing: boolean;
  lastSync: number | null;
  pendingCount: number;
  failedCount: number;
}

export class SyncManager {
  private readonly config: SyncConfig;
  private readonly status: SyncStatus;
  private syncInterval: number | null = null;
  
  constructor(config?: Partial<SyncConfig>) {
    this.config = {
      apiBaseUrl: '/api',
      syncInterval: 60000, // 1 minute
      maxRetries: 3,
      retryDelay: 5000,
      ...config
    };
    
    this.status = {
      isOnline: navigator.onLine,
      isSyncing: false,
      lastSync: null,
      pendingCount: 0,
      failedCount: 0
    };
    
    this.setupNetworkListeners();
  }
  
  /**
   * Setup network event listeners
   */
  private setupNetworkListeners(): void {
    globalThis.addEventListener('online', () => {
      console.log('[SyncManager] Network online');
      this.status.isOnline = true;
      this.syncNow();
    });
    
    globalThis.addEventListener('offline', () => {
      console.log('[SyncManager] Network offline');
      this.status.isOnline = false;
    });
  }
  
  /**
   * Start automatic sync
   */
  startAutoSync(): void {
    if (this.syncInterval) {
      console.warn('[SyncManager] Auto-sync already running');
      return;
    }
    
    console.log('[SyncManager] Starting auto-sync');
    
    this.syncInterval = globalThis.setInterval(() => {
      if (this.status.isOnline && !this.status.isSyncing) {
        this.syncNow();
      }
    }, this.config.syncInterval);
  }
  
  /**
   * Stop automatic sync
   */
  stopAutoSync(): void {
    if (this.syncInterval) {
      clearInterval(this.syncInterval);
      this.syncInterval = null;
      console.log('[SyncManager] Auto-sync stopped');
    }
  }
  
  /**
   * Sync now (manual trigger)
   */
  async syncNow(): Promise<void> {
    if (!this.status.isOnline) {
      console.log('[SyncManager] Cannot sync - offline');
      return;
    }
    
    if (this.status.isSyncing) {
      console.log('[SyncManager] Sync already in progress');
      return;
    }
    
    this.status.isSyncing = true;
    console.log('[SyncManager] Starting sync...');
    
    try {
      await storageManager.initialize();
      
      // Get unsynced progress items
      const unsyncedItems = await storageManager.getUnsyncedProgress();
      this.status.pendingCount = unsyncedItems.length;
      
      console.log(`[SyncManager] Found ${unsyncedItems.length} items to sync`);
      
      let syncedCount = 0;
      let failedCount = 0;
      
      for (const item of unsyncedItems) {
        try {
          await this.syncProgressItem(item);
          await storageManager.markProgressSynced(item.id);
          syncedCount++;
        } catch (error) {
          console.error('[SyncManager] Failed to sync item:', error);
          failedCount++;
        }
      }
      
      this.status.lastSync = Date.now();
      this.status.pendingCount = 0;
      this.status.failedCount = failedCount;
      
      console.log(`[SyncManager] Sync complete: ${syncedCount} synced, ${failedCount} failed`);
      
      // Trigger background sync if service worker is available
      if ('serviceWorker' in navigator && 'sync' in (globalThis as any).registration) {
        const registration = await navigator.serviceWorker.ready;
        await (registration as any).sync.register('sync-progress');
      }
      
    } catch (error) {
      console.error('[SyncManager] Sync failed:', error);
    } finally {
      this.status.isSyncing = false;
    }
  }
  
  /**
   * Sync single progress item to server
   */
  private async syncProgressItem(item: ProgressItem): Promise<void> {
    const url = `${this.config.apiBaseUrl}/progress/sync`;
    
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        userId: item.userId,
        contentId: item.contentId,
        progressData: item.progressData,
        completed: item.completed,
        score: item.score,
        timeSpent: item.timeSpent,
        timestamp: item.timestamp
      })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    console.log('[SyncManager] Synced item:', item.id);
  }
  
  /**
   * Get current sync status
   */
  getStatus(): SyncStatus {
    return { ...this.status };
  }
  
  /**
   * Check if online
   */
  isOnline(): boolean {
    return this.status.isOnline;
  }
}

// Singleton instance
export const syncManager = new SyncManager();
