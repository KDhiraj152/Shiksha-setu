/**
 * Offline Storage Manager
 * 
 * Manages IndexedDB storage for offline content and progress data.
 */

interface StorageConfig {
  dbName: string;
  version: number;
  maxContentSize: number; // MB
  maxProgressItems: number;
}

interface ContentItem {
  id: string;
  type: string;
  title: string;
  content: any;
  size: number;
  timestamp: number;
  accessed: number;
  accessCount: number;
}

export interface ProgressItem {
  id: string;
  userId: string;
  contentId: string;
  progressData: any;
  completed: boolean;
  score?: number;
  timeSpent?: number;
  timestamp: number;
  synced: boolean;
}

export class StorageManager {
  private readonly config: StorageConfig;
  private db: IDBDatabase | null = null;
  
  constructor(config?: Partial<StorageConfig>) {
    this.config = {
      dbName: 'ShikshaSetu',
      version: 1,
      maxContentSize: 500, // 500MB
      maxProgressItems: 1000,
      ...config
    };
  }
  
  /**
   * Initialize IndexedDB
   */
  async initialize(): Promise<void> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.config.dbName, this.config.version);
      
      request.onerror = () => {
        console.error('Failed to open IndexedDB:', request.error);
        reject(new Error(request.error?.message || 'Failed to open IndexedDB'));
      };
      
      request.onsuccess = () => {
        this.db = request.result;
        console.log('IndexedDB initialized');
        resolve();
      };
      
      request.onupgradeneeded = (event: any) => {
        const db = event.target.result;
        
        // Content store
        if (!db.objectStoreNames.contains('content')) {
          const contentStore = db.createObjectStore('content', { keyPath: 'id' });
          contentStore.createIndex('type', 'type', { unique: false });
          contentStore.createIndex('timestamp', 'timestamp', { unique: false });
          contentStore.createIndex('accessed', 'accessed', { unique: false });
        }
        
        // Progress store
        if (!db.objectStoreNames.contains('progress')) {
          const progressStore = db.createObjectStore('progress', { keyPath: 'id' });
          progressStore.createIndex('userId', 'userId', { unique: false });
          progressStore.createIndex('contentId', 'contentId', { unique: false });
          progressStore.createIndex('synced', 'synced', { unique: false });
          progressStore.createIndex('timestamp', 'timestamp', { unique: false });
        }
        
        // Settings store
        if (!db.objectStoreNames.contains('settings')) {
          db.createObjectStore('settings', { keyPath: 'key' });
        }
        
        // Downloaded models store
        if (!db.objectStoreNames.contains('models')) {
          const modelsStore = db.createObjectStore('models', { keyPath: 'name' });
          modelsStore.createIndex('downloaded', 'downloaded', { unique: false });
        }
        
        console.log('IndexedDB schema created');
      };
    });
  }
  
  /**
   * Save content for offline access
   */
  async saveContent(item: Omit<ContentItem, 'timestamp' | 'accessed' | 'accessCount'>): Promise<void> {
    if (!this.db) throw new Error('Database not initialized');
    
    const contentItem: ContentItem = {
      ...item,
      timestamp: Date.now(),
      accessed: Date.now(),
      accessCount: 0
    };
    
    // Check storage quota
    const currentSize = await this.getStorageSize();
    if (currentSize + item.size > this.config.maxContentSize * 1024 * 1024) {
      await this.evictOldContent();
    }
    
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['content'], 'readwrite');
      const store = transaction.objectStore('content');
      const request = store.put(contentItem);
      
      request.onsuccess = () => {
        console.log('Content saved:', item.id);
        resolve();
      };
      request.onerror = () => reject(new Error(request.error?.message || 'Failed to save content'));
    });
  }
  
  /**
   * Get content by ID
   */
  async getContent(id: string): Promise<ContentItem | null> {
    if (!this.db) throw new Error('Database not initialized');
    
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['content'], 'readwrite');
      const store = transaction.objectStore('content');
      const request = store.get(id);
      
      request.onsuccess = () => {
        const item = request.result;
        
        if (item) {
          // Update access statistics
          item.accessed = Date.now();
          item.accessCount++;
          store.put(item);
          console.log('Content retrieved:', id);
        }
        
        resolve(item || null);
      };
      request.onerror = () => reject(new Error(request.error?.message || 'Failed to get content'));
    });
  }
  
  /**
   * Get all content of a specific type
   */
  async getContentByType(type: string): Promise<ContentItem[]> {
    if (!this.db) throw new Error('Database not initialized');
    
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['content'], 'readonly');
      const store = transaction.objectStore('content');
      const index = store.index('type');
      const request = index.getAll(type);
      
      request.onsuccess = () => resolve(request.result || []);
      request.onerror = () => reject(new Error(request.error?.message || 'Failed to get content by type'));
    });
  }
  
  /**
   * Save progress data
   */
  async saveProgress(item: Omit<ProgressItem, 'timestamp' | 'synced'>): Promise<void> {
    if (!this.db) throw new Error('Database not initialized');
    
    const progressItem: ProgressItem = {
      ...item,
      timestamp: Date.now(),
      synced: false
    };
    
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['progress'], 'readwrite');
      const store = transaction.objectStore('progress');
      const request = store.put(progressItem);
      
      request.onsuccess = () => {
        console.log('Progress saved:', item.id);
        resolve();
      };
      request.onerror = () => reject(new Error(request.error?.message || 'Failed to save progress'));
    });
  }
  
  /**
   * Get progress for user and content
   */
  async getProgress(userId: string, contentId: string): Promise<ProgressItem | null> {
    if (!this.db) throw new Error('Database not initialized');
    
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['progress'], 'readonly');
      const store = transaction.objectStore('progress');
      const request = store.get(`${userId}_${contentId}`);
      
      request.onsuccess = () => resolve(request.result || null);
      request.onerror = () => reject(new Error(request.error?.message || 'Failed to get progress'));
    });
  }
  
  /**
   * Get all progress for user
   */
  async getUserProgress(userId: string): Promise<ProgressItem[]> {
    if (!this.db) throw new Error('Database not initialized');
    
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['progress'], 'readonly');
      const store = transaction.objectStore('progress');
      const index = store.index('userId');
      const request = index.getAll(userId);
      
      request.onsuccess = () => resolve(request.result || []);
      request.onerror = () => reject(new Error(request.error?.message || 'Failed to get user progress'));
    });
  }
  
  /**
   * Get unsynced progress items
   */
  async getUnsyncedProgress(): Promise<ProgressItem[]> {
    if (!this.db) throw new Error('Database not initialized');
    
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['progress'], 'readonly');
      const store = transaction.objectStore('progress');
      const index = store.index('synced');
      const request = index.getAll(IDBKeyRange.only(false));
      
      request.onsuccess = () => resolve(request.result || []);
      request.onerror = () => reject(new Error(request.error?.message || 'Failed to get unsynced progress'));
    });
  }
  
  /**
   * Mark progress as synced
   */
  async markProgressSynced(id: string): Promise<void> {
    if (!this.db) throw new Error('Database not initialized');
    
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['progress'], 'readwrite');
      const store = transaction.objectStore('progress');
      const request = store.get(id);
      
      request.onsuccess = () => {
        const item = request.result;
        if (item) {
          item.synced = true;
          store.put(item);
        }
        resolve();
      };
      request.onerror = () => reject(new Error(request.error?.message || 'Failed to mark progress synced'));
    });
  }
  
  /**
   * Get storage size in bytes
   */
  async getStorageSize(): Promise<number> {
    if (!this.db) throw new Error('Database not initialized');
    
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['content'], 'readonly');
      const store = transaction.objectStore('content');
      const request = store.getAll();
      
      request.onsuccess = () => {
        const items: ContentItem[] = request.result || [];
        const totalSize = items.reduce((sum, item) => sum + item.size, 0);
        resolve(totalSize);
      };
      request.onerror = () => reject(new Error(request.error?.message || 'Failed to get storage size'));
    });
  }
  
  /**
   * Evict old content based on LRU
   */
  async evictOldContent(): Promise<void> {
    if (!this.db) throw new Error('Database not initialized');
    
    console.log('Evicting old content...');
    
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(['content'], 'readwrite');
      const store = transaction.objectStore('content');
      const index = store.index('accessed');
      const request = index.openCursor();
      
      let evictedCount = 0;
      const maxEvict = 10;
      
      request.onsuccess = (event: any) => {
        const cursor = event.target.result;
        
        if (cursor && evictedCount < maxEvict) {
          console.log('Evicting content:', cursor.value.id);
          cursor.delete();
          evictedCount++;
          cursor.continue();
        } else {
          console.log(`Evicted ${evictedCount} items`);
          resolve();
        }
      };
      request.onerror = () => reject(new Error(request.error?.message || 'Failed to evict content'));
    });
  }
  
  /**
   * Clear all data
   */
  async clearAll(): Promise<void> {
    if (!this.db) throw new Error('Database not initialized');
    
    const stores = ['content', 'progress', 'settings', 'models'];
    
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction(stores, 'readwrite');
      
      for (const storeName of stores) {
        const store = transaction.objectStore(storeName);
        store.clear();
      }
      
      transaction.oncomplete = () => {
        console.log('All data cleared');
        resolve();
      };
      transaction.onerror = () => reject(new Error(transaction.error?.message || 'Failed to clear all data'));
    });
  }
  
  /**
   * Get storage statistics
   */
  async getStats(): Promise<any> {
    if (!this.db) throw new Error('Database not initialized');
    
    const contentStore = await this._getStoreStats('content');
    const progressStore = await this._getStoreStats('progress');
    const storageSize = await this.getStorageSize();
    
    return {
      content: contentStore,
      progress: progressStore,
      storageSize: storageSize,
      storageSizeMB: (storageSize / (1024 * 1024)).toFixed(2),
      maxSizeMB: this.config.maxContentSize
    };
  }
  
  private async _getStoreStats(storeName: string): Promise<any> {
    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([storeName], 'readonly');
      const store = transaction.objectStore(storeName);
      const countRequest = store.count();
      
      countRequest.onsuccess = () => {
        resolve({ count: countRequest.result });
      };
      countRequest.onerror = () => reject(new Error(countRequest.error?.message || 'Failed to get store stats'));
    });
  }
}

// Singleton instance
export const storageManager = new StorageManager();
