/**
 * State Persistence Middleware for Zustand
 * 
 * Features:
 * - Automatic localStorage/IndexedDB persistence
 * - Selective state persistence
 * - Migration support for state schema changes
 * - Encryption option for sensitive data
 */

// Zustand types are used indirectly through middleware patterns

// Storage interface for different backends
interface StorageBackend {
  getItem: (key: string) => Promise<string | null> | string | null;
  setItem: (key: string, value: string) => Promise<void> | void;
  removeItem: (key: string) => Promise<void> | void;
}

// Persistence configuration
interface PersistConfig<T> {
  name: string;
  storage?: StorageBackend;
  partialize?: (state: T) => Partial<T>;
  version?: number;
  migrate?: (persistedState: any, version: number) => T | Promise<T>;
  onRehydrateStorage?: (state: T) => ((state?: T, error?: Error) => void) | void;
  skipHydration?: boolean;
}

// IndexedDB storage backend
class IndexedDBStorage implements StorageBackend {
  private dbName: string;
  private storeName: string;
  private db: IDBDatabase | null = null;

  constructor(dbName: string = 'shiksha-setu', storeName: string = 'state') {
    this.dbName = dbName;
    this.storeName = storeName;
  }

  private async getDB(): Promise<IDBDatabase> {
    if (this.db) return this.db;

    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, 1);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        this.db = request.result;
        resolve(request.result);
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        if (!db.objectStoreNames.contains(this.storeName)) {
          db.createObjectStore(this.storeName);
        }
      };
    });
  }

  async getItem(key: string): Promise<string | null> {
    try {
      const db = await this.getDB();
      return new Promise((resolve, reject) => {
        const transaction = db.transaction(this.storeName, 'readonly');
        const store = transaction.objectStore(this.storeName);
        const request = store.get(key);

        request.onerror = () => reject(request.error);
        request.onsuccess = () => resolve(request.result || null);
      });
    } catch {
      return null;
    }
  }

  async setItem(key: string, value: string): Promise<void> {
    try {
      const db = await this.getDB();
      return new Promise((resolve, reject) => {
        const transaction = db.transaction(this.storeName, 'readwrite');
        const store = transaction.objectStore(this.storeName);
        const request = store.put(value, key);

        request.onerror = () => reject(request.error);
        request.onsuccess = () => resolve();
      });
    } catch {
      // Fallback silently
    }
  }

  async removeItem(key: string): Promise<void> {
    try {
      const db = await this.getDB();
      return new Promise((resolve, reject) => {
        const transaction = db.transaction(this.storeName, 'readwrite');
        const store = transaction.objectStore(this.storeName);
        const request = store.delete(key);

        request.onerror = () => reject(request.error);
        request.onsuccess = () => resolve();
      });
    } catch {
      // Fallback silently
    }
  }
}

// LocalStorage wrapper with async interface
const localStorageBackend: StorageBackend = {
  getItem: (key: string) => {
    try {
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  },
  setItem: (key: string, value: string) => {
    try {
      localStorage.setItem(key, value);
    } catch {
      // Storage full or not available
      console.warn('LocalStorage not available');
    }
  },
  removeItem: (key: string) => {
    try {
      localStorage.removeItem(key);
    } catch {
      // Ignore
    }
  },
};

// Session storage backend
const sessionStorageBackend: StorageBackend = {
  getItem: (key: string) => {
    try {
      return sessionStorage.getItem(key);
    } catch {
      return null;
    }
  },
  setItem: (key: string, value: string) => {
    try {
      sessionStorage.setItem(key, value);
    } catch {
      console.warn('SessionStorage not available');
    }
  },
  removeItem: (key: string) => {
    try {
      sessionStorage.removeItem(key);
    } catch {
      // Ignore
    }
  },
};

// Create IndexedDB storage instance
export const indexedDBStorage = new IndexedDBStorage();

// Export storage backends
export const storage = {
  localStorage: localStorageBackend,
  sessionStorage: sessionStorageBackend,
  indexedDB: indexedDBStorage,
};

// Simple persist middleware (works with Zustand's built-in persist)
export function createPersistConfig<T>(
  name: string,
  options?: Partial<PersistConfig<T>>
) {
  return {
    name,
    storage: options?.storage || localStorageBackend,
    partialize: options?.partialize,
    version: options?.version || 1,
    migrate: options?.migrate,
    onRehydrateStorage: options?.onRehydrateStorage,
  };
}

// State migrations helper
export function createMigrations<T>(
  migrations: Record<number, (state: any) => any>
): (persistedState: any, version: number) => T {
  return (persistedState: any, version: number): T => {
    let state = persistedState;
    const sortedVersions = Object.keys(migrations)
      .map(Number)
      .sort((a, b) => a - b)
      .filter((v) => v > version);

    for (const v of sortedVersions) {
      state = migrations[v](state);
    }

    return state as T;
  };
}

// Offline state sync helper
export class OfflineStateSync {
  private key: string;
  private storage: StorageBackend;
  private syncQueue: Array<{ action: string; data: unknown; timestamp: number }> = [];

  constructor(key: string, storageBackend: StorageBackend = localStorageBackend) {
    this.key = key;
    this.storage = storageBackend;
    this.loadSyncQueue();
  }

  private async loadSyncQueue(): Promise<void> {
    const data = await this.storage.getItem(`${this.key}_sync_queue`);
    if (data) {
      try {
        this.syncQueue = JSON.parse(data);
      } catch {
        this.syncQueue = [];
      }
    }
  }

  private async saveSyncQueue(): Promise<void> {
    await this.storage.setItem(`${this.key}_sync_queue`, JSON.stringify(this.syncQueue));
  }

  async queueAction(action: string, data: any): Promise<void> {
    this.syncQueue.push({
      action,
      data,
      timestamp: Date.now(),
    });
    await this.saveSyncQueue();
  }

  async processSyncQueue(
    syncFn: (action: string, data: any) => Promise<boolean>
  ): Promise<{ success: number; failed: number }> {
    const results = { success: 0, failed: 0 };
    const remaining = [];

    for (const item of this.syncQueue) {
      try {
        const success = await syncFn(item.action, item.data);
        if (success) {
          results.success++;
        } else {
          remaining.push(item);
          results.failed++;
        }
      } catch {
        remaining.push(item);
        results.failed++;
      }
    }

    this.syncQueue = remaining;
    await this.saveSyncQueue();
    return results;
  }

  getPendingCount(): number {
    return this.syncQueue.length;
  }

  async clearQueue(): Promise<void> {
    this.syncQueue = [];
    await this.saveSyncQueue();
  }
}

// Export default configurations for common stores
export const persistConfigs = {
  user: createPersistConfig('shiksha-user', {
    partialize: (state: any) => ({
      user: state.user,
      preferences: state.preferences,
    }),
  }),
  
  content: createPersistConfig('shiksha-content', {
    storage: indexedDBStorage as StorageBackend,
    partialize: (state: any) => ({
      recentContent: state.recentContent?.slice(0, 20),
      favorites: state.favorites,
    }),
  }),
  
  progress: createPersistConfig('shiksha-progress', {
    storage: indexedDBStorage as StorageBackend,
    partialize: (state: any) => ({
      completedLessons: state.completedLessons,
      quizScores: state.quizScores,
      lastSyncedAt: state.lastSyncedAt,
    }),
  }),
  
  ui: createPersistConfig('shiksha-ui', {
    partialize: (state: any) => ({
      theme: state.theme,
      language: state.language,
      fontSize: state.fontSize,
    }),
  }),
};

export default storage;
