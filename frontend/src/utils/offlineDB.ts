/**
 * IndexedDB Client for Offline Data Storage
 * 
 * Issue: CODE-REVIEW-GPT #4 (CRITICAL) + #10 (HIGH)
 * TypeScript migration with type safety
 */

const DB_NAME = 'shiksha-setu-offline';
const DB_VERSION = 2;

interface Content {
    id: string;
    title: string;
    content: string;
    subject: string;
    grade_level: number;
    language: string;
    downloaded_at?: number;
    offline_available?: boolean;
    [key: string]: any;
}

interface Progress {
    content_id: string;
    progress_percentage: number;
    last_position?: number;
    completed: boolean;
    updated_at?: number;
    synced?: boolean;
}

interface QuizResult {
    id?: number;
    content_id: string;
    score: number;
    answers: any[];
    timestamp?: number;
    synced?: boolean;
}

interface SyncQueueItem {
    type: 'progress' | 'quiz-result' | 'content';
    data: any;
    timestamp?: number;
    retries?: number;
}

interface StorageEstimate {
    usage: number;
    quota: number;
    percentage: number;
}

class OfflineDB {
    private db: IDBDatabase | null = null;

    async init(): Promise<IDBDatabase> {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(DB_NAME, DB_VERSION);

            request.onerror = () => reject(request.error);
            request.onsuccess = () => {
                this.db = request.result;
                resolve(this.db);
            };

            request.onupgradeneeded = (event: IDBVersionChangeEvent) => {
                const db = (event.target as IDBOpenDBRequest).result;

                // Content store
                if (!db.objectStoreNames.contains('content')) {
                    const contentStore = db.createObjectStore('content', { keyPath: 'id' });
                    contentStore.createIndex('subject', 'subject', { unique: false });
                    contentStore.createIndex('grade_level', 'grade_level', { unique: false });
                    contentStore.createIndex('language', 'language', { unique: false });
                    contentStore.createIndex('downloaded_at', 'downloaded_at', { unique: false });
                }

                // Progress store
                if (!db.objectStoreNames.contains('progress')) {
                    const progressStore = db.createObjectStore('progress', { keyPath: 'content_id' });
                    progressStore.createIndex('updated_at', 'updated_at', { unique: false });
                }

                // Sync queue store
                if (!db.objectStoreNames.contains('sync-queue')) {
                    db.createObjectStore('sync-queue', { keyPath: 'timestamp' });
                }

                // Quiz results store
                if (!db.objectStoreNames.contains('quiz-results')) {
                    const quizStore = db.createObjectStore('quiz-results', { keyPath: 'id', autoIncrement: true });
                    quizStore.createIndex('content_id', 'content_id', { unique: false });
                    quizStore.createIndex('synced', 'synced', { unique: false });
                }

                // User preferences store
                if (!db.objectStoreNames.contains('preferences')) {
                    db.createObjectStore('preferences', { keyPath: 'key' });
                }
            };
        });
    }

    // =============================================================================
    // CONTENT MANAGEMENT
    // =============================================================================

    async saveContent(content: Content): Promise<Content> {
        if (!this.db) throw new Error('Database not initialized');
        
        const tx = this.db.transaction('content', 'readwrite');
        const store = tx.objectStore('content');
        
        const contentData: Content = {
            ...content,
            downloaded_at: Date.now(),
            offline_available: true
        };
        
        await store.put(contentData);
        return contentData;
    }

    async getContent(id: string): Promise<Content | undefined> {
        if (!this.db) throw new Error('Database not initialized');
        
        const tx = this.db.transaction('content', 'readonly');
        const store = tx.objectStore('content');
        return new Promise((resolve, reject) => {
            const request = store.get(id);
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async getAllContent(): Promise<Content[]> {
        if (!this.db) throw new Error('Database not initialized');
        
        const tx = this.db.transaction('content', 'readonly');
        const store = tx.objectStore('content');
        return new Promise((resolve, reject) => {
            const request = store.getAll();
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async getContentBySubject(subject: string): Promise<Content[]> {
        if (!this.db) throw new Error('Database not initialized');
        
        const tx = this.db.transaction('content', 'readonly');
        const store = tx.objectStore('content');
        const index = store.index('subject');
        return new Promise((resolve, reject) => {
            const request = index.getAll(subject);
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async deleteContent(id: string): Promise<void> {
        if (!this.db) throw new Error('Database not initialized');
        
        const tx = this.db.transaction('content', 'readwrite');
        const store = tx.objectStore('content');
        await store.delete(id);
    }

    // =============================================================================
    // PROGRESS TRACKING
    // =============================================================================

    async saveProgress(contentId: string, progressData: Omit<Progress, 'content_id'>): Promise<Progress> {
        if (!this.db) throw new Error('Database not initialized');
        
        const tx = this.db.transaction('progress', 'readwrite');
        const store = tx.objectStore('progress');
        
        const progress: Progress = {
            content_id: contentId,
            ...progressData,
            updated_at: Date.now(),
            synced: false
        };
        
        await store.put(progress);
        
        // Queue for sync
        await this.queueForSync({
            type: 'progress',
            data: progress
        });
        
        return progress;
    }

    async getProgress(contentId: string): Promise<Progress | undefined> {
        if (!this.db) throw new Error('Database not initialized');
        
        const tx = this.db.transaction('progress', 'readonly');
        const store = tx.objectStore('progress');
        return new Promise((resolve, reject) => {
            const request = store.get(contentId);
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async getAllProgress(): Promise<Progress[]> {
        if (!this.db) throw new Error('Database not initialized');
        
        const tx = this.db.transaction('progress', 'readonly');
        const store = tx.objectStore('progress');
        return new Promise((resolve, reject) => {
            const request = store.getAll();
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    // =============================================================================
    // QUIZ RESULTS
    // =============================================================================

    async saveQuizResult(quizData: Omit<QuizResult, 'id' | 'timestamp' | 'synced'>): Promise<QuizResult> {
        if (!this.db) throw new Error('Database not initialized');
        
        const tx = this.db.transaction('quiz-results', 'readwrite');
        const store = tx.objectStore('quiz-results');
        
        const result: QuizResult = {
            ...quizData,
            timestamp: Date.now(),
            synced: false
        };
        
        const request = store.add(result);
        const id = await new Promise<number>((resolve, reject) => {
            request.onsuccess = () => resolve(request.result as number);
            request.onerror = () => reject(request.error);
        });
        
        const completeResult = { ...result, id };
        
        // Queue for sync
        await this.queueForSync({
            type: 'quiz-result',
            data: completeResult
        });
        
        return completeResult;
    }

    async getQuizResults(contentId: string): Promise<QuizResult[]> {
        if (!this.db) throw new Error('Database not initialized');
        
        const tx = this.db.transaction('quiz-results', 'readonly');
        const store = tx.objectStore('quiz-results');
        const index = store.index('content_id');
        return new Promise((resolve, reject) => {
            const request = index.getAll(contentId);
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    // =============================================================================
    // SYNC QUEUE
    // =============================================================================

    async queueForSync(item: SyncQueueItem): Promise<void> {
        if (!this.db) throw new Error('Database not initialized');
        
        const tx = this.db.transaction('sync-queue', 'readwrite');
        const store = tx.objectStore('sync-queue');
        
        const queueItem: SyncQueueItem = {
            ...item,
            timestamp: Date.now(),
            retries: 0
        };
        
        await store.put(queueItem);
        
        // Trigger background sync if available
        if ('serviceWorker' in navigator) {
            const registration = await navigator.serviceWorker.ready;
            // Background Sync API may not be available in all browsers
            if ('sync' in registration) {
                await (registration as unknown as { sync: { register: (tag: string) => Promise<void> } }).sync.register('sync-requests');
            }
        }
    }

    async getSyncQueue(): Promise<SyncQueueItem[]> {
        if (!this.db) throw new Error('Database not initialized');
        
        const tx = this.db.transaction('sync-queue', 'readonly');
        const store = tx.objectStore('sync-queue');
        return new Promise((resolve, reject) => {
            const request = store.getAll();
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    }

    async removeSyncItem(timestamp: number): Promise<void> {
        if (!this.db) throw new Error('Database not initialized');
        
        const tx = this.db.transaction('sync-queue', 'readwrite');
        const store = tx.objectStore('sync-queue');
        await store.delete(timestamp);
    }

    async clearSyncQueue(): Promise<void> {
        if (!this.db) throw new Error('Database not initialized');
        
        const tx = this.db.transaction('sync-queue', 'readwrite');
        const store = tx.objectStore('sync-queue');
        await store.clear();
    }

    // =============================================================================
    // PREFERENCES
    // =============================================================================

    async savePreference(key: string, value: any): Promise<void> {
        if (!this.db) throw new Error('Database not initialized');
        
        const tx = this.db.transaction('preferences', 'readwrite');
        const store = tx.objectStore('preferences');
        await store.put({ key, value });
    }

    async getPreference(key: string): Promise<any> {
        if (!this.db) throw new Error('Database not initialized');
        
        const tx = this.db.transaction('preferences', 'readonly');
        const store = tx.objectStore('preferences');
        return new Promise((resolve, reject) => {
            const request = store.get(key);
            request.onsuccess = () => resolve(request.result?.value);
            request.onerror = () => reject(request.error);
        });
    }

    // =============================================================================
    // UTILITY
    // =============================================================================

    async clearAll(): Promise<void> {
        if (!this.db) throw new Error('Database not initialized');
        
        const storeNames: string[] = ['content', 'progress', 'quiz-results', 'sync-queue'];
        for (const storeName of storeNames) {
            const tx = this.db.transaction(storeName, 'readwrite');
            const store = tx.objectStore(storeName);
            await store.clear();
        }
    }

    async getStorageSize(): Promise<StorageEstimate | null> {
        if ('estimate' in navigator.storage) {
            const estimate = await navigator.storage.estimate();
            return {
                usage: estimate.usage || 0,
                quota: estimate.quota || 0,
                percentage: estimate.usage && estimate.quota ? (estimate.usage / estimate.quota) * 100 : 0
            };
        }
        return null;
    }
}

// Export singleton instance
const offlineDB = new OfflineDB();
export default offlineDB;
export type { Content, Progress, QuizResult, SyncQueueItem, StorageEstimate };
