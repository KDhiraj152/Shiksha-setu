/**
 * Offline status hook
 * Tracks online/offline status and provides queue management
 */

import { useState, useEffect, useCallback } from 'react';
import { 
  offlineStatus, 
  offlineQueue, 
  processOfflineQueue,
  type QueuedAction,
} from '../lib/offline';

export interface UseOfflineReturn {
  /** Current online status */
  isOnline: boolean;
  /** Number of queued actions */
  queueCount: number;
  /** Add action to offline queue */
  addToQueue: (action: Omit<QueuedAction, 'id' | 'createdAt' | 'retries'>) => Promise<void>;
  /** Process all queued actions */
  processQueue: (processor: (action: QueuedAction) => Promise<boolean>) => Promise<void>;
  /** Clear the offline queue */
  clearQueue: () => Promise<void>;
}

/**
 * Hook for managing offline status and queued actions
 * 
 * @example
 * const { isOnline, queueCount, addToQueue, processQueue } = useOffline();
 * 
 * // Add action when offline
 * if (!isOnline) {
 *   await addToQueue({ type: 'feedback', payload: feedbackData, maxRetries: 3 });
 * }
 * 
 * // Process queue when back online
 * useEffect(() => {
 *   if (isOnline) {
 *     processQueue(async (action) => {
 *       await api.submitFeedback(action.payload);
 *       return true;
 *     });
 *   }
 * }, [isOnline]);
 */
export function useOffline(): UseOfflineReturn {
  const [isOnline, setIsOnline] = useState(offlineStatus.isOnline());
  const [queueCount, setQueueCount] = useState(0);

  // Subscribe to online/offline changes
  useEffect(() => {
    const unsubscribe = offlineStatus.subscribe(setIsOnline);
    return unsubscribe;
  }, []);

  // Update queue count
  const updateQueueCount = useCallback(async () => {
    const count = await offlineQueue.count();
    setQueueCount(count);
  }, []);

  // Load initial queue count
  useEffect(() => {
    updateQueueCount();
  }, [updateQueueCount]);

  // Add action to queue
  const addToQueue = useCallback(async (
    action: Omit<QueuedAction, 'id' | 'createdAt' | 'retries'>
  ) => {
    await offlineQueue.add(action);
    await updateQueueCount();
  }, [updateQueueCount]);

  // Process queue
  const processQueue = useCallback(async (
    processor: (action: QueuedAction) => Promise<boolean>
  ) => {
    await processOfflineQueue(processor);
    await updateQueueCount();
  }, [updateQueueCount]);

  // Clear queue
  const clearQueue = useCallback(async () => {
    await offlineQueue.clear();
    setQueueCount(0);
  }, []);

  return {
    isOnline,
    queueCount,
    addToQueue,
    processQueue,
    clearQueue,
  };
}
