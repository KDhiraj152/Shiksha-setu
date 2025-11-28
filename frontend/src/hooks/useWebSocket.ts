import { useEffect, useRef, useCallback, useState } from 'react';
import type { TaskStage } from '../components/molecules/TaskProgress';

export type WebSocketStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

export interface TaskUpdate {
  taskId: string;
  status: 'PENDING' | 'STARTED' | 'PROGRESS' | 'SUCCESS' | 'FAILURE' | 'REVOKED';
  stage?: TaskStage;
  progress?: number;
  progressInfo?: {
    current?: number;
    total?: number;
    message?: string;
  };
  result?: unknown;
  error?: string;
}

export interface UseWebSocketOptions {
  /** WebSocket endpoint URL */
  url?: string;
  /** Reconnect on disconnect */
  reconnect?: boolean;
  /** Reconnection interval in ms */
  reconnectInterval?: number;
  /** Maximum reconnection attempts */
  maxReconnectAttempts?: number;
  /** Callback when connected */
  onConnect?: () => void;
  /** Callback when disconnected */
  onDisconnect?: () => void;
  /** Callback when error occurs */
  onError?: (error: Event) => void;
}

const DEFAULT_WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws';

/**
 * WebSocket hook for real-time task progress updates.
 * 
 * @example
 * const { status, taskUpdates, subscribe, unsubscribe } = useWebSocket({
 *   onConnect: () => console.log('Connected'),
 * });
 * 
 * // Subscribe to a task
 * subscribe('task-123');
 * 
 * // Get updates for the task
 * const update = taskUpdates['task-123'];
 */
export function useWebSocket(options: UseWebSocketOptions = {}) {
  const {
    url = DEFAULT_WS_URL,
    reconnect = true,
    reconnectInterval = 3000,
    maxReconnectAttempts = 5,
    onConnect,
    onDisconnect,
    onError,
  } = options;

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptsRef = useRef(0);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const subscribedTasksRef = useRef<Set<string>>(new Set());

  const [status, setStatus] = useState<WebSocketStatus>('disconnected');
  const [taskUpdates, setTaskUpdates] = useState<Record<string, TaskUpdate>>({});

  // Connect to WebSocket
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    setStatus('connecting');

    try {
      const token = localStorage.getItem('access_token');
      const wsUrl = token ? `${url}?token=${token}` : url;
      const ws = new WebSocket(wsUrl);

      ws.onopen = () => {
        setStatus('connected');
        reconnectAttemptsRef.current = 0;
        onConnect?.();

        // Re-subscribe to all tasks
        subscribedTasksRef.current.forEach((taskId) => {
          ws.send(JSON.stringify({ type: 'subscribe', taskId }));
        });
      };

      ws.onclose = () => {
        setStatus('disconnected');
        wsRef.current = null;
        onDisconnect?.();

        // Attempt reconnection
        if (reconnect && reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current += 1;
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, reconnectInterval * reconnectAttemptsRef.current);
        }
      };

      ws.onerror = (error) => {
        setStatus('error');
        onError?.(error);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as TaskUpdate;
          
          if (data.taskId) {
            setTaskUpdates((prev) => ({
              ...prev,
              [data.taskId]: data,
            }));
          }
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      };

      wsRef.current = ws;
    } catch (err) {
      setStatus('error');
      console.error('WebSocket connection error:', err);
    }
  }, [url, reconnect, reconnectInterval, maxReconnectAttempts, onConnect, onDisconnect, onError]);

  // Disconnect from WebSocket
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    reconnectAttemptsRef.current = maxReconnectAttempts; // Prevent reconnection
    wsRef.current?.close();
    wsRef.current = null;
    setStatus('disconnected');
  }, [maxReconnectAttempts]);

  // Subscribe to task updates
  const subscribe = useCallback((taskId: string) => {
    subscribedTasksRef.current.add(taskId);
    
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'subscribe', taskId }));
    }
  }, []);

  // Unsubscribe from task updates
  const unsubscribe = useCallback((taskId: string) => {
    subscribedTasksRef.current.delete(taskId);
    
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'unsubscribe', taskId }));
    }

    // Clear task from state
    setTaskUpdates((prev) => {
      const { [taskId]: _, ...rest } = prev;
      return rest;
    });
  }, []);

  // Clear all task updates
  const clearUpdates = useCallback(() => {
    setTaskUpdates({});
  }, []);

  // Get update for specific task
  const getTaskUpdate = useCallback((taskId: string): TaskUpdate | undefined => {
    return taskUpdates[taskId];
  }, [taskUpdates]);

  // Auto-connect on mount
  useEffect(() => {
    connect();

    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    /** Current WebSocket connection status */
    status,
    /** All task updates keyed by taskId */
    taskUpdates,
    /** Subscribe to a task's updates */
    subscribe,
    /** Unsubscribe from a task's updates */
    unsubscribe,
    /** Get update for a specific task */
    getTaskUpdate,
    /** Clear all task updates */
    clearUpdates,
    /** Manually connect */
    connect,
    /** Manually disconnect */
    disconnect,
    /** Check if connected */
    isConnected: status === 'connected',
  };
}
