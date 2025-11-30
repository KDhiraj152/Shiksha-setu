/**
 * WebSocket hook for real-time streaming translation
 * 
 * Features:
 * - Auto-reconnect on disconnect
 * - Request queuing during reconnect
 * - Partial result handling every 500ms
 * - Connection health monitoring
 */

import { useEffect, useRef, useState, useCallback } from 'react';

export interface TranslationMessage {
  text: string;
  source_lang?: string;
  request_id?: string;
}

export interface TranslationResponse {
  type: 'ack' | 'partial' | 'final' | 'error' | 'connected';
  text?: string;
  source_lang?: string;
  target_lang?: string;
  progress?: number;
  latency_ms?: number;
  error?: string;
  timestamp: string;
  client_id?: string;
  text_length?: number;
  current_connections?: number;
  max_connections?: number;
}

export interface UseStreamingTranslationOptions {
  targetLang?: string;
  autoConnect?: boolean;
  token?: string | null;
  onPartialResult?: (text: string, progress: number) => void;
  onFinalResult?: (text: string) => void;
  onError?: (error: string) => void;
}

export interface UseStreamingTranslationReturn {
  isConnected: boolean;
  isConnecting: boolean;
  error: string | null;
  sendMessage: (message: TranslationMessage) => void;
  connect: () => void;
  disconnect: () => void;
  currentTranslation: string;
  translationProgress: number;
  connectionStats: {
    activeConnections: number;
    maxConnections: number;
  } | null;
}

export function useStreamingTranslation(
  options: UseStreamingTranslationOptions = {}
): UseStreamingTranslationReturn {
  const {
    targetLang = 'hi',
    autoConnect = true,
    token = null,
    onPartialResult,
    onFinalResult,
    onError
  } = options;

  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentTranslation, setCurrentTranslation] = useState('');
  const [translationProgress, setTranslationProgress] = useState(0);
  const [connectionStats, setConnectionStats] = useState<{
    activeConnections: number;
    maxConnections: number;
  } | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const messageQueueRef = useRef<TranslationMessage[]>([]);
  const reconnectAttemptsRef = useRef(0);
  const maxReconnectAttempts = 5;
  const reconnectDelay = 2000; // 2 seconds

  const getWebSocketUrl = useCallback(() => {
    const protocol = globalThis.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = import.meta.env.VITE_API_URL?.replace(/^https?:\/\//, '') || 'localhost:8000';
    const tokenParam = token ? `?token=${token}` : '';
    return `${protocol}//${host}/api/v1/streaming/translate${tokenParam}&target_lang=${targetLang}`;
  }, [token, targetLang]);

  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const response: TranslationResponse = JSON.parse(event.data);

      switch (response.type) {
        case 'connected':
          console.log('WebSocket connected:', response.client_id);
          setConnectionStats({
            activeConnections: response.current_connections || 0,
            maxConnections: response.max_connections || 1000
          });
          setError(null);
          reconnectAttemptsRef.current = 0;
          
          // Send queued messages
          while (messageQueueRef.current.length > 0) {
            const message = messageQueueRef.current.shift();
            if (message && wsRef.current?.readyState === WebSocket.OPEN) {
              wsRef.current.send(JSON.stringify(message));
            }
          }
          break;

        case 'ack':
          console.log('Translation acknowledged:', response.text_length, 'chars');
          setCurrentTranslation('');
          setTranslationProgress(0);
          break;

        case 'partial':
          if (response.text) {
            setCurrentTranslation(response.text);
            setTranslationProgress(response.progress || 0);
            onPartialResult?.(response.text, response.progress || 0);
          }
          break;

        case 'final':
          if (response.text) {
            setCurrentTranslation(response.text);
            setTranslationProgress(1);
            onFinalResult?.(response.text);
            console.log(`Translation completed in ${response.latency_ms}ms`);
          }
          break;

        case 'error': {
          const errorMsg = response.error || 'Unknown error';
          setError(errorMsg);
          onError?.(errorMsg);
          console.error('Translation error:', errorMsg);
          break;
        }
      }
    } catch (err) {
      console.error('Failed to parse WebSocket message:', err);
    }
  }, [onPartialResult, onFinalResult, onError]);

  const handleClose = useCallback(() => {
    setIsConnected(false);
    setIsConnecting(false);
    
    if (reconnectAttemptsRef.current < maxReconnectAttempts) {
      console.log(`WebSocket closed. Reconnecting in ${reconnectDelay}ms... (attempt ${reconnectAttemptsRef.current + 1}/${maxReconnectAttempts})`);
      
      reconnectTimeoutRef.current = setTimeout(() => {
        reconnectAttemptsRef.current += 1;
        connect();
      }, reconnectDelay);
    } else {
      setError('Connection lost. Maximum reconnect attempts reached.');
    }
  }, []);

  const handleError = useCallback((event: Event) => {
    console.error('WebSocket error:', event);
    setError('WebSocket connection error');
    setIsConnecting(false);
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return; // Already connected
    }

    if (wsRef.current?.readyState === WebSocket.CONNECTING) {
      return; // Already connecting
    }

    setIsConnecting(true);
    setError(null);

    try {
      const ws = new WebSocket(getWebSocketUrl());
      
      ws.onopen = () => {
        setIsConnected(true);
        setIsConnecting(false);
      };

      ws.onmessage = handleMessage;
      ws.onclose = handleClose;
      ws.onerror = handleError;

      wsRef.current = ws;
    } catch (err) {
      setError(`Failed to connect: ${err}`);
      setIsConnecting(false);
    }
  }, [getWebSocketUrl, handleMessage, handleClose, handleError]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setIsConnected(false);
    setIsConnecting(false);
    reconnectAttemptsRef.current = 0;
    messageQueueRef.current = [];
  }, []);

  const sendMessage = useCallback((message: TranslationMessage) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    } else {
      // Queue message if not connected
      messageQueueRef.current.push(message);
      
      // Try to reconnect if not already connecting
      if (!isConnecting && !isConnected) {
        connect();
      }
    }
  }, [isConnecting, isConnected, connect]);

  // Auto-connect on mount if enabled
  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    // Cleanup on unmount
    return () => {
      disconnect();
    };
  }, [autoConnect]); // Only run on mount/unmount

  return {
    isConnected,
    isConnecting,
    error,
    sendMessage,
    connect,
    disconnect,
    currentTranslation,
    translationProgress,
    connectionStats
  };
}
