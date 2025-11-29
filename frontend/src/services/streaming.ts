/**
 * WebSocket Streaming Service
 * 
 * Real-time communication for translations, progress updates, and notifications
 */

type MessageHandler = (data: any) => void;
type ErrorHandler = (error: Event | CloseEvent) => void;
type ConnectionHandler = () => void;

interface StreamingOptions {
  onMessage?: MessageHandler;
  onError?: ErrorHandler;
  onOpen?: ConnectionHandler;
  onClose?: ConnectionHandler;
  reconnect?: boolean;
  maxReconnectAttempts?: number;
}

class StreamingService {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private handlers: Map<string, Set<MessageHandler>> = new Map();
  private options: StreamingOptions = {};
  private isConnecting = false;
  private messageQueue: any[] = [];

  private getBaseUrl(): string {
    const apiUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';
    return apiUrl.replace(/^http/, 'ws');
  }

  /**
   * Connect to WebSocket streaming endpoint
   */
  async connect(options: StreamingOptions = {}): Promise<void> {
    if (this.ws?.readyState === WebSocket.OPEN || this.isConnecting) {
      return;
    }

    this.options = options;
    this.isConnecting = true;

    return new Promise((resolve, reject) => {
      const token = localStorage.getItem('access_token');
      const wsUrl = `${this.getBaseUrl()}/api/v1/streaming/ws?token=${token}`;

      try {
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
          console.log('[Streaming] Connected');
          this.isConnecting = false;
          this.reconnectAttempts = 0;
          
          // Send queued messages
          while (this.messageQueue.length > 0) {
            const msg = this.messageQueue.shift();
            this.send(msg);
          }
          
          this.options.onOpen?.();
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
            this.options.onMessage?.(data);
          } catch (e) {
            console.error('[Streaming] Parse error:', e);
          }
        };

        this.ws.onerror = (error) => {
          console.error('[Streaming] Error:', error);
          this.isConnecting = false;
          this.options.onError?.(error);
          reject(error);
        };

        this.ws.onclose = (event) => {
          console.log('[Streaming] Closed:', event.code, event.reason);
          this.isConnecting = false;
          this.options.onClose?.();
          
          // Auto-reconnect if enabled
          if (this.options.reconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.scheduleReconnect();
          }
        };
      } catch (error) {
        this.isConnecting = false;
        reject(error);
      }
    });
  }

  /**
   * Disconnect from WebSocket
   */
  disconnect(): void {
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    this.handlers.clear();
    this.messageQueue = [];
  }

  /**
   * Send message through WebSocket
   */
  send(data: any): boolean {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
      return true;
    }
    
    // Queue message if not connected
    this.messageQueue.push(data);
    return false;
  }

  /**
   * Subscribe to specific message types
   */
  subscribe(type: string, handler: MessageHandler): () => void {
    if (!this.handlers.has(type)) {
      this.handlers.set(type, new Set());
    }
    this.handlers.get(type)!.add(handler);

    // Return unsubscribe function
    return () => {
      this.handlers.get(type)?.delete(handler);
    };
  }

  /**
   * Start real-time translation stream
   */
  startTranslationStream(text: string, targetLanguages: string[]): void {
    this.send({
      type: 'translate_stream',
      text,
      target_languages: targetLanguages,
    });
  }

  /**
   * Subscribe to task progress updates
   */
  subscribeToTask(taskId: string, handler: MessageHandler): () => void {
    this.send({
      type: 'subscribe_task',
      task_id: taskId,
    });
    return this.subscribe(`task:${taskId}`, handler);
  }

  /**
   * Unsubscribe from task updates
   */
  unsubscribeFromTask(taskId: string): void {
    this.send({
      type: 'unsubscribe_task',
      task_id: taskId,
    });
    this.handlers.delete(`task:${taskId}`);
  }

  private handleMessage(data: any): void {
    const { type, task_id } = data;

    // Handle task-specific messages
    if (task_id) {
      const taskHandlers = this.handlers.get(`task:${task_id}`);
      taskHandlers?.forEach(handler => handler(data));
    }

    // Handle type-specific messages
    if (type) {
      const typeHandlers = this.handlers.get(type);
      typeHandlers?.forEach(handler => handler(data));
    }

    // Handle global handlers
    const globalHandlers = this.handlers.get('*');
    globalHandlers?.forEach(handler => handler(data));
  }

  private scheduleReconnect(): void {
    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
    
    console.log(`[Streaming] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
    
    setTimeout(() => {
      this.connect(this.options);
    }, delay);
  }

  /**
   * Check if connected
   */
  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

export const streamingService = new StreamingService();
export default streamingService;
