/**
 * Unified API Client for ShikshaSetu
 * 
 * This is the single source of truth for all API interactions.
 * All endpoints match the backend exactly.
 * 
 * Key fixes:
 * - Language names match backend (Hindi, Tamil, Telugu, etc. - NOT codes)
 * - Progress API uses /api/progress/ (NOT /api/v1/progress/)
 * - Q&A endpoints use FormData as backend expects
 * - Proper error handling and retry logic
 */

import axios, { 
  type AxiosInstance, 
  type AxiosError, 
  type InternalAxiosRequestConfig 
} from 'axios';
import type {
  User,
  TokenResponse,
  LoginRequest,
  RegisterRequest,
  ProcessRequest,
  SimplifyRequest,
  TranslateRequest,
  ValidateRequest,
  TTSRequest,
  TaskStatus,
  ProcessedContent,
  FeedbackRequest,
  HealthCheck,
  DetailedHealthCheck,
  PaginatedResponse,
  LibraryFilters,
  SearchParams,
} from '../types/api';

// =============================================================================
// Configuration
// =============================================================================

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

// Retry configuration
const MAX_RETRIES = 3;
const RETRY_DELAY_BASE = 1000; // 1 second

// =============================================================================
// Token Refresh Queue
// =============================================================================

let isRefreshing = false;
let refreshSubscribers: ((token: string) => void)[] = [];

const subscribeTokenRefresh = (callback: (token: string) => void) => {
  refreshSubscribers.push(callback);
};

const onTokenRefreshed = (token: string) => {
  refreshSubscribers.forEach(callback => callback(token));
  refreshSubscribers = [];
};

// =============================================================================
// Axios Instance
// =============================================================================

export const apiClient: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000, // 60 seconds for long operations
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor - Add auth token
apiClient.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const token = localStorage.getItem('access_token');
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor - Handle errors and token refresh
apiClient.interceptors.response.use(
  (response) => response,
  async (error: AxiosError<{ detail?: string; message?: string }>) => {
    const originalRequest = error.config as InternalAxiosRequestConfig & { 
      _retry?: boolean;
      _retryCount?: number;
    };

    // Handle 401 - Token expired
    if (error.response?.status === 401 && !originalRequest._retry) {
      if (isRefreshing) {
        // Wait for token refresh
        return new Promise((resolve) => {
          subscribeTokenRefresh((token: string) => {
            if (originalRequest.headers) {
              originalRequest.headers.Authorization = `Bearer ${token}`;
            }
            resolve(apiClient(originalRequest));
          });
        });
      }

      originalRequest._retry = true;
      isRefreshing = true;

      try {
        const refreshToken = localStorage.getItem('refresh_token');
        if (!refreshToken) throw new Error('No refresh token');

        const response = await axios.post<TokenResponse>(
          `${API_BASE_URL}/api/v1/auth/refresh`,
          null,
          {
            headers: { 'Content-Type': 'application/json' },
            params: { refresh_token: refreshToken }
          }
        );

        const { access_token, refresh_token } = response.data;
        localStorage.setItem('access_token', access_token);
        localStorage.setItem('refresh_token', refresh_token);

        onTokenRefreshed(access_token);
        isRefreshing = false;

        if (originalRequest.headers) {
          originalRequest.headers.Authorization = `Bearer ${access_token}`;
        }
        return apiClient(originalRequest);
      } catch {
        isRefreshing = false;
        localStorage.removeItem('access_token');
        localStorage.removeItem('refresh_token');
        window.location.href = '/login';
        throw error;
      }
    }

    // Handle 429 - Rate limit with exponential backoff
    if (error.response?.status === 429) {
      const retryCount = originalRequest._retryCount || 0;
      if (retryCount < MAX_RETRIES) {
        originalRequest._retryCount = retryCount + 1;
        const retryAfter = error.response.headers['retry-after'];
        const delay = retryAfter 
          ? parseInt(retryAfter) * 1000 
          : RETRY_DELAY_BASE * Math.pow(2, retryCount);
        
        await new Promise(resolve => setTimeout(resolve, delay));
        return apiClient(originalRequest);
      }
    }

    throw error;
  }
);

// =============================================================================
// Helper Functions
// =============================================================================

export function getApiBaseUrl(): string {
  return API_BASE_URL;
}

// Build FormData from object
function toFormData(data: Record<string, unknown>): FormData {
  const formData = new FormData();
  Object.entries(data).forEach(([key, value]) => {
    if (value !== undefined && value !== null) {
      if (typeof value === 'boolean') {
        formData.append(key, value.toString());
      } else if (typeof value === 'number') {
        formData.append(key, value.toString());
      } else if (typeof value === 'string') {
        formData.append(key, value);
      } else if (value instanceof File) {
        formData.append(key, value);
      } else if (Array.isArray(value)) {
        formData.append(key, JSON.stringify(value));
      } else {
        formData.append(key, JSON.stringify(value));
      }
    }
  });
  return formData;
}

// =============================================================================
// API Service
// =============================================================================

export const unifiedApi = {
  // ===========================================================================
  // Authentication
  // ===========================================================================

  async register(data: RegisterRequest): Promise<TokenResponse> {
    const response = await apiClient.post<TokenResponse>('/api/v1/auth/register', data);
    return response.data;
  },

  async login(data: LoginRequest): Promise<TokenResponse> {
    const response = await apiClient.post<TokenResponse>('/api/v1/auth/login', data);
    return response.data;
  },

  async refreshToken(refreshToken: string): Promise<TokenResponse> {
    const response = await apiClient.post<TokenResponse>(
      '/api/v1/auth/refresh',
      null,
      { params: { refresh_token: refreshToken } }
    );
    return response.data;
  },

  async getCurrentUser(): Promise<User> {
    const response = await apiClient.get<User>('/api/v1/auth/me');
    return response.data;
  },

  // ===========================================================================
  // File Upload
  // ===========================================================================

  async uploadFile(
    file: File,
    options?: {
      gradeLevel?: number;
      subject?: string;
      processForQA?: boolean;
      onProgress?: (progress: number) => void;
    }
  ): Promise<{
    status: string;
    content_id: string;
    file_path: string;
    filename: string;
    size: number;
    extracted_text?: string;
    qa_processing?: {
      enabled: boolean;
      task_id: string | null;
    };
  }> {
    const formData = new FormData();
    formData.append('file', file);
    
    if (options?.gradeLevel) {
      formData.append('grade_level', options.gradeLevel.toString());
    }
    if (options?.subject) {
      formData.append('subject', options.subject);
    }
    formData.append('process_for_qa', (options?.processForQA ?? true).toString());

    const response = await apiClient.post('/api/v1/content/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total && options?.onProgress) {
          const progress = (progressEvent.loaded / progressEvent.total) * 100;
          options.onProgress(progress);
        }
      },
    });
    return response.data;
  },

  async uploadChunk(
    chunk: Blob,
    metadata: {
      filename: string;
      uploadId: string;
      chunkIndex: number;
      totalChunks: number;
      checksum?: string;
    },
    onProgress?: (progress: number) => void
  ): Promise<{ status: string; message: string }> {
    const formData = new FormData();
    formData.append('file', chunk, metadata.filename);
    formData.append('metadata', JSON.stringify({
      filename: metadata.filename,
      upload_id: metadata.uploadId,
      chunk_index: metadata.chunkIndex,
      total_chunks: metadata.totalChunks,
      ...(metadata.checksum && { checksum: metadata.checksum }),
    }));

    const response = await apiClient.post('/api/v1/content/upload/chunked', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total && onProgress) {
          const progress = (progressEvent.loaded / progressEvent.total) * 100;
          onProgress(progress);
        }
      },
    });
    return response.data;
  },

  // ===========================================================================
  // Content Processing
  // ===========================================================================

  async processContent(
    filePath: string,
    data: ProcessRequest
  ): Promise<{ task_id: string; state: string; message?: string }> {
    const response = await apiClient.post(
      '/api/v1/content/process',
      {
        grade_level: data.grade_level,
        subject: data.subject,
        target_languages: data.target_languages,
        output_format: data.output_format || 'both',
        validation_threshold: data.validation_threshold || 0.8,
      },
      { params: { file_path: filePath } }
    );
    return response.data;
  },

  /**
   * Simplify text - Uses full language names (Hindi, Tamil, etc.)
   */
  async simplifyText(
    data: SimplifyRequest,
    waitForResult = false
  ): Promise<{ task_id: string; state: string; simplified_text?: string }> {
    const response = await apiClient.post(
      '/api/v1/content/simplify',
      data,
      { params: { wait: waitForResult } }
    );
    return response.data;
  },

  /**
   * Translate text - Uses full language names (Hindi, Tamil, Telugu, etc.)
   * NOT language codes!
   */
  async translateText(
    data: TranslateRequest,
    waitForResult = false
  ): Promise<{ 
    task_id: string; 
    state: string; 
    translated_text?: string;
    translations?: Record<string, string>;
  }> {
    // Ensure we send target_languages as backend expects
    const payload = {
      text: data.text,
      target_languages: data.target_languages || (data.target_language ? [data.target_language] : ['Hindi']),
      source_language: data.source_language,
      subject: data.subject,
    };
    
    const response = await apiClient.post(
      '/api/v1/content/translate',
      payload,
      { params: { wait: waitForResult } }
    );
    return response.data;
  },

  async validateContent(
    data: ValidateRequest,
    waitForResult = false
  ): Promise<{ 
    task_id: string; 
    state: string;
    is_valid?: boolean;
    accuracy_score?: number;
    issues?: Array<{ severity: string; message: string }>;
  }> {
    const response = await apiClient.post(
      '/api/v1/content/validate',
      data,
      { params: { wait: waitForResult } }
    );
    return response.data;
  },

  async generateAudio(
    data: TTSRequest,
    waitForResult = true
  ): Promise<{ 
    task_id: string; 
    state: string;
    audio_url?: string;
    audio_path?: string;
    duration?: number;
  }> {
    const response = await apiClient.post(
      '/api/v1/content/tts',
      data,
      { params: { wait: waitForResult } }
    );
    return response.data;
  },

  // ===========================================================================
  // Task Management
  // ===========================================================================

  async getTaskStatus(taskId: string): Promise<TaskStatus> {
    const response = await apiClient.get<TaskStatus>(`/api/v1/content/tasks/${taskId}`);
    return response.data;
  },

  async cancelTask(taskId: string, terminate = false): Promise<{ message: string }> {
    const response = await apiClient.delete(`/api/v1/content/tasks/${taskId}`, {
      params: { terminate },
    });
    return response.data;
  },

  /**
   * Poll task until completion with adaptive intervals
   */
  async pollTaskUntilComplete(
    taskId: string,
    options?: {
      maxWait?: number;
      onProgress?: (status: TaskStatus) => void;
    }
  ): Promise<TaskStatus> {
    const maxWait = options?.maxWait || 120000; // 2 minutes default
    const startTime = Date.now();
    let pollInterval = 500; // Start at 500ms
    
    while (Date.now() - startTime < maxWait) {
      const status = await this.getTaskStatus(taskId);
      options?.onProgress?.(status);
      
      if (status.state === 'SUCCESS' || status.state === 'FAILURE' || status.state === 'REVOKED') {
        return status;
      }
      
      // Adaptive polling - increase interval up to 2 seconds
      await new Promise(resolve => setTimeout(resolve, pollInterval));
      pollInterval = Math.min(pollInterval * 1.5, 2000);
    }
    
    throw new Error('Task polling timeout');
  },

  // ===========================================================================
  // Content Library
  // ===========================================================================

  async getContent(contentId: string): Promise<ProcessedContent> {
    const response = await apiClient.get<ProcessedContent>(
      `/api/v1/content/content/${contentId}`
    );
    return response.data;
  },

  async getLibrary(filters: LibraryFilters): Promise<PaginatedResponse<ProcessedContent>> {
    const response = await apiClient.get<PaginatedResponse<ProcessedContent>>(
      '/api/v1/content/library',
      { params: filters }
    );
    return response.data;
  },

  async searchContent(params: SearchParams): Promise<{ results: ProcessedContent[] }> {
    const response = await apiClient.get('/api/v1/content/content/search', { params });
    return response.data;
  },

  getAudioUrl(contentId: string, language?: string): string {
    const params = language ? `?language=${language}` : '';
    return `${API_BASE_URL}/api/v1/content/audio/${contentId}${params}`;
  },

  async submitFeedback(data: FeedbackRequest): Promise<{ message: string }> {
    const response = await apiClient.post('/api/v1/content/feedback', data);
    return response.data;
  },

  // ===========================================================================
  // Q&A (Uses FormData as backend expects)
  // ===========================================================================

  async processDocumentForQA(
    contentId: string,
    chunkSize = 512,
    overlap = 50
  ): Promise<{ task_id: string; message: string; content_id: string }> {
    const formData = toFormData({
      content_id: contentId,
      chunk_size: chunkSize,
      overlap: overlap,
    });

    const response = await apiClient.post('/api/v1/qa/process', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  async askQuestion(
    contentId: string,
    question: string,
    options?: {
      wait?: boolean;
      topK?: number;
    }
  ): Promise<{
    answer?: string;
    confidence_score?: number;
    task_id: string;
    message?: string;
  }> {
    const formData = toFormData({
      content_id: contentId,
      question: question,
      wait: options?.wait ?? false,
      top_k: options?.topK ?? 3,
    });

    const response = await apiClient.post('/api/v1/qa/ask', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  async getQAHistory(
    contentId: string,
    limit = 10
  ): Promise<{ history: Array<{ role: string; content: string; created_at: string }>; count: number }> {
    const response = await apiClient.get(`/api/v1/qa/history/${contentId}`, {
      params: { limit },
    });
    return response.data;
  },

  // ===========================================================================
  // Progress Tracking (Note: Uses /api/progress/ NOT /api/v1/progress/)
  // ===========================================================================

  async updateProgress(data: {
    user_id: string;
    content_id: string;
    progress_percent: number;
    time_spent_seconds: number;
    progress_data?: Record<string, unknown>;
  }): Promise<{
    id: number;
    user_id: string;
    content_id: string;
    progress_percent: number;
    completed: boolean;
    time_spent_seconds: number;
  }> {
    const response = await apiClient.post('/api/progress/update', data);
    return response.data;
  },

  async getUserProgress(userId: string): Promise<Array<{
    id: number;
    user_id: string;
    content_id: string;
    progress_percent: number;
    completed: boolean;
    time_spent_seconds: number;
    started_at: string;
    last_accessed: string;
  }>> {
    const response = await apiClient.get(`/api/progress/user/${userId}`);
    return response.data;
  },

  async submitQuiz(data: {
    user_id: string;
    content_id: string;
    quiz_id: string;
    score: number;
    max_score?: number;
    time_taken_seconds?: number;
    answers?: Record<string, unknown>;
  }): Promise<{ id: number; score: number; passed: boolean }> {
    const response = await apiClient.post('/api/progress/quiz/submit', data);
    return response.data;
  },

  // ===========================================================================
  // Health
  // ===========================================================================

  async getHealth(): Promise<HealthCheck> {
    const response = await apiClient.get<HealthCheck>('/health');
    return response.data;
  },

  async getDetailedHealth(): Promise<DetailedHealthCheck> {
    const response = await apiClient.get<DetailedHealthCheck>('/health/detailed');
    return response.data;
  },
};

// Export as default for backward compatibility
export default unifiedApi;
