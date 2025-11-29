import { apiClient, getApiBaseUrl } from './client';
import type {
  ProcessRequest,
  SimplifyRequest,
  TranslateRequest,
  ValidateRequest,
  TTSRequest,
  TaskStatus,
  ProcessedContent,
  FeedbackRequest,
  PaginatedResponse,
  LibraryFilters,
  SearchParams,
} from '../types/api';

/**
 * Content API endpoints
 */
export const contentApi = {
  /**
   * Upload a file for processing
   */
  upload: async (
    file: File,
    onProgress?: (progress: number) => void
  ): Promise<{ file_path: string; content_id: string; status: string; extracted_text?: string }> => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await apiClient.post(
      '/api/v1/content/upload',
      formData,
      {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const progress = (progressEvent.loaded / progressEvent.total) * 100;
            onProgress?.(progress);
          }
        },
      }
    );
    return response.data;
  },

  /**
   * Upload file in chunks for large files
   */
  uploadChunk: async (
    chunk: Blob,
    fileName: string,
    uploadId: string,
    chunkIndex: number,
    totalChunks: number,
    checksum?: string,
    onProgress?: (progress: number) => void
  ): Promise<{ status: string; message: string }> => {
    const formData = new FormData();
    formData.append('file', chunk, fileName);

    const metadata: Record<string, string | number> = {
      filename: fileName,
      upload_id: uploadId,
      chunk_index: chunkIndex,
      total_chunks: totalChunks,
    };

    if (checksum) {
      metadata.checksum = checksum;
    }

    formData.append('metadata', JSON.stringify(metadata));

    const response = await apiClient.post(
      '/api/v1/content/upload/chunked',
      formData,
      {
        headers: { 'Content-Type': 'multipart/form-data' },
        onUploadProgress: (progressEvent) => {
          if (progressEvent.total) {
            const progress = (progressEvent.loaded / progressEvent.total) * 100;
            onProgress?.(progress);
          }
        },
      }
    );
    return response.data;
  },

  /**
   * Process uploaded content
   */
  process: async (
    data: ProcessRequest
  ): Promise<{ task_id: string; status: string; message?: string }> => {
    const response = await apiClient.post(
      '/api/v1/content/process',
      {
        grade_level: data.grade_level,
        subject: data.subject,
        target_languages: data.target_languages,
        output_format: data.output_format || 'both',
        validation_threshold: data.validation_threshold || 0.8,
      },
      {
        params: { file_path: data.file_path },
      }
    );
    return response.data;
  },

  /**
   * Simplify text content
   */
  simplify: async (data: SimplifyRequest): Promise<{ task_id: string; status: string }> => {
    const response = await apiClient.post('/api/v1/content/simplify', data);
    return response.data;
  },

  /**
   * Translate text content
   */
  translate: async (data: TranslateRequest): Promise<{ task_id: string; status: string }> => {
    const response = await apiClient.post('/api/v1/content/translate', data);
    return response.data;
  },

  /**
   * Validate content against curriculum standards
   */
  validate: async (data: ValidateRequest): Promise<{ task_id: string; status: string }> => {
    const response = await apiClient.post('/api/v1/content/validate', data);
    return response.data;
  },

  /**
   * Generate audio from text (TTS)
   */
  generateAudio: async (data: TTSRequest): Promise<{ task_id: string; status: string }> => {
    const response = await apiClient.post('/api/v1/content/tts', data);
    return response.data;
  },

  /**
   * Get task status
   */
  getTaskStatus: async (taskId: string): Promise<TaskStatus> => {
    const response = await apiClient.get<TaskStatus>(`/api/v1/content/tasks/${taskId}`);
    return response.data;
  },

  /**
   * Cancel a running task
   */
  cancelTask: async (taskId: string, terminate = false): Promise<{ message: string }> => {
    const response = await apiClient.delete(`/api/v1/content/tasks/${taskId}`, {
      params: { terminate },
    });
    return response.data;
  },

  /**
   * Get processed content by ID
   */
  getContent: async (contentId: string): Promise<ProcessedContent> => {
    const response = await apiClient.get<ProcessedContent>(
      `/api/v1/content/content/${contentId}`
    );
    return response.data;
  },

  /**
   * Get audio URL for content
   */
  getAudioUrl: (contentId: string, language?: string): string => {
    const params = language ? `?language=${language}` : '';
    return `${getApiBaseUrl()}/api/v1/content/audio/${contentId}${params}`;
  },

  /**
   * Submit feedback for content
   */
  submitFeedback: async (data: FeedbackRequest): Promise<{ message: string }> => {
    const response = await apiClient.post('/api/v1/content/feedback', data);
    return response.data;
  },

  /**
   * Get content library with filters
   */
  getLibrary: async (filters: LibraryFilters): Promise<PaginatedResponse<ProcessedContent>> => {
    const response = await apiClient.get<PaginatedResponse<ProcessedContent>>(
      '/api/v1/content/library',
      { params: filters }
    );
    return response.data;
  },

  /**
   * Search content
   */
  search: async (params: SearchParams): Promise<{ results: ProcessedContent[] }> => {
    const response = await apiClient.get<{ results: ProcessedContent[] }>(
      '/api/v1/content/content/search',
      { params }
    );
    return response.data;
  },

  /**
   * Delete content
   */
  delete: async (contentId: string): Promise<{ message: string }> => {
    const response = await apiClient.delete(`/api/v1/content/content/${contentId}`);
    return response.data;
  },
};

export default contentApi;
