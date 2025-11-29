import { apiClient } from './client';

/**
 * Q&A API endpoints for document-based question answering
 */
export const qaApi = {
  /**
   * Process a document for Q&A (creates embeddings)
   */
  processDocument: async (
    contentId: string,
    chunkSize = 512,
    overlap = 50
  ): Promise<{ task_id: string; message: string }> => {
    const formData = new FormData();
    formData.append('content_id', contentId);
    formData.append('chunk_size', chunkSize.toString());
    formData.append('overlap', overlap.toString());

    const response = await apiClient.post('/api/v1/qa/process', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  /**
   * Ask a question about a document
   */
  ask: async (
    contentId: string,
    question: string,
    wait = false,
    topK = 3
  ): Promise<{
    task_id?: string;
    answer?: string;
    sources?: Array<{ text: string; score: number }>;
    status: string;
  }> => {
    const formData = new FormData();
    formData.append('content_id', contentId);
    formData.append('question', question);
    formData.append('wait', wait.toString());
    formData.append('top_k', topK.toString());

    const response = await apiClient.post('/api/v1/qa/ask', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  /**
   * Get Q&A history for a document
   */
  getHistory: async (
    contentId: string,
    limit = 10
  ): Promise<{
    history: Array<{
      id: string;
      question: string;
      answer: string;
      created_at: string;
    }>;
    count: number;
  }> => {
    const response = await apiClient.get(`/api/v1/qa/history/${contentId}`, {
      params: { limit },
    });
    return response.data;
  },

  /**
   * Clear Q&A history for a document
   */
  clearHistory: async (contentId: string): Promise<{ message: string }> => {
    const response = await apiClient.delete(`/api/v1/qa/history/${contentId}`);
    return response.data;
  },
};

export default qaApi;
