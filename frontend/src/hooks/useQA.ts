import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { qaApi } from '../services/qa';

// Query keys for Q&A related queries
export const qaKeys = {
  all: ['qa'] as const,
  history: (contentId: string) => [...qaKeys.all, 'history', contentId] as const,
};

/**
 * Hook to get Q&A history for a document
 */
export function useQAHistory(contentId: string, limit = 10, enabled = true) {
  return useQuery({
    queryKey: qaKeys.history(contentId),
    queryFn: () => qaApi.getHistory(contentId, limit),
    enabled: enabled && !!contentId,
    staleTime: 1 * 60 * 1000, // 1 minute
  });
}

/**
 * Hook to process a document for Q&A
 */
export function useProcessForQA() {
  return useMutation({
    mutationFn: ({
      contentId,
      chunkSize = 512,
      overlap = 50,
    }: {
      contentId: string;
      chunkSize?: number;
      overlap?: number;
    }) => qaApi.processDocument(contentId, chunkSize, overlap),
  });
}

/**
 * Hook to ask a question
 */
export function useAskQuestion() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({
      contentId,
      question,
      wait = false,
      topK = 3,
    }: {
      contentId: string;
      question: string;
      wait?: boolean;
      topK?: number;
    }) => qaApi.ask(contentId, question, wait, topK),
    onSuccess: (_, variables) => {
      // Invalidate history to show new Q&A
      queryClient.invalidateQueries({ queryKey: qaKeys.history(variables.contentId) });
    },
  });
}

/**
 * Hook to clear Q&A history
 */
export function useClearQAHistory() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (contentId: string) => qaApi.clearHistory(contentId),
    onSuccess: (_, contentId) => {
      queryClient.setQueryData(qaKeys.history(contentId), { history: [], count: 0 });
    },
  });
}
