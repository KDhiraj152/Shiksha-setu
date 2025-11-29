import { useMutation, useQuery, useQueryClient, useInfiniteQuery } from '@tanstack/react-query';
import { contentApi } from '../services/content';
import type {
  ProcessRequest,
  SimplifyRequest,
  TranslateRequest,
  ValidateRequest,
  TTSRequest,
  LibraryFilters,
  SearchParams,
  FeedbackRequest,
} from '../types/api';

// Query keys for content-related queries
export const contentKeys = {
  all: ['content'] as const,
  library: (filters: LibraryFilters) => [...contentKeys.all, 'library', filters] as const,
  search: (params: SearchParams) => [...contentKeys.all, 'search', params] as const,
  detail: (id: string) => [...contentKeys.all, 'detail', id] as const,
  task: (id: string) => [...contentKeys.all, 'task', id] as const,
};

/**
 * Hook to get content library with pagination and filters
 */
export function useLibrary(filters: LibraryFilters) {
  return useQuery({
    queryKey: contentKeys.library(filters),
    queryFn: () => contentApi.getLibrary(filters),
    staleTime: 2 * 60 * 1000, // 2 minutes
  });
}

/**
 * Hook for infinite scrolling library
 */
export function useInfiniteLibrary(filters: Omit<LibraryFilters, 'offset'>) {
  return useInfiniteQuery({
    queryKey: [...contentKeys.all, 'infinite-library', filters],
    queryFn: ({ pageParam = 0 }) => 
      contentApi.getLibrary({ ...filters, offset: pageParam }),
    initialPageParam: 0,
    getNextPageParam: (lastPage, pages) => {
      const pageSize = filters.limit || 20;
      const currentOffset = pages.length * pageSize;
      return currentOffset < lastPage.total ? currentOffset : undefined;
    },
    staleTime: 2 * 60 * 1000,
  });
}

/**
 * Hook to search content
 */
export function useSearchContent(params: SearchParams, enabled = true) {
  return useQuery({
    queryKey: contentKeys.search(params),
    queryFn: () => contentApi.search(params),
    enabled: enabled && !!params.q,
    staleTime: 1 * 60 * 1000, // 1 minute
  });
}

/**
 * Hook to get single content item
 */
export function useContent(contentId: string, enabled = true) {
  return useQuery({
    queryKey: contentKeys.detail(contentId),
    queryFn: () => contentApi.getContent(contentId),
    enabled: enabled && !!contentId,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
}

/**
 * Hook to get task status with polling
 */
export function useTaskStatus(taskId: string, enabled = true) {
  return useQuery({
    queryKey: contentKeys.task(taskId),
    queryFn: () => contentApi.getTaskStatus(taskId),
    enabled: enabled && !!taskId,
    refetchInterval: (query) => {
      const state = query.state.data?.state;
      // Stop polling when task is complete or failed
      if (state === 'SUCCESS' || state === 'FAILURE' || state === 'REVOKED') {
        return false;
      }
      return 2000; // Poll every 2 seconds
    },
    staleTime: 0, // Always fetch fresh data
  });
}

/**
 * Hook to upload file
 */
export function useUploadFile() {
  return useMutation({
    mutationFn: ({ 
      file, 
      onProgress 
    }: { 
      file: File; 
      onProgress?: (progress: number) => void 
    }) => contentApi.upload(file, onProgress),
  });
}

/**
 * Hook to process content
 */
export function useProcessContent() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (data: ProcessRequest) => contentApi.process(data),
    onSuccess: () => {
      // Invalidate library to show new processing task
      queryClient.invalidateQueries({ queryKey: contentKeys.all });
    },
  });
}

/**
 * Hook to simplify text
 */
export function useSimplifyText() {
  return useMutation({
    mutationFn: (data: SimplifyRequest) => contentApi.simplify(data),
  });
}

/**
 * Hook to translate text
 */
export function useTranslateText() {
  return useMutation({
    mutationFn: (data: TranslateRequest) => contentApi.translate(data),
  });
}

/**
 * Hook to validate content
 */
export function useValidateContent() {
  return useMutation({
    mutationFn: (data: ValidateRequest) => contentApi.validate(data),
  });
}

/**
 * Hook to generate audio (TTS)
 */
export function useGenerateAudio() {
  return useMutation({
    mutationFn: (data: TTSRequest) => contentApi.generateAudio(data),
  });
}

/**
 * Hook to cancel a task
 */
export function useCancelTask() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: ({ taskId, terminate = false }: { taskId: string; terminate?: boolean }) =>
      contentApi.cancelTask(taskId, terminate),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: contentKeys.task(variables.taskId) });
    },
  });
}

/**
 * Hook to submit feedback
 */
export function useSubmitFeedback() {
  return useMutation({
    mutationFn: (data: FeedbackRequest) => contentApi.submitFeedback(data),
  });
}

/**
 * Hook to delete content
 */
export function useDeleteContent() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: (contentId: string) => contentApi.delete(contentId),
    onSuccess: (_, contentId) => {
      queryClient.removeQueries({ queryKey: contentKeys.detail(contentId) });
      queryClient.invalidateQueries({ queryKey: [...contentKeys.all, 'library'] });
    },
  });
}
