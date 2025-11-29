/**
 * TanStack Query hooks for all API endpoints
 * Provides type-safe data fetching with caching, refetching, and error handling
 */

import { useQuery, useMutation, useQueryClient, useInfiniteQuery } from '@tanstack/react-query';
import type { UseQueryOptions, UseMutationOptions, UseInfiniteQueryOptions } from '@tanstack/react-query';
import api from '../services/api';
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
  LibraryFilters,
  SearchParams,
  PaginatedResponse,
} from '../types/api';
import { useAuthStore } from '../store/authStore';

// ============================================================================
// Query Keys - Centralized for cache management
// ============================================================================

export const queryKeys = {
  // Auth
  user: ['user'] as const,
  
  // Content
  content: (id: string) => ['content', id] as const,
  library: (filters: LibraryFilters) => ['library', filters] as const,
  search: (params: SearchParams) => ['search', params] as const,
  
  // Tasks
  task: (id: string) => ['task', id] as const,
  
  // Q&A
  qaHistory: (contentId: string) => ['qaHistory', contentId] as const,
  
  // Health
  health: ['health'] as const,
  detailedHealth: ['detailedHealth'] as const,
} as const;

// ============================================================================
// Auth Hooks
// ============================================================================

/**
 * Get current authenticated user
 */
export function useCurrentUser(
  options?: Omit<UseQueryOptions<User, Error>, 'queryKey' | 'queryFn'>
) {
  const { isAuthenticated } = useAuthStore();
  
  return useQuery({
    queryKey: queryKeys.user,
    queryFn: () => api.getCurrentUser(),
    enabled: isAuthenticated,
    staleTime: 5 * 60 * 1000, // 5 minutes
    ...options,
  });
}

/**
 * Login mutation
 */
export function useLogin(
  options?: UseMutationOptions<TokenResponse, Error, LoginRequest>
) {
  const queryClient = useQueryClient();
  const { setTokens } = useAuthStore();
  
  return useMutation({
    mutationFn: (data: LoginRequest) => api.login(data),
    onSuccess: (response) => {
      setTokens(response.access_token, response.refresh_token);
      queryClient.invalidateQueries({ queryKey: queryKeys.user });
    },
    ...options,
  });
}

/**
 * Register mutation
 */
export function useRegister(
  options?: UseMutationOptions<TokenResponse, Error, RegisterRequest>
) {
  const queryClient = useQueryClient();
  const { setTokens } = useAuthStore();
  
  return useMutation({
    mutationFn: (data: RegisterRequest) => api.register(data),
    onSuccess: (response) => {
      setTokens(response.access_token, response.refresh_token);
      queryClient.invalidateQueries({ queryKey: queryKeys.user });
    },
    ...options,
  });
}

/**
 * Logout mutation
 */
export function useLogout() {
  const queryClient = useQueryClient();
  const { logout } = useAuthStore();
  
  return useMutation({
    mutationFn: async () => {
      // Handled in onSuccess
    },
    onSuccess: () => {
      logout();
      queryClient.clear();
    },
  });
}

// ============================================================================
// Content Hooks
// ============================================================================

/**
 * Upload file mutation with progress tracking
 */
export function useUploadFile(
  options?: UseMutationOptions<
    { status: string; content_id: string; file_path: string; filename: string; size: number; extracted_text?: string },
    Error,
    { file: File; onProgress?: (progress: number) => void; gradeLevel?: number; subject?: string; processForQA?: boolean }
  >
) {
  return useMutation({
    mutationFn: ({ file, onProgress, gradeLevel, subject, processForQA }) => api.uploadFile(file, { onProgress, gradeLevel, subject, processForQA }),
    ...options,
  });
}

/**
 * Process content mutation
 */
export function useProcessContent(
  options?: UseMutationOptions<
    { task_id: string; state: string; message?: string },
    Error,
    ProcessRequest & { file_path: string }
  >
) {
  return useMutation({
    mutationFn: (data: ProcessRequest & { file_path: string }) => api.processContent(data.file_path, data),
    ...options,
  });
}

/**
 * Simplify text mutation
 */
export function useSimplifyText(
  options?: UseMutationOptions<{ task_id: string; state: string; simplified_text?: string }, Error, SimplifyRequest>
) {
  return useMutation({
    mutationFn: (data: SimplifyRequest) => api.simplifyText(data),
    ...options,
  });
}

/**
 * Translate text mutation
 */
export function useTranslateText(
  options?: UseMutationOptions<{ task_id: string; state: string; translated_text?: string; translations?: Record<string, string> }, Error, TranslateRequest>
) {
  return useMutation({
    mutationFn: (data: TranslateRequest) => api.translateText(data),
    ...options,
  });
}

/**
 * Validate content mutation
 */
export function useValidateContent(
  options?: UseMutationOptions<{ task_id: string; state: string; is_valid?: boolean; accuracy_score?: number; issues?: Array<{ severity: string; message: string }> }, Error, ValidateRequest>
) {
  return useMutation({
    mutationFn: (data: ValidateRequest) => api.validateContent(data),
    ...options,
  });
}

/**
 * Generate audio mutation
 */
export function useGenerateAudio(
  options?: UseMutationOptions<{ task_id: string; state: string; audio_url?: string; audio_path?: string; duration?: number }, Error, TTSRequest>
) {
  return useMutation({
    mutationFn: (data: TTSRequest) => api.generateAudio(data),
    ...options,
  });
}

/**
 * Get content by ID
 */
export function useContent(
  contentId: string,
  options?: Omit<UseQueryOptions<ProcessedContent, Error>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: queryKeys.content(contentId),
    queryFn: () => api.getContent(contentId),
    enabled: !!contentId,
    ...options,
  });
}

/**
 * Get library with filters (paginated)
 */
export function useLibrary(
  filters: LibraryFilters,
  options?: Omit<UseQueryOptions<PaginatedResponse<ProcessedContent>, Error>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: queryKeys.library(filters),
    queryFn: () => api.getLibrary(filters),
    ...options,
  });
}

/**
 * Infinite library query for infinite scroll
 */
export function useInfiniteLibrary(
  filters: Omit<LibraryFilters, 'offset'>,
  options?: Omit<
    UseInfiniteQueryOptions<PaginatedResponse<ProcessedContent>, Error>,
    'queryKey' | 'queryFn' | 'getNextPageParam' | 'initialPageParam'
  >
) {
  const limit = filters.limit || 20;
  
  return useInfiniteQuery({
    queryKey: ['library', 'infinite', filters],
    queryFn: ({ pageParam }) => api.getLibrary({ ...filters, offset: pageParam as number }),
    initialPageParam: 0,
    getNextPageParam: (lastPage, pages) => {
      const currentOffset = pages.length * limit;
      return lastPage.has_more ? currentOffset : undefined;
    },
    ...options,
  });
}

/**
 * Search content
 */
export function useSearchContent(
  params: SearchParams,
  options?: Omit<UseQueryOptions<{ results: ProcessedContent[] }, Error>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: queryKeys.search(params),
    queryFn: () => api.searchContent(params),
    enabled: !!params.q && params.q.length >= 2,
    ...options,
  });
}

/**
 * Submit feedback mutation
 */
export function useSubmitFeedback(
  options?: UseMutationOptions<{ message: string }, Error, FeedbackRequest>
) {
  return useMutation({
    mutationFn: (data: FeedbackRequest) => api.submitFeedback(data),
    ...options,
  });
}

// ============================================================================
// Task Hooks
// ============================================================================

/**
 * Get task status with polling
 */
export function useTaskStatus(
  taskId: string | null,
  options?: Omit<UseQueryOptions<TaskStatus, Error>, 'queryKey' | 'queryFn'> & {
    pollingInterval?: number;
  }
) {
  const { pollingInterval = 2000, ...queryOptions } = options || {};
  
  return useQuery({
    queryKey: queryKeys.task(taskId || ''),
    queryFn: () => api.getTaskStatus(taskId!),
    enabled: !!taskId,
    refetchInterval: (query) => {
      const state = query.state.data?.state;
      // Stop polling when task is complete or failed
      if (state === 'SUCCESS' || state === 'FAILURE' || state === 'REVOKED') {
        return false;
      }
      return pollingInterval;
    },
    ...queryOptions,
  });
}

/**
 * Cancel task mutation
 */
export function useCancelTask(
  options?: UseMutationOptions<{ message: string }, Error, { taskId: string; terminate?: boolean }>
) {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ taskId, terminate }) => api.cancelTask(taskId, terminate),
    onSuccess: (_, { taskId }) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.task(taskId) });
    },
    ...options,
  });
}

// ============================================================================
// Q&A Hooks
// ============================================================================

/**
 * Process document for Q&A
 */
export function useProcessDocumentForQA(
  options?: UseMutationOptions<
    { task_id: string; message: string },
    Error,
    { contentId: string; chunkSize?: number; overlap?: number }
  >
) {
  return useMutation({
    mutationFn: ({ contentId, chunkSize, overlap }) =>
      api.processDocumentForQA(contentId, chunkSize, overlap),
    ...options,
  });
}

/**
 * Ask question mutation
 */
export function useAskQuestion(
  options?: UseMutationOptions<
    { answer?: string; confidence_score?: number; task_id: string; message?: string },
    Error,
    { contentId: string; question: string; wait?: boolean; topK?: number }
  >
) {
  const queryClient = useQueryClient();
  
  return useMutation({
    mutationFn: ({ contentId, question, wait, topK }) =>
      api.askQuestion(contentId, question, { wait, topK }),
    onSuccess: (_, { contentId }) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.qaHistory(contentId) });
    },
    ...options,
  });
}

/**
 * Get Q&A history
 */
export function useQAHistory(
  contentId: string,
  limit?: number,
  options?: Omit<UseQueryOptions<{ history: unknown[]; count: number }, Error>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: queryKeys.qaHistory(contentId),
    queryFn: () => api.getQAHistory(contentId, limit),
    enabled: !!contentId,
    ...options,
  });
}

// ============================================================================
// Health Hooks
// ============================================================================

/**
 * Basic health check
 */
export function useHealth(
  options?: Omit<UseQueryOptions<HealthCheck, Error>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: queryKeys.health,
    queryFn: () => api.getHealth(),
    staleTime: 30 * 1000, // 30 seconds
    ...options,
  });
}

/**
 * Detailed health check
 */
export function useDetailedHealth(
  options?: Omit<UseQueryOptions<DetailedHealthCheck, Error>, 'queryKey' | 'queryFn'>
) {
  return useQuery({
    queryKey: queryKeys.detailedHealth,
    queryFn: () => api.getDetailedHealth(),
    staleTime: 30 * 1000, // 30 seconds
    ...options,
  });
}

// ============================================================================
// Utility Hooks
// ============================================================================

/**
 * Prefetch content for better UX
 */
export function usePrefetchContent() {
  const queryClient = useQueryClient();
  
  return (contentId: string) => {
    queryClient.prefetchQuery({
      queryKey: queryKeys.content(contentId),
      queryFn: () => api.getContent(contentId),
    });
  };
}

/**
 * Invalidate all content queries
 */
export function useInvalidateContent() {
  const queryClient = useQueryClient();
  
  return () => {
    queryClient.invalidateQueries({ queryKey: ['content'] });
    queryClient.invalidateQueries({ queryKey: ['library'] });
  };
}
