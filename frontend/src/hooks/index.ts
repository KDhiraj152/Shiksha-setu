/**
 * Custom React Hooks
 * 
 * Centralized export for all application hooks
 */

// API Hooks (TanStack Query)
export {
  // Query Keys
  queryKeys,
  
  // Auth
  useCurrentUser,
  useLogin,
  useRegister,
  useLogout,
  
  // Content
  useUploadFile,
  useProcessContent,
  useSimplifyText,
  useTranslateText,
  useValidateContent,
  useGenerateAudio,
  useContent,
  useLibrary,
  useInfiniteLibrary,
  useSearchContent,
  useSubmitFeedback,
  
  // Tasks
  useTaskStatus,
  useCancelTask,
  
  // Q&A
  useProcessDocumentForQA,
  useAskQuestion,
  useQAHistory,
  
  // Health
  useHealth,
  useDetailedHealth,
  
  // Utils
  usePrefetchContent,
  useInvalidateContent,
} from './useApi';

// Modular hooks (new architecture)
export { authKeys } from './useAuth';
export { contentKeys } from './useContent';
export { qaKeys } from './useQA';

// WebSocket Hook
export { useWebSocket, type WebSocketStatus, type TaskUpdate, type UseWebSocketOptions } from './useWebSocket';

// Offline Hook
export { useOffline, type UseOfflineReturn } from './useOffline';
