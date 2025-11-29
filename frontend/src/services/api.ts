/**
 * @deprecated Use unifiedApi from './unifiedApi' instead.
 * This file is kept for backward compatibility but will be removed in a future version.
 * 
 * Migration:
 *   // Before
 *   import { api } from './api';
 *   api.login(...)
 * 
 *   // After
 *   import { unifiedApi } from './unifiedApi';
 *   unifiedApi.login(...)
 */

// Re-export everything from unifiedApi for backward compatibility
export { unifiedApi as api, unifiedApi as default, apiClient } from './unifiedApi';

// Re-export types
export type {
  User,
  TokenResponse,
  LoginRequest,
  RegisterRequest,
  RefreshRequest,
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
  RateLimitInfo,
  ApiError,
  PaginatedResponse,
  LibraryFilters,
  SearchParams
} from '../types/api';
