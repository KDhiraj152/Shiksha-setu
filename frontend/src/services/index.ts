/**
 * API Services - ShikshaSetu
 * 
 * Modular API layer with separate services for each domain
 */

// Base client
export { apiClient, getApiBaseUrl } from './client';

// Domain APIs
export { authApi } from './auth';
export { contentApi } from './content';
export { qaApi } from './qa';
export { healthApi } from './health';

// New services
export { streamingService } from './streaming';
export { progressService } from './progress';
export { reviewService } from './reviews';
export type { ReviewStatus } from './reviews';

// Re-export types
export type {
  ProgressUpdate,
  QuizSubmission,
  ProgressEntry,
  QuizScore,
  Achievement,
  LearningStats,
  ParentReport,
} from './progress';

export type {
  Review,
  ReviewCreate,
  ReviewUpdate,
  ReviewComment,
  ReviewVersion,
} from './reviews';

// Primary unified API (recommended)
export { unifiedApi, apiClient as unifiedApiClient } from './unifiedApi';

// Legacy export for backward compatibility
// @deprecated Use unifiedApi instead
export { api } from './api';
