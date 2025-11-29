/**
 * Translation Review Service
 * 
 * Collaborative translation review, versioning, and approval workflows
 */

import { apiClient } from './client';

// Types
export interface ReviewCreate {
  translation_id: string;
  original_text: string;
  translated_text: string;
  source_lang: string;
  target_lang: string;
  metadata?: Record<string, any>;
}

export interface ReviewUpdate {
  translated_text?: string;
  status?: ReviewStatus;
  feedback?: string;
}

// Use union type instead of enum to avoid erasableSyntaxOnly error
export type ReviewStatus = 'pending' | 'approved' | 'rejected' | 'revised';

export const ReviewStatusValues = {
  Pending: 'pending' as ReviewStatus,
  Approved: 'approved' as ReviewStatus,
  Rejected: 'rejected' as ReviewStatus,
  Revised: 'revised' as ReviewStatus,
};

export interface Review {
  id: string;
  translation_id: string;
  original_text: string;
  translated_text: string;
  source_lang: string;
  target_lang: string;
  reviewer_id: string;
  reviewer_name?: string;
  status: ReviewStatus;
  feedback?: string;
  created_at: string;
  updated_at: string;
  metadata?: Record<string, any>;
  // Extended fields for UI
  content_title?: string;
  comments_count?: number;
  version?: number;
}

export interface ReviewComment {
  id: string;
  review_id: string;
  user_id: string;
  author_name?: string;
  text: string;
  position?: { start: number; end: number };
  resolved: boolean;
  created_at: string;
}

export interface ReviewVersion {
  id: string;
  review_id: string;
  version: number;
  translated_text: string;
  created_by: string;
  created_at: string;
  change_summary?: string;
}

export interface ReviewFilters {
  status?: ReviewStatus;
  reviewer_id?: string;
  source_lang?: string;
  target_lang?: string;
  skip?: number;
  limit?: number;
}

// Review Service
class ReviewService {
  /**
   * Create a new review request
   */
  async createReview(data: ReviewCreate): Promise<Review> {
    const response = await apiClient.post<Review>('/api/v1/reviews', data);
    return response.data;
  }

  /**
   * Get review by ID
   */
  async getReview(reviewId: string): Promise<Review> {
    const response = await apiClient.get<Review>(`/api/v1/reviews/${reviewId}`);
    return response.data;
  }

  /**
   * List reviews with filters
   */
  async listReviews(filters?: ReviewFilters): Promise<Review[]> {
    const response = await apiClient.get<Review[]>('/api/v1/reviews', {
      params: {
        status_filter: filters?.status,
        reviewer_id: filters?.reviewer_id,
        skip: filters?.skip || 0,
        limit: filters?.limit || 50,
      },
    });
    return response.data;
  }

  /**
   * Update review (status, text, feedback)
   */
  async updateReview(reviewId: string, data: ReviewUpdate): Promise<Review> {
    const response = await apiClient.patch<Review>(`/api/v1/reviews/${reviewId}`, data);
    return response.data;
  }

  /**
   * Approve a review
   */
  async approveReview(reviewId: string, feedback?: string): Promise<Review> {
    return this.updateReview(reviewId, {
      status: 'approved',
      feedback,
    });
  }

  /**
   * Reject a review
   */
  async rejectReview(reviewId: string, feedback: string): Promise<Review> {
    return this.updateReview(reviewId, {
      status: 'rejected',
      feedback,
    });
  }

  /**
   * Request revision
   */
  async requestRevision(reviewId: string, feedback: string): Promise<Review> {
    return this.updateReview(reviewId, {
      status: 'revised',
      feedback,
    });
  }

  /**
   * Add comment to review
   */
  async addComment(
    reviewId: string,
    content: string,
    position?: { start: number; end: number }
  ): Promise<ReviewComment> {
    const response = await apiClient.post<ReviewComment>(`/api/v1/reviews/${reviewId}/comments`, {
      content,
      position,
    });
    return response.data;
  }

  /**
   * Get comments for review
   */
  async getComments(reviewId: string): Promise<ReviewComment[]> {
    const response = await apiClient.get<ReviewComment[]>(`/api/v1/reviews/${reviewId}/comments`);
    return response.data;
  }

  /**
   * Resolve a comment
   */
  async resolveComment(reviewId: string, commentId: string): Promise<ReviewComment> {
    const response = await apiClient.patch<ReviewComment>(
      `/api/v1/reviews/${reviewId}/comments/${commentId}`,
      { resolved: true }
    );
    return response.data;
  }

  /**
   * Get version history
   */
  async getVersionHistory(reviewId: string): Promise<ReviewVersion[]> {
    const response = await apiClient.get<ReviewVersion[]>(`/api/v1/reviews/${reviewId}/versions`);
    return response.data;
  }

  /**
   * Get versions (alias for getVersionHistory)
   */
  async getVersions(reviewId: string): Promise<ReviewVersion[]> {
    return this.getVersionHistory(reviewId);
  }

  /**
   * Get reviews with filters (alias for listReviews)
   */
  async getReviews(filters?: ReviewFilters): Promise<Review[]> {
    return this.listReviews(filters);
  }

  /**
   * Create new version
   */
  async createVersion(
    reviewId: string,
    translatedText: string,
    changeSummary?: string
  ): Promise<ReviewVersion> {
    const response = await apiClient.post<ReviewVersion>(`/api/v1/reviews/${reviewId}/versions`, {
      translated_text: translatedText,
      change_summary: changeSummary,
    });
    return response.data;
  }

  /**
   * Get review statistics
   */
  async getStats(): Promise<{
    pending: number;
    approved: number;
    rejected: number;
    revised: number;
    total: number;
  }> {
    const response = await apiClient.get('/api/v1/reviews/stats');
    return response.data;
  }

  /**
   * Get reviews assigned to current user
   */
  async getMyReviews(status?: ReviewStatus): Promise<Review[]> {
    const response = await apiClient.get<Review[]>('/api/v1/reviews/my', {
      params: { status_filter: status },
    });
    return response.data;
  }
}

export const reviewService = new ReviewService();
export default reviewService;
