/**
 * Progress Tracking Service
 * 
 * Student progress, quiz scores, achievements, and parent reports
 */

import { apiClient } from './client';

// Types
export interface ProgressUpdate {
  user_id: string;
  content_id: string;
  progress_percent: number;
  time_spent_seconds: number;
  progress_data?: Record<string, any>;
}

export interface QuizSubmission {
  user_id: string;
  content_id: string;
  quiz_id: string;
  score: number;
  max_score?: number;
  time_taken_seconds?: number;
  answers?: Record<string, any>;
}

export interface SessionStart {
  user_id: string;
  content_id: string;
  device_type?: string;
}

export interface SessionEnd {
  session_id: number;
  interactions?: number;
  pages_viewed?: number;
  videos_watched?: number;
  exercises_completed?: number;
}

export interface ProgressEntry {
  id: number;
  user_id: string;
  content_id: string;
  progress_percent: number;
  completed: boolean;
  time_spent_seconds: number;
  started_at: string;
  last_accessed: string;
}

export interface QuizScore {
  id: number;
  user_id: string;
  content_id: string;
  quiz_id: string;
  score: number;
  max_score: number;
  percentage: number;
  passed: boolean;
  submitted_at: string;
}

export interface Achievement {
  id: number;
  name: string;
  description: string;
  badge_url: string;
  earned_at: string;
  category: string;
}

export interface LearningStats {
  total_time_spent: number;
  contents_completed: number;
  average_score: number;
  current_streak: number;
  total_achievements: number;
}

export interface ParentReport {
  id: number;
  user_id: string;
  period_start: string;
  period_end: string;
  summary: {
    time_spent: number;
    contents_completed: number;
    quiz_average: number;
    strengths: string[];
    areas_for_improvement: string[];
  };
  created_at: string;
  pdf_url?: string;
}

// Progress Service
class ProgressService {
  /**
   * Update learning progress
   */
  async updateProgress(data: ProgressUpdate): Promise<ProgressEntry> {
    const response = await apiClient.post<ProgressEntry>('/api/progress/update', data);
    return response.data;
  }

  /**
   * Get user's progress on specific content
   */
  async getProgress(userId: string, contentId: string): Promise<ProgressEntry | null> {
    try {
      const response = await apiClient.get<ProgressEntry>(`/api/progress/${userId}/${contentId}`);
      return response.data;
    } catch (error: any) {
      if (error.response?.status === 404) return null;
      throw error;
    }
  }

  /**
   * Get all progress entries for a user
   */
  async getUserProgress(
    userId: string,
    options?: { completed?: boolean; limit?: number; offset?: number }
  ): Promise<{ items: ProgressEntry[]; total: number }> {
    const response = await apiClient.get(`/api/progress/user/${userId}`, {
      params: options,
    });
    return response.data;
  }

  /**
   * Submit quiz score
   */
  async submitQuiz(data: QuizSubmission): Promise<QuizScore> {
    const response = await apiClient.post<QuizScore>('/api/progress/quiz/submit', data);
    return response.data;
  }

  /**
   * Get quiz scores for user
   */
  async getQuizScores(
    userId: string,
    options?: { contentId?: string; limit?: number }
  ): Promise<QuizScore[]> {
    const response = await apiClient.get<QuizScore[]>(`/api/progress/quiz/scores/${userId}`, {
      params: options,
    });
    return response.data;
  }

  /**
   * Start a learning session
   */
  async startSession(data: SessionStart): Promise<{ session_id: number }> {
    const response = await apiClient.post<{ session_id: number }>('/api/progress/session/start', data);
    return response.data;
  }

  /**
   * End a learning session
   */
  async endSession(data: SessionEnd): Promise<{ success: boolean; duration_seconds: number }> {
    const response = await apiClient.post('/api/progress/session/end', data);
    return response.data;
  }

  /**
   * Get learning statistics
   */
  async getStats(userId: string, period?: 'week' | 'month' | 'year'): Promise<LearningStats> {
    const response = await apiClient.get<LearningStats>(`/api/progress/stats/${userId}`, {
      params: { period },
    });
    return response.data;
  }

  /**
   * Get user achievements
   */
  async getAchievements(userId: string): Promise<Achievement[]> {
    const response = await apiClient.get<Achievement[]>(`/api/progress/achievements/${userId}`);
    return response.data;
  }

  /**
   * Generate parent report
   */
  async generateParentReport(
    userId: string,
    periodStart: string,
    periodEnd: string
  ): Promise<ParentReport> {
    const response = await apiClient.post<ParentReport>('/api/progress/report/generate', {
      user_id: userId,
      period_start: periodStart,
      period_end: periodEnd,
    });
    return response.data;
  }

  /**
   * Get parent reports
   */
  async getParentReports(userId: string, limit?: number): Promise<ParentReport[]> {
    const response = await apiClient.get<ParentReport[]>(`/api/progress/reports/${userId}`, {
      params: { limit },
    });
    return response.data;
  }
}

export const progressService = new ProgressService();
export default progressService;
