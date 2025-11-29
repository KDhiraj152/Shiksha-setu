import { apiClient } from './client';
import type { HealthCheck, DetailedHealthCheck } from '../types/api';

/**
 * AI Services health status
 */
export interface AIHealthStatus {
  status: 'healthy' | 'degraded' | 'unavailable';
  orchestrator?: string;
  memory?: {
    max_memory_mb: number;
    used_memory_mb: number;
    available_memory_mb: number;
    loaded_services: Record<string, number>;
  };
  services?: Record<string, string>;
  config?: {
    device: string;
    compute_type: string;
    max_memory_gb: number;
  };
  loaded_models?: {
    translator: boolean;
    tts: boolean;
    simplifier: boolean;
    embeddings: boolean;
  };
  error?: string;
  message?: string;
  timestamp: string;
}

/**
 * Health check API endpoints
 */
export const healthApi = {
  /**
   * Basic health check
   */
  check: async (): Promise<HealthCheck> => {
    const response = await apiClient.get<HealthCheck>('/health');
    return response.data;
  },

  /**
   * Detailed health check with component status
   */
  detailed: async (): Promise<DetailedHealthCheck> => {
    const response = await apiClient.get<DetailedHealthCheck>('/health/detailed');
    return response.data;
  },

  /**
   * AI services health check (new optimized stack)
   */
  aiServices: async (): Promise<AIHealthStatus> => {
    const response = await apiClient.get<AIHealthStatus>('/health/ai');
    return response.data;
  },

  /**
   * Preload AI models into memory
   */
  preloadAI: async (services?: string[]): Promise<{
    success: boolean;
    loaded: string[];
    errors?: Array<{ service: string; error: string }>;
    memory_usage?: Record<string, number>;
  }> => {
    const response = await apiClient.post('/health/ai/preload', { services });
    return response.data;
  },

  /**
   * Readiness probe (Kubernetes)
   */
  ready: async (): Promise<{ ready: boolean }> => {
    const response = await apiClient.get('/health/ready');
    return response.data;
  },

  /**
   * Liveness probe (Kubernetes)
   */
  live: async (): Promise<{ alive: boolean }> => {
    const response = await apiClient.get('/health/live');
    return response.data;
  },
};

export default healthApi;
