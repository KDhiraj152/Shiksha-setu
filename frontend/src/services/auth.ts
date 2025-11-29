import { apiClient } from './client';
import type {
  User,
  TokenResponse,
  LoginRequest,
  RegisterRequest,
  RefreshRequest,
} from '../types/api';

/**
 * Authentication API endpoints
 */
export const authApi = {
  /**
   * Register a new user
   */
  register: async (data: RegisterRequest): Promise<TokenResponse> => {
    const response = await apiClient.post<TokenResponse>('/api/v1/auth/register', data);
    return response.data;
  },

  /**
   * Login user
   */
  login: async (data: LoginRequest): Promise<TokenResponse> => {
    const response = await apiClient.post<TokenResponse>('/api/v1/auth/login', data);
    return response.data;
  },

  /**
   * Refresh access token
   */
  refresh: async (data: RefreshRequest): Promise<TokenResponse> => {
    const response = await apiClient.post<TokenResponse>('/api/v1/auth/refresh', data);
    return response.data;
  },

  /**
   * Get current user profile
   */
  getCurrentUser: async (): Promise<User> => {
    const response = await apiClient.get<User>('/api/v1/auth/me');
    return response.data;
  },

  /**
   * Update user profile
   */
  updateProfile: async (data: Partial<User>): Promise<User> => {
    const response = await apiClient.patch<User>('/api/v1/auth/me', data);
    return response.data;
  },

  /**
   * Change password
   */
  changePassword: async (data: { 
    current_password: string; 
    new_password: string 
  }): Promise<{ message: string }> => {
    const response = await apiClient.post<{ message: string }>(
      '/api/v1/auth/change-password', 
      data
    );
    return response.data;
  },

  /**
   * Logout - clear local storage
   */
  logout: (): void => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
  },

  /**
   * Store tokens after login/register
   */
  storeTokens: (tokens: TokenResponse): void => {
    localStorage.setItem('access_token', tokens.access_token);
    localStorage.setItem('refresh_token', tokens.refresh_token);
  },

  /**
   * Check if user is authenticated
   */
  isAuthenticated: (): boolean => {
    return !!localStorage.getItem('access_token');
  },
};

export default authApi;
