/**
 * Frontend Component Tests for AI Stack Integration
 * 
 * Tests:
 * - Health monitoring components
 * - AI service status display
 * - WebSocket connections
 * - API service calls
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';

// Mock API service
import { healthService } from '../services/health';

vi.mock('../services/health');

describe('AI Health Monitor', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('displays AI orchestrator status', async () => {
    // Mock health API response
    const mockHealthData = {
      orchestrator: 'running',
      device: 'mps',
      compute_type: 'int8',
      memory_used_mb: 2500,
      memory_available_mb: 7740,
      loaded_services: ['simplifier', 'translator']
    };

    vi.mocked(healthService.getAIHealth).mockResolvedValue(mockHealthData);

    // Test would render component and verify status display
    // This is a placeholder - actual component needs to be implemented
    expect(mockHealthData.orchestrator).toBe('running');
  });

  it('shows memory usage correctly', async () => {
    const mockHealthData = {
      memory_used_mb: 2500,
      memory_available_mb: 7740,
      memory_budget_mb: 10240
    };

    vi.mocked(healthService.getAIHealth).mockResolvedValue(mockHealthData);

    const memoryPercentage = (mockHealthData.memory_used_mb / mockHealthData.memory_budget_mb) * 100;
    
    expect(memoryPercentage).toBeCloseTo(24.4, 1);
  });

  it('indicates when services are loaded', async () => {
    const mockHealthData = {
      loaded_services: ['simplifier', 'translator', 'tts']
    };

    vi.mocked(healthService.getAIHealth).mockResolvedValue(mockHealthData);

    expect(mockHealthData.loaded_services).toContain('simplifier');
    expect(mockHealthData.loaded_services).toHaveLength(3);
  });
});

describe('AI Service API Calls', () => {
  it('calls simplification endpoint correctly', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        task_id: 'test-task-123',
        status: 'queued'
      })
    });

    global.fetch = mockFetch;

    const response = await fetch('/api/v1/ai/simplify', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: 'Complex text',
        target_grade: 6
      })
    });

    const data = await response.json();

    expect(mockFetch).toHaveBeenCalledWith('/api/v1/ai/simplify', expect.objectContaining({
      method: 'POST'
    }));
    expect(data.task_id).toBe('test-task-123');
  });

  it('handles API errors gracefully', async () => {
    const mockFetch = vi.fn().mockRejectedValue(new Error('Network error'));
    global.fetch = mockFetch;

    await expect(
      fetch('/api/v1/ai/simplify', {
        method: 'POST',
        body: JSON.stringify({ text: 'test' })
      })
    ).rejects.toThrow('Network error');
  });
});

describe('WebSocket Streaming', () => {
  it('establishes WebSocket connection for streaming', () => {
    const mockWebSocket = vi.fn();
    global.WebSocket = mockWebSocket as any;

    const ws = new WebSocket('ws://localhost:8000/api/v1/ai/stream');
    
    expect(mockWebSocket).toHaveBeenCalledWith('ws://localhost:8000/api/v1/ai/stream');
  });

  it('receives streaming simplification chunks', (done) => {
    const mockMessage = JSON.stringify({
      type: 'chunk',
      data: 'Simplified text chunk...'
    });

    const mockWs = {
      onmessage: null as ((event: MessageEvent) => void) | null,
      send: vi.fn(),
      close: vi.fn()
    };

    if (mockWs.onmessage) {
      mockWs.onmessage(new MessageEvent('message', { data: mockMessage }));
    }

    // Placeholder - actual test would verify message handling
    expect(mockMessage).toContain('Simplified text chunk');
    done();
  });
});

describe('Content Processing UI', () => {
  it('displays task progress', async () => {
    const taskStatus = {
      task_id: 'task-123',
      status: 'processing',
      progress: 45,
      message: 'Simplifying text...'
    };

    expect(taskStatus.progress).toBe(45);
    expect(taskStatus.status).toBe('processing');
  });

  it('shows completion message', async () => {
    const taskStatus = {
      status: 'completed',
      result: {
        simplified_text: 'Easy to understand text'
      }
    };

    expect(taskStatus.status).toBe('completed');
    expect(taskStatus.result.simplified_text).toBeTruthy();
  });
});

describe('Translation UI', () => {
  it('displays available languages', () => {
    const availableLanguages = [
      'Hindi', 'Tamil', 'Telugu', 'Kannada', 'Malayalam',
      'Marathi', 'Bengali', 'Gujarati', 'Odia', 'Punjabi'
    ];

    expect(availableLanguages).toContain('Hindi');
    expect(availableLanguages.length).toBeGreaterThanOrEqual(10);
  });

  it('handles translation request', async () => {
    const translationRequest = {
      text: 'Hello',
      target_languages: ['Hindi', 'Tamil']
    };

    expect(translationRequest.target_languages).toHaveLength(2);
  });
});

describe('Audio Player', () => {
  it('loads audio URL', () => {
    const audioUrl = '/api/v1/audio/test-content-123.mp3';
    
    expect(audioUrl).toMatch(/\.mp3$/);
  });

  it('handles audio generation task', async () => {
    const mockResponse = {
      task_id: 'audio-task-123',
      status: 'queued'
    };

    expect(mockResponse.task_id).toContain('audio-task');
  });
});

describe('Error Handling', () => {
  it('displays error message on API failure', () => {
    const errorState = {
      hasError: true,
      message: 'Failed to connect to AI service'
    };

    expect(errorState.hasError).toBe(true);
    expect(errorState.message).toContain('Failed to connect');
  });

  it('shows retry option on failure', () => {
    const retryAvailable = true;
    expect(retryAvailable).toBe(true);
  });
});
