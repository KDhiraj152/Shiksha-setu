import { afterEach, vi } from 'vitest';
import { cleanup } from '@testing-library/react';
import '@testing-library/jest-dom';

// Cleanup after each test
afterEach(() => {
  cleanup();
});

// Mock environment variables
vi.stubGlobal('import', {
  meta: {
    env: {
      VITE_API_BASE_URL: 'http://localhost:8000'
    }
  }
});

// Mock window.matchMedia
Object.defineProperty(globalThis, 'matchMedia', {
  writable: true,
  value: vi.fn().mockImplementation(query => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: vi.fn(),
    removeListener: vi.fn(),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    dispatchEvent: vi.fn(),
  })),
});

// Mock IntersectionObserver
globalThis.IntersectionObserver = class IntersectionObserver {
  disconnect(): void {
    // Mock implementation - no-op
  }
  observe(): void {
    // Mock implementation - no-op
  }
  takeRecords() {
    return [];
  }
  unobserve(): void {
    // Mock implementation - no-op
  }
} as any;
