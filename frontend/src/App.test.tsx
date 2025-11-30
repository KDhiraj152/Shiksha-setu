import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import App from './App';

describe('App Component', () => {
  it('renders without crashing', () => {
    const { container } = render(<App />);
    expect(container).toBeTruthy();
  });

  it('contains the main app div', () => {
    const { container } = render(<App />);
    const mainDiv = container.querySelector('div');
    expect(mainDiv).toHaveClass('min-h-screen');
  });
});
