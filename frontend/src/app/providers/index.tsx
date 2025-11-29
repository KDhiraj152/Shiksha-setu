import type { ReactNode } from 'react';
import { QueryProvider } from './QueryProvider';

interface AppProvidersProps {
  children: ReactNode;
}

/**
 * Root provider component that wraps all providers needed by the app.
 * Order matters - providers are nested from outside to inside.
 */
export function AppProviders({ children }: AppProvidersProps) {
  return (
    <QueryProvider>
      {/* Add other providers here as needed */}
      {/* ErrorBoundary, ToastProvider, etc. */}
      {children}
    </QueryProvider>
  );
}

export { QueryProvider } from './QueryProvider';

export default AppProviders;
