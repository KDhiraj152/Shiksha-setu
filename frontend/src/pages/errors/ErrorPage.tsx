import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import { Home, RefreshCw, AlertTriangle } from 'lucide-react';
import { Button } from '../../components/ui/Button/Button';

interface ErrorPageProps {
  error?: Error;
  resetErrorBoundary?: () => void;
}

/**
 * Generic Error Page
 */
export function ErrorPage({ error, resetErrorBoundary }: ErrorPageProps) {
  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center max-w-md"
      >
        {/* Error Illustration */}
        <div className="mb-8">
          <div className="w-24 h-24 mx-auto rounded-full bg-red-100 dark:bg-red-900/30 flex items-center justify-center">
            <AlertTriangle className="w-12 h-12 text-red-600 dark:text-red-400" />
          </div>
        </div>

        <h1 className="text-2xl font-bold text-foreground mb-2">
          Something Went Wrong
        </h1>
        <p className="text-muted-foreground mb-4">
          We encountered an unexpected error. Please try again.
        </p>

        {error && (
          <div className="mb-8 p-4 rounded-lg bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800">
            <p className="text-sm text-red-700 dark:text-red-300 font-mono break-all">
              {error.message}
            </p>
          </div>
        )}

        <div className="flex flex-col sm:flex-row items-center justify-center gap-3">
          {resetErrorBoundary && (
            <Button variant="gradient" onClick={resetErrorBoundary}>
              <RefreshCw className="w-4 h-4 mr-2" />
              Try Again
            </Button>
          )}
          <Link to="/" className="btn btn-outline inline-flex items-center">
            <Home className="w-4 h-4 mr-2" />
            Go Home
          </Link>
        </div>
      </motion.div>
    </div>
  );
}

export default ErrorPage;
