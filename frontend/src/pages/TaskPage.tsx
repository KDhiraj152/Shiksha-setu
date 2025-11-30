import { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import { api } from '../services/api';
import type { TaskStatus } from '../types/api';
import { Button, Spinner, Badge } from '../components/ui';
import { TaskProgress } from '../components/molecules';

type TaskStage = 'uploading' | 'processing' | 'simplifying' | 'translating' | 'validating' | 'generating' | 'complete';

// Map API stages to our TaskProgress stages
const mapStageToTaskStage = (stage?: string): TaskStage | undefined => {
  if (!stage) return undefined;
  
  const stageMap: Record<string, TaskStage> = {
    'upload': 'uploading',
    'uploading': 'uploading',
    'extraction': 'processing',
    'processing': 'processing',
    'simplification': 'simplifying',
    'simplifying': 'simplifying',
    'translation': 'translating',
    'translating': 'translating',
    'validation': 'validating',
    'validating': 'validating',
    'audio': 'generating',
    'generating': 'generating',
    'audio_generation': 'generating',
    'complete': 'complete',
    'completed': 'complete',
    'done': 'complete',
  };
  
  return stageMap[stage.toLowerCase()] || 'processing';
};

// Icons
const ArrowLeftIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 19l-7-7m0 0l7-7m-7 7h18" />
  </svg>
);

const CheckCircleIcon = () => (
  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
);

const ExclamationCircleIcon = () => (
  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
  </svg>
);

export default function TaskPage() {
  const { taskId } = useParams<{ taskId: string }>();
  const navigate = useNavigate();
  const [taskStatus, setTaskStatus] = useState<TaskStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isRedirecting, setIsRedirecting] = useState(false);

  useEffect(() => {
    if (!taskId) return;

    const pollTask = async () => {
      try {
        const status = await api.getTaskStatus(taskId);
        setTaskStatus(status);

        // Check if task completed successfully
        if (status.state === 'SUCCESS' && status.result) {
          // Different result types have different structures
          const result = status.result;
          
          // If result has an id (from file upload/process), redirect to content page
          if (result.id || result.content_id) {
            setIsRedirecting(true);
            setTimeout(() => {
              navigate(`/content/${result.id || result.content_id}`);
            }, 2000);
          }
          // Otherwise, task completed but show results on this page (simplify/translate/etc)
          // Don't redirect - just show success state
        } else if (status.state === 'FAILURE') {
          setError(status.error || 'Task failed');
        }
      } catch (err: unknown) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to get task status';
        setError(errorMessage);
      }
    };

    const interval = setInterval(pollTask, 2000);
    pollTask();

    return () => clearInterval(interval);
  }, [taskId, navigate]);

  // Error state
  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-primary-50/50 via-surface-50 to-secondary-50/50 dark:from-surface-950 dark:via-surface-900 dark:to-surface-950 p-4 sm:p-8">
        <div className="max-w-2xl mx-auto">
          <Link
            to="/upload"
            className="inline-flex items-center gap-2 text-sm text-muted-600 hover:text-surface-900 dark:text-muted-400 dark:hover:text-surface-100 mb-6 transition-colors"
          >
            <ArrowLeftIcon />
            Back to Upload
          </Link>

          <div className="glass-card p-6 sm:p-8">
            <div className="flex items-start gap-4">
              <div className="flex-shrink-0 w-12 h-12 rounded-full bg-error-100 dark:bg-error-900/30 flex items-center justify-center">
                <ExclamationCircleIcon />
              </div>
              <div className="flex-1 min-w-0">
                <h1 className="text-xl font-semibold text-error-900 dark:text-error-200">
                  Processing Failed
                </h1>
                <p className="mt-2 text-error-700 dark:text-error-300">
                  {error}
                </p>
              </div>
            </div>

            <div className="mt-6 flex flex-col sm:flex-row gap-3">
              <Button onClick={() => navigate('/upload')} className="flex-1">
                Try Again
              </Button>
              <Button variant="outline" onClick={() => navigate('/')} className="flex-1">
                Go Home
              </Button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Loading state (no task status yet)
  if (!taskStatus) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-primary-50/50 via-surface-50 to-secondary-50/50 dark:from-surface-950 dark:via-surface-900 dark:to-surface-950 p-4 sm:p-8">
        <div className="max-w-2xl mx-auto">
          <div className="glass-card p-8 sm:p-12 text-center">
            <Spinner size="lg" className="mx-auto" />
            <p className="mt-4 text-muted-600 dark:text-muted-400">
              Connecting to processing server...
            </p>
          </div>
        </div>
      </div>
    );
  }

  const currentStage = taskStatus.state === 'SUCCESS' 
    ? 'complete' 
    : mapStageToTaskStage(taskStatus.stage);

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50/50 via-surface-50 to-secondary-50/50 dark:from-surface-950 dark:via-surface-900 dark:to-surface-950 p-4 sm:p-8">
      <div className="max-w-3xl mx-auto">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-2xl sm:text-3xl font-bold bg-gradient-to-r from-primary-600 to-secondary-600 bg-clip-text text-transparent">
            Processing Content
          </h1>
          <p className="mt-1 text-muted-600 dark:text-muted-400">
            Task ID: <code className="text-xs bg-surface-100 dark:bg-surface-800 px-2 py-1 rounded">{taskId}</code>
          </p>
        </div>

        {/* Main Content */}
        <div className="glass-card p-6 sm:p-8">
          {/* Status Badge */}
          <div className="flex items-center justify-between mb-6">
            <span className="text-sm font-medium text-surface-700 dark:text-surface-300">
              Status
            </span>
            <Badge
              variant={
                taskStatus.state === 'SUCCESS' ? 'success' :
                taskStatus.state === 'FAILURE' ? 'error' :
                taskStatus.state === 'PENDING' ? 'warning' :
                'info'
              }
            >
              {taskStatus.state}
            </Badge>
          </div>

          {/* Task Progress Component */}
          <TaskProgress
            currentStage={currentStage}
            progress={taskStatus.progress || 0}
            message={taskStatus.message}
          />

          {/* Success State */}
          {taskStatus.state === 'SUCCESS' && (
            <div className="mt-6 p-4 bg-success-50 dark:bg-success-900/20 border border-success-200 dark:border-success-800 rounded-lg">
              <div className="flex items-center gap-3">
                <div className="text-success-600 dark:text-success-400">
                  <CheckCircleIcon />
                </div>
                <div>
                  <p className="font-medium text-success-800 dark:text-success-200">
                    Processing Complete!
                  </p>
                  <p className="text-sm text-success-600 dark:text-success-400">
                    {isRedirecting ? 'Redirecting to results...' : 'Ready to view results'}
                  </p>
                </div>
              </div>
              
              {isRedirecting && (
                <div className="mt-4 flex justify-center">
                  <Spinner size="sm" />
                </div>
              )}

              {!isRedirecting && taskStatus.result && (
                <div className="mt-4 space-y-3">
                  {/* Show results directly for simple tasks (simplify, translate, etc) */}
                  {taskStatus.result.simplified_text && (
                    <div className="bg-surface-50 dark:bg-surface-800/50 p-4 rounded-lg">
                      <p className="text-sm font-medium text-surface-700 dark:text-surface-300 mb-2">Simplified Text:</p>
                      <p className="text-sm text-surface-600 dark:text-surface-400">{taskStatus.result.simplified_text}</p>
                    </div>
                  )}
                  
                  {taskStatus.result.translations && (
                    <div className="bg-surface-50 dark:bg-surface-800/50 p-4 rounded-lg">
                      <p className="text-sm font-medium text-surface-700 dark:text-surface-300 mb-2">Translations:</p>
                      {Object.entries(taskStatus.result.translations).map(([lang, text]) => (
                        <div key={lang} className="mt-2">
                          <span className="text-xs font-medium text-surface-500 dark:text-surface-400">{lang}:</span>
                          <p className="text-sm text-surface-600 dark:text-surface-400">{text as string}</p>
                        </div>
                      ))}
                    </div>
                  )}
                  
                  {/* Only show View Results button if there's a content_id or id */}
                  {(taskStatus.result.id || taskStatus.result.content_id) && (
                    <Button
                      onClick={() => navigate(`/content/${taskStatus.result.id || taskStatus.result.content_id}`)}
                      className="w-full"
                    >
                      View Full Results
                    </Button>
                  )}
                  
                  {/* If no content_id, show action buttons */}
                  {!(taskStatus.result.id || taskStatus.result.content_id) && (
                    <div className="flex gap-3">
                      <Button
                        onClick={() => navigate('/simplify')}
                        variant="outline"
                        className="flex-1"
                      >
                        Simplify More
                      </Button>
                      <Button
                        onClick={() => navigate('/translate')}
                        variant="outline"
                        className="flex-1"
                      >
                        Translate More
                      </Button>
                    </div>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Estimated Time */}
          {(taskStatus.state === 'PROCESSING' || taskStatus.state === 'STARTED') && (
            <div className="mt-6 text-center">
              <p className="text-sm text-muted-500 dark:text-muted-400">
                Estimated time remaining: ~{Math.max(1, Math.round((100 - (taskStatus.progress || 0)) / 20))} minute(s)
              </p>
            </div>
          )}
        </div>

        {/* Info Cards */}
        <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="glass-card p-4">
            <h3 className="text-sm font-medium text-surface-900 dark:text-surface-100 mb-1">
              What's happening?
            </h3>
            <p className="text-xs text-muted-600 dark:text-muted-400">
              Your content is being processed through our AI pipeline: text extraction, 
              simplification for the target grade level, translation, and audio generation.
            </p>
          </div>
          <div className="glass-card p-4">
            <h3 className="text-sm font-medium text-surface-900 dark:text-surface-100 mb-1">
              Can I leave this page?
            </h3>
            <p className="text-xs text-muted-600 dark:text-muted-400">
              Yes! Processing continues in the background. You can find your content 
              in the Library once complete, or we'll redirect you automatically.
            </p>
          </div>
        </div>

        {/* Back Link */}
        <div className="mt-6 text-center">
          <Link
            to="/library"
            className="text-sm text-primary-600 hover:text-primary-700 dark:text-primary-400 dark:hover:text-primary-300 transition-colors"
          >
            Go to Library →
          </Link>
        </div>
      </div>
    </div>
  );
}
