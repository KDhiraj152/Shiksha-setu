import { useMemo } from 'react';
import { cn } from '../ui/utils';
import { Progress } from '../ui/Progress';
import { Spinner } from '../ui/Spinner';

export type TaskStage = 
  | 'uploading'
  | 'processing'
  | 'simplifying'
  | 'translating'
  | 'validating'
  | 'generating'
  | 'complete'
  | 'failed';

export interface TaskProgressProps {
  /** Current stage of processing */
  currentStage?: TaskStage;
  /** Overall progress percentage (0-100) */
  progress?: number;
  /** Current message */
  message?: string;
  /** Error message if failed */
  error?: string;
  /** Show detailed stages */
  showStages?: boolean;
  /** Additional class name */
  className?: string;
}

// Stage configuration
const STAGES: { key: TaskStage; label: string; description: string }[] = [
  { key: 'uploading', label: 'Uploading', description: 'Uploading file...' },
  { key: 'processing', label: 'Processing', description: 'Extracting text content...' },
  { key: 'simplifying', label: 'Simplifying', description: 'Simplifying complex terms...' },
  { key: 'translating', label: 'Translating', description: 'Translating to target language...' },
  { key: 'validating', label: 'Validating', description: 'Checking quality...' },
  { key: 'generating', label: 'Generating', description: 'Generating audio...' },
  { key: 'complete', label: 'Complete', description: 'All done!' },
];

// Icons
const CheckIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
  </svg>
);

const XIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
  </svg>
);

/**
 * Task progress tracker with visual stage indicators and progress bar.
 * 
 * @example
 * <TaskProgress
 *   currentStage="translating"
 *   progress={65}
 *   message="Translating to Hindi..."
 *   showStages
 * />
 */
export function TaskProgress({
  currentStage = 'processing',
  progress = 0,
  message,
  error,
  showStages = true,
  className,
}: TaskProgressProps) {
  // Get current stage index
  const currentStageIndex = useMemo(() => {
    if (currentStage === 'complete') return STAGES.length - 1;
    if (currentStage === 'failed') return -1;
    return STAGES.findIndex(s => s.key === currentStage);
  }, [currentStage]);

  // Get stage description
  const stageDescription = useMemo(() => {
    if (error) return error;
    if (message) return message;
    const stageInfo = STAGES.find(s => s.key === currentStage);
    return stageInfo?.description || 'Processing...';
  }, [currentStage, message, error]);

  return (
    <div className={cn('space-y-6', className)}>
      {/* Progress Bar */}
      <div>
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium text-surface-700 dark:text-surface-300">
            {stageDescription}
          </span>
          <span className="text-sm font-medium text-primary-600 dark:text-primary-400">
            {progress}%
          </span>
        </div>
        <Progress 
          value={progress} 
          variant={error ? 'error' : currentStage === 'complete' ? 'success' : 'default'}
          size="md"
        />
      </div>

      {/* Stage Indicators */}
      {showStages && (
        <div className="relative">
          {/* Progress Line */}
          <div className="absolute top-5 left-0 right-0 h-0.5 bg-surface-200 dark:bg-surface-700">
            <div
              className="h-full bg-primary-500 transition-all duration-500"
              style={{ 
                width: `${((currentStageIndex + 1) / STAGES.length) * 100}%` 
              }}
            />
          </div>

          {/* Stages */}
          <div className="relative grid grid-cols-7 gap-1">
            {STAGES.map((stage, index) => {
              const isComplete = index < currentStageIndex;
              const isCurrent = index === currentStageIndex;
              const isFailed = currentStage === 'failed' && isCurrent;
              
              return (
                <div key={stage.key} className="flex flex-col items-center">
                  {/* Stage Icon */}
                  <div
                    className={cn(
                      'relative z-10 flex items-center justify-center w-10 h-10 rounded-full border-2 transition-all duration-300',
                      isComplete && 'bg-primary-500 border-primary-500',
                      isCurrent && !isFailed && 'bg-primary-100 border-primary-500 dark:bg-primary-900/30',
                      isFailed && 'bg-error-100 border-error-500 dark:bg-error-900/30',
                      !isComplete && !isCurrent && !isFailed && 'bg-surface-100 border-surface-300 dark:bg-surface-800 dark:border-surface-600'
                    )}
                  >
                    {isComplete ? (
                      <CheckIcon />
                    ) : isFailed ? (
                      <span className="text-error-600 dark:text-error-400">
                        <XIcon />
                      </span>
                    ) : isCurrent ? (
                      <Spinner size="sm" />
                    ) : (
                      <span className={cn(
                        'text-xs font-medium',
                        'text-surface-500 dark:text-surface-400'
                      )}>
                        {index + 1}
                      </span>
                    )}
                  </div>

                  {/* Stage Label */}
                  <span
                    className={cn(
                      'mt-2 text-xs font-medium text-center transition-colors',
                      (isComplete || isCurrent) && !isFailed && 'text-primary-600 dark:text-primary-400',
                      isFailed && 'text-error-600 dark:text-error-400',
                      !isComplete && !isCurrent && !isFailed && 'text-surface-500 dark:text-surface-400'
                    )}
                  >
                    {stage.label}
                  </span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className="p-4 bg-error-50 dark:bg-error-900/20 border border-error-200 dark:border-error-800 rounded-lg">
          <div className="flex items-start gap-3">
            <span className="text-error-500 flex-shrink-0 mt-0.5">
              <XIcon />
            </span>
            <div>
              <p className="text-sm font-medium text-error-800 dark:text-error-200">
                Processing Failed
              </p>
              <p className="text-xs text-error-600 dark:text-error-400 mt-1">
                {error}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

TaskProgress.displayName = 'TaskProgress';
