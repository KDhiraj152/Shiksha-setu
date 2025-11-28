import { type HTMLAttributes } from 'react';
import { cn } from './utils';

export interface ProgressProps extends HTMLAttributes<HTMLDivElement> {
  /** Current progress value (0-100) */
  value: number;
  /** Maximum value (default 100) */
  max?: number;
  /** Show percentage text */
  showLabel?: boolean;
  /** Progress size variant */
  size?: 'sm' | 'md' | 'lg';
  /** Progress color variant */
  variant?: 'primary' | 'success' | 'warning' | 'error';
  /** Indeterminate loading state */
  indeterminate?: boolean;
  /** Custom label text (overrides percentage) */
  label?: string;
  /** Animated progress bar */
  animated?: boolean;
}

/**
 * Progress bar component with multiple variants and sizes.
 * Supports determinate and indeterminate states.
 * 
 * @example
 * <Progress value={65} showLabel variant="success" />
 * <Progress indeterminate label="Processing..." />
 */
export function Progress({
  className,
  value,
  max = 100,
  showLabel = false,
  size = 'md',
  variant = 'primary',
  indeterminate = false,
  label,
  animated = false,
  ...props
}: ProgressProps) {
  const percentage = Math.min(100, Math.max(0, (value / max) * 100));
  
  const sizeStyles = {
    sm: 'h-1',
    md: 'h-2',
    lg: 'h-3',
  };

  const colorStyles = {
    primary: 'bg-primary-500',
    success: 'bg-success-500',
    warning: 'bg-warning-500',
    error: 'bg-error-500',
  };

  return (
    <div className={cn('w-full', className)} {...props}>
      {(showLabel || label) && (
        <div className="flex justify-between mb-1.5">
          <span className="text-sm font-medium text-surface-700 dark:text-surface-300">
            {label || 'Progress'}
          </span>
          {showLabel && !indeterminate && (
            <span className="text-sm font-medium text-surface-500">
              {Math.round(percentage)}%
            </span>
          )}
        </div>
      )}
      
      <div
        className={cn(
          'w-full bg-surface-200 dark:bg-surface-700 rounded-full overflow-hidden',
          sizeStyles[size]
        )}
        role="progressbar"
        aria-valuenow={indeterminate ? undefined : value}
        aria-valuemin={0}
        aria-valuemax={max}
        aria-label={label || `${Math.round(percentage)}% complete`}
      >
        {indeterminate ? (
          <div
            className={cn(
              'h-full rounded-full animate-progress',
              colorStyles[variant],
              'w-1/3'
            )}
          />
        ) : (
          <div
            className={cn(
              'h-full rounded-full transition-all duration-300 ease-out',
              colorStyles[variant],
              animated && 'animate-pulse-slow'
            )}
            style={{ width: `${percentage}%` }}
          />
        )}
      </div>
    </div>
  );
}

Progress.displayName = 'Progress';
