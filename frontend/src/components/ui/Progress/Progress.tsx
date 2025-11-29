import { type HTMLAttributes } from 'react';
import { cn } from '../../../lib/cn';

export interface ProgressProps extends HTMLAttributes<HTMLDivElement> {
  /** Progress value (0-100) */
  value?: number;
  /** Maximum value */
  max?: number;
  /** Progress bar size */
  size?: 'sm' | 'md' | 'lg';
  /** Show indeterminate animation */
  indeterminate?: boolean;
  /** Show label */
  showLabel?: boolean;
  /** Custom label text */
  label?: string;
  /** Progress bar variant */
  variant?: 'default' | 'success' | 'warning' | 'error';
}

/**
 * Premium progress bar with gradient fill and animations.
 * 
 * @example
 * <Progress value={75} showLabel />
 * <Progress indeterminate />
 */
export function Progress({
  className,
  value = 0,
  max = 100,
  size = 'md',
  indeterminate = false,
  showLabel = false,
  label,
  variant = 'default',
  ...props
}: ProgressProps) {
  const percentage = Math.min(100, Math.max(0, (value / max) * 100));

  const sizeStyles = {
    sm: 'progress-sm',
    md: '',
    lg: 'progress-lg',
  };

  const variantStyles = {
    default: 'bg-gradient-to-r from-[rgb(var(--color-primary-500))] to-[rgb(var(--color-secondary-500))]',
    success: 'bg-[rgb(var(--color-success-500))]',
    warning: 'bg-[rgb(var(--color-warning-500))]',
    error: 'bg-[rgb(var(--color-error-500))]',
  };

  return (
    <div className={cn('w-full', className)} {...props}>
      {(showLabel || label) && (
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-[rgb(var(--color-fg-secondary))]">
            {label || 'Progress'}
          </span>
          {!indeterminate && (
            <span className="text-sm font-medium text-[rgb(var(--color-fg-muted))] tabular-nums">
              {Math.round(percentage)}%
            </span>
          )}
        </div>
      )}
      <div
        className={cn('progress', sizeStyles[size], indeterminate && 'progress-indeterminate')}
        role="progressbar"
        aria-valuenow={indeterminate ? undefined : percentage}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <div
          className={cn('progress-bar', variantStyles[variant])}
          style={indeterminate ? undefined : { width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

export default Progress;
