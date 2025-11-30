import { type HTMLAttributes } from 'react';
import { cn } from './utils';

export interface SpinnerProps extends HTMLAttributes<HTMLDivElement> {
  /** Spinner size */
  size?: 'sm' | 'md' | 'lg' | 'xl';
  /** Spinner color variant */
  variant?: 'primary' | 'white' | 'muted';
  /** Accessible label for screen readers */
  label?: string;
}

/**
 * Loading spinner component with multiple sizes and colors.
 * 
 * @example
 * <Spinner size="lg" label="Loading content..." />
 */
export function Spinner({
  className,
  size = 'md',
  variant = 'primary',
  label = 'Loading...',
  ...props
}: SpinnerProps) {
  const sizeStyles = {
    sm: 'w-4 h-4 border-2',
    md: 'w-6 h-6 border-2',
    lg: 'w-8 h-8 border-[3px]',
    xl: 'w-12 h-12 border-4',
  };

  const colorStyles = {
    primary: 'border-primary-500/30 border-t-primary-500',
    white: 'border-white/30 border-t-white',
    muted: 'border-surface-300 border-t-surface-500',
  };

  return (
    <div
      className={cn(
        'animate-spin rounded-full',
        sizeStyles[size],
        colorStyles[variant],
        className
      )}
      role="status"
      aria-label={label}
      {...props}
    >
      <span className="sr-only">{label}</span>
    </div>
  );
}

Spinner.displayName = 'Spinner';
