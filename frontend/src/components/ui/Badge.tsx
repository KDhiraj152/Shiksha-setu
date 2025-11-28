import { type HTMLAttributes, type ReactNode } from 'react';
import { cn } from './utils';

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  /** Badge visual style variant */
  variant?: 'primary' | 'secondary' | 'success' | 'warning' | 'error' | 'neutral' | 'info';
  /** Badge size */
  size?: 'sm' | 'md' | 'lg';
  /** Icon to show before text */
  icon?: ReactNode;
  /** Make badge a dot indicator (no text) */
  dot?: boolean;
}

/**
 * Badge component for status indicators and labels.
 * 
 * @example
 * <Badge variant="success" icon={<CheckIcon />}>Completed</Badge>
 * <Badge variant="warning">Processing</Badge>
 * <Badge variant="error" dot />
 */
export function Badge({
  className,
  variant = 'primary',
  size = 'md',
  icon,
  dot = false,
  children,
  ...props
}: BadgeProps) {
  const variantStyles = {
    primary: 'badge-primary',
    secondary: 'bg-secondary-100 text-secondary-700 dark:bg-secondary-900/30 dark:text-secondary-400',
    success: 'badge-success',
    warning: 'badge-warning',
    error: 'badge-error',
    neutral: 'badge-neutral',
    info: 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400',
  };

  const sizeStyles = {
    sm: 'px-1.5 py-0.5 text-2xs',
    md: 'px-2 py-0.5 text-xs',
    lg: 'px-2.5 py-1 text-sm',
  };

  if (dot) {
    const dotColors: Record<string, string> = {
      primary: 'bg-primary-500',
      secondary: 'bg-secondary-500',
      success: 'bg-success-500',
      warning: 'bg-warning-500',
      error: 'bg-error-500',
      neutral: 'bg-surface-400',
      info: 'bg-blue-500',
    };

    return (
      <span
        className={cn(
          'inline-block rounded-full',
          dotColors[variant],
          size === 'sm' && 'w-1.5 h-1.5',
          size === 'md' && 'w-2 h-2',
          size === 'lg' && 'w-2.5 h-2.5',
          className
        )}
        aria-hidden="true"
        {...props}
      />
    );
  }

  return (
    <span
      className={cn(
        'badge',
        variantStyles[variant],
        sizeStyles[size],
        className
      )}
      {...props}
    >
      {icon && <span className="shrink-0">{icon}</span>}
      {children}
    </span>
  );
}

Badge.displayName = 'Badge';
