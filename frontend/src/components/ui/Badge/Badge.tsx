import { type HTMLAttributes, type ReactNode } from 'react';
import { cn } from '../../../lib/cn';

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  /** Badge visual variant */
  variant?: 'primary' | 'secondary' | 'success' | 'warning' | 'error' | 'neutral';
  /** Badge size */
  size?: 'sm' | 'md' | 'lg';
  /** Optional icon to show before text */
  icon?: ReactNode;
  /** Optional dot indicator */
  dot?: boolean;
}

/**
 * Premium badge component for labels, tags, and status indicators.
 * 
 * @example
 * <Badge variant="success" icon={<CheckIcon />}>Completed</Badge>
 * <Badge variant="warning" dot>Pending</Badge>
 */
export function Badge({
  className,
  variant = 'primary',
  size = 'md',
  icon,
  dot,
  children,
  ...props
}: BadgeProps) {
  const variantStyles = {
    primary: 'badge-primary',
    secondary: 'badge-secondary',
    success: 'badge-success',
    warning: 'badge-warning',
    error: 'badge-error',
    neutral: 'badge-neutral',
  };

  const sizeStyles = {
    sm: 'badge-sm',
    md: '',
    lg: 'badge-lg',
  };

  const dotColors = {
    primary: 'bg-[rgb(var(--color-primary-500))]',
    secondary: 'bg-[rgb(var(--color-secondary-500))]',
    success: 'bg-[rgb(var(--color-success-500))]',
    warning: 'bg-[rgb(var(--color-warning-500))]',
    error: 'bg-[rgb(var(--color-error-500))]',
    neutral: 'bg-[rgb(var(--color-fg-muted))]',
  };

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
      {dot && (
        <span className={cn('w-1.5 h-1.5 rounded-full', dotColors[variant])} />
      )}
      {icon && <span className="shrink-0">{icon}</span>}
      {children}
    </span>
  );
}

export default Badge;
