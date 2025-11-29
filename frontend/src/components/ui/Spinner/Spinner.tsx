import { cn } from '../../../lib/cn';

export interface SpinnerProps {
  /** Spinner size */
  size?: 'sm' | 'md' | 'lg' | 'xl';
  /** Custom className */
  className?: string;
  /** Accessible label */
  label?: string;
}

/**
 * Premium loading spinner with smooth animation.
 * 
 * @example
 * <Spinner size="lg" />
 * <Spinner size="sm" className="text-white" />
 */
export function Spinner({ size = 'md', className, label = 'Loading' }: SpinnerProps) {
  const sizeStyles = {
    sm: 'spinner-sm',
    md: 'spinner-md',
    lg: 'spinner-lg',
    xl: 'spinner-xl',
  };

  return (
    <span
      className={cn('spinner', sizeStyles[size], className)}
      role="status"
      aria-label={label}
    >
      <span className="sr-only">{label}</span>
    </span>
  );
}

export default Spinner;
