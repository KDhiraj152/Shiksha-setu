import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from 'react';
import { cn } from '../../../lib/cn';

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  /** Visual style variant */
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost' | 'danger' | 'success' | 'gradient';
  /** Button size */
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl';
  /** Show loading spinner */
  isLoading?: boolean;
  /** Loading text (replaces children when loading) */
  loadingText?: string;
  /** Icon to show before children */
  leftIcon?: ReactNode;
  /** Icon to show after children */
  rightIcon?: ReactNode;
  /** Make button full width */
  fullWidth?: boolean;
  /** Icon-only button (square) */
  iconOnly?: boolean;
}

/**
 * Premium button component with gradient primary, multiple variants and sizes.
 * Features smooth hover animations, focus states, and loading support.
 * 
 * @example
 * <Button variant="primary" size="lg" leftIcon={<Upload />}>
 *   Upload File
 * </Button>
 */
export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      className,
      variant = 'primary',
      size = 'md',
      isLoading = false,
      loadingText,
      leftIcon,
      rightIcon,
      fullWidth = false,
      iconOnly = false,
      disabled,
      children,
      ...props
    },
    ref
  ) => {
    const variantStyles = {
      primary: 'btn-primary',
      secondary: 'btn-secondary',
      outline: 'btn-outline',
      ghost: 'btn-ghost',
      danger: 'btn-danger',
      success: 'btn-success',
      gradient: 'btn-gradient bg-gradient-to-r from-primary-500 to-secondary-500 text-white hover:from-primary-600 hover:to-secondary-600 shadow-lg shadow-primary-500/25',
    };

    const sizeStyles = {
      xs: iconOnly ? 'btn-icon w-7 h-7' : 'btn-xs',
      sm: iconOnly ? 'btn-icon btn-sm' : 'btn-sm',
      md: iconOnly ? 'btn-icon' : 'btn-md',
      lg: iconOnly ? 'btn-icon btn-lg' : 'btn-lg',
      xl: iconOnly ? 'btn-icon w-14 h-14' : 'btn-xl',
    };

    return (
      <button
        ref={ref}
        className={cn(
          'btn',
          variantStyles[variant],
          sizeStyles[size],
          fullWidth && 'w-full',
          iconOnly && 'p-0',
          className
        )}
        disabled={disabled || isLoading}
        aria-busy={isLoading}
        {...props}
      >
        {isLoading ? (
          <>
            <Spinner size={size === 'xs' || size === 'sm' ? 'sm' : 'md'} />
            {!iconOnly && <span>{loadingText || children}</span>}
          </>
        ) : (
          <>
            {leftIcon && <span className="shrink-0 -ml-0.5">{leftIcon}</span>}
            {!iconOnly && children}
            {rightIcon && <span className="shrink-0 -mr-0.5">{rightIcon}</span>}
          </>
        )}
      </button>
    );
  }
);

Button.displayName = 'Button';

// Inline spinner to avoid circular dependency
function Spinner({ size = 'md' }: { size?: 'sm' | 'md' | 'lg' }) {
  const sizeClasses = {
    sm: 'w-4 h-4 border-2',
    md: 'w-5 h-5 border-2',
    lg: 'w-6 h-6 border-2',
  };

  return (
    <span
      className={cn(
        'inline-block rounded-full border-current border-t-transparent animate-spin',
        sizeClasses[size]
      )}
      role="status"
      aria-label="Loading"
    />
  );
}

export default Button;
