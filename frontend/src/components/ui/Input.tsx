import { forwardRef, type InputHTMLAttributes, type ReactNode, useId } from 'react';
import { cn } from './utils';

export interface InputProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'size'> {
  /** Input label */
  label?: string;
  /** Helper text shown below input */
  helperText?: string;
  /** Error message (also sets error state) */
  error?: string;
  /** Input size variant */
  size?: 'sm' | 'md' | 'lg';
  /** Icon to show on the left */
  leftIcon?: ReactNode;
  /** Icon to show on the right */
  rightIcon?: ReactNode;
  /** Make input full width */
  fullWidth?: boolean;
}

/**
 * Input component with label, helper text, and error states.
 * Supports icons and multiple sizes.
 * 
 * @example
 * <Input
 *   label="Email"
 *   type="email"
 *   placeholder="Enter your email"
 *   helperText="We'll never share your email"
 *   error={errors.email?.message}
 * />
 */
export const Input = forwardRef<HTMLInputElement, InputProps>(
  (
    {
      className,
      label,
      helperText,
      error,
      size = 'md',
      leftIcon,
      rightIcon,
      fullWidth = true,
      id: providedId,
      ...props
    },
    ref
  ) => {
    const generatedId = useId();
    const id = providedId || generatedId;
    const errorId = `${id}-error`;
    const helperId = `${id}-helper`;

    const sizeStyles = {
      sm: 'py-1.5 px-2.5 text-sm',
      md: 'py-2 px-3 text-sm',
      lg: 'py-2.5 px-4 text-base',
    };

    const iconSizeStyles = {
      sm: 'w-4 h-4',
      md: 'w-4 h-4',
      lg: 'w-5 h-5',
    };

    return (
      <div className={cn('flex flex-col gap-1.5', fullWidth && 'w-full')}>
        {label && (
          <label
            htmlFor={id}
            className="text-sm font-medium text-surface-700 dark:text-surface-300"
          >
            {label}
            {props.required && <span className="text-error-500 ml-0.5">*</span>}
          </label>
        )}
        
        <div className="relative">
          {leftIcon && (
            <span
              className={cn(
                'absolute left-3 top-1/2 -translate-y-1/2 text-surface-400 pointer-events-none',
                iconSizeStyles[size]
              )}
            >
              {leftIcon}
            </span>
          )}
          
          <input
            ref={ref}
            id={id}
            className={cn(
              'input-field',
              sizeStyles[size],
              leftIcon && 'pl-10',
              rightIcon && 'pr-10',
              error && 'input-error',
              className
            )}
            aria-invalid={!!error}
            aria-describedby={
              error ? errorId : helperText ? helperId : undefined
            }
            {...props}
          />
          
          {rightIcon && (
            <span
              className={cn(
                'absolute right-3 top-1/2 -translate-y-1/2 text-surface-400 pointer-events-none',
                iconSizeStyles[size]
              )}
            >
              {rightIcon}
            </span>
          )}
        </div>
        
        {error && (
          <p id={errorId} className="text-sm text-error-500" role="alert">
            {error}
          </p>
        )}
        
        {helperText && !error && (
          <p id={helperId} className="text-sm text-surface-500">
            {helperText}
          </p>
        )}
      </div>
    );
  }
);

Input.displayName = 'Input';
