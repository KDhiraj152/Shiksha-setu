import { forwardRef, type InputHTMLAttributes, type ReactNode, useId } from 'react';
import { cn } from '../../../lib/cn';

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
  /** Container className */
  containerClassName?: string;
}

/**
 * Premium input component with floating labels, icons, and validation states.
 * Features smooth focus animations and accessible error handling.
 * 
 * @example
 * <Input
 *   label="Email Address"
 *   type="email"
 *   placeholder="you@example.com"
 *   leftIcon={<Mail className="w-5 h-5" />}
 *   error={errors.email?.message}
 * />
 */
export const Input = forwardRef<HTMLInputElement, InputProps>(
  (
    {
      className,
      containerClassName,
      label,
      helperText,
      error,
      size = 'md',
      leftIcon,
      rightIcon,
      fullWidth = true,
      id: providedId,
      disabled,
      ...props
    },
    ref
  ) => {
    const generatedId = useId();
    const id = providedId || generatedId;
    const errorId = `${id}-error`;
    const helperId = `${id}-helper`;

    const sizeStyles = {
      sm: 'input-sm',
      md: '',
      lg: 'input-lg',
    };

    return (
      <div className={cn('flex flex-col gap-1.5', fullWidth && 'w-full', containerClassName)}>
        {label && (
          <label
            htmlFor={id}
            className="text-sm font-medium text-[rgb(var(--color-fg-secondary))]"
          >
            {label}
            {props.required && <span className="text-[rgb(var(--color-error-500))] ml-0.5">*</span>}
          </label>
        )}
        
        <div className="relative">
          {leftIcon && (
            <span className="absolute left-4 top-1/2 -translate-y-1/2 text-[rgb(var(--color-fg-muted))] pointer-events-none">
              {leftIcon}
            </span>
          )}
          
          <input
            ref={ref}
            id={id}
            disabled={disabled}
            aria-invalid={!!error}
            aria-describedby={error ? errorId : helperText ? helperId : undefined}
            className={cn(
              'input',
              sizeStyles[size],
              leftIcon && 'pl-11',
              rightIcon && 'pr-11',
              error && 'input-error',
              className
            )}
            {...props}
          />
          
          {rightIcon && (
            <span className="absolute right-4 top-1/2 -translate-y-1/2 text-[rgb(var(--color-fg-muted))]">
              {rightIcon}
            </span>
          )}
        </div>
        
        {error && (
          <p id={errorId} className="text-sm text-[rgb(var(--color-error-500))] flex items-center gap-1">
            <svg className="w-4 h-4 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            {error}
          </p>
        )}
        
        {helperText && !error && (
          <p id={helperId} className="text-sm text-[rgb(var(--color-fg-muted))]">
            {helperText}
          </p>
        )}
      </div>
    );
  }
);

Input.displayName = 'Input';

export default Input;
