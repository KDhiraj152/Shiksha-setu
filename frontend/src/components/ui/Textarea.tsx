import { forwardRef, type TextareaHTMLAttributes, useId } from 'react';
import { cn } from './utils';

export interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  /** Textarea label */
  label?: string;
  /** Helper text shown below textarea */
  helperText?: string;
  /** Error message (also sets error state) */
  error?: string;
  /** Show character count */
  showCount?: boolean;
  /** Make textarea resizable */
  resize?: 'none' | 'vertical' | 'horizontal' | 'both';
  /** Make textarea full width */
  fullWidth?: boolean;
}

/**
 * Textarea component with label, helper text, character count, and error states.
 * 
 * @example
 * <Textarea
 *   label="Description"
 *   placeholder="Enter content description"
 *   maxLength={500}
 *   showCount
 *   rows={4}
 * />
 */
export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  (
    {
      className,
      label,
      helperText,
      error,
      showCount = false,
      resize = 'vertical',
      fullWidth = true,
      maxLength,
      value,
      defaultValue,
      id: providedId,
      ...props
    },
    ref
  ) => {
    const generatedId = useId();
    const id = providedId || generatedId;
    const errorId = `${id}-error`;
    const helperId = `${id}-helper`;

    const resizeStyles = {
      none: 'resize-none',
      vertical: 'resize-y',
      horizontal: 'resize-x',
      both: 'resize',
    };

    const currentLength = String(value ?? defaultValue ?? '').length;

    return (
      <div className={cn('flex flex-col gap-1.5', fullWidth && 'w-full')}>
        <div className="flex items-center justify-between">
          {label && (
            <label
              htmlFor={id}
              className="text-sm font-medium text-surface-700 dark:text-surface-300"
            >
              {label}
              {props.required && <span className="text-error-500 ml-0.5">*</span>}
            </label>
          )}
          
          {showCount && maxLength && (
            <span
              className={cn(
                'text-xs',
                currentLength > maxLength * 0.9
                  ? 'text-error-500'
                  : 'text-surface-400'
              )}
            >
              {currentLength}/{maxLength}
            </span>
          )}
        </div>
        
        <textarea
          ref={ref}
          id={id}
          className={cn(
            'input-field min-h-[80px]',
            resizeStyles[resize],
            error && 'input-error',
            className
          )}
          value={value}
          defaultValue={defaultValue}
          maxLength={maxLength}
          aria-invalid={!!error}
          aria-describedby={
            error ? errorId : helperText ? helperId : undefined
          }
          {...props}
        />
        
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

Textarea.displayName = 'Textarea';
