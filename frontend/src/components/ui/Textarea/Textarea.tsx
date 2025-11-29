import { forwardRef, type TextareaHTMLAttributes, useId } from 'react';
import { cn } from '../../../lib/cn';

export interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  /** Textarea label */
  label?: string;
  /** Helper text shown below textarea */
  helperText?: string;
  /** Error message (also sets error state) */
  error?: string;
  /** Make textarea full width */
  fullWidth?: boolean;
  /** Container className */
  containerClassName?: string;
  /** Show character count */
  showCount?: boolean;
}

/**
 * Premium textarea component with label, validation, and character count.
 * 
 * @example
 * <Textarea
 *   label="Description"
 *   placeholder="Enter your content..."
 *   maxLength={500}
 *   showCount
 * />
 */
export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  (
    {
      className,
      containerClassName,
      label,
      helperText,
      error,
      fullWidth = true,
      id: providedId,
      disabled,
      maxLength,
      showCount,
      value,
      ...props
    },
    ref
  ) => {
    const generatedId = useId();
    const id = providedId || generatedId;
    const errorId = `${id}-error`;
    const helperId = `${id}-helper`;
    
    const currentLength = typeof value === 'string' ? value.length : 0;

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
        
        <textarea
          ref={ref}
          id={id}
          disabled={disabled}
          maxLength={maxLength}
          value={value}
          aria-invalid={!!error}
          aria-describedby={error ? errorId : helperText ? helperId : undefined}
          className={cn(
            'input textarea min-h-[120px] resize-y',
            error && 'input-error',
            className
          )}
          {...props}
        />
        
        <div className="flex items-center justify-between gap-2">
          <div className="flex-1">
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
          
          {showCount && maxLength && (
            <p className={cn(
              'text-sm tabular-nums',
              currentLength >= maxLength 
                ? 'text-[rgb(var(--color-error-500))]' 
                : 'text-[rgb(var(--color-fg-muted))]'
            )}>
              {currentLength}/{maxLength}
            </p>
          )}
        </div>
      </div>
    );
  }
);

Textarea.displayName = 'Textarea';

export default Textarea;
