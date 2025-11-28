import { forwardRef, type SelectHTMLAttributes, useId } from 'react';
import { cn } from './utils';

export interface SelectOption {
  value: string;
  label: string;
  disabled?: boolean;
}

export interface SelectProps extends Omit<SelectHTMLAttributes<HTMLSelectElement>, 'size'> {
  /** Select label */
  label?: string;
  /** Placeholder option text */
  placeholder?: string;
  /** Options to display */
  options: SelectOption[];
  /** Helper text shown below select */
  helperText?: string;
  /** Error message (also sets error state) */
  error?: string;
  /** Select size variant */
  size?: 'sm' | 'md' | 'lg';
  /** Make select full width */
  fullWidth?: boolean;
}

/**
 * Select component with options, label, and error states.
 * 
 * @example
 * <Select
 *   label="Language"
 *   placeholder="Select a language"
 *   options={[
 *     { value: 'hi', label: 'Hindi' },
 *     { value: 'en', label: 'English' },
 *     { value: 'ta', label: 'Tamil' },
 *   ]}
 * />
 */
export const Select = forwardRef<HTMLSelectElement, SelectProps>(
  (
    {
      className,
      label,
      placeholder,
      options,
      helperText,
      error,
      size = 'md',
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
          <select
            ref={ref}
            id={id}
            className={cn(
              'input-field appearance-none pr-10 cursor-pointer',
              sizeStyles[size],
              error && 'input-error',
              className
            )}
            aria-invalid={!!error}
            aria-describedby={
              error ? errorId : helperText ? helperId : undefined
            }
            {...props}
          >
            {placeholder && (
              <option value="" disabled>
                {placeholder}
              </option>
            )}
            {options.map((option) => (
              <option
                key={option.value}
                value={option.value}
                disabled={option.disabled}
              >
                {option.label}
              </option>
            ))}
          </select>
          
          {/* Custom dropdown arrow */}
          <svg
            className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-surface-400 pointer-events-none"
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 20 20"
            fill="currentColor"
          >
            <path
              fillRule="evenodd"
              d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
              clipRule="evenodd"
            />
          </svg>
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

Select.displayName = 'Select';
