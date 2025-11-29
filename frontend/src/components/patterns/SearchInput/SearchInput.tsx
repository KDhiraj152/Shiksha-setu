import { forwardRef, type InputHTMLAttributes } from 'react';
import { cn } from '../../../lib/cn';

export interface SearchInputProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'type' | 'size'> {
  /** Size variant */
  inputSize?: 'sm' | 'md' | 'lg';
  /** Show clear button when has value */
  clearable?: boolean;
  /** Show loading state */
  isLoading?: boolean;
  /** Callback when clear button is clicked */
  onClear?: () => void;
  /** Container className */
  containerClassName?: string;
}

/**
 * Premium search input with icon, clear button, and loading states.
 * 
 * @example
 * <SearchInput
 *   placeholder="Search content..."
 *   value={query}
 *   onChange={(e) => setQuery(e.target.value)}
 *   clearable
 *   onClear={() => setQuery('')}
 * />
 */
export const SearchInput = forwardRef<HTMLInputElement, SearchInputProps>(
  (
    {
      className,
      containerClassName,
      inputSize = 'md',
      clearable = true,
      isLoading = false,
      onClear,
      value,
      ...props
    },
    ref
  ) => {
    const hasValue = value !== undefined && value !== '';

    const sizeStyles = {
      sm: 'h-9 text-sm pl-9 pr-8',
      md: 'h-11 text-sm pl-11 pr-10',
      lg: 'h-12 text-base pl-12 pr-11',
    };

    const iconSizes = {
      sm: 'w-4 h-4 left-2.5',
      md: 'w-5 h-5 left-3.5',
      lg: 'w-5 h-5 left-4',
    };

    const clearSizes = {
      sm: 'w-4 h-4 right-2',
      md: 'w-4 h-4 right-3',
      lg: 'w-5 h-5 right-3.5',
    };

    return (
      <div className={cn('relative', containerClassName)}>
        <SearchIcon className={cn(
          'absolute top-1/2 -translate-y-1/2 text-[rgb(var(--color-fg-muted))] pointer-events-none',
          iconSizes[inputSize]
        )} />
        
        <input
          ref={ref}
          type="search"
          value={value}
          className={cn(
            'input w-full',
            sizeStyles[inputSize],
            className
          )}
          {...props}
        />
        
        {isLoading ? (
          <span className={cn(
            'absolute top-1/2 -translate-y-1/2 text-[rgb(var(--color-fg-muted))]',
            clearSizes[inputSize]
          )}>
            <LoadingSpinner />
          </span>
        ) : clearable && hasValue ? (
          <button
            type="button"
            onClick={onClear}
            className={cn(
              'absolute top-1/2 -translate-y-1/2 text-[rgb(var(--color-fg-muted))] hover:text-[rgb(var(--color-fg-secondary))] transition-colors',
              clearSizes[inputSize]
            )}
            aria-label="Clear search"
          >
            <ClearIcon />
          </button>
        ) : null}
      </div>
    );
  }
);

SearchInput.displayName = 'SearchInput';

function SearchIcon({ className }: { className?: string }) {
  return (
    <svg className={className} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
    </svg>
  );
}

function ClearIcon() {
  return (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
    </svg>
  );
}

function LoadingSpinner() {
  return (
    <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
    </svg>
  );
}

export default SearchInput;
