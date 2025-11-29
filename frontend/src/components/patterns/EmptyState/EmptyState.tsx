import { type ReactNode } from 'react';
import { cn } from '../../../lib/cn';
import { Button } from '../../ui/Button/Button';

export interface EmptyStateProps {
  /** Main title */
  title: string;
  /** Description text */
  description?: string;
  /** Custom icon or illustration */
  icon?: ReactNode;
  /** Primary action button */
  action?: {
    label: string;
    onClick: () => void;
  };
  /** Secondary action button */
  secondaryAction?: {
    label: string;
    onClick: () => void;
  };
  /** Custom className */
  className?: string;
}

/**
 * Premium empty state component for when no content is available.
 * Features subtle animations and actionable CTAs.
 * 
 * @example
 * <EmptyState
 *   title="No content yet"
 *   description="Start by creating your first piece of content"
 *   icon={<FileIcon />}
 *   action={{ label: "Create Content", onClick: () => {} }}
 * />
 */
export function EmptyState({
  title,
  description,
  icon,
  action,
  secondaryAction,
  className,
}: EmptyStateProps) {
  return (
    <div
      className={cn(
        'flex flex-col items-center justify-center text-center py-16 px-6',
        className
      )}
    >
      {icon ? (
        <div className="mb-6 text-[rgb(var(--color-fg-muted))]">
          {icon}
        </div>
      ) : (
        <div className="mb-6">
          <DefaultEmptyIcon />
        </div>
      )}
      
      <h3 className="text-xl font-semibold text-[rgb(var(--color-fg))] mb-2">
        {title}
      </h3>
      
      {description && (
        <p className="text-sm text-[rgb(var(--color-fg-muted))] max-w-sm mb-6">
          {description}
        </p>
      )}
      
      {(action || secondaryAction) && (
        <div className="flex items-center gap-3">
          {action && (
            <Button variant="primary" onClick={action.onClick}>
              {action.label}
            </Button>
          )}
          {secondaryAction && (
            <Button variant="outline" onClick={secondaryAction.onClick}>
              {secondaryAction.label}
            </Button>
          )}
        </div>
      )}
    </div>
  );
}

function DefaultEmptyIcon() {
  return (
    <svg 
      className="w-20 h-20 text-[rgb(var(--color-border))]" 
      fill="none" 
      viewBox="0 0 80 80"
    >
      <rect x="16" y="16" width="48" height="48" rx="8" stroke="currentColor" strokeWidth="2" strokeDasharray="4 4" />
      <path 
        d="M35 40h10M40 35v10" 
        stroke="rgb(var(--color-fg-muted))" 
        strokeWidth="2" 
        strokeLinecap="round" 
      />
    </svg>
  );
}

export default EmptyState;
