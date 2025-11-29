import { type ReactNode } from 'react';
import { cn } from '../../../lib/cn';

export interface PageHeaderProps {
  /** Page title */
  title: string;
  /** Page description/subtitle */
  description?: string;
  /** Optional icon to display */
  icon?: ReactNode;
  /** Actions to display on the right */
  actions?: ReactNode;
  /** Breadcrumb navigation */
  breadcrumb?: ReactNode;
  /** Custom className */
  className?: string;
}

/**
 * Consistent page header component with title, description, and actions.
 * 
 * @example
 * <PageHeader
 *   title="Dashboard"
 *   description="Welcome back! Here's what's happening today."
 *   icon={<Home className="w-8 h-8" />}
 *   actions={<Button>New Content</Button>}
 * />
 */
export function PageHeader({
  title,
  description,
  icon,
  actions,
  breadcrumb,
  className,
}: PageHeaderProps) {
  return (
    <div className={cn('mb-8', className)}>
      {breadcrumb && (
        <div className="mb-4">
          {breadcrumb}
        </div>
      )}
      
      <div className="flex items-start justify-between gap-4">
        <div className="flex items-start gap-4">
          {icon && (
            <div className="text-[rgb(var(--color-primary-500))] shrink-0">
              {icon}
            </div>
          )}
          <div>
            <h1 className="text-2xl md:text-3xl font-bold text-[rgb(var(--color-fg))] tracking-tight">
              {title}
            </h1>
            {description && (
              <p className="mt-1 text-sm md:text-base text-[rgb(var(--color-fg-secondary))]">
                {description}
              </p>
            )}
          </div>
        </div>
        
        {actions && (
          <div className="shrink-0 flex items-center gap-3">
            {actions}
          </div>
        )}
      </div>
    </div>
  );
}

export default PageHeader;
