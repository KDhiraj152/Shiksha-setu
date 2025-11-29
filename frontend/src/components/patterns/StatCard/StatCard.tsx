import { type ReactNode } from 'react';
import { cn } from '../../../lib/cn';

export interface StatCardProps {
  /** Card title/label */
  label: string;
  /** Main value to display */
  value: string | number;
  /** Optional subtitle or description */
  description?: string;
  /** Icon to display */
  icon?: ReactNode;
  /** Trend indicator: positive, negative, or neutral */
  trend?: {
    value: number;
    direction: 'up' | 'down' | 'neutral';
  };
  /** Gradient colors for icon background */
  gradient?: string;
  /** Custom className */
  className?: string;
  /** onClick handler for interactive cards */
  onClick?: () => void;
}

/**
 * Premium stat card for dashboard metrics with trend indicators.
 * Features gradient icon backgrounds and smooth hover animations.
 * 
 * @example
 * <StatCard
 *   label="Total Content"
 *   value={1234}
 *   icon={<FileText className="w-6 h-6" />}
 *   gradient="from-blue-500 to-cyan-500"
 *   trend={{ value: 12, direction: 'up' }}
 * />
 */
export function StatCard({
  label,
  value,
  description,
  icon,
  trend,
  gradient = 'from-purple-500 to-pink-500',
  className,
  onClick,
}: StatCardProps) {
  const TrendIcon = trend?.direction === 'up' ? TrendUpIcon : trend?.direction === 'down' ? TrendDownIcon : null;

  const trendColor = {
    up: 'text-[rgb(var(--color-success-500))]',
    down: 'text-[rgb(var(--color-error-500))]',
    neutral: 'text-[rgb(var(--color-fg-muted))]',
  };

  return (
    <div
      className={cn(
        'card p-6 transition-all duration-200',
        onClick && 'card-interactive cursor-pointer',
        className
      )}
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
    >
      <div className="flex items-start justify-between mb-4">
        {icon && (
          <div className={cn(
            'w-12 h-12 rounded-xl flex items-center justify-center text-white bg-gradient-to-br',
            gradient
          )}>
            {icon}
          </div>
        )}
        {trend && TrendIcon && (
          <div className={cn('flex items-center gap-1 text-sm font-medium', trendColor[trend.direction])}>
            <TrendIcon />
            <span>{Math.abs(trend.value)}%</span>
          </div>
        )}
      </div>
      
      <div className="space-y-1">
        <p className="text-3xl font-bold text-[rgb(var(--color-fg))] tracking-tight tabular-nums">
          {typeof value === 'number' ? value.toLocaleString() : value}
        </p>
        <p className="text-sm font-medium text-[rgb(var(--color-fg-secondary))]">
          {label}
        </p>
        {description && (
          <p className="text-xs text-[rgb(var(--color-fg-muted))]">
            {description}
          </p>
        )}
      </div>
    </div>
  );
}

// Inline icons to avoid import issues
function TrendUpIcon() {
  return (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
    </svg>
  );
}

function TrendDownIcon() {
  return (
    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M13 17h8m0 0V9m0 8l-8-8-4 4-6-6" />
    </svg>
  );
}

export default StatCard;
