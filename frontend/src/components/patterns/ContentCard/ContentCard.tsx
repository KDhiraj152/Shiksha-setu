
import { Link } from 'react-router-dom';
import { cn } from '../../../lib/cn';
import { Badge } from '../../ui/Badge';
import { formatRelativeTime } from '../../../lib/formatters';

export interface ContentCardProps {
  /** Unique content ID */
  id: string;
  /** Content title or subject */
  title: string;
  /** Preview text */
  preview: string;
  /** Grade level */
  gradeLevel: number;
  /** Language */
  language: string;
  /** Subject category */
  subject?: string;
  /** Whether audio is available */
  hasAudio?: boolean;
  /** Creation/update timestamp */
  timestamp?: string;
  /** Custom className */
  className?: string;
  /** Click handler */
  onClick?: () => void;
  /** Link destination */
  href?: string;
}

/**
 * Premium content card for library items with metadata badges.
 * Features hover animations and optional audio indicator.
 * 
 * @example
 * <ContentCard
 *   id="123"
 *   title="Science - Chapter 5"
 *   preview="This chapter explains the concept of photosynthesis..."
 *   gradeLevel={8}
 *   language="Hindi"
 *   hasAudio
 *   href="/app/library/123"
 * />
 */
export function ContentCard({
  title,
  preview,
  gradeLevel,
  language,
  subject,
  hasAudio,
  timestamp,
  className,
  onClick,
  href,
}: ContentCardProps) {
  const content = (
    <>
      <div className="flex items-start justify-between gap-3 mb-3">
        <h3 className="text-lg font-semibold text-[rgb(var(--color-fg))] line-clamp-1">
          {title}
        </h3>
        {hasAudio && (
          <span className="shrink-0 text-[rgb(var(--color-success-500))]" title="Audio available">
            <AudioIcon />
          </span>
        )}
      </div>
      
      <p className="text-sm text-[rgb(var(--color-fg-secondary))] line-clamp-2 mb-4">
        {preview}
      </p>
      
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 flex-wrap">
          <Badge variant="primary" size="sm">Grade {gradeLevel}</Badge>
          <Badge variant="neutral" size="sm">{language}</Badge>
          {subject && (
            <Badge variant="secondary" size="sm">{subject}</Badge>
          )}
        </div>
        
        {timestamp && (
          <span className="text-xs text-[rgb(var(--color-fg-muted))]">
            {formatRelativeTime(timestamp)}
          </span>
        )}
      </div>
    </>
  );

  const cardClasses = cn(
    'card p-5 card-interactive group',
    className
  );

  if (href) {
    return (
      <Link to={href} className={cardClasses}>
        {content}
      </Link>
    );
  }

  return (
    <div 
      className={cardClasses}
      onClick={onClick}
      role={onClick ? 'button' : undefined}
      tabIndex={onClick ? 0 : undefined}
    >
      {content}
    </div>
  );
}

function AudioIcon() {
  return (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M15.536 8.464a5 5 0 010 7.072m2.828-9.9a9 9 0 010 12.728M5.586 15H4a1 1 0 01-1-1v-4a1 1 0 011-1h1.586l4.707-4.707C10.923 3.663 12 4.109 12 5v14c0 .891-1.077 1.337-1.707.707L5.586 15z" />
    </svg>
  );
}

export default ContentCard;
