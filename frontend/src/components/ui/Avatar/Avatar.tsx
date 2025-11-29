import { cn } from '../../../lib/cn';
import { getInitials } from '../../../lib/formatters';

export interface AvatarProps {
  /** User's name for initials fallback */
  name?: string;
  /** Image source URL */
  src?: string;
  /** Alt text for image */
  alt?: string;
  /** Avatar size */
  size?: 'xs' | 'sm' | 'md' | 'lg' | 'xl';
  /** Custom className */
  className?: string;
}

/**
 * Premium avatar component with gradient fallback and smooth image loading.
 * 
 * @example
 * <Avatar name="John Doe" src="/avatar.jpg" size="lg" />
 * <Avatar name="Jane Smith" /> // Shows "JS" initials
 */
export function Avatar({
  name,
  src,
  alt,
  size = 'md',
  className,
}: AvatarProps) {
  const sizeStyles = {
    xs: 'avatar-xs',
    sm: 'avatar-sm',
    md: 'avatar-md',
    lg: 'avatar-lg',
    xl: 'avatar-xl',
  };

  const initials = name ? getInitials(name) : '?';

  return (
    <span
      className={cn('avatar', sizeStyles[size], className)}
      title={name}
    >
      {src ? (
        <img
          src={src}
          alt={alt || name || 'Avatar'}
          loading="lazy"
          onError={(e) => {
            // Hide broken image, show initials
            e.currentTarget.style.display = 'none';
          }}
        />
      ) : (
        initials
      )}
    </span>
  );
}

export default Avatar;
