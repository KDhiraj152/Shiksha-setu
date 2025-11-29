/**
 * UI Component Library - ShikshaSetu Design System v2.0
 * Premium, Apple-quality components for educational excellence
 * 
 * Export all UI primitives from a single entry point
 */

// Core Atoms - New Architecture (folder-based)
export { Button, type ButtonProps } from './Button/Button';
export { Input, type InputProps } from './Input/Input';
export { Textarea, type TextareaProps } from './Textarea/Textarea';
export { Badge, type BadgeProps } from './Badge/Badge';
export { Spinner, type SpinnerProps } from './Spinner/Spinner';
export { Progress, type ProgressProps } from './Progress/Progress';
export { Avatar, type AvatarProps } from './Avatar/Avatar';
export { ToastContainer, useToast } from './Toast/Toast';

// Animation Components
export { PageTransition } from './PageTransition';
export { AnimatedList, AnimatedListItem } from './AnimatedList';
export { AnimatedCard } from './AnimatedCard';
export { 
  Skeleton, 
  CardSkeleton,
  StatCardSkeleton,
  TableRowSkeleton,
  AvatarSkeleton,
  ButtonSkeleton,
  TextBlockSkeleton,
  ListSkeleton,
  GridSkeleton,
} from './Skeleton';

// Overlay Components
export { Modal, ConfirmModal } from './Modal';
export { Dropdown, DropdownButton } from './Dropdown';

// Legacy exports (for backward compatibility during migration)
export { Select, type SelectProps, type SelectOption } from './Select/Select';
export { IconButton, type IconButtonProps } from './IconButton/IconButton';
export { Tooltip, type TooltipProps } from './Tooltip/Tooltip';

// Utils
export { cn } from './utils';
