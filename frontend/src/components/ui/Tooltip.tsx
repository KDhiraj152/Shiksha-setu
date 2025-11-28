import { type ReactNode } from 'react';
import * as TooltipPrimitive from '@radix-ui/react-tooltip';
import { cn } from './utils';

export interface TooltipProps {
  /** Tooltip content */
  content: ReactNode;
  /** Trigger element */
  children: ReactNode;
  /** Side to show tooltip */
  side?: 'top' | 'right' | 'bottom' | 'left';
  /** Alignment of tooltip */
  align?: 'start' | 'center' | 'end';
  /** Delay before showing (ms) */
  delayDuration?: number;
  /** Additional class names */
  className?: string;
}

/**
 * Tooltip component using Radix UI for accessibility.
 * 
 * @example
 * <Tooltip content="Copy to clipboard" side="top">
 *   <IconButton icon={<CopyIcon />} aria-label="Copy" />
 * </Tooltip>
 */
export function Tooltip({
  content,
  children,
  side = 'top',
  align = 'center',
  delayDuration = 200,
  className,
}: TooltipProps) {
  return (
    <TooltipPrimitive.Provider>
      <TooltipPrimitive.Root delayDuration={delayDuration}>
        <TooltipPrimitive.Trigger asChild>{children}</TooltipPrimitive.Trigger>
        <TooltipPrimitive.Portal>
          <TooltipPrimitive.Content
            side={side}
            align={align}
            sideOffset={5}
            className={cn(
              'z-50 px-3 py-1.5 text-xs font-medium',
              'bg-surface-900 text-white dark:bg-surface-100 dark:text-surface-900',
              'rounded-md shadow-lg',
              'animate-fade-in',
              'max-w-xs',
              className
            )}
          >
            {content}
            <TooltipPrimitive.Arrow className="fill-surface-900 dark:fill-surface-100" />
          </TooltipPrimitive.Content>
        </TooltipPrimitive.Portal>
      </TooltipPrimitive.Root>
    </TooltipPrimitive.Provider>
  );
}

Tooltip.displayName = 'Tooltip';
