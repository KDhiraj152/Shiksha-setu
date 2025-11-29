import { motion, AnimatePresence } from 'framer-motion';
import { X, CheckCircle, AlertCircle, AlertTriangle, Info } from 'lucide-react';
import { useUIStore, type Toast as ToastType } from '../../../store/uiStore';
import { cn } from '../../../lib/cn';

const icons = {
  success: CheckCircle,
  error: AlertCircle,
  warning: AlertTriangle,
  info: Info,
};

const styles = {
  success: {
    container: 'bg-green-50 dark:bg-green-950/50 border-green-200 dark:border-green-800',
    icon: 'text-green-600 dark:text-green-400',
    title: 'text-green-900 dark:text-green-100',
    description: 'text-green-700 dark:text-green-300',
  },
  error: {
    container: 'bg-red-50 dark:bg-red-950/50 border-red-200 dark:border-red-800',
    icon: 'text-red-600 dark:text-red-400',
    title: 'text-red-900 dark:text-red-100',
    description: 'text-red-700 dark:text-red-300',
  },
  warning: {
    container: 'bg-amber-50 dark:bg-amber-950/50 border-amber-200 dark:border-amber-800',
    icon: 'text-amber-600 dark:text-amber-400',
    title: 'text-amber-900 dark:text-amber-100',
    description: 'text-amber-700 dark:text-amber-300',
  },
  info: {
    container: 'bg-blue-50 dark:bg-blue-950/50 border-blue-200 dark:border-blue-800',
    icon: 'text-blue-600 dark:text-blue-400',
    title: 'text-blue-900 dark:text-blue-100',
    description: 'text-blue-700 dark:text-blue-300',
  },
};

interface ToastItemProps {
  toast: ToastType;
  onRemove: () => void;
}

function ToastItem({ toast, onRemove }: ToastItemProps) {
  const Icon = icons[toast.type];
  const style = styles[toast.type];

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: -20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, x: 100, scale: 0.95 }}
      transition={{ type: 'spring', stiffness: 500, damping: 40 }}
      className={cn(
        'pointer-events-auto w-full max-w-sm rounded-xl border p-4 shadow-lg backdrop-blur-sm',
        style.container
      )}
    >
      <div className="flex items-start gap-3">
        <Icon className={cn('h-5 w-5 shrink-0 mt-0.5', style.icon)} />
        
        <div className="flex-1 min-w-0">
          <p className={cn('font-medium text-sm', style.title)}>
            {toast.title}
          </p>
          {toast.description && (
            <p className={cn('mt-1 text-sm', style.description)}>
              {toast.description}
            </p>
          )}
          {toast.action && (
            <button
              onClick={toast.action.onClick}
              className={cn(
                'mt-2 text-sm font-medium underline-offset-2 hover:underline',
                style.icon
              )}
            >
              {toast.action.label}
            </button>
          )}
        </div>
        
        <button
          onClick={onRemove}
          className={cn(
            'shrink-0 rounded-lg p-1 opacity-60 transition-opacity hover:opacity-100',
            style.icon
          )}
        >
          <X className="h-4 w-4" />
        </button>
      </div>
    </motion.div>
  );
}

export function ToastContainer() {
  const toasts = useUIStore((state) => state.toasts);
  const removeToast = useUIStore((state) => state.removeToast);

  return (
    <div className="fixed top-4 right-4 z-[100] flex flex-col gap-2 pointer-events-none">
      <AnimatePresence mode="popLayout">
        {toasts.map((toast) => (
          <ToastItem
            key={toast.id}
            toast={toast}
            onRemove={() => removeToast(toast.id)}
          />
        ))}
      </AnimatePresence>
    </div>
  );
}

// Hook for easy toast creation
export function useToast() {
  const addToast = useUIStore((state) => state.addToast);
  
  return {
    success: (title: string, description?: string) =>
      addToast({ type: 'success', title, description }),
    error: (title: string, description?: string) =>
      addToast({ type: 'error', title, description }),
    warning: (title: string, description?: string) =>
      addToast({ type: 'warning', title, description }),
    info: (title: string, description?: string) =>
      addToast({ type: 'info', title, description }),
    custom: (toast: Omit<ToastType, 'id'>) => addToast(toast),
  };
}

export default ToastContainer;
