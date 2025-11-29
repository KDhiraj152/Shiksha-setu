import { motion } from 'framer-motion';
import { 
  CheckCircle, 
  XCircle, 
  Clock, 
  Loader2,
  FileText,
  Languages,
  Volume2,
  Download,
  RefreshCw
} from 'lucide-react';
import { Button } from '../../ui/Button/Button';
import { Progress } from '../../ui/Progress';
import { Badge } from '../../ui/Badge';
import { cn } from '../../../lib/cn';

interface ProcessingStep {
  id: string;
  label: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress?: number;
  message?: string;
}

interface ProcessingStatusProps {
  taskId: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  steps: ProcessingStep[];
  result?: {
    simplifiedText?: string;
    translations?: Record<string, string>;
    audioUrls?: Record<string, string>;
  };
  onCancel?: () => void;
  onRetry?: () => void;
  onDownload?: (type: 'text' | 'audio', language?: string) => void;
}

const statusConfig = {
  pending: { icon: Clock, color: 'text-amber-500', label: 'Pending' },
  processing: { icon: Loader2, color: 'text-blue-500', label: 'Processing' },
  completed: { icon: CheckCircle, color: 'text-green-500', label: 'Completed' },
  failed: { icon: XCircle, color: 'text-red-500', label: 'Failed' },
};

export function ProcessingStatus({
  taskId,
  status,
  progress,
  steps,
  result,
  onCancel,
  onRetry,
  onDownload,
}: ProcessingStatusProps) {
  const StatusIcon = statusConfig[status].icon;
  const isActive = status === 'pending' || status === 'processing';

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div
            className={cn(
              'w-12 h-12 rounded-xl flex items-center justify-center',
              status === 'completed' && 'bg-green-100 dark:bg-green-900/30',
              status === 'failed' && 'bg-red-100 dark:bg-red-900/30',
              isActive && 'bg-blue-100 dark:bg-blue-900/30'
            )}
          >
            <StatusIcon
              className={cn(
                'w-6 h-6',
                statusConfig[status].color,
                status === 'processing' && 'animate-spin'
              )}
            />
          </div>
          <div>
            <h3 className="font-semibold text-foreground">
              {statusConfig[status].label}
            </h3>
            <p className="text-sm text-muted-foreground">
              Task ID: {taskId.slice(0, 8)}...
            </p>
          </div>
        </div>

        <Badge
          variant={
            status === 'completed' ? 'success' :
            status === 'failed' ? 'error' :
            'primary'
          }
        >
          {progress}%
        </Badge>
      </div>

      {/* Overall Progress */}
      <div>
        <Progress 
          value={progress} 
          indeterminate={status === 'processing' && progress === 0}
          className={cn(
            status === 'completed' && '[&>div]:bg-green-500',
            status === 'failed' && '[&>div]:bg-red-500'
          )}
        />
      </div>

      {/* Steps */}
      <div className="space-y-3">
        {steps.map((step, index) => {
          const StepIcon = 
            step.id === 'simplify' ? FileText :
            step.id === 'translate' ? Languages :
            step.id === 'tts' ? Volume2 :
            statusConfig[step.status].icon;

          return (
            <motion.div
              key={step.id}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className={cn(
                'flex items-center gap-3 p-3 rounded-lg',
                step.status === 'completed' && 'bg-green-50 dark:bg-green-950/20',
                step.status === 'failed' && 'bg-red-50 dark:bg-red-950/20',
                step.status === 'processing' && 'bg-blue-50 dark:bg-blue-950/20',
                step.status === 'pending' && 'bg-muted/30'
              )}
            >
              <div
                className={cn(
                  'w-8 h-8 rounded-lg flex items-center justify-center',
                  step.status === 'completed' && 'bg-green-100 dark:bg-green-900/30 text-green-600',
                  step.status === 'failed' && 'bg-red-100 dark:bg-red-900/30 text-red-600',
                  step.status === 'processing' && 'bg-blue-100 dark:bg-blue-900/30 text-blue-600',
                  step.status === 'pending' && 'bg-muted text-muted-foreground'
                )}
              >
                <StepIcon 
                  className={cn(
                    'w-4 h-4',
                    step.status === 'processing' && 'animate-spin'
                  )} 
                />
              </div>

              <div className="flex-1 min-w-0">
                <p className="font-medium text-sm text-foreground">
                  {step.label}
                </p>
                {step.message && (
                  <p className="text-xs text-muted-foreground truncate">
                    {step.message}
                  </p>
                )}
              </div>

              {step.status !== 'pending' && (
                <Badge
                  variant={
                    step.status === 'completed' ? 'success' :
                    step.status === 'failed' ? 'error' :
                    'primary'
                  }
                  size="sm"
                >
                  {step.status === 'processing' 
                    ? `${step.progress || 0}%` 
                    : step.status}
                </Badge>
              )}
            </motion.div>
          );
        })}
      </div>

      {/* Actions */}
      {isActive && onCancel && (
        <Button variant="outline" onClick={onCancel} className="w-full">
          Cancel Processing
        </Button>
      )}

      {status === 'failed' && onRetry && (
        <Button variant="outline" onClick={onRetry} className="w-full">
          <RefreshCw className="w-4 h-4 mr-2" />
          Retry Processing
        </Button>
      )}

      {/* Results */}
      {status === 'completed' && result && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-4 pt-4 border-t border-border"
        >
          <h4 className="font-medium text-foreground">Downloads</h4>
          
          <div className="grid grid-cols-2 gap-2">
            {result.translations && Object.keys(result.translations).map((lang) => (
              <Button
                key={`text-${lang}`}
                variant="outline"
                size="sm"
                onClick={() => onDownload?.('text', lang)}
              >
                <Download className="w-4 h-4 mr-2" />
                {lang.toUpperCase()} Text
              </Button>
            ))}
            
            {result.audioUrls && Object.keys(result.audioUrls).map((lang) => (
              <Button
                key={`audio-${lang}`}
                variant="outline"
                size="sm"
                onClick={() => onDownload?.('audio', lang)}
              >
                <Download className="w-4 h-4 mr-2" />
                {lang.toUpperCase()} Audio
              </Button>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  );
}

export default ProcessingStatus;
