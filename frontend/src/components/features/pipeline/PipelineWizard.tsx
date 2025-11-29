/**
 * Pipeline Wizard - Unified visual component for AI content processing
 * 
 * Shows the complete flow: Upload → Extract → Simplify → Translate → Validate → TTS
 * with real-time progress tracking and stage indicators.
 */

import { motion } from 'framer-motion';
import {
  Upload,
  FileText,
  Sparkles,
  Languages,
  CheckCircle,
  Volume2,
  Loader2,
  AlertCircle,
} from 'lucide-react';
import { cn } from '../../../lib/cn';
import { usePipelineStore, pipelineSelectors, type PipelineStage } from '../../../store/pipelineStore';interface PipelineStageConfig {
  id: PipelineStage;
  name: string;
  shortName: string;
  icon: React.ElementType;
  description: string;
  gradient: string;
}

const PIPELINE_STAGES: PipelineStageConfig[] = [
  {
    id: 'uploading',
    name: 'Upload',
    shortName: 'Upload',
    icon: Upload,
    description: 'Upload your document',
    gradient: 'from-blue-500 to-blue-600',
  },
  {
    id: 'extracting',
    name: 'Extract Text',
    shortName: 'Extract',
    icon: FileText,
    description: 'Extract text from document',
    gradient: 'from-purple-500 to-purple-600',
  },
  {
    id: 'simplifying',
    name: 'Simplify',
    shortName: 'Simplify',
    icon: Sparkles,
    description: 'Adapt to grade level',
    gradient: 'from-pink-500 to-pink-600',
  },
  {
    id: 'translating',
    name: 'Translate',
    shortName: 'Translate',
    icon: Languages,
    description: 'Convert to regional languages',
    gradient: 'from-green-500 to-green-600',
  },
  {
    id: 'validating',
    name: 'Validate',
    shortName: 'Validate',
    icon: CheckCircle,
    description: 'Check NCERT alignment',
    gradient: 'from-amber-500 to-amber-600',
  },
  {
    id: 'generating-audio',
    name: 'Generate Audio',
    shortName: 'Audio',
    icon: Volume2,
    description: 'Create text-to-speech',
    gradient: 'from-cyan-500 to-cyan-600',
  },
];

interface PipelineWizardProps {
  /** Layout orientation */
  orientation?: 'horizontal' | 'vertical';
  /** Show stage descriptions */
  showDescriptions?: boolean;
  /** Compact mode for smaller spaces */
  compact?: boolean;
  /** Callback when stage is clicked */
  onStageClick?: (stage: PipelineStage) => void;
  /** Custom class name */
  className?: string;
}

/**
 * Visual pipeline wizard showing progress through AI processing stages
 */
export function PipelineWizard({
  orientation = 'horizontal',
  showDescriptions = true,
  compact = false,
  onStageClick,
  className,
}: PipelineWizardProps) {
  const pipelineState = usePipelineStore();

  const getStageStatus = (stageId: PipelineStage) => {
    return pipelineSelectors.getStageStatus(pipelineState, stageId);
  };

  const renderStageIcon = (stage: PipelineStageConfig, status: string) => {
    const IconComponent = stage.icon;
    
    if (status === 'completed') {
      return (
        <motion.div
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          className="text-white"
        >
          <CheckCircle className={compact ? 'w-4 h-4' : 'w-5 h-5'} />
        </motion.div>
      );
    }
    
    if (status === 'active') {
      return (
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
        >
          <Loader2 className={compact ? 'w-4 h-4' : 'w-5 h-5'} />
        </motion.div>
      );
    }
    
    if (status === 'error') {
      return <AlertCircle className={cn(compact ? 'w-4 h-4' : 'w-5 h-5', 'text-red-500')} />;
    }
    
    return <IconComponent className={compact ? 'w-4 h-4' : 'w-5 h-5'} />;
  };

  const renderStage = (stage: PipelineStageConfig, index: number) => {
    const status = getStageStatus(stage.id);
    const isClickable = onStageClick && pipelineState.canProceedToStage(stage.id);
    
    const stageContent = (
      <motion.div
        key={stage.id}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: index * 0.1 }}
        className={cn(
          'flex items-center gap-3',
          orientation === 'vertical' && 'flex-row',
          isClickable && 'cursor-pointer',
        )}
        onClick={() => isClickable && onStageClick?.(stage.id)}
      >
        {/* Stage Icon */}
        <div
          className={cn(
            'relative flex items-center justify-center rounded-xl transition-all duration-300',
            compact ? 'w-8 h-8' : 'w-12 h-12',
            status === 'completed' && `bg-gradient-to-br ${stage.gradient} shadow-lg`,
            status === 'active' && `bg-gradient-to-br ${stage.gradient} shadow-lg shadow-${stage.gradient.split('-')[1]}-500/30 animate-pulse`,
            status === 'error' && 'bg-red-100 dark:bg-red-900/30 border-2 border-red-500',
            status === 'pending' && 'bg-muted border-2 border-dashed border-border',
            isClickable && 'hover:scale-105',
          )}
        >
          <span className={cn(
            status === 'completed' || status === 'active' ? 'text-white' : 'text-muted-foreground'
          )}>
            {renderStageIcon(stage, status)}
          </span>
          
          {/* Progress ring for active stage */}
          {status === 'active' && pipelineState.stageProgress > 0 && (
            <svg
              className="absolute inset-0 w-full h-full -rotate-90"
              viewBox="0 0 48 48"
            >
              <circle
                cx="24"
                cy="24"
                r="22"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                className="text-white/30"
              />
              <circle
                cx="24"
                cy="24"
                r="22"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeDasharray={`${(pipelineState.stageProgress / 100) * 138} 138`}
                className="text-white"
              />
            </svg>
          )}
        </div>

        {/* Stage Info */}
        {!compact && (
          <div className={cn(
            'flex flex-col',
            orientation === 'horizontal' && 'hidden lg:flex',
          )}>
            <span className={cn(
              'font-medium text-sm',
              status === 'completed' && 'text-foreground',
              status === 'active' && 'text-primary-600',
              status === 'pending' && 'text-muted-foreground',
              status === 'error' && 'text-red-600',
            )}>
              {stage.shortName}
            </span>
            {showDescriptions && (
              <span className="text-xs text-muted-foreground">
                {status === 'active' && `${pipelineState.stageProgress}%`}
                {status === 'completed' && 'Done'}
                {status === 'pending' && stage.description}
                {status === 'error' && 'Failed'}
              </span>
            )}
          </div>
        )}
      </motion.div>
    );

    return stageContent;
  };

  const renderConnector = (index: number) => {
    if (index === PIPELINE_STAGES.length - 1) return null;
    
    const currentStatus = getStageStatus(PIPELINE_STAGES[index].id);
    const isActive = currentStatus === 'completed' || currentStatus === 'active';
    
    if (orientation === 'horizontal') {
      return (
        <div className={cn(
          'flex-1 h-0.5 mx-2 transition-colors duration-500',
          compact ? 'min-w-4' : 'min-w-8',
          isActive ? 'bg-gradient-to-r from-primary-500 to-primary-400' : 'bg-border',
        )} />
      );
    }
    
    return (
      <div className={cn(
        'w-0.5 h-8 ml-5 transition-colors duration-500',
        isActive ? 'bg-gradient-to-b from-primary-500 to-primary-400' : 'bg-border',
      )} />
    );
  };

  if (orientation === 'vertical') {
    return (
      <div className={cn('flex flex-col', className)}>
        {PIPELINE_STAGES.map((stage, index) => (
          <div key={stage.id}>
            {renderStage(stage, index)}
            {renderConnector(index)}
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className={cn(
      'flex items-center justify-between',
      compact ? 'gap-2' : 'gap-4',
      className,
    )}>
      {PIPELINE_STAGES.map((stage, index) => (
        <div key={stage.id} className="flex items-center flex-1 last:flex-none">
          {renderStage(stage, index)}
          {renderConnector(index)}
        </div>
      ))}
    </div>
  );
}

/**
 * Minimal pipeline indicator for headers/toolbars
 */
export function PipelineIndicator({ className }: { className?: string }) {
  const pipelineState = usePipelineStore();
  const isProcessing = pipelineSelectors.isProcessing(pipelineState);
  
  if (!isProcessing && pipelineState.currentStage === 'idle') {
    return null;
  }

  const currentStage = PIPELINE_STAGES.find(s => s.id === pipelineState.currentStage);
  
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9 }}
      className={cn(
        'flex items-center gap-2 px-3 py-1.5 rounded-full',
        'bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300',
        className,
      )}
    >
      {isProcessing && (
        <Loader2 className="w-3 h-3 animate-spin" />
      )}
      <span className="text-xs font-medium">
        {currentStage?.shortName || 'Processing'} {pipelineState.stageProgress > 0 && `${pipelineState.stageProgress}%`}
      </span>
    </motion.div>
  );
}

/**
 * Pipeline progress bar for inline display
 */
export function PipelineProgressBar({ className }: { className?: string }) {
  const pipelineState = usePipelineStore();
  const completedCount = pipelineState.completedStages.length;
  const totalStages = PIPELINE_STAGES.length;
  const progressPercent = (completedCount / totalStages) * 100;

  return (
    <div className={cn('w-full', className)}>
      <div className="flex items-center justify-between mb-1">
        <span className="text-xs font-medium text-muted-foreground">
          Pipeline Progress
        </span>
        <span className="text-xs text-muted-foreground">
          {completedCount}/{totalStages} stages
        </span>
      </div>
      <div className="h-2 bg-muted rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${progressPercent}%` }}
          className="h-full bg-gradient-to-r from-primary-500 to-secondary-500 rounded-full"
        />
      </div>
    </div>
  );
}

export default PipelineWizard;
