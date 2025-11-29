/**
 * Validation Results - Display NCERT curriculum validation feedback
 * 
 * Shows validation score, issues, and suggestions for content improvement.
 */

import { motion, AnimatePresence } from 'framer-motion';
import { 
  CheckCircle2, 
  AlertTriangle, 
  AlertCircle, 
  Info,
  Shield,
  BookOpen,
  TrendingUp,
  ChevronDown,
  ChevronRight,
  RefreshCw
} from 'lucide-react';
import { useState } from 'react';
import { Badge } from '../../ui/Badge/Badge';
import { Button } from '../../ui/Button/Button';
import { cn } from '../../../lib/cn';
import type { ValidationResult } from '../../../store/pipelineStore';

interface ValidationResultsProps {
  /** Validation result data */
  result: ValidationResult;
  /** Grade level being validated against */
  gradeLevel?: number;
  /** Subject being validated */
  subject?: string;
  /** Callback to re-run validation */
  onRevalidate?: () => void;
  /** Loading state */
  isLoading?: boolean;
  /** Custom class name */
  className?: string;
}

const getScoreColor = (score: number) => {
  if (score >= 0.9) return 'text-green-600 dark:text-green-400';
  if (score >= 0.7) return 'text-amber-600 dark:text-amber-400';
  return 'text-red-600 dark:text-red-400';
};

const getScoreGradient = (score: number) => {
  if (score >= 0.9) return 'from-green-500 to-emerald-500';
  if (score >= 0.7) return 'from-amber-500 to-yellow-500';
  return 'from-red-500 to-orange-500';
};

const getSeverityIcon = (severity: 'error' | 'warning' | 'info') => {
  switch (severity) {
    case 'error':
      return AlertCircle;
    case 'warning':
      return AlertTriangle;
    case 'info':
      return Info;
  }
};

const getSeverityColor = (severity: 'error' | 'warning' | 'info') => {
  switch (severity) {
    case 'error':
      return 'text-red-600 bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800';
    case 'warning':
      return 'text-amber-600 bg-amber-50 dark:bg-amber-900/20 border-amber-200 dark:border-amber-800';
    case 'info':
      return 'text-blue-600 bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800';
  }
};

export function ValidationResults({
  result,
  gradeLevel = 6,
  subject = 'General',
  onRevalidate,
  isLoading = false,
  className,
}: ValidationResultsProps) {
  const [showDetails, setShowDetails] = useState(true);
  const [showSuggestions, setShowSuggestions] = useState(true);

  const scorePercent = Math.round(result.score * 100);
  const errorCount = result.issues.filter(i => i.severity === 'error').length;
  const warningCount = result.issues.filter(i => i.severity === 'warning').length;
  const infoCount = result.issues.filter(i => i.severity === 'info').length;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        'bg-card border border-border rounded-xl overflow-hidden',
        className
      )}
    >
      {/* Header */}
      <div className="p-6 border-b border-border">
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <div className={cn(
              'w-12 h-12 rounded-xl flex items-center justify-center',
              `bg-gradient-to-br ${getScoreGradient(result.score)}`,
              'shadow-lg'
            )}>
              <Shield className="w-6 h-6 text-white" />
            </div>
            <div>
              <h3 className="font-semibold text-foreground">NCERT Validation</h3>
              <p className="text-sm text-muted-foreground">
                Grade {gradeLevel} • {subject}
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            <Badge 
              variant={result.isValid ? 'success' : 'warning'}
              className="text-sm"
            >
              {result.isValid ? 'Passed' : 'Needs Review'}
            </Badge>
            
            {onRevalidate && (
              <Button
                variant="outline"
                size="sm"
                onClick={onRevalidate}
                disabled={isLoading}
                leftIcon={<RefreshCw className={cn('w-4 h-4', isLoading && 'animate-spin')} />}
              >
                Revalidate
              </Button>
            )}
          </div>
        </div>
        
        {/* Score Display */}
        <div className="mt-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm font-medium text-muted-foreground">
              Curriculum Alignment Score
            </span>
            <span className={cn('text-2xl font-bold', getScoreColor(result.score))}>
              {scorePercent}%
            </span>
          </div>
          <div className="h-3 bg-muted rounded-full overflow-hidden">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${scorePercent}%` }}
              transition={{ duration: 0.8, ease: 'easeOut' }}
              className={cn(
                'h-full rounded-full',
                `bg-gradient-to-r ${getScoreGradient(result.score)}`
              )}
            />
          </div>
          
          {/* Quick Stats */}
          <div className="flex items-center gap-4 mt-4">
            {errorCount > 0 && (
              <div className="flex items-center gap-1.5 text-red-600">
                <AlertCircle className="w-4 h-4" />
                <span className="text-sm font-medium">{errorCount} errors</span>
              </div>
            )}
            {warningCount > 0 && (
              <div className="flex items-center gap-1.5 text-amber-600">
                <AlertTriangle className="w-4 h-4" />
                <span className="text-sm font-medium">{warningCount} warnings</span>
              </div>
            )}
            {infoCount > 0 && (
              <div className="flex items-center gap-1.5 text-blue-600">
                <Info className="w-4 h-4" />
                <span className="text-sm font-medium">{infoCount} suggestions</span>
              </div>
            )}
            {result.issues.length === 0 && (
              <div className="flex items-center gap-1.5 text-green-600">
                <CheckCircle2 className="w-4 h-4" />
                <span className="text-sm font-medium">No issues found</span>
              </div>
            )}
          </div>
        </div>
      </div>
      
      {/* Issues Section */}
      {result.issues.length > 0 && (
        <div className="border-b border-border">
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="w-full flex items-center justify-between p-4 hover:bg-muted/50 transition-colors"
          >
            <span className="font-medium text-foreground">Issues & Warnings</span>
            {showDetails ? (
              <ChevronDown className="w-4 h-4 text-muted-foreground" />
            ) : (
              <ChevronRight className="w-4 h-4 text-muted-foreground" />
            )}
          </button>
          
          <AnimatePresence>
            {showDetails && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="overflow-hidden"
              >
                <div className="px-4 pb-4 space-y-2">
                  {result.issues.map((issue, index) => {
                    const Icon = getSeverityIcon(issue.severity);
                    return (
                      <motion.div
                        key={index}
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: index * 0.05 }}
                        className={cn(
                          'flex items-start gap-3 p-3 rounded-lg border',
                          getSeverityColor(issue.severity)
                        )}
                      >
                        <Icon className="w-4 h-4 mt-0.5 shrink-0" />
                        <span className="text-sm">{issue.message}</span>
                      </motion.div>
                    );
                  })}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
      
      {/* Suggestions Section */}
      {result.suggestions.length > 0 && (
        <div>
          <button
            onClick={() => setShowSuggestions(!showSuggestions)}
            className="w-full flex items-center justify-between p-4 hover:bg-muted/50 transition-colors"
          >
            <span className="font-medium text-foreground flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-primary-500" />
              Improvement Suggestions
            </span>
            {showSuggestions ? (
              <ChevronDown className="w-4 h-4 text-muted-foreground" />
            ) : (
              <ChevronRight className="w-4 h-4 text-muted-foreground" />
            )}
          </button>
          
          <AnimatePresence>
            {showSuggestions && (
              <motion.div
                initial={{ height: 0, opacity: 0 }}
                animate={{ height: 'auto', opacity: 1 }}
                exit={{ height: 0, opacity: 0 }}
                className="overflow-hidden"
              >
                <div className="px-4 pb-4 space-y-2">
                  {result.suggestions.map((suggestion, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, x: -10 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.05 }}
                      className="flex items-start gap-3 p-3 rounded-lg bg-primary-50 dark:bg-primary-900/20 border border-primary-200 dark:border-primary-800"
                    >
                      <BookOpen className="w-4 h-4 mt-0.5 shrink-0 text-primary-600" />
                      <span className="text-sm text-primary-800 dark:text-primary-200">
                        {suggestion}
                      </span>
                    </motion.div>
                  ))}
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      )}
      
      {/* Empty State */}
      {result.issues.length === 0 && result.suggestions.length === 0 && (
        <div className="p-8 text-center">
          <CheckCircle2 className="w-12 h-12 mx-auto text-green-500 mb-3" />
          <h4 className="font-medium text-foreground mb-1">
            Content Validated Successfully
          </h4>
          <p className="text-sm text-muted-foreground">
            Your content meets NCERT curriculum standards for Grade {gradeLevel}.
          </p>
        </div>
      )}
    </motion.div>
  );
}

export default ValidationResults;
