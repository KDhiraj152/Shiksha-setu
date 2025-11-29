/**
 * Content Comparison - Side-by-side view of original vs processed content
 * 
 * Shows original text alongside simplified/translated versions with
 * visual diff highlighting and easy copying.
 */

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Copy, 
  Check, 
  ArrowLeftRight, 
  Languages, 
  FileText,
  Volume2,
  Maximize2
} from 'lucide-react';
import { Button } from '../../ui/Button/Button';
import { Badge } from '../../ui/Badge/Badge';
import { cn } from '../../../lib/cn';

interface ContentComparisonProps {
  /** Original text content */
  originalText: string;
  /** Processed/simplified text */
  processedText?: string;
  /** Translations by language */
  translations?: Record<string, string>;
  /** Current grade level */
  gradeLevel?: number;
  /** Audio URL for processed text */
  audioUrl?: string;
  /** Show word count comparison */
  showStats?: boolean;
  /** Allow expanding to full screen */
  allowExpand?: boolean;
  /** Custom class name */
  className?: string;
}

type ViewMode = 'simplified' | 'translations';

export function ContentComparison({
  originalText,
  processedText,
  translations = {},
  gradeLevel = 6,
  audioUrl,
  showStats = true,
  allowExpand = true,
  className,
}: ContentComparisonProps) {
  const [viewMode, setViewMode] = useState<ViewMode>('simplified');
  const [selectedLanguage, setSelectedLanguage] = useState<string>(
    Object.keys(translations)[0] || 'Hindi'
  );
  const [copiedSide, setCopiedSide] = useState<'original' | 'processed' | null>(null);
  const [isExpanded, setIsExpanded] = useState(false);

  const availableLanguages = Object.keys(translations);
  const hasTranslations = availableLanguages.length > 0;

  // Calculate stats
  const originalWords = originalText.split(/\s+/).filter(Boolean).length;
  const processedWords = processedText?.split(/\s+/).filter(Boolean).length || 0;
  const reductionPercent = originalWords > 0 
    ? Math.round((1 - processedWords / originalWords) * 100)
    : 0;

  const handleCopy = async (text: string, side: 'original' | 'processed') => {
    await navigator.clipboard.writeText(text);
    setCopiedSide(side);
    setTimeout(() => setCopiedSide(null), 2000);
  };

  const getRightPanelContent = () => {
    if (viewMode === 'simplified') {
      return processedText || 'No simplified content yet. Click "Simplify" to process.';
    }
    return translations[selectedLanguage] || `No ${selectedLanguage} translation yet.`;
  };

  const ContentPanel = ({ 
    title, 
    content, 
    side,
    badge,
    icon: Icon,
    gradient,
  }: { 
    title: string; 
    content: string; 
    side: 'original' | 'processed';
    badge?: string;
    icon?: React.ElementType;
    gradient?: string;
  }) => (
    <div className="flex flex-col h-full">
      {/* Panel Header */}
      <div className={cn(
        'flex items-center justify-between p-4 border-b border-border',
        gradient && `bg-gradient-to-r ${gradient} text-white`
      )}>
        <div className="flex items-center gap-2">
          {Icon && <Icon className="w-4 h-4" />}
          <span className="font-medium">{title}</span>
          {badge && (
            <Badge variant={gradient ? 'secondary' : 'neutral'} size="sm">
              {badge}
            </Badge>
          )}
        </div>
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="sm"
            iconOnly
            onClick={() => handleCopy(content, side)}
            className={gradient ? 'text-white/80 hover:text-white hover:bg-white/20' : ''}
          >
            {copiedSide === side ? (
              <Check className="w-4 h-4 text-green-500" />
            ) : (
              <Copy className="w-4 h-4" />
            )}
          </Button>
          {side === 'processed' && audioUrl && (
            <Button
              variant="ghost"
              size="sm"
              iconOnly
              onClick={() => new Audio(audioUrl).play()}
              className={gradient ? 'text-white/80 hover:text-white hover:bg-white/20' : ''}
            >
              <Volume2 className="w-4 h-4" />
            </Button>
          )}
        </div>
      </div>
      
      {/* Panel Content */}
      <div className="flex-1 p-4 overflow-y-auto bg-muted/30">
        <p className="whitespace-pre-wrap text-sm leading-relaxed text-foreground">
          {content}
        </p>
      </div>
      
      {/* Panel Footer Stats */}
      {showStats && (
        <div className="flex items-center justify-between px-4 py-2 text-xs text-muted-foreground border-t border-border bg-muted/20">
          <span>{content.split(/\s+/).filter(Boolean).length} words</span>
          <span>{content.length} characters</span>
        </div>
      )}
    </div>
  );

  return (
    <AnimatePresence>
      <motion.div
        layout
        className={cn(
          'bg-card border border-border rounded-xl overflow-hidden',
          isExpanded && 'fixed inset-4 z-50',
          className
        )}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-border bg-muted/30">
          <div className="flex items-center gap-4">
            <h3 className="font-semibold text-foreground flex items-center gap-2">
              <ArrowLeftRight className="w-4 h-4" />
              Content Comparison
            </h3>
            
            {/* View Mode Toggle */}
            {hasTranslations && (
              <div className="flex border border-border rounded-lg p-1">
                <button
                  onClick={() => setViewMode('simplified')}
                  className={cn(
                    'px-3 py-1 text-xs font-medium rounded transition-colors',
                    viewMode === 'simplified'
                      ? 'bg-primary-100 dark:bg-primary-900/30 text-primary-600'
                      : 'text-muted-foreground hover:text-foreground'
                  )}
                >
                  <FileText className="w-3 h-3 inline mr-1" />
                  Simplified
                </button>
                <button
                  onClick={() => setViewMode('translations')}
                  className={cn(
                    'px-3 py-1 text-xs font-medium rounded transition-colors',
                    viewMode === 'translations'
                      ? 'bg-primary-100 dark:bg-primary-900/30 text-primary-600'
                      : 'text-muted-foreground hover:text-foreground'
                  )}
                >
                  <Languages className="w-3 h-3 inline mr-1" />
                  Translations
                </button>
              </div>
            )}
            
            {/* Language Selector */}
            {viewMode === 'translations' && hasTranslations && (
              <select
                value={selectedLanguage}
                onChange={(e) => setSelectedLanguage(e.target.value)}
                className="px-3 py-1 text-xs border border-border rounded-lg bg-background"
              >
                {availableLanguages.map(lang => (
                  <option key={lang} value={lang}>{lang}</option>
                ))}
              </select>
            )}
          </div>
          
          <div className="flex items-center gap-2">
            {showStats && processedText && (
              <Badge variant="secondary" className="text-xs">
                {reductionPercent > 0 ? `${reductionPercent}% simpler` : 'Same length'}
              </Badge>
            )}
            
            {allowExpand && (
              <Button
                variant="ghost"
                size="sm"
                iconOnly
                onClick={() => setIsExpanded(!isExpanded)}
              >
                <Maximize2 className="w-4 h-4" />
              </Button>
            )}
          </div>
        </div>
        
        {/* Comparison Grid */}
        <div className={cn(
          'grid grid-cols-1 md:grid-cols-2 divide-y md:divide-y-0 md:divide-x divide-border',
          isExpanded ? 'h-[calc(100%-60px)]' : 'h-[400px]'
        )}>
          {/* Original Content */}
          <ContentPanel
            title="Original"
            content={originalText}
            side="original"
            icon={FileText}
          />
          
          {/* Processed Content */}
          <ContentPanel
            title={viewMode === 'simplified' ? `Simplified (Grade ${gradeLevel})` : selectedLanguage}
            content={getRightPanelContent()}
            side="processed"
            icon={viewMode === 'simplified' ? Sparkles : Languages}
            gradient={viewMode === 'simplified' 
              ? 'from-primary-500 to-secondary-500' 
              : 'from-green-500 to-emerald-500'
            }
          />
        </div>
      </motion.div>
    </AnimatePresence>
  );
}

// Import Sparkles for the component
import { Sparkles } from 'lucide-react';

export default ContentComparison;
