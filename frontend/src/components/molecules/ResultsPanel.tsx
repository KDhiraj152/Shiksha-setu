import { useState, useCallback } from 'react';
import * as Tabs from '@radix-ui/react-tabs';
import { cn } from '../ui/utils';
import { Button } from '../ui/Button';
import { IconButton } from '../ui/IconButton';
import { Badge } from '../ui/Badge';
import { Tooltip } from '../ui/Tooltip';
import { Textarea } from '../ui/Textarea';
import { AudioPlayer } from './AudioPlayer';

export interface ResultContent {
  /** Original text */
  original?: string;
  /** Simplified text */
  simplified?: string;
  /** Translated text */
  translated?: string;
  /** Audio URL */
  audioUrl?: string;
  /** Source language */
  sourceLanguage?: string;
  /** Target language */
  targetLanguage?: string;
  /** Processing metadata */
  metadata?: {
    wordCount?: number;
    processingTime?: number;
    confidence?: number;
  };
}

export interface ResultsPanelProps {
  /** Result content */
  content: ResultContent;
  /** Task ID for reference */
  taskId?: string;
  /** Enable feedback form */
  enableFeedback?: boolean;
  /** Callback when feedback is submitted */
  onFeedback?: (feedback: { rating: number; comment: string; taskId?: string }) => void;
  /** Additional class name */
  className?: string;
}

// Icons
const CopyIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
  </svg>
);

const CheckIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
  </svg>
);

const DownloadIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
  </svg>
);

const StarIcon = ({ filled }: { filled: boolean }) => (
  <svg 
    className={cn('w-6 h-6', filled ? 'fill-warning-400 text-warning-400' : 'text-surface-300')} 
    fill={filled ? 'currentColor' : 'none'} 
    stroke="currentColor" 
    viewBox="0 0 24 24"
  >
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z" />
  </svg>
);

/**
 * Results panel with tabbed content, copy/download actions, and feedback form.
 * 
 * @example
 * <ResultsPanel
 *   content={{
 *     original: "Complex text...",
 *     simplified: "Simple text...",
 *     translated: "अनुवादित पाठ...",
 *     audioUrl: "/audio/task123.mp3",
 *     targetLanguage: "hi",
 *   }}
 *   enableFeedback
 *   onFeedback={handleFeedback}
 * />
 */
export function ResultsPanel({
  content,
  taskId,
  enableFeedback = true,
  onFeedback,
  className,
}: ResultsPanelProps) {
  const [copiedTab, setCopiedTab] = useState<string | null>(null);
  const [rating, setRating] = useState(0);
  const [feedbackComment, setFeedbackComment] = useState('');
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);

  const { original, simplified, translated, audioUrl, targetLanguage, metadata } = content;

  // Determine available tabs
  const tabs = [];
  if (original) tabs.push({ id: 'original', label: 'Original', content: original });
  if (simplified) tabs.push({ id: 'simplified', label: 'Simplified', content: simplified });
  if (translated) tabs.push({ id: 'translated', label: 'Translated', content: translated });

  // Copy to clipboard
  const handleCopy = useCallback(async (text: string, tabId: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedTab(tabId);
      setTimeout(() => setCopiedTab(null), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  }, []);

  // Download as text file
  const handleDownload = useCallback((text: string, filename: string) => {
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, []);

  // Submit feedback
  const handleSubmitFeedback = useCallback(() => {
    if (rating > 0) {
      onFeedback?.({ rating, comment: feedbackComment, taskId });
      setFeedbackSubmitted(true);
    }
  }, [rating, feedbackComment, taskId, onFeedback]);

  if (tabs.length === 0 && !audioUrl) {
    return (
      <div className={cn('p-6 text-center text-surface-500', className)}>
        No results to display
      </div>
    );
  }

  return (
    <div className={cn('w-full', className)}>
      {/* Tabs */}
      {tabs.length > 0 && (
        <Tabs.Root defaultValue={tabs[tabs.length - 1].id} className="w-full">
          <Tabs.List className="flex border-b border-surface-200 dark:border-surface-700 mb-4">
            {tabs.map((tab) => (
              <Tabs.Trigger
                key={tab.id}
                value={tab.id}
                className={cn(
                  'px-4 py-2.5 text-sm font-medium transition-colors',
                  'border-b-2 border-transparent -mb-px',
                  'text-surface-500 hover:text-surface-700 dark:hover:text-surface-300',
                  'data-[state=active]:border-primary-500 data-[state=active]:text-primary-600 dark:data-[state=active]:text-primary-400'
                )}
              >
                {tab.label}
                {tab.id === 'translated' && targetLanguage && (
                  <Badge variant="primary" size="sm" className="ml-2">
                    {targetLanguage.toUpperCase()}
                  </Badge>
                )}
              </Tabs.Trigger>
            ))}
          </Tabs.List>

          {tabs.map((tab) => (
            <Tabs.Content key={tab.id} value={tab.id}>
              <div className="relative">
                {/* Actions */}
                <div className="absolute top-2 right-2 flex gap-1">
                  <Tooltip content={copiedTab === tab.id ? 'Copied!' : 'Copy to clipboard'}>
                    <IconButton
                      icon={copiedTab === tab.id ? <CheckIcon /> : <CopyIcon />}
                      variant="secondary"
                      size="sm"
                      aria-label="Copy to clipboard"
                      onClick={() => handleCopy(tab.content, tab.id)}
                    />
                  </Tooltip>
                  <Tooltip content="Download as text">
                    <IconButton
                      icon={<DownloadIcon />}
                      variant="secondary"
                      size="sm"
                      aria-label="Download as text file"
                      onClick={() => handleDownload(tab.content, `${tab.id}-${taskId || 'content'}.txt`)}
                    />
                  </Tooltip>
                </div>

                {/* Content */}
                <div
                  className={cn(
                    'p-4 pr-20 rounded-lg bg-surface-50 dark:bg-surface-800/50',
                    'border border-surface-200 dark:border-surface-700',
                    'min-h-[200px] max-h-[400px] overflow-auto',
                    'text-surface-800 dark:text-surface-200 leading-relaxed',
                    'whitespace-pre-wrap',
                    tab.id === 'translated' && targetLanguage === 'hi' && 'font-hindi'
                  )}
                >
                  {tab.content}
                </div>
              </div>
            </Tabs.Content>
          ))}
        </Tabs.Root>
      )}

      {/* Audio player */}
      {audioUrl && (
        <div className={cn(tabs.length > 0 && 'mt-6')}>
          <h3 className="text-sm font-medium text-surface-700 dark:text-surface-300 mb-3">
            Generated Audio
          </h3>
          <AudioPlayer
            src={audioUrl}
            title="Audio Narration"
            language={targetLanguage}
            showDownload
            showSpeedControl
          />
        </div>
      )}

      {/* Metadata */}
      {metadata && (
        <div className="mt-6 flex flex-wrap gap-4 text-xs text-surface-500">
          {metadata.wordCount && (
            <span>{metadata.wordCount} words</span>
          )}
          {metadata.processingTime && (
            <span>Processed in {metadata.processingTime}s</span>
          )}
          {metadata.confidence && (
            <span>Confidence: {Math.round(metadata.confidence * 100)}%</span>
          )}
        </div>
      )}

      {/* Feedback */}
      {enableFeedback && !feedbackSubmitted && (
        <div className="mt-8 p-4 rounded-lg bg-surface-50 dark:bg-surface-800/50 border border-surface-200 dark:border-surface-700">
          <h3 className="text-sm font-medium text-surface-700 dark:text-surface-300 mb-3">
            How was the translation quality?
          </h3>
          
          {/* Star rating */}
          <div className="flex gap-1 mb-4">
            {[1, 2, 3, 4, 5].map((star) => (
              <button
                key={star}
                onClick={() => setRating(star)}
                className="p-0.5 hover:scale-110 transition-transform"
                aria-label={`Rate ${star} stars`}
              >
                <StarIcon filled={star <= rating} />
              </button>
            ))}
          </div>

          {/* Comment */}
          <Textarea
            placeholder="Tell us more about your experience (optional)"
            value={feedbackComment}
            onChange={(e) => setFeedbackComment(e.target.value)}
            rows={2}
            className="mb-3"
          />

          <Button
            variant="primary"
            size="sm"
            onClick={handleSubmitFeedback}
            disabled={rating === 0}
          >
            Submit Feedback
          </Button>
        </div>
      )}

      {feedbackSubmitted && (
        <div className="mt-6 p-4 rounded-lg bg-success-50 dark:bg-success-900/20 border border-success-200 dark:border-success-800">
          <p className="text-sm text-success-700 dark:text-success-300 flex items-center gap-2">
            <CheckIcon />
            Thank you for your feedback!
          </p>
        </div>
      )}
    </div>
  );
}

ResultsPanel.displayName = 'ResultsPanel';
