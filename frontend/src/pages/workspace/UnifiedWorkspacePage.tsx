/**
 * Unified AI Workspace Page
 * 
 * The flagship interface for ShikshaSetu - a unified AI-first workspace
 * that lets users work with content through the complete pipeline:
 * Upload → Extract → Simplify → Translate → Validate → TTS
 * 
 * Features:
 * - Drag & drop file upload or text paste
 * - Visual pipeline wizard showing progress
 * - Side-by-side content comparison
 * - One-click access to all AI features
 * - Real-time processing with progress feedback
 * - NCERT validation results
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import {
  Sparkles,
  Languages,
  Volume2,
  Upload,
  FileText,
  Copy,
  Check,
  RotateCcw,
  Loader2,
  ChevronDown,
  Settings2,
  Zap,
  Shield,
  X,
} from 'lucide-react';
import { Button } from '../../components/ui/Button/Button';
import { Textarea } from '../../components/ui/Textarea/Textarea';
import { Badge } from '../../components/ui/Badge/Badge';
import { PipelineWizard } from '../../components/features/pipeline';
import { ContentComparison } from '../../components/features/content';
import { ValidationResults } from '../../components/features/validation';
import { usePipelineStore, pipelineSelectors } from '../../store/pipelineStore';
import { unifiedApi } from '../../services/unifiedApi';
import { cn } from '../../lib/cn';
import { pageVariants, staggerItem } from '../../lib/animations';
import toast from 'react-hot-toast';

// =============================================================================
// Constants
// =============================================================================

const LANGUAGES = [
  { code: 'Hindi', name: 'हिंदी', native: 'Hindi' },
  { code: 'Tamil', name: 'தமிழ்', native: 'Tamil' },
  { code: 'Telugu', name: 'తెలుగు', native: 'Telugu' },
  { code: 'Bengali', name: 'বাংলা', native: 'Bengali' },
  { code: 'Marathi', name: 'मराठी', native: 'Marathi' },
  { code: 'Gujarati', name: 'ગુજરાતી', native: 'Gujarati' },
  { code: 'Kannada', name: 'ಕನ್ನಡ', native: 'Kannada' },
  { code: 'Malayalam', name: 'മലയാളം', native: 'Malayalam' },
  { code: 'Punjabi', name: 'ਪੰਜਾਬੀ', native: 'Punjabi' },
  { code: 'Odia', name: 'ଓଡ଼ିଆ', native: 'Odia' },
];

const GRADES = [5, 6, 7, 8, 9, 10, 11, 12];

const SUBJECTS = [
  'General',
  'Science',
  'Mathematics',
  'Social Studies',
  'English',
  'Hindi',
  'Computer Science',
];

// =============================================================================
// Main Component
// =============================================================================

export function UnifiedWorkspacePage() {
  // Pipeline store
  const pipeline = usePipelineStore();
  const isProcessing = pipelineSelectors.isProcessing(pipeline);
  const hasContent = pipelineSelectors.hasContent(pipeline);
  const hasResults = pipelineSelectors.hasResults(pipeline);

  // Local UI state
  const [inputText, setInputText] = useState('');
  const [showSettings, setShowSettings] = useState(false);
  const [copied, setCopied] = useState(false);

  // Refs
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Sync input text with pipeline store
  useEffect(() => {
    if (pipeline.originalText && !inputText) {
      setInputText(pipeline.originalText);
    }
  }, [pipeline.originalText]);

  // ==========================================================================
  // File Upload
  // ==========================================================================

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    pipeline.setFile(file);
    pipeline.startStage('uploading');

    try {
      const result = await unifiedApi.uploadFile(file, {
        gradeLevel: pipeline.gradeLevel,
        subject: pipeline.subject,
        onProgress: (progress) => {
          pipeline.updateProgress('uploading', progress);
        },
      });

      pipeline.setContentId(result.content_id, result.file_path);
      pipeline.completeStage('uploading');

      // If text was extracted, use it
      if (result.extracted_text) {
        pipeline.setOriginalText(result.extracted_text);
        setInputText(result.extracted_text);
        pipeline.completeStage('extracting');
        toast.success('Document uploaded and text extracted!');
      } else {
        toast.success('Document uploaded successfully!');
      }
    } catch (error: any) {
      pipeline.failStage('uploading', error.message || 'Upload failed');
      toast.error(error.message || 'Failed to upload file');
    }
  }, [pipeline]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'text/plain': ['.txt'],
      'application/msword': ['.doc'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
    },
    maxSize: 100 * 1024 * 1024, // 100MB
    multiple: false,
  });

  // ==========================================================================
  // Processing Actions
  // ==========================================================================

  const handleTextInput = () => {
    if (inputText.trim()) {
      pipeline.setOriginalText(inputText);
    }
  };

  const handleSimplify = async () => {
    const text = pipeline.originalText || inputText;
    if (!text.trim()) {
      toast.error('Please enter or upload some text first');
      return;
    }

    if (!pipeline.originalText) {
      pipeline.setOriginalText(text);
    }

    pipeline.startStage('simplifying');

    try {
      const result = await unifiedApi.simplifyText({
        text,
        grade_level: pipeline.gradeLevel,
        subject: pipeline.subject,
      });

      // Poll for result
      const status = await unifiedApi.pollTaskUntilComplete(result.task_id, {
        onProgress: (s) => {
          pipeline.updateProgress('simplifying', s.progress || 0);
        },
      });

      if (status.state === 'SUCCESS' && status.result?.simplified_text) {
        pipeline.setSimplifiedText(status.result.simplified_text);
        pipeline.completeStage('simplifying');
        toast.success('Text simplified successfully!');
      } else {
        throw new Error(status.error || 'Simplification failed');
      }
    } catch (error: any) {
      pipeline.failStage('simplifying', error.message);
      toast.error(error.message || 'Failed to simplify text');
    }
  };

  const handleTranslate = async () => {
    const text = pipeline.simplifiedText || pipeline.originalText || inputText;
    if (!text.trim()) {
      toast.error('Please enter or upload some text first');
      return;
    }

    if (!pipeline.originalText) {
      pipeline.setOriginalText(text);
    }

    pipeline.startStage('translating');

    try {
      const result = await unifiedApi.translateText({
        text,
        target_languages: pipeline.targetLanguages,
        subject: pipeline.subject,
      });

      const status = await unifiedApi.pollTaskUntilComplete(result.task_id, {
        onProgress: (s) => {
          pipeline.updateProgress('translating', s.progress || 0);
        },
      });

      if (status.state === 'SUCCESS' && status.result?.translations) {
        pipeline.setTranslations(status.result.translations);
        pipeline.completeStage('translating');
        toast.success('Translation completed!');
      } else {
        throw new Error(status.error || 'Translation failed');
      }
    } catch (error: any) {
      pipeline.failStage('translating', error.message);
      toast.error(error.message || 'Failed to translate text');
    }
  };

  const handleValidate = async () => {
    if (!pipeline.simplifiedText && !pipeline.originalText) {
      toast.error('Please simplify content first');
      return;
    }

    pipeline.startStage('validating');

    try {
      const result = await unifiedApi.validateContent({
        original_text: pipeline.originalText,
        processed_text: pipeline.simplifiedText || pipeline.originalText,
        grade_level: pipeline.gradeLevel,
        subject: pipeline.subject,
        language: pipeline.targetLanguages[0] || 'Hindi',
      }, true);

      if (result.is_valid !== undefined) {
        pipeline.setValidation({
          isValid: result.is_valid,
          score: result.accuracy_score || 0,
          issues: (result.issues || []).map((issue: { severity: string; message: string }) => ({
            severity: issue.severity as 'error' | 'warning' | 'info',
            message: issue.message,
          })),
          suggestions: [],
        });
        pipeline.completeStage('validating');
        toast.success('Validation complete!');
      } else {
        throw new Error('Validation failed');
      }
    } catch (error: any) {
      pipeline.failStage('validating', error.message);
      toast.error(error.message || 'Failed to validate content');
    }
  };

  const handleGenerateAudio = async () => {
    const text = pipeline.simplifiedText || pipeline.translations[pipeline.targetLanguages[0]] || pipeline.originalText;
    if (!text) {
      toast.error('No content to convert to speech');
      return;
    }

    pipeline.startStage('generating-audio');

    try {
      const result = await unifiedApi.generateAudio({
        text,
        language: pipeline.targetLanguages[0] || 'Hindi',
        subject: pipeline.subject,
      }, true);

      if (result.audio_url) {
        pipeline.setAudio(result.audio_url, result.duration);
        pipeline.completeStage('generating-audio');
        toast.success('Audio generated!');
      } else {
        throw new Error('Audio generation failed');
      }
    } catch (error: any) {
      pipeline.failStage('generating-audio', error.message);
      toast.error(error.message || 'Failed to generate audio');
    }
  };

  const handleReset = () => {
    pipeline.reset();
    setInputText('');
  };

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
    toast.success('Copied to clipboard!');
  };

  // ==========================================================================
  // Render
  // ==========================================================================

  return (
    <motion.div
      variants={pageVariants}
      initial="initial"
      animate="animate"
      className="min-h-screen bg-gradient-to-b from-background to-muted/20"
    >
      <div className="max-w-6xl mx-auto p-4 md:p-8 space-y-6">
        {/* Header */}
        <motion.div variants={staggerItem} className="text-center space-y-2">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-primary-500 to-secondary-500 mb-4 shadow-lg shadow-primary-500/25">
            <Sparkles className="w-8 h-8 text-white" />
          </div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-primary-500 to-secondary-500 bg-clip-text text-transparent">
            AI Workspace
          </h1>
          <p className="text-muted-foreground max-w-2xl mx-auto">
            Transform educational content with AI — simplify, translate, validate, and convert to speech
          </p>
        </motion.div>

        {/* Pipeline Progress */}
        <motion.div variants={staggerItem}>
          <PipelineWizard 
            compact={false} 
            showDescriptions={true}
            className="bg-card p-4 rounded-xl border border-border"
          />
        </motion.div>

        {/* Main Content Area */}
        <motion.div variants={staggerItem} className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          {/* Left: Input Area (3 columns) */}
          <div className="lg:col-span-3 space-y-4">
            {/* Drop Zone / Text Input */}
            <div
              {...getRootProps()}
              className={cn(
                'relative rounded-xl border-2 border-dashed transition-all duration-200',
                'bg-card shadow-sm',
                isDragActive 
                  ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20' 
                  : 'border-border hover:border-primary-500/50',
              )}
            >
              <input {...getInputProps()} />
              
              <Textarea
                placeholder="Paste or type any educational text here... or drag & drop a PDF/document"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onBlur={handleTextInput}
                className="min-h-[300px] border-0 bg-transparent resize-none text-base focus:ring-0"
                onClick={(e) => e.stopPropagation()}
              />

              {/* File indicator */}
              {pipeline.fileName && (
                <div className="absolute top-3 right-3">
                  <Badge variant="secondary" className="gap-1">
                    <FileText className="w-3 h-3" />
                    {pipeline.fileName}
                    <button onClick={(e) => { e.stopPropagation(); handleReset(); }}>
                      <X className="w-3 h-3 ml-1" />
                    </button>
                  </Badge>
                </div>
              )}

              {/* Drag overlay */}
              {isDragActive && (
                <div className="absolute inset-0 flex items-center justify-center bg-primary-500/10 rounded-xl">
                  <div className="text-center">
                    <Upload className="w-12 h-12 mx-auto text-primary-500 mb-2" />
                    <p className="text-primary-600 font-medium">Drop your file here</p>
                  </div>
                </div>
              )}

              {/* Bottom toolbar */}
              <div className="flex items-center justify-between p-3 border-t border-border/50">
                <div className="flex items-center gap-2">
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf,.txt,.doc,.docx"
                    className="hidden"
                    onChange={(e) => e.target.files?.[0] && onDrop([e.target.files[0]])}
                  />
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }}
                  >
                    <Upload className="w-4 h-4 mr-1" />
                    Upload
                  </Button>

                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={(e) => { e.stopPropagation(); setShowSettings(!showSettings); }}
                  >
                    <Settings2 className="w-4 h-4 mr-1" />
                    Settings
                    <ChevronDown className={cn('w-4 h-4 ml-1 transition-transform', showSettings && 'rotate-180')} />
                  </Button>
                </div>

                <div className="flex items-center gap-2">
                  {inputText && (
                    <span className="text-xs text-muted-foreground">
                      {inputText.split(/\s+/).filter(Boolean).length} words
                    </span>
                  )}
                  {(inputText || pipeline.originalText) && (
                    <Button variant="ghost" size="sm" onClick={handleReset}>
                      <RotateCcw className="w-4 h-4 mr-1" />
                      Clear
                    </Button>
                  )}
                </div>
              </div>
            </div>

            {/* Settings Panel */}
            <AnimatePresence>
              {showSettings && (
                <motion.div
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  className="overflow-hidden"
                >
                  <div className="p-4 bg-card rounded-xl border border-border grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div>
                      <label className="text-sm font-medium text-muted-foreground mb-2 block">
                        Grade Level
                      </label>
                      <select
                        value={pipeline.gradeLevel}
                        onChange={(e) => pipeline.setGradeLevel(Number(e.target.value))}
                        className="w-full rounded-lg border border-input bg-background px-3 py-2 text-sm"
                      >
                        {GRADES.map(g => (
                          <option key={g} value={g}>Grade {g}</option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="text-sm font-medium text-muted-foreground mb-2 block">
                        Subject
                      </label>
                      <select
                        value={pipeline.subject}
                        onChange={(e) => pipeline.setSubject(e.target.value)}
                        className="w-full rounded-lg border border-input bg-background px-3 py-2 text-sm"
                      >
                        {SUBJECTS.map(s => (
                          <option key={s} value={s}>{s}</option>
                        ))}
                      </select>
                    </div>
                    <div>
                      <label className="text-sm font-medium text-muted-foreground mb-2 block">
                        Target Languages
                      </label>
                      <select
                        value={pipeline.targetLanguages[0]}
                        onChange={(e) => pipeline.setTargetLanguages([e.target.value])}
                        className="w-full rounded-lg border border-input bg-background px-3 py-2 text-sm"
                      >
                        {LANGUAGES.map(l => (
                          <option key={l.code} value={l.code}>{l.name} ({l.code})</option>
                        ))}
                      </select>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Action Buttons */}
            <div className="flex flex-wrap justify-center gap-3">
              <Button
                size="lg"
                variant="gradient"
                onClick={handleSimplify}
                disabled={(!inputText.trim() && !pipeline.originalText) || isProcessing}
                className="min-w-[140px]"
              >
                {pipeline.currentStage === 'simplifying' ? (
                  <Loader2 className="w-5 h-5 animate-spin mr-2" />
                ) : (
                  <Sparkles className="w-5 h-5 mr-2" />
                )}
                Simplify
              </Button>

              <Button
                size="lg"
                variant="secondary"
                onClick={handleTranslate}
                disabled={(!inputText.trim() && !pipeline.originalText) || isProcessing}
                className="min-w-[140px]"
              >
                {pipeline.currentStage === 'translating' ? (
                  <Loader2 className="w-5 h-5 animate-spin mr-2" />
                ) : (
                  <Languages className="w-5 h-5 mr-2" />
                )}
                Translate
              </Button>

              <Button
                size="lg"
                variant="secondary"
                onClick={handleValidate}
                disabled={!pipeline.simplifiedText || isProcessing}
                className="min-w-[140px]"
              >
                {pipeline.currentStage === 'validating' ? (
                  <Loader2 className="w-5 h-5 animate-spin mr-2" />
                ) : (
                  <Shield className="w-5 h-5 mr-2" />
                )}
                Validate
              </Button>

              <Button
                size="lg"
                variant="outline"
                onClick={handleGenerateAudio}
                disabled={(!pipeline.simplifiedText && !pipeline.originalText) || isProcessing}
                className="min-w-[140px]"
              >
                {pipeline.currentStage === 'generating-audio' ? (
                  <Loader2 className="w-5 h-5 animate-spin mr-2" />
                ) : (
                  <Volume2 className="w-5 h-5 mr-2" />
                )}
                Audio
              </Button>
            </div>
          </div>

          {/* Right: Results Panel (2 columns) */}
          <div className="lg:col-span-2 space-y-4">
            {/* Results Card */}
            <div className="bg-card rounded-xl border border-border overflow-hidden">
              <div className="p-4 border-b border-border bg-muted/30">
                <h3 className="font-semibold text-foreground">Results</h3>
              </div>

              <div className="p-4 min-h-[300px]">
                {!hasResults ? (
                  <div className="flex flex-col items-center justify-center h-full text-center py-12">
                    <div className="w-16 h-16 rounded-2xl bg-muted flex items-center justify-center mb-4">
                      <Zap className="w-8 h-8 text-muted-foreground" />
                    </div>
                    <p className="text-muted-foreground">
                      Results will appear here after processing
                    </p>
                    <p className="text-xs text-muted-foreground mt-2">
                      Try simplifying or translating some text
                    </p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {/* Simplified Text */}
                    {pipeline.simplifiedText && (
                      <div>
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium text-foreground flex items-center gap-2">
                            <Sparkles className="w-4 h-4 text-primary-500" />
                            Simplified (Grade {pipeline.gradeLevel})
                          </span>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleCopy(pipeline.simplifiedText)}
                          >
                            {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                          </Button>
                        </div>
                        <div className="p-3 bg-muted/50 rounded-lg text-sm leading-relaxed max-h-[200px] overflow-y-auto">
                          {pipeline.simplifiedText}
                        </div>
                      </div>
                    )}

                    {/* Translations */}
                    {Object.entries(pipeline.translations).map(([lang, text]) => (
                      <div key={lang}>
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium text-foreground flex items-center gap-2">
                            <Languages className="w-4 h-4 text-green-500" />
                            {lang}
                          </span>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleCopy(text)}
                          >
                            <Copy className="w-4 h-4" />
                          </Button>
                        </div>
                        <div className="p-3 bg-muted/50 rounded-lg text-sm leading-relaxed max-h-[200px] overflow-y-auto font-hindi">
                          {text}
                        </div>
                      </div>
                    ))}

                    {/* Audio Player */}
                    {pipeline.audioUrl && (
                      <div>
                        <span className="text-sm font-medium text-foreground flex items-center gap-2 mb-2">
                          <Volume2 className="w-4 h-4 text-purple-500" />
                          Generated Audio
                        </span>
                        <audio 
                          controls 
                          src={pipeline.audioUrl} 
                          className="w-full rounded-lg"
                        />
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>

            {/* Validation Results */}
            {pipeline.validation && (
              <ValidationResults
                result={pipeline.validation}
                gradeLevel={pipeline.gradeLevel}
                subject={pipeline.subject}
                onRevalidate={handleValidate}
                isLoading={pipeline.currentStage === 'validating'}
              />
            )}
          </div>
        </motion.div>

        {/* Content Comparison (when we have both original and processed) */}
        {pipeline.originalText && (pipeline.simplifiedText || Object.keys(pipeline.translations).length > 0) && (
          <motion.div variants={staggerItem}>
            <ContentComparison
              originalText={pipeline.originalText}
              processedText={pipeline.simplifiedText}
              translations={pipeline.translations}
              gradeLevel={pipeline.gradeLevel}
              audioUrl={pipeline.audioUrl || undefined}
            />
          </motion.div>
        )}

        {/* Quick Tip */}
        {!hasContent && !inputText && (
          <motion.div
            variants={staggerItem}
            className="text-center text-sm text-muted-foreground"
          >
            <Zap className="w-4 h-4 inline-block mr-1" />
            Tip: Just paste any educational content and click a button — AI handles the rest!
          </motion.div>
        )}
      </div>
    </motion.div>
  );
}

export default UnifiedWorkspacePage;
