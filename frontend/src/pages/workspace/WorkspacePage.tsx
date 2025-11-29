/**
 * AI Workspace Page
 * 
 * A flexible, AI-first interface that lets users work with content immediately
 * without being forced through rigid workflows.
 * 
 * Features:
 * - Direct text input (paste or type)
 * - Optional file upload
 * - One-click access to all AI features
 * - Smart defaults that just work
 * - Results appear in-place
 */

import { useState, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Sparkles, 
  Languages, 
  Volume2, 
  MessageSquare,
  Upload,
  FileText,
  Copy,
  Check,
  RotateCcw,
  Loader2,
  ChevronDown,
  Settings2,
  Zap
} from 'lucide-react';
import { api } from '../../services/api';
import { Button } from '../../components/ui/Button/Button';
import { Textarea } from '../../components/ui/Textarea/Textarea';
import { Badge } from '../../components/ui/Badge/Badge';
import { cn } from '../../lib/cn';

// Smart defaults
const DEFAULT_GRADE = 6;
const DEFAULT_SUBJECT = 'General';
const DEFAULT_LANGUAGE = 'Hindi';

// Languages
const LANGUAGES = [
  { code: 'Hindi', name: 'हिंदी' },
  { code: 'Tamil', name: 'தமிழ்' },
  { code: 'Telugu', name: 'తెలుగు' },
  { code: 'Bengali', name: 'বাংলা' },
  { code: 'Marathi', name: 'मराठी' },
];

// Grades
const GRADES = [5, 6, 7, 8, 9, 10];

// Subjects
const SUBJECTS = ['General', 'Science', 'Mathematics', 'Social Studies', 'English', 'History'];

type ActiveFeature = 'simplify' | 'translate' | 'tts' | 'qa' | null;

interface Result {
  type: ActiveFeature;
  content: string;
  audioUrl?: string;
  language?: string;
}

export function WorkspacePage() {
  // Input state
  const [inputText, setInputText] = useState('');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [contentId, setContentId] = useState<string | null>(null);
  
  // Settings (with smart defaults)
  const [gradeLevel, setGradeLevel] = useState(DEFAULT_GRADE);
  const [subject, setSubject] = useState(DEFAULT_SUBJECT);
  const [targetLanguage, setTargetLanguage] = useState(DEFAULT_LANGUAGE);
  const [showSettings, setShowSettings] = useState(false);
  
  // Processing state
  const [activeFeature, setActiveFeature] = useState<ActiveFeature>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [results, setResults] = useState<Result[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  
  // File input ref
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Handle file upload
  const handleFileUpload = async (file: File) => {
    setUploadedFile(file);
    setError(null);
    
    try {
      const result = await api.uploadFile(file);
      setContentId(result.content_id);
      
      // If text was extracted, use it (cast to access optional property)
      const uploadResult = result as { file_path: string; content_id: string; status: string; extracted_text?: string };
      if (uploadResult.extracted_text) {
        setInputText(uploadResult.extracted_text);
      }
    } catch (e: any) {
      setError(e.message || 'Failed to upload file');
    }
  };

  // Handle file drop
  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleFileUpload(file);
  }, []);

  // Simplify text
  const handleSimplify = async () => {
    if (!inputText.trim()) return;
    
    setActiveFeature('simplify');
    setIsProcessing(true);
    setError(null);
    
    try {
      const result = await api.simplifyText({
        text: inputText,
        grade_level: gradeLevel,
        subject,
      });
      
      // Poll for result
      await pollForResult(result.task_id, 'simplify');
    } catch (e: any) {
      setError(e.message || 'Simplification failed');
      setIsProcessing(false);
    }
  };

  // Translate text
  const handleTranslate = async () => {
    if (!inputText.trim()) return;
    
    setActiveFeature('translate');
    setIsProcessing(true);
    setError(null);
    
    try {
      const result = await api.translateText({
        text: inputText,
        target_language: targetLanguage,
        subject,
      });
      
      await pollForResult(result.task_id, 'translate');
    } catch (e: any) {
      setError(e.message || 'Translation failed');
      setIsProcessing(false);
    }
  };

  // Text to Speech
  const handleTTS = async () => {
    if (!inputText.trim()) return;
    
    setActiveFeature('tts');
    setIsProcessing(true);
    setError(null);
    
    try {
      const result = await api.generateAudio({
        text: inputText,
        language: targetLanguage,
      });
      
      await pollForResult(result.task_id, 'tts');
    } catch (e: any) {
      setError(e.message || 'Speech generation failed');
      setIsProcessing(false);
    }
  };

  // Ask question about content
  const handleAskQuestion = async () => {
    if (!inputText.trim()) return;
    
    setActiveFeature('qa');
    setIsProcessing(true);
    setError(null);
    
    try {
      // For Q&A, we treat the input as a question about previously uploaded content
      // If no content uploaded, we'll use the text as context
      const result = await api.askQuestion(
        contentId || 'direct-input',
        inputText,
        { wait: true }
      );
      
      setResults(prev => [...prev, {
        type: 'qa',
        content: result.answer || 'No answer available',
      }]);
      setIsProcessing(false);
    } catch (e: any) {
      setError(e.message || 'Q&A failed');
      setIsProcessing(false);
    }
  };

  // Poll for task result
  const pollForResult = async (taskId: string, type: ActiveFeature) => {
    const maxAttempts = 60;
    let attempts = 0;
    
    const poll = async () => {
      try {
        const status = await api.getTaskStatus(taskId);
        
        if (status.state === 'SUCCESS') {
          let content = '';
          let audioUrl = '';
          
          if (type === 'simplify') {
            content = status.result?.simplified_text || '';
          } else if (type === 'translate') {
            content = status.result?.translated_text || '';
          } else if (type === 'tts') {
            audioUrl = status.result?.audio_url || '';
            content = 'Audio generated successfully';
          }
          
          setResults(prev => [...prev, {
            type,
            content,
            audioUrl,
            language: type === 'translate' ? targetLanguage : undefined,
          }]);
          setIsProcessing(false);
          return;
        }
        
        if (status.state === 'FAILURE') {
          setError(status.error || 'Processing failed');
          setIsProcessing(false);
          return;
        }
        
        attempts++;
        if (attempts < maxAttempts) {
          setTimeout(poll, 1000);
        } else {
          setError('Request timed out');
          setIsProcessing(false);
        }
      } catch (e) {
        attempts++;
        if (attempts < maxAttempts) {
          setTimeout(poll, 1000);
        }
      }
    };
    
    poll();
  };

  // Copy to clipboard
  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Reset
  const handleReset = () => {
    setInputText('');
    setUploadedFile(null);
    setContentId(null);
    setResults([]);
    setError(null);
    setActiveFeature(null);
  };

  const hasInput = inputText.trim().length > 0;
  const latestResult = results[results.length - 1];

  return (
    <div className="min-h-screen bg-gradient-to-b from-background to-muted/20 p-4 md:p-8">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Header */}
        <motion.div 
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center space-y-2"
        >
          <h1 className="text-3xl font-bold bg-gradient-to-r from-primary-500 to-primary-700 bg-clip-text text-transparent">
            AI Workspace
          </h1>
          <p className="text-muted-foreground">
            Paste text, upload a file, or just start typing — then use any AI feature instantly
          </p>
        </motion.div>

        {/* Main Input Area */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="relative"
        >
          <div
            className={cn(
              "relative rounded-xl border-2 border-dashed transition-colors",
              "bg-card shadow-sm",
              !hasInput && "border-muted-foreground/25 hover:border-primary-500/50"
            )}
            onDrop={handleDrop}
            onDragOver={(e) => e.preventDefault()}
          >
            <Textarea
              placeholder="Paste or type any text here... or drag & drop a PDF/document"
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              className="min-h-[200px] border-0 bg-transparent resize-none text-base focus:ring-0"
            />
            
            {/* File upload indicator */}
            {uploadedFile && (
              <div className="absolute top-3 right-3">
                <Badge variant="secondary" className="gap-1">
                  <FileText className="w-3 h-3" />
                  {uploadedFile.name}
                </Badge>
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
                  onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
                />
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => fileInputRef.current?.click()}
                  className="text-muted-foreground"
                >
                  <Upload className="w-4 h-4 mr-1" />
                  Upload
                </Button>
                
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowSettings(!showSettings)}
                  className="text-muted-foreground"
                >
                  <Settings2 className="w-4 h-4 mr-1" />
                  Settings
                  <ChevronDown className={cn(
                    "w-4 h-4 ml-1 transition-transform",
                    showSettings && "rotate-180"
                  )} />
                </Button>
              </div>
              
              {hasInput && (
                <Button variant="ghost" size="sm" onClick={handleReset}>
                  <RotateCcw className="w-4 h-4 mr-1" />
                  Clear
                </Button>
              )}
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
                <div className="mt-3 p-4 bg-card rounded-lg border shadow-sm grid grid-cols-3 gap-4">
                  <div>
                    <label className="text-sm font-medium text-muted-foreground mb-2 block">
                      Grade Level
                    </label>
                    <select
                      value={gradeLevel}
                      onChange={(e) => setGradeLevel(Number(e.target.value))}
                      className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
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
                      value={subject}
                      onChange={(e) => setSubject(e.target.value)}
                      className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
                    >
                      {SUBJECTS.map(s => (
                        <option key={s} value={s}>{s}</option>
                      ))}
                    </select>
                  </div>
                  <div>
                    <label className="text-sm font-medium text-muted-foreground mb-2 block">
                      Target Language
                    </label>
                    <select
                      value={targetLanguage}
                      onChange={(e) => setTargetLanguage(e.target.value)}
                      className="w-full rounded-md border border-input bg-background px-3 py-2 text-sm"
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
        </motion.div>

        {/* Action Buttons - Always visible */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="flex flex-wrap justify-center gap-3"
        >
          <Button
            size="lg"
            onClick={handleSimplify}
            disabled={!hasInput || isProcessing}
            className={cn(
              "gap-2 min-w-[140px]",
              activeFeature === 'simplify' && isProcessing && "animate-pulse"
            )}
          >
            {activeFeature === 'simplify' && isProcessing ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Sparkles className="w-5 h-5" />
            )}
            Simplify
          </Button>
          
          <Button
            size="lg"
            variant="secondary"
            onClick={handleTranslate}
            disabled={!hasInput || isProcessing}
            className={cn(
              "gap-2 min-w-[140px]",
              activeFeature === 'translate' && isProcessing && "animate-pulse"
            )}
          >
            {activeFeature === 'translate' && isProcessing ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Languages className="w-5 h-5" />
            )}
            Translate
          </Button>
          
          <Button
            size="lg"
            variant="secondary"
            onClick={handleTTS}
            disabled={!hasInput || isProcessing}
            className={cn(
              "gap-2 min-w-[140px]",
              activeFeature === 'tts' && isProcessing && "animate-pulse"
            )}
          >
            {activeFeature === 'tts' && isProcessing ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <Volume2 className="w-5 h-5" />
            )}
            Read Aloud
          </Button>
          
          <Button
            size="lg"
            variant="outline"
            onClick={handleAskQuestion}
            disabled={!hasInput || isProcessing}
            className={cn(
              "gap-2 min-w-[140px]",
              activeFeature === 'qa' && isProcessing && "animate-pulse"
            )}
          >
            {activeFeature === 'qa' && isProcessing ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <MessageSquare className="w-5 h-5" />
            )}
            Ask AI
          </Button>
        </motion.div>

        {/* Error Display */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              className="p-4 bg-destructive/10 border border-destructive/20 rounded-lg text-destructive text-center"
            >
              {error}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Results Display */}
        <AnimatePresence>
          {latestResult && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="bg-card rounded-xl border shadow-sm overflow-hidden"
            >
              <div className="flex items-center justify-between p-4 border-b bg-muted/30">
                <div className="flex items-center gap-2">
                  {latestResult.type === 'simplify' && <Sparkles className="w-5 h-5 text-primary-500" />}
                  {latestResult.type === 'translate' && <Languages className="w-5 h-5 text-blue-500" />}
                  {latestResult.type === 'tts' && <Volume2 className="w-5 h-5 text-green-500" />}
                  {latestResult.type === 'qa' && <MessageSquare className="w-5 h-5 text-purple-500" />}
                  <span className="font-medium capitalize">{latestResult.type} Result</span>
                  {latestResult.language && (
                    <Badge variant="secondary">{latestResult.language}</Badge>
                  )}
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleCopy(latestResult.content)}
                >
                  {copied ? (
                    <Check className="w-4 h-4 text-green-500" />
                  ) : (
                    <Copy className="w-4 h-4" />
                  )}
                </Button>
              </div>
              
              <div className="p-4">
                {latestResult.audioUrl ? (
                  <audio controls src={latestResult.audioUrl} className="w-full" />
                ) : (
                  <p className="whitespace-pre-wrap text-foreground leading-relaxed">
                    {latestResult.content}
                  </p>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Quick tip */}
        {!hasInput && !latestResult && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
            className="text-center text-sm text-muted-foreground"
          >
            <Zap className="w-4 h-4 inline-block mr-1" />
            Tip: Just paste any educational content and click a button — no setup required!
          </motion.div>
        )}
      </div>
    </div>
  );
}

export default WorkspacePage;
