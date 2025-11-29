/**
 * Content Detail Page
 * 
 * View and interact with processed educational content
 * Features: simplification, translation, audio, Q&A
 */

import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  ArrowLeft,
  Languages, 
  MessageCircle,
  FileText,
  Download,
  Share2,
  ThumbsUp,
  ThumbsDown,
  Loader2,
  Play,
  Pause,
  ChevronDown,
  Send,
  Sparkles
} from 'lucide-react';
import { api } from '../../services/api';
import { Button } from '../../components/ui/Button/Button';
import { Badge } from '../../components/ui/Badge/Badge';
import { Spinner } from '../../components/ui/Spinner/Spinner';
import { pageVariants, staggerItem } from '../../lib/animations';

// Types
interface ContentData {
  id: string;
  original_text: string;
  simplified_text?: string;
  translated_text?: string;
  translations?: Record<string, string>;
  language: string;
  grade_level: number;
  subject: string;
  audio_url?: string;
  metadata?: Record<string, any>;
}

interface QAMessage {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  confidence?: number;
}

// Supported languages
const LANGUAGES = [
  { code: 'Hindi', name: 'हिंदी' },
  { code: 'Tamil', name: 'தமிழ்' },
  { code: 'Telugu', name: 'తెలుగు' },
  { code: 'Bengali', name: 'বাংলা' },
  { code: 'Marathi', name: 'मराठी' },
  { code: 'Gujarati', name: 'ગુજરાતી' },
  { code: 'Kannada', name: 'ಕನ್ನಡ' },
  { code: 'Malayalam', name: 'മലയാളം' },
  { code: 'Punjabi', name: 'ਪੰਜਾਬੀ' },
  { code: 'Odia', name: 'ଓଡ଼ିଆ' },
];

export function ContentDetailPage() {
  const { contentId } = useParams<{ contentId: string }>();
  const navigate = useNavigate();
  
  // State
  const [content, setContent] = useState<ContentData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'original' | 'simplified' | 'translated'>('original');
  const [selectedLanguage, setSelectedLanguage] = useState('Hindi');
  const [isTranslating, setIsTranslating] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [qaMessages, setQaMessages] = useState<QAMessage[]>([]);
  const [qaInput, setQaInput] = useState('');
  const [isAskingQuestion, setIsAskingQuestion] = useState(false);
  const [showQA, setShowQA] = useState(false);

  // Load content
  useEffect(() => {
    if (!contentId) return;
    
    const loadContent = async () => {
      try {
        setLoading(true);
        const data = await api.getContent(contentId);
        setContent(data);
        
        // Load Q&A history
        try {
          const history = await api.getQAHistory(contentId);
          if (history.history?.length > 0) {
            setQaMessages(history.history.map((h: any) => ({
              id: h.id,
              type: h.role === 'user' ? 'user' : 'assistant',
              content: h.content,
              timestamp: new Date(h.created_at),
              confidence: h.confidence_score,
            })));
          }
        } catch (e) {
          // Q&A history might not exist yet
        }
      } catch (e: any) {
        setError(e.message || 'Failed to load content');
      } finally {
        setLoading(false);
      }
    };

    loadContent();
  }, [contentId]);

  // Handle translation
  const handleTranslate = async () => {
    if (!content || isTranslating) return;
    
    try {
      setIsTranslating(true);
      const text = content.simplified_text || content.original_text;
      const result = await api.translateText({
        text,
        target_language: selectedLanguage,
      });

      // Poll for result
      if (result.task_id) {
        const pollInterval = setInterval(async () => {
          const status = await api.getTaskStatus(result.task_id);
          if (status.state === 'SUCCESS') {
            clearInterval(pollInterval);
            setContent(prev => prev ? {
              ...prev,
              translations: {
                ...prev.translations,
                [selectedLanguage]: status.result?.translations?.[selectedLanguage] || '',
              },
              translated_text: status.result?.translations?.[selectedLanguage] || '',
            } : null);
            setActiveTab('translated');
            setIsTranslating(false);
          } else if (status.state === 'FAILURE') {
            clearInterval(pollInterval);
            setIsTranslating(false);
          }
        }, 1000);
      }
    } catch (e) {
      setIsTranslating(false);
    }
  };

  // Handle Q&A
  const handleAskQuestion = async () => {
    if (!qaInput.trim() || !contentId || isAskingQuestion) return;

    const userMessage: QAMessage = {
      id: `user-${Date.now()}`,
      type: 'user',
      content: qaInput,
      timestamp: new Date(),
    };

    setQaMessages(prev => [...prev, userMessage]);
    setQaInput('');
    setIsAskingQuestion(true);

    try {
      const result = await api.askQuestion(contentId, qaInput, { wait: true });
      
      const assistantMessage: QAMessage = {
        id: `assistant-${Date.now()}`,
        type: 'assistant',
        content: result.answer || 'I could not find an answer to your question.',
        timestamp: new Date(),
        confidence: result.confidence_score,
      };

      setQaMessages(prev => [...prev, assistantMessage]);
    } catch (e: any) {
      const errorMessage: QAMessage = {
        id: `error-${Date.now()}`,
        type: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date(),
      };
      setQaMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsAskingQuestion(false);
    }
  };

  // Handle feedback
  const handleFeedback = async (rating: number) => {
    if (!contentId) return;
    try {
      await api.submitFeedback({
        content_id: contentId,
        rating,
        feedback_text: rating > 3 ? 'Helpful' : 'Needs improvement',
      });
    } catch (e) {
      console.error('Feedback failed:', e);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="text-center">
          <Spinner size="lg" className="mb-4" />
          <p className="text-muted-foreground">Loading content...</p>
        </div>
      </div>
    );
  }

  if (error || !content) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[60vh]">
        <div className="text-center max-w-md">
          <FileText className="w-16 h-16 text-muted-foreground/50 mx-auto mb-4" />
          <h2 className="text-xl font-semibold mb-2">Content Not Found</h2>
          <p className="text-muted-foreground mb-6">{error || 'The content you are looking for does not exist.'}</p>
          <Button onClick={() => navigate('/library')}>
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back to Library
          </Button>
        </div>
      </div>
    );
  }

  const displayText = activeTab === 'original' 
    ? content.original_text 
    : activeTab === 'simplified' 
      ? content.simplified_text 
      : content.translations?.[selectedLanguage] || content.translated_text;

  return (
    <motion.div 
      variants={pageVariants}
      initial="initial"
      animate="enter"
      exit="exit"
      className="space-y-6"
    >
      {/* Header */}
      <motion.div variants={staggerItem} className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="sm" onClick={() => navigate(-1)}>
            <ArrowLeft className="w-4 h-4" />
          </Button>
          <div>
            <h1 className="text-2xl font-bold text-foreground">{content.subject}</h1>
            <div className="flex items-center gap-2 mt-1">
              <Badge variant="primary">Grade {content.grade_level}</Badge>
              <Badge variant="secondary">{content.language}</Badge>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm">
            <Share2 className="w-4 h-4 mr-2" />
            Share
          </Button>
          <Button variant="outline" size="sm">
            <Download className="w-4 h-4 mr-2" />
            Export
          </Button>
        </div>
      </motion.div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Content Panel */}
        <motion.div variants={staggerItem} className="lg:col-span-2 space-y-4">
          {/* Tabs */}
          <div className="flex items-center gap-2 border-b border-border pb-2">
            <button
              onClick={() => setActiveTab('original')}
              className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${
                activeTab === 'original' 
                  ? 'bg-primary-50 text-primary-600 border-b-2 border-primary-500' 
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <FileText className="w-4 h-4 inline mr-2" />
              Original
            </button>
            <button
              onClick={() => setActiveTab('simplified')}
              disabled={!content.simplified_text}
              className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${
                activeTab === 'simplified' 
                  ? 'bg-primary-50 text-primary-600 border-b-2 border-primary-500' 
                  : 'text-muted-foreground hover:text-foreground disabled:opacity-50'
              }`}
            >
              <Sparkles className="w-4 h-4 inline mr-2" />
              Simplified
            </button>
            <button
              onClick={() => setActiveTab('translated')}
              className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors ${
                activeTab === 'translated' 
                  ? 'bg-primary-50 text-primary-600 border-b-2 border-primary-500' 
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <Languages className="w-4 h-4 inline mr-2" />
              Translated
            </button>
          </div>

          {/* Translation Controls */}
          {activeTab === 'translated' && (
            <div className="flex items-center gap-3 bg-muted/50 rounded-lg p-3">
              <select
                value={selectedLanguage}
                onChange={(e) => setSelectedLanguage(e.target.value)}
                className="px-3 py-2 rounded-lg border border-border bg-background text-sm"
              >
                {LANGUAGES.map(lang => (
                  <option key={lang.code} value={lang.code}>
                    {lang.name} ({lang.code})
                  </option>
                ))}
              </select>
              <Button 
                size="sm" 
                onClick={handleTranslate}
                isLoading={isTranslating}
                disabled={isTranslating}
              >
                {isTranslating ? 'Translating...' : 'Translate'}
              </Button>
            </div>
          )}

          {/* Content Display */}
          <div className="bg-card rounded-xl border border-border p-6 min-h-[400px]">
            <AnimatePresence mode="wait">
              <motion.div
                key={activeTab + selectedLanguage}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="prose prose-lg max-w-none"
              >
                {displayText ? (
                  <p className="whitespace-pre-wrap text-foreground leading-relaxed">
                    {displayText}
                  </p>
                ) : (
                  <div className="text-center py-12 text-muted-foreground">
                    <Languages className="w-12 h-12 mx-auto mb-3 opacity-50" />
                    <p>No translation available yet.</p>
                    <p className="text-sm">Select a language and click Translate.</p>
                  </div>
                )}
              </motion.div>
            </AnimatePresence>
          </div>

          {/* Audio Player */}
          {content.audio_url && (
            <div className="bg-gradient-to-r from-primary-50 to-secondary-50 rounded-xl p-4 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Button 
                  variant="primary" 
                  size="sm"
                  iconOnly
                  onClick={() => setIsPlaying(!isPlaying)}
                >
                  {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                </Button>
                <div>
                  <p className="font-medium text-sm">Audio Version</p>
                  <p className="text-xs text-muted-foreground">Listen to this content</p>
                </div>
              </div>
              <audio
                src={api.getAudioUrl(content.id)}
                onPlay={() => setIsPlaying(true)}
                onPause={() => setIsPlaying(false)}
                onEnded={() => setIsPlaying(false)}
                controls
                className="w-64"
              />
            </div>
          )}

          {/* Feedback */}
          <div className="flex items-center justify-between bg-muted/30 rounded-xl p-4">
            <span className="text-sm text-muted-foreground">Was this content helpful?</span>
            <div className="flex items-center gap-2">
              <Button variant="ghost" size="sm" onClick={() => handleFeedback(5)}>
                <ThumbsUp className="w-4 h-4 mr-1" />
                Yes
              </Button>
              <Button variant="ghost" size="sm" onClick={() => handleFeedback(1)}>
                <ThumbsDown className="w-4 h-4 mr-1" />
                No
              </Button>
            </div>
          </div>
        </motion.div>

        {/* Q&A Panel */}
        <motion.div variants={staggerItem} className="lg:col-span-1">
          <div className="bg-card rounded-xl border border-border overflow-hidden h-full flex flex-col">
            {/* Q&A Header */}
            <div 
              className="p-4 border-b border-border bg-gradient-to-r from-primary-500 to-secondary-500 text-white cursor-pointer"
              onClick={() => setShowQA(!showQA)}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <MessageCircle className="w-5 h-5" />
                  <span className="font-semibold">Ask Questions</span>
                </div>
                <ChevronDown className={`w-5 h-5 transition-transform ${showQA ? 'rotate-180' : ''}`} />
              </div>
              <p className="text-sm text-white/80 mt-1">
                AI-powered Q&A about this content
              </p>
            </div>

            {/* Q&A Content */}
            <AnimatePresence>
              {showQA && (
                <motion.div
                  initial={{ height: 0 }}
                  animate={{ height: 'auto' }}
                  exit={{ height: 0 }}
                  className="flex-1 flex flex-col overflow-hidden"
                >
                  {/* Messages */}
                  <div className="flex-1 overflow-y-auto p-4 space-y-4 max-h-[400px]">
                    {qaMessages.length === 0 ? (
                      <div className="text-center py-8 text-muted-foreground">
                        <MessageCircle className="w-10 h-10 mx-auto mb-3 opacity-50" />
                        <p className="text-sm">Ask questions about this content</p>
                      </div>
                    ) : (
                      qaMessages.map((msg) => (
                        <div
                          key={msg.id}
                          className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}
                        >
                          <div
                            className={`max-w-[85%] rounded-2xl px-4 py-2 ${
                              msg.type === 'user'
                                ? 'bg-primary-500 text-white rounded-br-sm'
                                : 'bg-muted rounded-bl-sm'
                            }`}
                          >
                            <p className="text-sm">{msg.content}</p>
                            {msg.confidence && (
                              <p className="text-xs opacity-70 mt-1">
                                Confidence: {Math.round(msg.confidence * 100)}%
                              </p>
                            )}
                          </div>
                        </div>
                      ))
                    )}
                    {isAskingQuestion && (
                      <div className="flex justify-start">
                        <div className="bg-muted rounded-2xl rounded-bl-sm px-4 py-3">
                          <Loader2 className="w-4 h-4 animate-spin" />
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Input */}
                  <div className="p-4 border-t border-border">
                    <div className="flex items-center gap-2">
                      <input
                        type="text"
                        value={qaInput}
                        onChange={(e) => setQaInput(e.target.value)}
                        onKeyPress={(e) => e.key === 'Enter' && handleAskQuestion()}
                        placeholder="Ask a question..."
                        className="flex-1 px-4 py-2 rounded-full border border-border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-primary-500"
                      />
                      <Button 
                        size="sm" 
                        iconOnly 
                        onClick={handleAskQuestion}
                        disabled={!qaInput.trim() || isAskingQuestion}
                      >
                        <Send className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </motion.div>
      </div>
    </motion.div>
  );
}

export default ContentDetailPage;
