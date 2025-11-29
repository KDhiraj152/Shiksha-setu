/**
 * Translate Page
 * 
 * Multi-language translation with real-time streaming support
 */

import { useState, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Languages, 
  Copy, 
  Check,
  RotateCcw,
  ArrowRight,
  Loader2,
  Volume2,
  Globe,
  Zap
} from 'lucide-react';
import { api } from '../../services/api';
import { streamingService } from '../../services/streaming';
import { Button } from '../../components/ui/Button/Button';
import { Textarea } from '../../components/ui/Textarea/Textarea';
import { Badge } from '../../components/ui/Badge/Badge';
import { Progress } from '../../components/ui/Progress/Progress';
import { pageVariants, staggerItem } from '../../lib/animations';

// Indian languages
const LANGUAGES = [
  { code: 'Hindi', name: 'हिंदी', native: 'Hindi', flag: '🇮🇳' },
  { code: 'Tamil', name: 'தமிழ்', native: 'Tamil', flag: '🇮🇳' },
  { code: 'Telugu', name: 'తెలుగు', native: 'Telugu', flag: '🇮🇳' },
  { code: 'Bengali', name: 'বাংলা', native: 'Bengali', flag: '🇮🇳' },
  { code: 'Marathi', name: 'मराठी', native: 'Marathi', flag: '🇮🇳' },
  { code: 'Gujarati', name: 'ગુજરાતી', native: 'Gujarati', flag: '🇮🇳' },
  { code: 'Kannada', name: 'ಕನ್ನಡ', native: 'Kannada', flag: '🇮🇳' },
  { code: 'Malayalam', name: 'മലയാളം', native: 'Malayalam', flag: '🇮🇳' },
  { code: 'Punjabi', name: 'ਪੰਜਾਬੀ', native: 'Punjabi', flag: '🇮🇳' },
  { code: 'Odia', name: 'ଓଡ଼ିଆ', native: 'Odia', flag: '🇮🇳' },
  { code: 'Assamese', name: 'অসমীয়া', native: 'Assamese', flag: '🇮🇳' },
  { code: 'Urdu', name: 'اردو', native: 'Urdu', flag: '🇮🇳' },
];

interface Translation {
  language: string;
  text: string;
  status: 'pending' | 'processing' | 'completed' | 'error';
  progress?: number;
}

export function TranslatePage() {
  // State
  const [inputText, setInputText] = useState('');
  const [selectedLanguages, setSelectedLanguages] = useState<string[]>(['Hindi']);
  const [translations, setTranslations] = useState<Map<string, Translation>>(new Map());
  const [isProcessing, setIsProcessing] = useState(false);
  const [copied, setCopied] = useState<string | null>(null);
  const [useStreaming, setUseStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Connect to WebSocket for streaming
  useEffect(() => {
    if (useStreaming) {
      streamingService.connect({
        reconnect: true,
        onMessage: (data) => {
          if (data.type === 'translation_chunk') {
            setTranslations(prev => {
              const newMap = new Map(prev);
              const existing = newMap.get(data.language);
              if (existing) {
                newMap.set(data.language, {
                  ...existing,
                  text: (existing.text || '') + data.chunk,
                  status: 'processing',
                });
              }
              return newMap;
            });
          } else if (data.type === 'translation_complete') {
            setTranslations(prev => {
              const newMap = new Map(prev);
              const existing = newMap.get(data.language);
              if (existing) {
                newMap.set(data.language, {
                  ...existing,
                  status: 'completed',
                });
              }
              return newMap;
            });
          }
        },
      });
    }
    
    return () => {
      if (useStreaming) {
        streamingService.disconnect();
      }
    };
  }, [useStreaming]);

  // Toggle language
  const toggleLanguage = (code: string) => {
    setSelectedLanguages(prev => 
      prev.includes(code)
        ? prev.filter(l => l !== code)
        : [...prev, code]
    );
  };

  // Handle translation
  const handleTranslate = useCallback(async () => {
    if (!inputText.trim() || selectedLanguages.length === 0 || isProcessing) return;

    setIsProcessing(true);
    setError(null);

    // Initialize translations
    const initialTranslations = new Map<string, Translation>();
    selectedLanguages.forEach(lang => {
      initialTranslations.set(lang, {
        language: lang,
        text: '',
        status: 'pending',
        progress: 0,
      });
    });
    setTranslations(initialTranslations);

    try {
      if (useStreaming && streamingService.isConnected) {
        // Use WebSocket streaming
        streamingService.startTranslationStream(inputText, selectedLanguages);
        
        // Set timeout for streaming
        setTimeout(() => {
          if (isProcessing) {
            setIsProcessing(false);
          }
        }, 60000);
      } else {
        // Use REST API
        const result = await api.translateText({
          text: inputText,
          target_languages: selectedLanguages,
        });

        if (result.task_id) {
          // Poll for results
          const pollInterval = setInterval(async () => {
            try {
              const status = await api.getTaskStatus(result.task_id);
              
              if (status.state === 'PENDING' || status.state === 'STARTED' || status.state === 'PROCESSING') {
                // Update progress for all languages
                setTranslations(prev => {
                  const newMap = new Map(prev);
                  selectedLanguages.forEach(lang => {
                    const existing = newMap.get(lang);
                    if (existing) {
                      newMap.set(lang, {
                        ...existing,
                        status: 'processing',
                        progress: status.progress || 50,
                      });
                    }
                  });
                  return newMap;
                });
              } else if (status.state === 'SUCCESS') {
                clearInterval(pollInterval);
                const resultTranslations = status.result?.translations || {};
                
                setTranslations(prev => {
                  const newMap = new Map(prev);
                  selectedLanguages.forEach(lang => {
                    newMap.set(lang, {
                      language: lang,
                      text: resultTranslations[lang] || '',
                      status: resultTranslations[lang] ? 'completed' : 'error',
                    });
                  });
                  return newMap;
                });
                setIsProcessing(false);
              } else if (status.state === 'FAILURE') {
                clearInterval(pollInterval);
                setError(status.error || 'Translation failed');
                setIsProcessing(false);
              }
            } catch (e) {
              // Keep polling
            }
          }, 1000);

          // Timeout
          setTimeout(() => {
            clearInterval(pollInterval);
            if (isProcessing) {
              setError('Request timed out');
              setIsProcessing(false);
            }
          }, 120000);
        }
      }
    } catch (e: any) {
      setError(e.message || 'Translation failed');
      setIsProcessing(false);
    }
  }, [inputText, selectedLanguages, isProcessing, useStreaming]);

  // Copy translation
  const handleCopy = (lang: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(lang);
    setTimeout(() => setCopied(null), 2000);
  };

  // Reset
  const handleReset = () => {
    setInputText('');
    setTranslations(new Map());
    setError(null);
  };

  // Get language info
  const getLangInfo = (code: string) => LANGUAGES.find(l => l.code === code);

  return (
    <motion.div 
      variants={pageVariants}
      initial="initial"
      animate="enter"
      className="max-w-7xl mx-auto space-y-6"
    >
      {/* Header */}
      <motion.div variants={staggerItem} className="text-center">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-purple-500 to-pink-500 mb-4">
          <Languages className="w-8 h-8 text-white" />
        </div>
        <h1 className="text-3xl font-bold text-foreground mb-2">Multi-Language Translation</h1>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          Translate educational content into 12+ Indian regional languages simultaneously.
          Preserves educational context and terminology.
        </p>
      </motion.div>

      {/* Language Selection */}
      <motion.div variants={staggerItem} className="bg-card rounded-xl border border-border p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold text-foreground flex items-center gap-2">
            <Globe className="w-5 h-5 text-primary-500" />
            Select Target Languages
            <Badge variant="secondary">{selectedLanguages.length} selected</Badge>
          </h3>
          <label className="flex items-center gap-2 text-sm">
            <input
              type="checkbox"
              checked={useStreaming}
              onChange={(e) => setUseStreaming(e.target.checked)}
              className="rounded"
            />
            <Zap className="w-4 h-4 text-yellow-500" />
            Real-time streaming
          </label>
        </div>
        
        <div className="flex flex-wrap gap-2">
          {LANGUAGES.map((lang) => (
            <button
              key={lang.code}
              onClick={() => toggleLanguage(lang.code)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-2 ${
                selectedLanguages.includes(lang.code)
                  ? 'bg-primary-500 text-white shadow-lg shadow-primary-500/25'
                  : 'bg-muted text-muted-foreground hover:bg-muted/80'
              }`}
            >
              <span>{lang.flag}</span>
              <span>{lang.native}</span>
              {selectedLanguages.includes(lang.code) && (
                <Check className="w-4 h-4" />
              )}
            </button>
          ))}
        </div>
      </motion.div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input */}
        <motion.div variants={staggerItem} className="space-y-3">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-foreground">
              Source Text (English)
            </label>
            <span className="text-xs text-muted-foreground">
              {inputText.split(/\s+/).filter(Boolean).length} words
            </span>
          </div>
          <Textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Enter or paste the text you want to translate..."
            className="min-h-[300px] resize-none"
          />
          
          {/* Actions */}
          <div className="flex items-center gap-3">
            <Button variant="outline" onClick={handleReset} disabled={isProcessing}>
              <RotateCcw className="w-4 h-4 mr-2" />
              Reset
            </Button>
            <Button 
              variant="gradient" 
              className="flex-1"
              onClick={handleTranslate}
              disabled={!inputText.trim() || selectedLanguages.length === 0 || isProcessing}
              isLoading={isProcessing}
            >
              <Languages className="w-5 h-5 mr-2" />
              Translate to {selectedLanguages.length} Language{selectedLanguages.length !== 1 ? 's' : ''}
              <ArrowRight className="w-5 h-5 ml-2" />
            </Button>
          </div>
        </motion.div>

        {/* Translations */}
        <motion.div variants={staggerItem} className="space-y-4">
          <h3 className="text-sm font-medium text-foreground">Translations</h3>
          
          {translations.size === 0 ? (
            <div className="bg-muted/30 rounded-xl p-12 text-center">
              <Languages className="w-12 h-12 text-muted-foreground/50 mx-auto mb-3" />
              <p className="text-muted-foreground">
                Select languages and click translate to see results here
              </p>
            </div>
          ) : (
            <div className="space-y-4 max-h-[500px] overflow-y-auto pr-2">
              <AnimatePresence mode="popLayout">
                {Array.from(translations.entries()).map(([lang, translation]) => {
                  const langInfo = getLangInfo(lang);
                  return (
                    <motion.div
                      key={lang}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, scale: 0.95 }}
                      className="bg-card rounded-xl border border-border overflow-hidden"
                    >
                      {/* Header */}
                      <div className="flex items-center justify-between px-4 py-3 bg-muted/30 border-b border-border">
                        <div className="flex items-center gap-2">
                          <span className="text-lg">{langInfo?.flag}</span>
                          <span className="font-medium">{langInfo?.name}</span>
                          <span className="text-sm text-muted-foreground">({lang})</span>
                        </div>
                        <div className="flex items-center gap-2">
                          {translation.status === 'processing' && (
                            <Loader2 className="w-4 h-4 animate-spin text-primary-500" />
                          )}
                          {translation.status === 'completed' && (
                            <>
                              <Button 
                                variant="ghost" 
                                size="sm"
                                onClick={() => handleCopy(lang, translation.text)}
                              >
                                {copied === lang ? (
                                  <Check className="w-4 h-4" />
                                ) : (
                                  <Copy className="w-4 h-4" />
                                )}
                              </Button>
                              <Button variant="ghost" size="sm">
                                <Volume2 className="w-4 h-4" />
                              </Button>
                            </>
                          )}
                          {translation.status === 'error' && (
                            <Badge variant="error">Failed</Badge>
                          )}
                        </div>
                      </div>
                      
                      {/* Content */}
                      <div className="p-4">
                        {translation.status === 'pending' && (
                          <div className="text-center py-4">
                            <div className="w-8 h-8 rounded-full border-2 border-muted border-t-primary-500 animate-spin mx-auto" />
                            <p className="text-sm text-muted-foreground mt-2">Waiting...</p>
                          </div>
                        )}
                        {translation.status === 'processing' && (
                          <div className="space-y-3">
                            {translation.text && (
                              <p className="text-foreground">{translation.text}</p>
                            )}
                            <Progress value={translation.progress || 50} className="h-1" />
                          </div>
                        )}
                        {translation.status === 'completed' && (
                          <p className="text-foreground whitespace-pre-wrap leading-relaxed">
                            {translation.text}
                          </p>
                        )}
                        {translation.status === 'error' && (
                          <p className="text-red-500 text-sm">
                            Failed to translate. Please try again.
                          </p>
                        )}
                      </div>
                    </motion.div>
                  );
                })}
              </AnimatePresence>
            </div>
          )}
        </motion.div>
      </div>

      {/* Error */}
      {error && (
        <motion.div 
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700 text-sm"
        >
          {error}
        </motion.div>
      )}

      {/* Info */}
      <motion.div variants={staggerItem} className="bg-gradient-to-r from-purple-50 to-pink-50 rounded-xl p-6 border border-purple-100">
        <h3 className="font-semibold text-purple-900 mb-2">About Translation Quality</h3>
        <ul className="text-sm text-purple-700 space-y-1">
          <li>• Translations are optimized for educational content</li>
          <li>• Technical terms and formulas are preserved</li>
          <li>• NCERT curriculum terminology is maintained</li>
          <li>• Real-time streaming provides faster initial results</li>
        </ul>
      </motion.div>
    </motion.div>
  );
}

export default TranslatePage;
