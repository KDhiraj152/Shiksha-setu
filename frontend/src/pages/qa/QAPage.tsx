/**
 * Q&A Page
 * 
 * Document Question & Answer interface with AI-powered responses
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import { useSearchParams } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  MessageCircle,
  Send,
  FileText,
  Upload,
  Loader2,
  Sparkles,
  Trash2,
  Copy,
  Check,
  Brain,
  Zap
} from 'lucide-react';
import { api } from '../../services/api';
import { Button } from '../../components/ui/Button/Button';
import { Badge } from '../../components/ui/Badge/Badge';
import { pageVariants, staggerItem } from '../../lib/animations';

// Types
interface Message {
  id: string;
  type: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
  confidence?: number;
  contextChunks?: number;
  processing?: boolean;
}

interface Document {
  id: string;
  name: string;
  subject?: string;
  gradeLevel?: number;
  qaReady: boolean;
}

export function QAPage() {
  const [searchParams] = useSearchParams();
  const contentIdParam = searchParams.get('contentId');
  
  // State
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedDoc, setSelectedDoc] = useState<string | null>(contentIdParam);
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isProcessingDoc, setIsProcessingDoc] = useState(false);
  const [, setUploadedFile] = useState<File | null>(null);
  const [copied, setCopied] = useState<string | null>(null);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load documents
  useEffect(() => {
    const loadDocuments = async () => {
      try {
        const library = await api.getLibrary({ limit: 50, offset: 0 });
        setDocuments(library.items?.map(item => ({
          id: item.id,
          name: item.metadata?.filename || `${item.subject} - Grade ${item.grade_level}`,
          subject: item.subject,
          gradeLevel: item.grade_level,
          qaReady: item.metadata?.qa_ready || false,
        })) || []);
      } catch (e) {
        console.error('Failed to load documents:', e);
      }
    };

    loadDocuments();
  }, []);

  // Load Q&A history when document changes
  useEffect(() => {
    if (!selectedDoc) return;

    const loadHistory = async () => {
      try {
        const history = await api.getQAHistory(selectedDoc);
        if (history.history?.length > 0) {
          setMessages(history.history.map((h: any) => ({
            id: h.id || `hist-${Date.now()}-${Math.random()}`,
            type: h.role === 'user' ? 'user' : 'assistant',
            content: h.content,
            timestamp: new Date(h.created_at),
            confidence: h.confidence_score,
          })));
        } else {
          setMessages([]);
        }
      } catch (e) {
        setMessages([]);
      }
    };

    loadHistory();
  }, [selectedDoc]);

  // Scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Handle file upload
  const handleFileUpload = async (file: File) => {
    setUploadedFile(file);
    setIsProcessingDoc(true);

    try {
      // Upload file
      const uploadResult = await api.uploadFile(file);
      const contentId = uploadResult.content_id;

      // Add system message
      setMessages([{
        id: `sys-${Date.now()}`,
        type: 'system',
        content: `Document "${file.name}" uploaded. Processing for Q&A...`,
        timestamp: new Date(),
      }]);

      // Process for Q&A
      const processResult = await api.processDocumentForQA(contentId);

      // Poll for processing completion
      if (processResult.task_id) {
        const pollInterval = setInterval(async () => {
          const status = await api.getTaskStatus(processResult.task_id);
          
          if (status.state === 'SUCCESS') {
            clearInterval(pollInterval);
            setSelectedDoc(contentId);
            setDocuments(prev => [...prev, {
              id: contentId,
              name: file.name,
              qaReady: true,
            }]);
            setMessages([{
              id: `sys-${Date.now()}`,
              type: 'system',
              content: `✓ Document ready! You can now ask questions about "${file.name}"`,
              timestamp: new Date(),
            }]);
            setIsProcessingDoc(false);
          } else if (status.state === 'FAILURE') {
            clearInterval(pollInterval);
            setMessages([{
              id: `sys-${Date.now()}`,
              type: 'system',
              content: `✗ Failed to process document. Please try again.`,
              timestamp: new Date(),
            }]);
            setIsProcessingDoc(false);
          }
        }, 2000);

        // Timeout after 5 minutes
        setTimeout(() => {
          clearInterval(pollInterval);
          setIsProcessingDoc(false);
        }, 300000);
      }
    } catch (e: any) {
      setMessages([{
        id: `sys-${Date.now()}`,
        type: 'system',
        content: `✗ Upload failed: ${e.message}`,
        timestamp: new Date(),
      }]);
      setIsProcessingDoc(false);
    }
  };

  // Handle question
  const handleAskQuestion = useCallback(async () => {
    if (!inputValue.trim() || !selectedDoc || isLoading) return;

    const userMessage: Message = {
      id: `user-${Date.now()}`,
      type: 'user',
      content: inputValue,
      timestamp: new Date(),
    };

    const assistantMessage: Message = {
      id: `assistant-${Date.now()}`,
      type: 'assistant',
      content: '',
      timestamp: new Date(),
      processing: true,
    };

    setMessages(prev => [...prev, userMessage, assistantMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const result = await api.askQuestion(selectedDoc, inputValue, { wait: true, topK: 3 });

      setMessages(prev => prev.map(msg => 
        msg.id === assistantMessage.id 
          ? {
              ...msg,
              content: result.answer || 'I could not find an answer to your question.',
              confidence: result.confidence_score,
              processing: false,
            }
          : msg
      ));
    } catch (e: any) {
      setMessages(prev => prev.map(msg => 
        msg.id === assistantMessage.id 
          ? {
              ...msg,
              content: 'Sorry, I encountered an error processing your question. Please try again.',
              processing: false,
            }
          : msg
      ));
    } finally {
      setIsLoading(false);
    }
  }, [inputValue, selectedDoc, isLoading]);

  // Copy message
  const handleCopy = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(id);
    setTimeout(() => setCopied(null), 2000);
  };

  // Clear chat
  const handleClearChat = () => {
    setMessages([]);
  };

  const selectedDocument = documents.find(d => d.id === selectedDoc);

  return (
    <motion.div 
      variants={pageVariants}
      initial="initial"
      animate="enter"
      className="h-[calc(100vh-180px)] flex flex-col"
    >
      {/* Header */}
      <motion.div variants={staggerItem} className="flex-shrink-0 pb-4 border-b border-border">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-foreground flex items-center gap-2">
              <Brain className="w-7 h-7 text-primary-500" />
              Document Q&A
            </h1>
            <p className="text-muted-foreground mt-1">
              Ask questions about your educational documents
            </p>
          </div>
          <div className="flex items-center gap-3">
            {/* Document selector */}
            <select
              value={selectedDoc || ''}
              onChange={(e) => setSelectedDoc(e.target.value || null)}
              className="px-4 py-2 rounded-lg border border-border bg-background text-foreground min-w-[200px]"
            >
              <option value="">Select a document...</option>
              {documents.map(doc => (
                <option key={doc.id} value={doc.id}>
                  {doc.name} {doc.qaReady ? '✓' : '(processing)'}
                </option>
              ))}
            </select>
            
            {/* Upload button */}
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.txt,.doc,.docx"
              onChange={(e) => e.target.files?.[0] && handleFileUpload(e.target.files[0])}
              className="hidden"
            />
            <Button 
              variant="outline"
              onClick={() => fileInputRef.current?.click()}
              disabled={isProcessingDoc}
            >
              {isProcessingDoc ? (
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              ) : (
                <Upload className="w-4 h-4 mr-2" />
              )}
              Upload
            </Button>
          </div>
        </div>
      </motion.div>

      {/* Main Content */}
      <div className="flex-1 flex gap-4 min-h-0 mt-4">
        {/* Chat Area */}
        <motion.div 
          variants={staggerItem}
          className="flex-1 flex flex-col bg-card rounded-xl border border-border overflow-hidden"
        >
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {!selectedDoc ? (
              <div className="flex items-center justify-center h-full">
                <div className="text-center max-w-md">
                  <MessageCircle className="w-16 h-16 text-muted-foreground/30 mx-auto mb-4" />
                  <h3 className="text-lg font-semibold text-foreground mb-2">
                    Select or Upload a Document
                  </h3>
                  <p className="text-muted-foreground text-sm mb-4">
                    Choose an existing document from the dropdown or upload a new one to start asking questions.
                  </p>
                  <Button onClick={() => fileInputRef.current?.click()}>
                    <Upload className="w-4 h-4 mr-2" />
                    Upload Document
                  </Button>
                </div>
              </div>
            ) : messages.length === 0 ? (
              <div className="flex items-center justify-center h-full">
                <div className="text-center max-w-md">
                  <Sparkles className="w-16 h-16 text-primary-500/30 mx-auto mb-4" />
                  <h3 className="text-lg font-semibold text-foreground mb-2">
                    Ready to Answer Questions
                  </h3>
                  <p className="text-muted-foreground text-sm">
                    Ask anything about <strong>{selectedDocument?.name}</strong>
                  </p>
                </div>
              </div>
            ) : (
              <AnimatePresence mode="popLayout">
                {messages.map((msg) => (
                  <motion.div
                    key={msg.id}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, scale: 0.95 }}
                    className={`flex ${msg.type === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    <div className={`max-w-[80%] ${msg.type === 'system' ? 'w-full' : ''}`}>
                      {msg.type === 'system' ? (
                        <div className="bg-muted/50 rounded-lg px-4 py-2 text-center text-sm text-muted-foreground">
                          {msg.content}
                        </div>
                      ) : (
                        <div
                          className={`rounded-2xl px-4 py-3 ${
                            msg.type === 'user'
                              ? 'bg-primary-500 text-white rounded-br-sm'
                              : 'bg-muted rounded-bl-sm'
                          }`}
                        >
                          {msg.processing ? (
                            <div className="flex items-center gap-2">
                              <Loader2 className="w-4 h-4 animate-spin" />
                              <span className="text-sm">Thinking...</span>
                            </div>
                          ) : (
                            <>
                              <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
                              
                              {msg.type === 'assistant' && (
                                <div className="mt-2 pt-2 border-t border-current/10 flex items-center justify-between text-xs opacity-70">
                                  <div className="flex items-center gap-2">
                                    {msg.confidence !== undefined && (
                                      <Badge 
                                        variant={msg.confidence > 0.7 ? 'success' : msg.confidence > 0.4 ? 'warning' : 'error'}
                                        className="text-[10px]"
                                      >
                                        {Math.round(msg.confidence * 100)}% confident
                                      </Badge>
                                    )}
                                    {msg.contextChunks && (
                                      <span>{msg.contextChunks} sources</span>
                                    )}
                                  </div>
                                  <button
                                    onClick={() => handleCopy(msg.id, msg.content)}
                                    className="hover:opacity-100 opacity-50 transition-opacity"
                                  >
                                    {copied === msg.id ? (
                                      <Check className="w-3 h-3" />
                                    ) : (
                                      <Copy className="w-3 h-3" />
                                    )}
                                  </button>
                                </div>
                              )}
                            </>
                          )}
                        </div>
                      )}
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          {selectedDoc && (
            <div className="flex-shrink-0 p-4 border-t border-border bg-muted/30">
              <div className="flex items-center gap-2">
                <input
                  type="text"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && handleAskQuestion()}
                  placeholder="Ask a question about this document..."
                  disabled={isLoading}
                  className="flex-1 px-4 py-3 rounded-xl border border-border bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-primary-500"
                />
                <Button 
                  variant="primary"
                  size="lg"
                  onClick={handleAskQuestion}
                  disabled={!inputValue.trim() || isLoading}
                  isLoading={isLoading}
                >
                  <Send className="w-5 h-5" />
                </Button>
                {messages.length > 0 && (
                  <Button variant="ghost" onClick={handleClearChat}>
                    <Trash2 className="w-4 h-4" />
                  </Button>
                )}
              </div>
            </div>
          )}
        </motion.div>

        {/* Sidebar - Document Info */}
        {selectedDoc && selectedDocument && (
          <motion.div 
            variants={staggerItem}
            className="w-72 flex-shrink-0 bg-card rounded-xl border border-border p-4 space-y-4"
          >
            <div>
              <h3 className="font-semibold text-foreground flex items-center gap-2">
                <FileText className="w-4 h-4 text-primary-500" />
                Document Info
              </h3>
            </div>

            <div className="space-y-3 text-sm">
              <div>
                <span className="text-muted-foreground">Name:</span>
                <p className="font-medium text-foreground truncate">{selectedDocument.name}</p>
              </div>
              {selectedDocument.subject && (
                <div>
                  <span className="text-muted-foreground">Subject:</span>
                  <p className="font-medium text-foreground">{selectedDocument.subject}</p>
                </div>
              )}
              {selectedDocument.gradeLevel && (
                <div>
                  <span className="text-muted-foreground">Grade Level:</span>
                  <p className="font-medium text-foreground">Grade {selectedDocument.gradeLevel}</p>
                </div>
              )}
              <div>
                <span className="text-muted-foreground">Status:</span>
                <Badge variant={selectedDocument.qaReady ? 'success' : 'warning'} className="ml-2">
                  {selectedDocument.qaReady ? 'Ready' : 'Processing'}
                </Badge>
              </div>
            </div>

            <div className="pt-4 border-t border-border">
              <h4 className="text-sm font-medium text-foreground mb-2">Suggested Questions</h4>
              <div className="space-y-2">
                {[
                  'What is the main topic?',
                  'Summarize the key points',
                  'Explain the concepts simply',
                ].map((q, i) => (
                  <button
                    key={i}
                    onClick={() => setInputValue(q)}
                    className="w-full text-left px-3 py-2 rounded-lg bg-muted/50 text-sm text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>

            <div className="pt-4 border-t border-border">
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <Zap className="w-4 h-4 text-yellow-500" />
                <span>AI-powered by advanced NLP</span>
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </motion.div>
  );
}

export default QAPage;
