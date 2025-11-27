import { useState, useEffect, useCallback } from 'react';
import api from '../services/api';

interface Content {
  id: string;
  subject: string;
  grade_level: number;
  language?: string;
  simplified_text?: string;
  created_at?: string;
}

interface Answer {
  answer: string;
  confidence_score?: number;
  context_chunks?: string[];
}

interface HistoryItem {
  id?: string;
  chat_id?: string;
  question: string;
  answer: string;
  confidence_score?: number;
  created_at?: string;
}

export default function QAPage() {
  const [contents, setContents] = useState<Content[]>([]);
  const [selectedContent, setSelectedContent] = useState('');
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isAsking, setIsAsking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);

  useEffect(() => {
    loadContents();
  }, []);

  const loadHistory = useCallback(async () => {
    if (!selectedContent) return;
    try {
      const response = await api.getQAHistory(selectedContent, 10);
      setHistory(response.history || []);
    } catch (err) {
      console.error('Failed to load history', err);
    }
  }, [selectedContent]);

  useEffect(() => {
    if (selectedContent) {
      loadHistory();
    }
  }, [selectedContent, loadHistory]);

  const loadContents = async () => {
    try {
      const response = await api.getLibrary({ limit: 50, offset: 0 });
      setContents(response.items);
    } catch (err) {
      console.error('Failed to load contents', err);
    }
  };

  const pollTaskStatus = useCallback(async (
    taskId: string,
    maxAttempts: number = 30,
    intervalMs: number = 2000
  ): Promise<boolean> => {
    let attempts = 0;
    let timeoutId: ReturnType<typeof setTimeout> | null = null;

    const checkStatus = async (): Promise<boolean> => {
      try {
        attempts++;
        
        // Query backend for task status
        // Adjust endpoint based on your actual API
        const response = await api.getTaskStatus?.(taskId);
        const status = (response as any)?.status || 'unknown';

        if (status === 'completed' || status === 'success') {
          return true;
        } else if (status === 'failed' || status === 'error') {
          throw new Error(response?.error || 'Task processing failed');
        }

        // Still processing
        if (attempts >= maxAttempts) {
          throw new Error(`Task ${taskId} timed out after ${maxAttempts} attempts`);
        }

        // Schedule next check
        await new Promise((resolve) => {
          timeoutId = setTimeout(resolve, intervalMs);
        });

        return await checkStatus();
      } catch (err) {
        if (timeoutId) clearTimeout(timeoutId);
        throw err;
      }
    };

    try {
      return await checkStatus();
    } catch (err) {
      if (timeoutId) clearTimeout(timeoutId);
      throw err;
    }
  }, []);

  const handleProcessDocument = async () => {
    if (!selectedContent) return;
    setIsProcessing(true);
    setError(null);

    try {
      const response = await api.processDocumentForQA(selectedContent);
      const taskId = response.task_id;
      
      // Show initial feedback
      setError(`Document processing started (Task ID: ${taskId}). Waiting for completion...`);

      // Poll for task completion
      const completed = await pollTaskStatus(taskId);
      
      if (completed) {
        // Refresh contents and history when done
        await loadContents();
        await loadHistory();
        setError(null);
        alert('Document processed successfully! You can now ask questions.');
      }
    } catch (err: any) {
      const errorMessage = err.response?.data?.detail || err.message || 'Failed to process document';
      setError(errorMessage);
      console.error('Document processing error:', err);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleAskQuestion = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedContent || !question.trim()) return;

    setIsAsking(true);
    setError(null);
    setAnswer(null);

    try {
      const response = await api.askQuestion(selectedContent, question, true, 3);
      setAnswer(response);
      setQuestion('');
      loadHistory();
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to ask question');
    } finally {
      setIsAsking(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-white to-purple-50 p-8">
      <div className="max-w-6xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent">
            Q&A System
          </h1>
          <p className="text-gray-600">Ask questions about your uploaded documents</p>
        </div>

        {error && (
          <div className="mb-6 p-4 bg-red-50 border-2 border-red-200 rounded-lg text-red-700">
            {error}
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left sidebar - Document selection */}
          <div className="lg:col-span-1">
            <div className="glass-card p-6">
              <h3 className="font-semibold text-gray-900 mb-4">📚 Select Document</h3>
              <select
                value={selectedContent}
                onChange={(e) => setSelectedContent(e.target.value)}
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 transition-colors mb-4"
              >
                <option value="">Choose a document...</option>
                {contents.map((content) => (
                  <option key={content.id} value={content.id}>
                    {content.subject} - Grade {content.grade_level}
                  </option>
                ))}
              </select>

              {selectedContent && (
                <button
                  onClick={handleProcessDocument}
                  disabled={isProcessing}
                  className="w-full btn-secondary disabled:opacity-50"
                >
                  {isProcessing ? 'Processing...' : '🔄 Process for Q&A'}
                </button>
              )}

              {history.length > 0 && (
                <div className="mt-6">
                  <h4 className="font-semibold text-gray-900 mb-3">📜 Recent Q&A</h4>
                  <div className="space-y-2 max-h-96 overflow-y-auto">
                    {history.map((item) => {
                      const preview = (item.answer ?? '').substring(0, 100);
                      const showEllipsis = item.answer && item.answer.length > 100;
                      return (
                        <div key={item.id || item.chat_id || Math.random()} className="p-3 bg-gray-50 rounded-lg text-sm">
                          <p className="font-medium text-gray-900 mb-1">Q: {item.question}</p>
                          <p className="text-gray-600">A: {preview}{showEllipsis ? '...' : ''}</p>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Main content - Q&A interface */}
          <div className="lg:col-span-2">
            <div className="glass-card p-8">
              <form onSubmit={handleAskQuestion} className="mb-6">
                <label htmlFor="question" className="block text-sm font-semibold text-gray-700 mb-2">
                  Ask a Question
                </label>
                <textarea
                  id="question"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  rows={4}
                  className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-indigo-500 focus:ring-2 focus:ring-indigo-200 transition-colors mb-4"
                  placeholder="What would you like to know about this document?"
                  disabled={!selectedContent}
                />
                <button
                  type="submit"
                  disabled={isAsking || !selectedContent || !question.trim()}
                  className="btn-primary w-full disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isAsking ? (
                    <>
                      <svg className="animate-spin h-5 w-5 mr-2 inline" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                      </svg>
                      Getting Answer...
                    </>
                  ) : (
                    '💬 Ask Question'
                  )}
                </button>
              </form>

              {answer && (
                <div className="p-6 bg-gradient-to-br from-indigo-50 to-purple-50 rounded-lg border-2 border-indigo-200">
                  <h3 className="font-semibold text-indigo-900 mb-3">Answer:</h3>
                  <p className="text-gray-800 leading-relaxed whitespace-pre-wrap">{answer.answer}</p>
                  {answer.confidence_score != null && (
                    <div className="mt-4 flex items-center gap-2 text-sm text-gray-600">
                      <span>Confidence:</span>
                      <div className="flex-1 h-2 bg-gray-200 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-indigo-500 to-purple-500"
                          style={{ width: `${answer.confidence_score * 100}%` }}
                        />
                      </div>
                      <span>{(answer.confidence_score * 100).toFixed(1)}%</span>
                    </div>
                  )}
                </div>
              )}

              {!selectedContent && (
                <div className="text-center py-12 text-gray-400">
                  <svg className="w-24 h-24 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                  </svg>
                  <p>Select a document to start asking questions</p>
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="mt-8 glass-card p-6">
          <h3 className="font-semibold text-gray-900 mb-3">💡 How it works:</h3>
          <ol className="space-y-2 text-gray-600">
            <li>1. Select a document from your library</li>
            <li>2. Process it for Q&A (creates searchable chunks)</li>
            <li>3. Ask questions in natural language</li>
            <li>4. Get AI-powered answers based on document content</li>
          </ol>
        </div>
      </div>
    </div>
  );
}
