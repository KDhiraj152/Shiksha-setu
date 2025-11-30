import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../services/api';

export default function TranslatePage() {
  const navigate = useNavigate();
  const [text, setText] = useState('');
  const [sourceLanguage, setSourceLanguage] = useState('en');
  const [targetLanguage, setTargetLanguage] = useState('hi');
  const [subject, setSubject] = useState('General');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const languages = [
    { code: 'en', name: 'English' },
    { code: 'hi', name: 'Hindi' },
    { code: 'bn', name: 'Bengali' },
    { code: 'ta', name: 'Tamil' },
    { code: 'te', name: 'Telugu' },
    { code: 'mr', name: 'Marathi' },
    { code: 'gu', name: 'Gujarati' },
    { code: 'kn', name: 'Kannada' },
    { code: 'ml', name: 'Malayalam' },
    { code: 'pa', name: 'Punjabi' },
    { code: 'or', name: 'Odia' },
  ];

  const subjects = ['General', 'Science', 'Mathematics', 'Social Studies', 'English', 'Hindi'];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!text.trim()) {
      setError('Please enter some text to translate');
      return;
    }

    if (sourceLanguage === targetLanguage) {
      setError('Source and target languages must differ');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await api.translateText({
        text,
        source_language: sourceLanguage,
        target_language: targetLanguage,
        subject
      });
      
      navigate(`/tasks/${response.task_id}`);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to start translation');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 via-white to-pink-50 p-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-purple-600 to-pink-600 bg-clip-text text-transparent">
            Multi-Language Translation
          </h1>
          <p className="text-gray-600">Translate educational content into Indian languages</p>
        </div>

        {error && (
          <div className="mb-6 p-4 bg-red-50 border-2 border-red-200 rounded-lg text-red-700">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="glass-card p-8">
          <div className="mb-6">
            <label htmlFor="text" className="block text-sm font-semibold text-gray-700 mb-2">
              Text to Translate *
            </label>
            <textarea
              id="text"
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={8}
              className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-purple-500 focus:ring-2 focus:ring-purple-200 transition-colors"
              placeholder="Enter the text you want to translate..."
              required
            />
            <p className="mt-2 text-sm text-gray-500">{text.length} characters</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
            <div>
              <label htmlFor="source" className="block text-sm font-semibold text-gray-700 mb-2">
                Source Language
              </label>
              <select
                id="source"
                value={sourceLanguage}
                onChange={(e) => setSourceLanguage(e.target.value)}
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-purple-500 focus:ring-2 focus:ring-purple-200 transition-colors"
              >
                {languages.map(lang => (
                  <option key={lang.code} value={lang.code}>{lang.name}</option>
                ))}
              </select>
            </div>

            <div>
              <label htmlFor="target" className="block text-sm font-semibold text-gray-700 mb-2">
                Target Language
              </label>
              <select
                id="target"
                value={targetLanguage}
                onChange={(e) => setTargetLanguage(e.target.value)}
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-purple-500 focus:ring-2 focus:ring-purple-200 transition-colors"
              >
                {languages.map(lang => (
                  <option key={lang.code} value={lang.code}>{lang.name}</option>
                ))}
              </select>
            </div>

            <div>
              <label htmlFor="subject" className="block text-sm font-semibold text-gray-700 mb-2">
                Subject
              </label>
              <select
                id="subject"
                value={subject}
                onChange={(e) => setSubject(e.target.value)}
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-purple-500 focus:ring-2 focus:ring-purple-200 transition-colors"
              >
                {subjects.map(subj => (
                  <option key={subj} value={subj}>{subj}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="flex gap-4">
            <button
              type="submit"
              disabled={isLoading || !text.trim()}
              className="btn-primary flex-1 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <>
                  <svg className="animate-spin h-5 w-5 mr-2 inline" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  Translating...
                </>
              ) : (
                '🌐 Translate Text'
              )}
            </button>
            <button
              type="button"
              onClick={() => navigate('/features')}
              className="btn-secondary"
            >
              Back to Features
            </button>
          </div>
        </form>

        <div className="mt-8 glass-card p-6">
          <h3 className="font-semibold text-gray-900 mb-3">🌐 Supported Languages:</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-2 text-gray-600">
            {languages.map(lang => (
              <div key={lang.code} className="flex items-center">
                <span className="text-purple-600 mr-2">→</span>
                {lang.name}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
