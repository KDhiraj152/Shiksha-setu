import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import api from '../services/api';

const SUBJECTS = ['General', 'Science', 'Mathematics', 'Social Studies', 'English', 'Hindi'];
const GRADES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
const MAX_TEXT_LENGTH = 5000;

export default function SimplifyPage() {
  const navigate = useNavigate();
  const [text, setText] = useState('');
  const [gradeLevel, setGradeLevel] = useState(6);
  const [subject, setSubject] = useState('General');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    const trimmedText = text.trim();
    
    if (!trimmedText) {
      setError('Please enter some text to simplify');
      return;
    }

    if (trimmedText.length > MAX_TEXT_LENGTH) {
      setError(`Text exceeds maximum length of ${MAX_TEXT_LENGTH} characters. Current length: ${trimmedText.length}`);
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await api.simplifyText({
        text,
        grade_level: gradeLevel,
        subject
      });
      
      navigate(`/tasks/${response.task_id}`);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to start simplification');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 p-8">
      <div className="max-w-4xl mx-auto">
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2 bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent">
            Text Simplification
          </h1>
          <p className="text-gray-600">Make educational content easier to understand for students</p>
        </div>

        {error && (
          <div className="mb-6 p-4 bg-red-50 border-2 border-red-200 rounded-lg text-red-700">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="glass-card p-8">
          <div className="mb-6">
            <label htmlFor="text" className="block text-sm font-semibold text-gray-700 mb-2">
              Text to Simplify *
            </label>
            <textarea
              id="text"
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={8}
              className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-primary-500 focus:ring-2 focus:ring-primary-200 transition-colors"
              placeholder="Enter the text you want to simplify..."
              required
              maxLength={MAX_TEXT_LENGTH}
            />
            <p className={`mt-2 text-sm ${text.length > MAX_TEXT_LENGTH * 0.9 ? 'text-orange-600 font-semibold' : 'text-gray-500'}`}>
              {text.length} / {MAX_TEXT_LENGTH} characters
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <div>
              <label htmlFor="grade" className="block text-sm font-semibold text-gray-700 mb-2">
                Target Grade Level
              </label>
              <select
                id="grade"
                value={gradeLevel}
                onChange={(e) => setGradeLevel(Number.parseInt(e.target.value))}
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-primary-500 focus:ring-2 focus:ring-primary-200 transition-colors"
              >
                {GRADES.map(grade => (
                  <option key={grade} value={grade}>Grade {grade}</option>
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
                className="w-full px-4 py-3 border-2 border-gray-200 rounded-lg focus:border-primary-500 focus:ring-2 focus:ring-primary-200 transition-colors"
              >
                {SUBJECTS.map(subj => (
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
                  Simplifying...
                </>
              ) : (
                '📝 Simplify Text'
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
          <h3 className="font-semibold text-gray-900 mb-3">💡 Tips for Best Results:</h3>
          <ul className="space-y-2 text-gray-600">
            <li>• Keep text focused on a single topic or concept</li>
            <li>• Select the appropriate grade level for your target audience</li>
            <li>• Choose the correct subject for domain-specific simplification</li>
            <li>• Longer texts may take more time to process</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
