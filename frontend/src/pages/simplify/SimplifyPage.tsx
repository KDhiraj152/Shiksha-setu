/**
 * Simplify Page
 * 
 * AI-powered text simplification tool with grade-level targeting
 */

import { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { 
  Sparkles, 
  Copy, 
  Check,
  RotateCcw,
  FileText,
  ArrowRight,
  Loader2,
  Info,
  BookOpen
} from 'lucide-react';
import { api } from '../../services/api';
import { Button } from '../../components/ui/Button/Button';
import { Textarea } from '../../components/ui/Textarea/Textarea';
import { Badge } from '../../components/ui/Badge/Badge';
import { Progress } from '../../components/ui/Progress/Progress';
import { pageVariants, staggerItem } from '../../lib/animations';

// Grade configurations
const GRADES = [
  { value: 1, label: 'Grade 1', age: '6-7 yrs', description: 'Very simple sentences' },
  { value: 2, label: 'Grade 2', age: '7-8 yrs', description: 'Simple vocabulary' },
  { value: 3, label: 'Grade 3', age: '8-9 yrs', description: 'Basic concepts' },
  { value: 4, label: 'Grade 4', age: '9-10 yrs', description: 'Clear explanations' },
  { value: 5, label: 'Grade 5', age: '10-11 yrs', description: 'Standard simplification' },
  { value: 6, label: 'Grade 6', age: '11-12 yrs', description: 'Moderate complexity' },
  { value: 7, label: 'Grade 7', age: '12-13 yrs', description: 'Some technical terms' },
  { value: 8, label: 'Grade 8', age: '13-14 yrs', description: 'Accessible complexity' },
  { value: 9, label: 'Grade 9', age: '14-15 yrs', description: 'Near-original' },
  { value: 10, label: 'Grade 10', age: '15-16 yrs', description: 'Minimal changes' },
];

const SUBJECTS = [
  'General',
  'Science',
  'Mathematics',
  'Social Studies',
  'English',
  'Environmental Science',
  'History',
  'Geography',
  'Civics',
  'Economics',
];

export function SimplifyPage() {
  // State
  const [inputText, setInputText] = useState('');
  const [outputText, setOutputText] = useState('');
  const [gradeLevel, setGradeLevel] = useState(5);
  const [subject, setSubject] = useState('General');
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [copied, setCopied] = useState(false);
  const [, setTaskId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState<{
    originalWords: number;
    simplifiedWords: number;
    readabilityImprovement: number;
  } | null>(null);

  // Handle simplification
  const handleSimplify = useCallback(async () => {
    if (!inputText.trim() || isProcessing) return;

    setIsProcessing(true);
    setError(null);
    setProgress(10);
    setOutputText('');
    setStats(null);

    try {
      // Start simplification task
      const result = await api.simplifyText({
        text: inputText,
        grade_level: gradeLevel,
        subject,
      });

      setTaskId(result.task_id);
      setProgress(30);

      // Poll for completion
      const pollInterval = setInterval(async () => {
        try {
          const status = await api.getTaskStatus(result.task_id);
          
          if (status.state === 'PENDING' || status.state === 'STARTED' || status.state === 'PROCESSING') {
            setProgress(Math.min(90, (status.progress || 50)));
          } else if (status.state === 'SUCCESS') {
            clearInterval(pollInterval);
            const simplified = status.result?.simplified_text || '';
            setOutputText(simplified);
            setProgress(100);
            
            // Calculate stats
            const originalWords = inputText.split(/\s+/).length;
            const simplifiedWords = simplified.split(/\s+/).length;
            setStats({
              originalWords,
              simplifiedWords,
              readabilityImprovement: Math.round((1 - simplifiedWords / originalWords) * 100 + 15),
            });
            
            setTimeout(() => setIsProcessing(false), 500);
          } else if (status.state === 'FAILURE') {
            clearInterval(pollInterval);
            setError(status.error || 'Simplification failed');
            setIsProcessing(false);
          }
        } catch (e) {
          // Keep polling
        }
      }, 1000);

      // Timeout after 2 minutes
      setTimeout(() => {
        clearInterval(pollInterval);
        if (isProcessing) {
          setError('Request timed out. Please try again.');
          setIsProcessing(false);
        }
      }, 120000);

    } catch (e: any) {
      setError(e.message || 'Failed to simplify text');
      setIsProcessing(false);
    }
  }, [inputText, gradeLevel, subject, isProcessing]);

  // Copy to clipboard
  const handleCopy = useCallback(() => {
    navigator.clipboard.writeText(outputText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [outputText]);

  // Reset
  const handleReset = () => {
    setInputText('');
    setOutputText('');
    setStats(null);
    setError(null);
    setProgress(0);
  };

  const selectedGrade = GRADES.find(g => g.value === gradeLevel);

  return (
    <motion.div 
      variants={pageVariants}
      initial="initial"
      animate="enter"
      className="max-w-6xl mx-auto space-y-6"
    >
      {/* Header */}
      <motion.div variants={staggerItem} className="text-center">
        <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-primary-500 to-secondary-500 mb-4">
          <Sparkles className="w-8 h-8 text-white" />
        </div>
        <h1 className="text-3xl font-bold text-foreground mb-2">Text Simplification</h1>
        <p className="text-muted-foreground max-w-2xl mx-auto">
          Transform complex educational content into grade-appropriate text using AI. 
          Perfect for making NCERT content accessible to all students.
        </p>
      </motion.div>

      {/* Settings */}
      <motion.div variants={staggerItem} className="bg-card rounded-xl border border-border p-6">
        <div className="flex flex-col md:flex-row gap-6">
          {/* Grade Selection */}
          <div className="flex-1">
            <label className="block text-sm font-medium text-foreground mb-3">
              Target Grade Level
            </label>
            <div className="flex flex-wrap gap-2">
              {GRADES.map((grade) => (
                <button
                  key={grade.value}
                  onClick={() => setGradeLevel(grade.value)}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                    gradeLevel === grade.value
                      ? 'bg-primary-500 text-white shadow-lg shadow-primary-500/25'
                      : 'bg-muted text-muted-foreground hover:bg-muted/80'
                  }`}
                >
                  {grade.value}
                </button>
              ))}
            </div>
            {selectedGrade && (
              <div className="mt-3 flex items-center gap-2 text-sm text-muted-foreground">
                <Info className="w-4 h-4" />
                <span>{selectedGrade.label} ({selectedGrade.age}): {selectedGrade.description}</span>
              </div>
            )}
          </div>

          {/* Subject */}
          <div className="md:w-64">
            <label className="block text-sm font-medium text-foreground mb-3">
              Subject Area
            </label>
            <select
              value={subject}
              onChange={(e) => setSubject(e.target.value)}
              className="w-full px-4 py-2.5 rounded-lg border border-border bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-primary-500"
            >
              {SUBJECTS.map((s) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
          </div>
        </div>
      </motion.div>

      {/* Editor */}
      <motion.div variants={staggerItem} className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Input */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-foreground flex items-center gap-2">
              <FileText className="w-4 h-4" />
              Original Text
            </label>
            <span className="text-xs text-muted-foreground">
              {inputText.split(/\s+/).filter(Boolean).length} words
            </span>
          </div>
          <Textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="Paste or type the text you want to simplify..."
            className="min-h-[400px] resize-none"
          />
        </div>

        {/* Output */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <label className="text-sm font-medium text-foreground flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-primary-500" />
              Simplified Text
              {stats && (
                <Badge variant="success" className="ml-2">
                  {stats.readabilityImprovement}% easier
                </Badge>
              )}
            </label>
            {outputText && (
              <Button variant="ghost" size="sm" onClick={handleCopy}>
                {copied ? <Check className="w-4 h-4 mr-1" /> : <Copy className="w-4 h-4 mr-1" />}
                {copied ? 'Copied!' : 'Copy'}
              </Button>
            )}
          </div>
          <div className="relative">
            <Textarea
              value={outputText}
              readOnly
              placeholder="Simplified text will appear here..."
              className="min-h-[400px] resize-none bg-muted/30"
            />
            {isProcessing && (
              <div className="absolute inset-0 flex items-center justify-center bg-background/80 rounded-lg">
                <div className="text-center">
                  <Loader2 className="w-8 h-8 animate-spin text-primary-500 mx-auto mb-3" />
                  <p className="text-sm text-muted-foreground mb-2">Simplifying content...</p>
                  <Progress value={progress} className="w-48" />
                </div>
              </div>
            )}
          </div>
        </div>
      </motion.div>

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

      {/* Actions */}
      <motion.div variants={staggerItem} className="flex items-center justify-center gap-4">
        <Button variant="outline" onClick={handleReset} disabled={isProcessing}>
          <RotateCcw className="w-4 h-4 mr-2" />
          Reset
        </Button>
        <Button 
          variant="gradient" 
          size="lg"
          onClick={handleSimplify}
          disabled={!inputText.trim() || isProcessing}
          isLoading={isProcessing}
        >
          <Sparkles className="w-5 h-5 mr-2" />
          Simplify for Grade {gradeLevel}
          <ArrowRight className="w-5 h-5 ml-2" />
        </Button>
      </motion.div>

      {/* Stats */}
      {stats && (
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="grid grid-cols-3 gap-4"
        >
          <div className="bg-card rounded-xl border border-border p-4 text-center">
            <div className="text-2xl font-bold text-foreground">{stats.originalWords}</div>
            <div className="text-sm text-muted-foreground">Original Words</div>
          </div>
          <div className="bg-card rounded-xl border border-border p-4 text-center">
            <div className="text-2xl font-bold text-primary-500">{stats.simplifiedWords}</div>
            <div className="text-sm text-muted-foreground">Simplified Words</div>
          </div>
          <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-xl border border-green-200 p-4 text-center">
            <div className="text-2xl font-bold text-green-600">{stats.readabilityImprovement}%</div>
            <div className="text-sm text-green-700">Readability Improvement</div>
          </div>
        </motion.div>
      )}

      {/* Tips */}
      <motion.div variants={staggerItem} className="bg-muted/30 rounded-xl p-6">
        <h3 className="font-semibold text-foreground mb-3 flex items-center gap-2">
          <BookOpen className="w-5 h-5 text-primary-500" />
          Tips for Best Results
        </h3>
        <ul className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm text-muted-foreground">
          <li className="flex items-start gap-2">
            <span className="text-primary-500">•</span>
            Choose the appropriate grade level for your target audience
          </li>
          <li className="flex items-start gap-2">
            <span className="text-primary-500">•</span>
            Select the correct subject for domain-specific terminology
          </li>
          <li className="flex items-start gap-2">
            <span className="text-primary-500">•</span>
            Longer texts may take more time to process
          </li>
          <li className="flex items-start gap-2">
            <span className="text-primary-500">•</span>
            Review output for accuracy before using with students
          </li>
        </ul>
      </motion.div>
    </motion.div>
  );
}

export default SimplifyPage;
