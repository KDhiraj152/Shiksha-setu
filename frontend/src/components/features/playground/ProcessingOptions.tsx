import { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  GraduationCap, 
  BookOpen, 
  Languages, 
  Settings2,
  Sparkles,
  AlertCircle
} from 'lucide-react';
import { Button } from '../../ui/Button/Button';
import { Badge } from '../../ui/Badge';
import { cn } from '../../../lib/cn';
import { GRADES, SUBJECTS, LANGUAGES } from '../../../lib/constants';

interface ProcessingConfig {
  gradeLevel: string;
  subject: string;
  targetLanguages: string[];
  outputFormat: 'text' | 'audio' | 'both';
  validationThreshold: number;
}

interface ProcessingOptionsProps {
  config: ProcessingConfig;
  onChange: (config: ProcessingConfig) => void;
  onProcess: () => void;
  isProcessing?: boolean;
  isDisabled?: boolean;
}

export function ProcessingOptions({
  config,
  onChange,
  onProcess,
  isProcessing,
  isDisabled,
}: ProcessingOptionsProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  const updateConfig = <K extends keyof ProcessingConfig>(
    key: K,
    value: ProcessingConfig[K]
  ) => {
    onChange({ ...config, [key]: value });
  };

  const toggleLanguage = (langCode: string) => {
    const current = config.targetLanguages;
    const updated = current.includes(langCode)
      ? current.filter((l) => l !== langCode)
      : [...current, langCode];
    updateConfig('targetLanguages', updated);
  };

  const isValid = config.gradeLevel && config.subject && config.targetLanguages.length > 0;

  return (
    <div className="space-y-6">
      {/* Grade Level */}
      <div>
        <label className="flex items-center gap-2 text-sm font-medium text-foreground mb-3">
          <GraduationCap className="w-4 h-4 text-primary-500" />
          Grade Level
        </label>
        <div className="grid grid-cols-4 gap-2">
          {GRADES.map((grade) => (
            <button
              key={grade.value}
              onClick={() => updateConfig('gradeLevel', grade.value)}
              className={cn(
                'px-3 py-2 rounded-lg text-sm font-medium transition-all',
                config.gradeLevel === grade.value
                  ? 'bg-primary-500 text-white shadow-md shadow-primary-500/25'
                  : 'bg-muted hover:bg-muted/80 text-foreground'
              )}
            >
              {grade.label}
            </button>
          ))}
        </div>
      </div>

      {/* Subject */}
      <div>
        <label className="flex items-center gap-2 text-sm font-medium text-foreground mb-3">
          <BookOpen className="w-4 h-4 text-primary-500" />
          Subject
        </label>
        <div className="grid grid-cols-3 gap-2">
          {SUBJECTS.map((subject) => (
            <button
              key={subject.value}
              onClick={() => updateConfig('subject', subject.value)}
              className={cn(
                'px-3 py-2 rounded-lg text-sm font-medium transition-all text-left',
                config.subject === subject.value
                  ? 'bg-primary-500 text-white shadow-md shadow-primary-500/25'
                  : 'bg-muted hover:bg-muted/80 text-foreground'
              )}
            >
              <span className="mr-1.5">{subject.icon}</span>
              {subject.label}
            </button>
          ))}
        </div>
      </div>

      {/* Target Languages */}
      <div>
        <label className="flex items-center gap-2 text-sm font-medium text-foreground mb-3">
          <Languages className="w-4 h-4 text-primary-500" />
          Target Languages
          <Badge variant="secondary" size="sm">
            {config.targetLanguages.length} selected
          </Badge>
        </label>
        <div className="flex flex-wrap gap-2">
          {LANGUAGES.map((lang) => {
            const isSelected = config.targetLanguages.includes(lang.code);
            return (
              <button
                key={lang.code}
                onClick={() => toggleLanguage(lang.code)}
                className={cn(
                  'px-3 py-1.5 rounded-full text-sm font-medium transition-all border',
                  isSelected
                    ? 'bg-primary-500 text-white border-primary-500 shadow-md shadow-primary-500/25'
                    : 'bg-background border-border hover:border-primary-300 text-foreground'
                )}
              >
                <span className="mr-1">{lang.flag}</span>
                {lang.label}
              </button>
            );
          })}
        </div>
      </div>

      {/* Advanced Options */}
      <div>
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
        >
          <Settings2 className="w-4 h-4" />
          Advanced Options
          <motion.span
            animate={{ rotate: showAdvanced ? 180 : 0 }}
            className="ml-auto"
          >
            ▼
          </motion.span>
        </button>

        {showAdvanced && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-4 space-y-4 pt-4 border-t border-border"
          >
            {/* Output Format */}
            <div>
              <label className="text-sm text-muted-foreground mb-2 block">
                Output Format
              </label>
              <div className="flex gap-2">
                {[
                  { value: 'text', label: 'Text Only' },
                  { value: 'audio', label: 'Audio Only' },
                  { value: 'both', label: 'Text + Audio' },
                ].map((option) => (
                  <button
                    key={option.value}
                    onClick={() =>
                      updateConfig('outputFormat', option.value as ProcessingConfig['outputFormat'])
                    }
                    className={cn(
                      'px-4 py-2 rounded-lg text-sm transition-all flex-1',
                      config.outputFormat === option.value
                        ? 'bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 border border-primary-200 dark:border-primary-800'
                        : 'bg-muted text-muted-foreground hover:text-foreground'
                    )}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Validation Threshold */}
            <div>
              <label className="text-sm text-muted-foreground mb-2 block">
                Validation Threshold: {(config.validationThreshold * 100).toFixed(0)}%
              </label>
              <input
                type="range"
                min="0.5"
                max="1"
                step="0.05"
                value={config.validationThreshold}
                onChange={(e) =>
                  updateConfig('validationThreshold', parseFloat(e.target.value))
                }
                className="w-full accent-primary-500"
              />
              <div className="flex justify-between text-xs text-muted-foreground mt-1">
                <span>Lenient (50%)</span>
                <span>Strict (100%)</span>
              </div>
            </div>
          </motion.div>
        )}
      </div>

      {/* Validation message */}
      {!isValid && (
        <div className="flex items-center gap-2 p-3 rounded-lg bg-amber-50 dark:bg-amber-950/30 text-amber-700 dark:text-amber-300 text-sm">
          <AlertCircle className="w-4 h-4 shrink-0" />
          Please select grade level, subject, and at least one target language
        </div>
      )}

      {/* Process Button */}
      <Button
        variant="gradient"
        size="lg"
        className="w-full"
        onClick={onProcess}
        isLoading={isProcessing}
        disabled={isDisabled || !isValid}
        leftIcon={<Sparkles className="w-4 h-4" />}
      >
        Process Content
      </Button>
    </div>
  );
}

export default ProcessingOptions;
