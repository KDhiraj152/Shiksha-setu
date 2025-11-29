import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ArrowRight, ArrowLeft } from 'lucide-react';
import { PageHeader } from '../../components/patterns/PageHeader';
import { Button } from '../../components/ui/Button/Button';
import { 
  FileUpload, 
  ProcessingOptions, 
  ProcessingStatus 
} from '../../components/features/playground';
import { useUploadFile, useProcessContent, useTaskStatus } from '../../hooks';
import { useToast } from '../../components/ui/Toast';
import { cn } from '../../lib/cn';

type Step = 'upload' | 'configure' | 'processing' | 'complete';

interface ProcessingConfig {
  gradeLevel: string;
  subject: string;
  targetLanguages: string[];
  outputFormat: 'text' | 'audio' | 'both';
  validationThreshold: number;
}

/**
 * Playground Page - Content processing workflow
 * Step 1: Upload file
 * Step 2: Configure processing options  
 * Step 3: Process and view status
 * Step 4: View results
 */
export function PlaygroundPage() {
  const toast = useToast();
  const uploadFile = useUploadFile();
  const processContent = useProcessContent();
  
  const [step, setStep] = useState<Step>('upload');
  const [uploadedFile, setUploadedFile] = useState<{
    filePath: string;
    contentId: string;
  } | null>(null);
  const [taskId, setTaskId] = useState<string | null>(null);
  const [config, setConfig] = useState<ProcessingConfig>({
    gradeLevel: '',
    subject: '',
    targetLanguages: [],
    outputFormat: 'both',
    validationThreshold: 0.8,
  });

  // Poll task status when we have a task
  const { data: taskStatus } = useTaskStatus(taskId, { enabled: taskId != null });

  // Handle file upload
  const handleUpload = async (file: File) => {
    try {
      const result = await uploadFile.mutateAsync({ file });
      setUploadedFile({
        filePath: result.file_path,
        contentId: result.content_id,
      });
      return result;
    } catch (error: unknown) {
      const err = error as Error;
      toast.error('Upload failed', err?.message || 'Please try again');
      throw error;
    }
  };

  const handleFileReady = (filePath: string, contentId: string) => {
    setUploadedFile({ filePath, contentId });
    toast.success('File uploaded', 'Ready to configure processing options');
  };

  // Handle processing
  const handleProcess = async () => {
    if (!uploadedFile) return;

    try {
      const result = await processContent.mutateAsync({
        file_path: uploadedFile.filePath,
        grade_level: parseInt(config.gradeLevel, 10) || 6,
        subject: config.subject,
        target_languages: config.targetLanguages,
        output_format: config.outputFormat,
        validation_threshold: config.validationThreshold,
      });
      
      setTaskId(result.task_id);
      setStep('processing');
      toast.info('Processing started', 'Your content is being processed');
    } catch (error: unknown) {
      const err = error as Error;
      toast.error('Processing failed', err?.message || 'Please try again');
    }
  };

  // Reset to start
  const handleReset = () => {
    setStep('upload');
    setUploadedFile(null);
    setTaskId(null);
    setConfig({
      gradeLevel: '',
      subject: '',
      targetLanguages: [],
      outputFormat: 'both',
      validationThreshold: 0.8,
    });
  };

  // Update step based on task status
  useEffect(() => {
    if (taskStatus?.state === 'SUCCESS' && step === 'processing') {
      setStep('complete');
    }
  }, [taskStatus?.state, step]);

  // Map task state to status string
  const getTaskStatusString = (): 'pending' | 'processing' | 'completed' | 'failed' => {
    if (!taskStatus) return 'pending';
    switch (taskStatus.state) {
      case 'SUCCESS': return 'completed';
      case 'FAILURE': return 'failed';
      case 'STARTED':
      case 'PROCESSING': return 'processing';
      default: return 'pending';
    }
  };

  const steps = [
    { id: 'upload', label: 'Upload', number: 1 },
    { id: 'configure', label: 'Configure', number: 2 },
    { id: 'processing', label: 'Process', number: 3 },
  ];

  const currentStepIndex = steps.findIndex(s => s.id === step);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-6"
    >
      {/* Page Header */}
      <PageHeader
        title="Content Playground"
        description="Upload, process, and transform educational content with AI"
        actions={
          step !== 'upload' && (
            <Button variant="ghost" onClick={handleReset}>
              Start Over
            </Button>
          )
        }
      />

      {/* Step Indicator */}
      <div className="flex items-center justify-center gap-2">
        {steps.map((s, index) => (
          <div key={s.id} className="flex items-center">
            <div
              className={cn(
                'w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium transition-all',
                currentStepIndex > index && 'bg-primary-500 text-white',
                currentStepIndex === index && 'bg-primary-500 text-white ring-4 ring-primary-500/20',
                currentStepIndex < index && 'bg-muted text-muted-foreground'
              )}
            >
              {s.number}
            </div>
            <span
              className={cn(
                'ml-2 text-sm font-medium',
                currentStepIndex >= index ? 'text-foreground' : 'text-muted-foreground'
              )}
            >
              {s.label}
            </span>
            {index < steps.length - 1 && (
              <div
                className={cn(
                  'w-12 h-0.5 mx-4 transition-colors',
                  currentStepIndex > index ? 'bg-primary-500' : 'bg-border'
                )}
              />
            )}
          </div>
        ))}
      </div>

      {/* Content Area */}
      <div className="max-w-2xl mx-auto">
        <AnimatePresence mode="wait">
          {/* Step 1: Upload */}
          {step === 'upload' && (
            <motion.div
              key="upload"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="card p-6"
            >
              <h2 className="text-lg font-semibold text-foreground mb-4">
                Upload Your Content
              </h2>
              
              <FileUpload
                onUpload={handleUpload}
                onFileReady={handleFileReady}
              />

              {uploadedFile && (
                <div className="mt-6 flex justify-end">
                  <Button
                    variant="gradient"
                    onClick={() => setStep('configure')}
                    rightIcon={<ArrowRight className="w-4 h-4" />}
                  >
                    Continue to Configure
                  </Button>
                </div>
              )}
            </motion.div>
          )}

          {/* Step 2: Configure */}
          {step === 'configure' && (
            <motion.div
              key="configure"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="card p-6"
            >
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-lg font-semibold text-foreground">
                  Configure Processing
                </h2>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setStep('upload')}
                  leftIcon={<ArrowLeft className="w-4 h-4" />}
                >
                  Back
                </Button>
              </div>

              <ProcessingOptions
                config={config}
                onChange={setConfig}
                onProcess={handleProcess}
                isProcessing={processContent.isPending}
                isDisabled={!uploadedFile}
              />
            </motion.div>
          )}

          {/* Step 3: Processing */}
          {(step === 'processing' || step === 'complete') && taskId && (
            <motion.div
              key="processing"
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="card p-6"
            >
              <ProcessingStatus
                taskId={taskId}
                status={getTaskStatusString()}
                progress={taskStatus?.progress || 0}
                steps={[
                  {
                    id: 'extract',
                    label: 'Text Extraction',
                    status: getStepStatus(taskStatus?.progress || 0, 0, 30),
                    progress: Math.min(100, Math.max(0, (taskStatus?.progress || 0) * 3.33)),
                  },
                  {
                    id: 'simplify',
                    label: 'Content Simplification',
                    status: getStepStatus(taskStatus?.progress || 0, 30, 60),
                    progress: Math.min(100, Math.max(0, ((taskStatus?.progress || 0) - 30) * 3.33)),
                  },
                  {
                    id: 'translate',
                    label: 'Translation',
                    status: getStepStatus(taskStatus?.progress || 0, 60, 90),
                    progress: Math.min(100, Math.max(0, ((taskStatus?.progress || 0) - 60) * 3.33)),
                  },
                  {
                    id: 'tts',
                    label: 'Audio Generation',
                    status: getStepStatus(taskStatus?.progress || 0, 90, 100),
                    progress: Math.min(100, Math.max(0, ((taskStatus?.progress || 0) - 90) * 10)),
                  },
                ]}
                result={taskStatus?.result}
                onRetry={handleReset}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
}

function getStepStatus(progress: number, start: number, end: number): 'pending' | 'processing' | 'completed' | 'failed' {
  if (progress >= end) return 'completed';
  if (progress >= start) return 'processing';
  return 'pending';
}

export default PlaygroundPage;
