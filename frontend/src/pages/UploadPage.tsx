import { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../services/api';
import type { ProcessRequest } from '../types/api';
import { Button, Select, Progress, Badge } from '../components/ui';
import { FileDropzone } from '../components/molecules';

const CHUNK_SIZE = 5 * 1024 * 1024; // 5MB chunks

// Icons
const CheckIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
  </svg>
);

const ProcessIcon = () => (
  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
  </svg>
);

// Options data
const LANGUAGES = [
  { value: 'Hindi', label: 'Hindi' },
  { value: 'Tamil', label: 'Tamil' },
  { value: 'Telugu', label: 'Telugu' },
  { value: 'Bengali', label: 'Bengali' },
  { value: 'Marathi', label: 'Marathi' },
  { value: 'Gujarati', label: 'Gujarati' },
  { value: 'Kannada', label: 'Kannada' },
  { value: 'Malayalam', label: 'Malayalam' },
  { value: 'Punjabi', label: 'Punjabi' },
  { value: 'Odia', label: 'Odia' },
];

const SUBJECTS = [
  { value: 'Mathematics', label: 'Mathematics' },
  { value: 'Science', label: 'Science' },
  { value: 'Social Studies', label: 'Social Studies' },
  { value: 'English', label: 'English' },
  { value: 'Hindi', label: 'Hindi' },
  { value: 'Computer Science', label: 'Computer Science' },
];

const GRADES = [5, 6, 7, 8, 9, 10, 11, 12].map(g => ({
  value: String(g),
  label: `Grade ${g}`,
}));

const OUTPUT_FORMATS = [
  { value: 'text', label: 'Text Only' },
  { value: 'audio', label: 'Audio Only' },
  { value: 'both', label: 'Text + Audio' },
];

export default function UploadPage() {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [gradeLevel, setGradeLevel] = useState('8');
  const [subject, setSubject] = useState('Mathematics');
  const [targetLanguage, setTargetLanguage] = useState('Hindi');
  const [outputFormat, setOutputFormat] = useState<'text' | 'audio' | 'both'>('both');
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [filePath, setFilePath] = useState<string | null>(null);

  const handleFileDrop = useCallback((acceptedFiles: File[]) => {
    const selectedFile = acceptedFiles[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
      setFilePath(null);
      setUploadProgress(0);
    }
  }, []);

  const uploadFileInChunks = async (file: File): Promise<string> => {
    const totalChunks = Math.ceil(file.size / CHUNK_SIZE);
    const uploadId = `${Date.now()}_${file.name}`;

    for (let chunkIndex = 0; chunkIndex < totalChunks; chunkIndex++) {
      const start = chunkIndex * CHUNK_SIZE;
      const end = Math.min(start + CHUNK_SIZE, file.size);
      const chunk = file.slice(start, end);

      const response = await api.uploadChunk(
        chunk,
        file.name,
        uploadId,
        chunkIndex,
        totalChunks,
        undefined,
        (progress) => {
          const totalProgress = ((chunkIndex + progress / 100) / totalChunks) * 100;
          setUploadProgress(Math.round(totalProgress));
        }
      );

      if (response.status === 'complete') {
        return `data/uploads/${file.name}`;
      }
    }

    throw new Error('Upload incomplete');
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file');
      return;
    }

    setIsUploading(true);
    setError(null);
    setUploadProgress(0);

    try {
      let uploadedFilePath: string;

      if (file.size > 10 * 1024 * 1024) {
        // Use chunked upload for files > 10MB
        uploadedFilePath = await uploadFileInChunks(file);
      } else {
        // Regular upload for smaller files
        const response = await api.uploadFile(file, (progress) => {
          setUploadProgress(Math.round(progress));
        });
        uploadedFilePath = response.file_path;
      }

      setFilePath(uploadedFilePath);
      setUploadProgress(100);
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : 'Upload failed';
      setError(errorMessage);
    } finally {
      setIsUploading(false);
    }
  };

  const handleProcess = async () => {
    if (!filePath) {
      setError('Please upload a file first');
      return;
    }

    setIsProcessing(true);
    setError(null);

    try {
      const processRequest: ProcessRequest = {
        file_path: filePath,
        grade_level: Number(gradeLevel),
        subject,
        target_languages: [targetLanguage],
        output_format: outputFormat,
        validation_threshold: 0.8,
      };

      const response = await api.processContent(processRequest);
      
      // Navigate to task tracking page
      navigate(`/tasks/${response.task_id}`);
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : 'Processing failed';
      setError(errorMessage);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleRemoveFile = () => {
    setFile(null);
    setFilePath(null);
    setUploadProgress(0);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50/50 via-surface-50 to-secondary-50/50 dark:from-surface-950 dark:via-surface-900 dark:to-surface-950 p-4 sm:p-8">
      <div className="max-w-3xl mx-auto space-y-6">
        {/* Header */}
        <div className="text-center sm:text-left">
          <h1 className="text-3xl sm:text-4xl font-bold mb-2 bg-gradient-to-r from-primary-600 to-secondary-600 bg-clip-text text-transparent">
            Upload Content
          </h1>
          <p className="text-muted-600 dark:text-muted-400">
            Upload educational content for AI-powered simplification and translation
          </p>
        </div>

        {/* Step 1: File Upload */}
        <section className="glass-card p-6 sm:p-8">
          <div className="flex items-center gap-3 mb-6">
            <div className="flex items-center justify-center w-8 h-8 rounded-full bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400 font-semibold text-sm">
              1
            </div>
            <h2 className="text-xl font-semibold text-surface-900 dark:text-surface-100">
              Select PDF File
            </h2>
            {filePath && (
              <Badge variant="success" className="ml-auto">
                <CheckIcon />
                <span className="ml-1">Uploaded</span>
              </Badge>
            )}
          </div>

          <FileDropzone
            onDrop={handleFileDrop}
            accept={{ 'application/pdf': ['.pdf'] }}
            maxSize={100 * 1024 * 1024} // 100MB
            disabled={isUploading || !!filePath}
          />

          {file && !filePath && (
            <div className="mt-4 space-y-4">
              <div className="flex items-center justify-between p-4 bg-surface-100 dark:bg-surface-800 rounded-lg">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-error-100 dark:bg-error-900/30 flex items-center justify-center">
                    <svg className="w-5 h-5 text-error-600" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M4 4a2 2 0 012-2h4.586A2 2 0 0112 2.586L15.414 6A2 2 0 0116 7.414V16a2 2 0 01-2 2H6a2 2 0 01-2-2V4z" clipRule="evenodd" />
                    </svg>
                  </div>
                  <div>
                    <p className="text-sm font-medium text-surface-900 dark:text-surface-100">
                      {file.name}
                    </p>
                    <p className="text-xs text-muted-500">
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                </div>
                <Button variant="ghost" size="sm" onClick={handleRemoveFile}>
                  Remove
                </Button>
              </div>

              {isUploading && (
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-600 dark:text-muted-400">Uploading...</span>
                    <span className="font-medium text-primary-600">{uploadProgress}%</span>
                  </div>
                  <Progress value={uploadProgress} variant="primary" animated />
                </div>
              )}

              {!isUploading && (
                <Button
                  onClick={handleUpload}
                  isLoading={isUploading}
                  className="w-full"
                >
                  Upload File
                </Button>
              )}
            </div>
          )}

          {filePath && (
            <div className="mt-4 p-4 bg-success-50 dark:bg-success-900/20 border border-success-200 dark:border-success-800 rounded-lg">
              <div className="flex items-center gap-2 text-success-700 dark:text-success-400">
                <CheckIcon />
                <span className="font-medium">File uploaded successfully!</span>
              </div>
              <p className="mt-1 text-sm text-success-600 dark:text-success-500">
                Continue to configure processing options below.
              </p>
            </div>
          )}
        </section>

        {/* Step 2: Processing Configuration */}
        <section className={`glass-card p-6 sm:p-8 transition-opacity duration-300 ${!filePath ? 'opacity-50 pointer-events-none' : ''}`}>
          <div className="flex items-center gap-3 mb-6">
            <div className={`flex items-center justify-center w-8 h-8 rounded-full font-semibold text-sm ${filePath ? 'bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400' : 'bg-surface-200 dark:bg-surface-700 text-surface-500'}`}>
              2
            </div>
            <h2 className="text-xl font-semibold text-surface-900 dark:text-surface-100">
              Processing Options
            </h2>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 sm:gap-6">
            <Select
              label="Grade Level"
              options={GRADES}
              value={gradeLevel}
              onChange={(e) => setGradeLevel(e.target.value)}
            />

            <Select
              label="Subject"
              options={SUBJECTS}
              value={subject}
              onChange={(e) => setSubject(e.target.value)}
            />

            <Select
              label="Target Language"
              options={LANGUAGES}
              value={targetLanguage}
              onChange={(e) => setTargetLanguage(e.target.value)}
            />

            <Select
              label="Output Format"
              options={OUTPUT_FORMATS}
              value={outputFormat}
              onChange={(e) => setOutputFormat(e.target.value as 'text' | 'audio' | 'both')}
            />
          </div>

          {/* Configuration Summary */}
          <div className="mt-6 p-4 bg-surface-100 dark:bg-surface-800 rounded-lg">
            <h3 className="text-sm font-medium text-surface-700 dark:text-surface-300 mb-3">
              Summary
            </h3>
            <div className="flex flex-wrap gap-2">
              <Badge variant="info">Grade {gradeLevel}</Badge>
              <Badge variant="info">{subject}</Badge>
              <Badge variant="info">{targetLanguage}</Badge>
              <Badge variant="info">
                {outputFormat === 'both' ? 'Text + Audio' : outputFormat === 'text' ? 'Text Only' : 'Audio Only'}
              </Badge>
            </div>
          </div>
        </section>

        {/* Error Display */}
        {error && (
          <div className="p-4 bg-error-50 dark:bg-error-900/20 border border-error-200 dark:border-error-800 rounded-lg">
            <div className="flex items-start gap-3">
              <svg className="w-5 h-5 text-error-500 flex-shrink-0 mt-0.5" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
              <div>
                <p className="font-medium text-error-800 dark:text-error-300">
                  Something went wrong
                </p>
                <p className="text-sm text-error-600 dark:text-error-400 mt-1">
                  {error}
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Submit Button */}
        <Button
          onClick={handleProcess}
          disabled={!filePath || isProcessing}
          isLoading={isProcessing}
          size="lg"
          className="w-full"
          leftIcon={<ProcessIcon />}
        >
          {isProcessing ? 'Starting Processing...' : 'Process Content'}
        </Button>

        {/* Info Footer */}
        <p className="text-center text-sm text-muted-500 dark:text-muted-400">
          Processing typically takes 1-3 minutes depending on document length.
          You'll be redirected to track progress.
        </p>
      </div>
    </div>
  );
}
