import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Upload, 
  FileText, 
  X, 
  CheckCircle, 
  AlertCircle,
  File,
  Image,
  FileAudio
} from 'lucide-react';
import { Button } from '../../ui/Button/Button';
import { Progress } from '../../ui/Progress';
import { cn } from '../../../lib/cn';
import { formatFileSize } from '../../../lib/formatters';

interface UploadedFile {
  file: File;
  id: string;
  status: 'uploading' | 'uploaded' | 'error';
  progress: number;
  error?: string;
  filePath?: string;
  contentId?: string;
}

interface FileUploadProps {
  onUpload: (file: File) => Promise<{ file_path: string; content_id: string }>;
  onFileReady?: (filePath: string, contentId: string) => void;
  accept?: Record<string, string[]>;
  maxSize?: number;
  maxFiles?: number;
}

const fileTypeIcons: Record<string, React.ElementType> = {
  'application/pdf': FileText,
  'text/plain': FileText,
  'image/': Image,
  'audio/': FileAudio,
};

const getFileIcon = (type: string): React.ElementType => {
  for (const [key, Icon] of Object.entries(fileTypeIcons)) {
    if (type.startsWith(key)) return Icon;
  }
  return File;
};

export function FileUpload({
  onUpload,
  onFileReady,
  accept = {
    'application/pdf': ['.pdf'],
    'text/plain': ['.txt'],
    'application/msword': ['.doc'],
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
  },
  maxSize = 50 * 1024 * 1024, // 50MB
  maxFiles = 1,
}: FileUploadProps) {
  const [files, setFiles] = useState<UploadedFile[]>([]);

  const handleUpload = async (file: File) => {
    const id = Math.random().toString(36).slice(2);
    
    setFiles((prev) => [
      ...prev,
      { file, id, status: 'uploading', progress: 0 },
    ]);

    try {
      // Simulate progress updates
      const progressInterval = setInterval(() => {
        setFiles((prev) =>
          prev.map((f) =>
            f.id === id && f.progress < 90
              ? { ...f, progress: f.progress + 10 }
              : f
          )
        );
      }, 200);

      const result = await onUpload(file);

      clearInterval(progressInterval);

      setFiles((prev) =>
        prev.map((f) =>
          f.id === id
            ? {
                ...f,
                status: 'uploaded',
                progress: 100,
                filePath: result.file_path,
                contentId: result.content_id,
              }
            : f
        )
      );

      onFileReady?.(result.file_path, result.content_id);
    } catch (error: any) {
      setFiles((prev) =>
        prev.map((f) =>
          f.id === id
            ? {
                ...f,
                status: 'error',
                error: error?.message || 'Upload failed',
              }
            : f
        )
      );
    }
  };

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      acceptedFiles.forEach((file) => handleUpload(file));
    },
    [handleUpload]
  );

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept,
    maxSize,
    maxFiles: maxFiles - files.length,
    disabled: files.length >= maxFiles,
  });

  const removeFile = (id: string) => {
    setFiles((prev) => prev.filter((f) => f.id !== id));
  };

  return (
    <div className="space-y-4">
      {/* Dropzone */}
      <div
        {...getRootProps()}
        className={cn(
          'relative rounded-2xl border-2 border-dashed transition-all duration-300 cursor-pointer',
          'hover:border-primary-400 hover:bg-primary-50/50 dark:hover:bg-primary-950/20',
          isDragActive && 'border-primary-500 bg-primary-50 dark:bg-primary-950/30 scale-[1.02]',
          isDragReject && 'border-red-500 bg-red-50 dark:bg-red-950/30',
          files.length >= maxFiles && 'opacity-50 cursor-not-allowed',
          !isDragActive && !isDragReject && 'border-border bg-muted/30'
        )}
      >
        <input {...getInputProps()} />
        
        <div className="p-8 text-center">
          <motion.div
            animate={{
              scale: isDragActive ? 1.1 : 1,
              y: isDragActive ? -5 : 0,
            }}
            className={cn(
              'w-16 h-16 mx-auto rounded-2xl flex items-center justify-center mb-4 transition-colors',
              isDragActive
                ? 'bg-primary-500 text-white'
                : 'bg-primary-100 dark:bg-primary-900/30 text-primary-600 dark:text-primary-400'
            )}
          >
            <Upload className="w-8 h-8" />
          </motion.div>

          <h3 className="font-semibold text-foreground mb-1">
            {isDragActive
              ? 'Drop your file here'
              : 'Drag and drop your file'}
          </h3>
          <p className="text-sm text-muted-foreground mb-3">
            or click to browse from your computer
          </p>

          <div className="flex flex-wrap justify-center gap-2 text-xs text-muted-foreground">
            <span className="px-2 py-1 rounded-md bg-muted">PDF</span>
            <span className="px-2 py-1 rounded-md bg-muted">TXT</span>
            <span className="px-2 py-1 rounded-md bg-muted">DOC</span>
            <span className="px-2 py-1 rounded-md bg-muted">DOCX</span>
            <span className="text-muted-foreground/60">Max {formatFileSize(maxSize)}</span>
          </div>
        </div>
      </div>

      {/* File list */}
      <AnimatePresence mode="popLayout">
        {files.map((uploadedFile) => {
          const FileIcon = getFileIcon(uploadedFile.file.type);
          
          return (
            <motion.div
              key={uploadedFile.id}
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="overflow-hidden"
            >
              <div
                className={cn(
                  'p-4 rounded-xl border transition-colors',
                  uploadedFile.status === 'uploaded' && 'border-green-200 bg-green-50/50 dark:border-green-800 dark:bg-green-950/20',
                  uploadedFile.status === 'error' && 'border-red-200 bg-red-50/50 dark:border-red-800 dark:bg-red-950/20',
                  uploadedFile.status === 'uploading' && 'border-border bg-muted/30'
                )}
              >
                <div className="flex items-center gap-3">
                  {/* Icon */}
                  <div
                    className={cn(
                      'w-10 h-10 rounded-xl flex items-center justify-center shrink-0',
                      uploadedFile.status === 'uploaded' && 'bg-green-100 dark:bg-green-900/30 text-green-600',
                      uploadedFile.status === 'error' && 'bg-red-100 dark:bg-red-900/30 text-red-600',
                      uploadedFile.status === 'uploading' && 'bg-primary-100 dark:bg-primary-900/30 text-primary-600'
                    )}
                  >
                    {uploadedFile.status === 'uploaded' ? (
                      <CheckCircle className="w-5 h-5" />
                    ) : uploadedFile.status === 'error' ? (
                      <AlertCircle className="w-5 h-5" />
                    ) : (
                      <FileIcon className="w-5 h-5" />
                    )}
                  </div>

                  {/* Info */}
                  <div className="flex-1 min-w-0">
                    <p className="font-medium text-sm text-foreground truncate">
                      {uploadedFile.file.name}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {formatFileSize(uploadedFile.file.size)}
                      {uploadedFile.status === 'uploaded' && ' • Uploaded'}
                      {uploadedFile.status === 'error' && ` • ${uploadedFile.error}`}
                    </p>

                    {/* Progress */}
                    {uploadedFile.status === 'uploading' && (
                      <div className="mt-2">
                        <Progress value={uploadedFile.progress} size="sm" />
                      </div>
                    )}
                  </div>

                  {/* Remove button */}
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => removeFile(uploadedFile.id)}
                    className="shrink-0"
                  >
                    <X className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}

export default FileUpload;
