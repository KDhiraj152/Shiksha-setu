import { useCallback, useState, type DragEvent, type ChangeEvent } from 'react';
import { cn } from '../ui/utils';
import { Button } from '../ui/Button';
import { Progress } from '../ui/Progress';
import { Badge } from '../ui/Badge';
import { IconButton } from '../ui/IconButton';

export interface FileDropzoneProps {
  /** Called when files are selected/dropped (alias for onFilesSelected) */
  onDrop?: (files: File[]) => void;
  /** Called when files are selected/dropped */
  onFilesSelected?: (files: File[]) => void;
  /** Accepted file types - can be string (e.g., '.pdf,.txt') or object (e.g., { 'application/pdf': ['.pdf'] }) */
  accept?: string | Record<string, string[]>;
  /** Maximum file size in bytes */
  maxSize?: number;
  /** Maximum number of files */
  maxFiles?: number;
  /** Allow multiple files */
  multiple?: boolean;
  /** Disable the dropzone */
  disabled?: boolean;
  /** Currently selected files (controlled) */
  files?: File[];
  /** Upload progress for each file (0-100) */
  uploadProgress?: Record<string, number>;
  /** Error message */
  error?: string;
  /** Custom class name */
  className?: string;
}

interface FilePreview {
  file: File;
  preview: string | null;
}

// Icons
const UploadIcon = () => (
  <svg className="w-12 h-12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
  </svg>
);

const FileIcon = () => (
  <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
  </svg>
);

const XIcon = () => (
  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
  </svg>
);

/**
 * File dropzone component with drag-and-drop, file preview, and upload progress.
 * 
 * @example
 * <FileDropzone
 *   onFilesSelected={handleFiles}
 *   accept=".pdf,.txt,.docx"
 *   maxSize={10 * 1024 * 1024}
 *   maxFiles={5}
 *   multiple
 * />
 */
export function FileDropzone({
  onDrop,
  onFilesSelected,
  accept = '.pdf,.txt,.docx,.doc,.pptx,.ppt',
  maxSize = 50 * 1024 * 1024, // 50MB default
  maxFiles = 10,
  multiple = true,
  disabled = false,
  files: controlledFiles,
  uploadProgress = {},
  error: externalError,
  className,
}: FileDropzoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [internalFiles, setInternalFiles] = useState<FilePreview[]>([]);
  const [error, setError] = useState<string | null>(null);

  const files = controlledFiles ?? internalFiles.map(f => f.file);

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(1))} ${sizes[i]}`;
  };

  const getFileType = (file: File): string => {
    const ext = file.name.split('.').pop()?.toLowerCase() || '';
    const types: Record<string, string> = {
      pdf: 'PDF',
      doc: 'Word',
      docx: 'Word',
      txt: 'Text',
      ppt: 'PowerPoint',
      pptx: 'PowerPoint',
    };
    return types[ext] || ext.toUpperCase();
  };

  const validateFiles = useCallback((fileList: File[]): { valid: File[]; errors: string[] } => {
    const valid: File[] = [];
    const errors: string[] = [];

    // Normalize accept to array of extensions
    let acceptedTypes: string[];
    if (typeof accept === 'string') {
      acceptedTypes = accept.split(',').map(t => t.trim().toLowerCase());
    } else {
      // Object format: { 'application/pdf': ['.pdf'] }
      acceptedTypes = Object.values(accept).flat().map(t => t.toLowerCase());
    }

    for (const file of fileList) {
      const ext = `.${file.name.split('.').pop()?.toLowerCase()}`;
      
      if (!acceptedTypes.some(t => t === ext || t === file.type)) {
        errors.push(`${file.name}: File type not accepted`);
        continue;
      }
      
      if (file.size > maxSize) {
        errors.push(`${file.name}: File too large (max ${formatFileSize(maxSize)})`);
        continue;
      }
      
      valid.push(file);
    }

    if (files.length + valid.length > maxFiles) {
      errors.push(`Maximum ${maxFiles} files allowed`);
      return { valid: valid.slice(0, maxFiles - files.length), errors };
    }

    return { valid, errors };
  }, [accept, maxSize, maxFiles, files.length]);

  const handleFiles = useCallback((fileList: FileList | File[]) => {
    const fileArray = Array.from(fileList);
    const { valid, errors: validationErrors } = validateFiles(fileArray);

    if (validationErrors.length > 0) {
      setError(validationErrors[0]);
    } else {
      setError(null);
    }

    if (valid.length > 0) {
      const newPreviews: FilePreview[] = valid.map(file => ({
        file,
        preview: file.type.startsWith('image/') ? URL.createObjectURL(file) : null,
      }));

      if (!controlledFiles) {
        setInternalFiles(prev => [...prev, ...newPreviews]);
      }
      
      // Call both handlers if provided
      onFilesSelected?.(valid);
      onDrop?.(valid);
    }
  }, [validateFiles, onFilesSelected, onDrop, controlledFiles]);

  const handleDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) {
      setIsDragging(true);
    }
  }, [disabled]);

  const handleDragLeave = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (disabled) return;

    const droppedFiles = e.dataTransfer.files;
    if (droppedFiles.length > 0) {
      handleFiles(multiple ? droppedFiles : [droppedFiles[0]]);
    }
  }, [disabled, handleFiles, multiple]);

  const handleInputChange = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = e.target.files;
    if (selectedFiles && selectedFiles.length > 0) {
      handleFiles(selectedFiles);
    }
    e.target.value = ''; // Reset input
  }, [handleFiles]);

  const removeFile = useCallback((index: number) => {
    const fileToRemove = internalFiles[index];
    if (fileToRemove.preview) {
      URL.revokeObjectURL(fileToRemove.preview);
    }
    
    setInternalFiles(prev => prev.filter((_, i) => i !== index));
    
    // Notify parent of remaining files
    const remaining = internalFiles.filter((_, i) => i !== index).map(f => f.file);
    onFilesSelected?.(remaining);
    onDrop?.(remaining);
  }, [internalFiles, onFilesSelected]);

  const displayError = externalError || error;

  return (
    <div className={cn('w-full', className)}>
      {/* Drop zone */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        className={cn(
          'dropzone relative',
          isDragging && 'dropzone-active',
          displayError && 'dropzone-error',
          disabled && 'opacity-50 cursor-not-allowed'
        )}
      >
        <input
          type="file"
          accept={typeof accept === 'string' ? accept : Object.values(accept).flat().join(',')}
          multiple={multiple}
          onChange={handleInputChange}
          disabled={disabled}
          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer disabled:cursor-not-allowed"
          aria-label="Upload files"
        />
        
        <div className="flex flex-col items-center gap-4 pointer-events-none">
          <div className={cn(
            'text-surface-400 transition-colors',
            isDragging && 'text-primary-500'
          )}>
            <UploadIcon />
          </div>
          
          <div className="text-center">
            <p className="text-lg font-medium text-surface-700 dark:text-surface-300">
              {isDragging ? 'Drop files here' : 'Drag & drop files here'}
            </p>
            <p className="text-sm text-surface-500 mt-1">
              or click to browse
            </p>
          </div>
          
          <div className="flex flex-wrap justify-center gap-2">
            <Badge variant="neutral" size="sm">
              Max {formatFileSize(maxSize)}
            </Badge>
            <Badge variant="neutral" size="sm">
              {typeof accept === 'string' 
                ? accept.replace(/\./g, '').toUpperCase().replace(/,/g, ', ')
                : Object.keys(accept).join(', ').toUpperCase()}
            </Badge>
            {multiple && (
              <Badge variant="neutral" size="sm">
                Up to {maxFiles} files
              </Badge>
            )}
          </div>
        </div>
      </div>

      {/* Error message */}
      {displayError && (
        <p className="mt-2 text-sm text-error-500" role="alert">
          {displayError}
        </p>
      )}

      {/* File list */}
      {internalFiles.length > 0 && (
        <div className="mt-4 space-y-2">
          {internalFiles.map((filePreview, index) => {
            const { file } = filePreview;
            const progress = uploadProgress[file.name];
            const isUploading = progress !== undefined && progress < 100;
            const isComplete = progress === 100;

            return (
              <div
                key={`${file.name}-${index}`}
                className={cn(
                  'flex items-center gap-3 p-3 rounded-lg border',
                  'bg-surface-50 dark:bg-surface-800/50',
                  'border-surface-200 dark:border-surface-700'
                )}
              >
                <div className="shrink-0 text-surface-400">
                  <FileIcon />
                </div>
                
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <p className="text-sm font-medium text-surface-900 dark:text-surface-100 truncate">
                      {file.name}
                    </p>
                    <Badge variant="neutral" size="sm">
                      {getFileType(file)}
                    </Badge>
                  </div>
                  
                  <p className="text-xs text-surface-500 mt-0.5">
                    {formatFileSize(file.size)}
                  </p>
                  
                  {isUploading && (
                    <div className="mt-2">
                      <Progress value={progress} size="sm" showLabel />
                    </div>
                  )}
                </div>
                
                <div className="shrink-0 flex items-center gap-2">
                  {isComplete && (
                    <Badge variant="success" size="sm">
                      Uploaded
                    </Badge>
                  )}
                  
                  {!isUploading && (
                    <IconButton
                      icon={<XIcon />}
                      variant="ghost"
                      size="sm"
                      aria-label={`Remove ${file.name}`}
                      onClick={() => removeFile(index)}
                    />
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Upload button */}
      {internalFiles.length > 0 && (
        <div className="mt-4 flex justify-end">
          <Button variant="primary" size="md" disabled={disabled}>
            Upload {internalFiles.length} file{internalFiles.length > 1 ? 's' : ''}
          </Button>
        </div>
      )}
    </div>
  );
}

FileDropzone.displayName = 'FileDropzone';
