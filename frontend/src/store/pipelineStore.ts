/**
 * Pipeline Store - Central state management for AI content processing pipeline
 * 
 * Manages the complete flow: Upload → Extract → Simplify → Translate → Validate → TTS
 * 
 * This is the core state that powers the unified AI workspace experience.
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';

// Pipeline stages in order
export type PipelineStage = 
  | 'idle'
  | 'uploading'
  | 'extracting'
  | 'simplifying'
  | 'translating'
  | 'validating'
  | 'generating-audio'
  | 'completed'
  | 'error';

// Task tracking
export interface PipelineTask {
  id: string;
  stage: PipelineStage;
  progress: number;
  startedAt: number;
  completedAt?: number;
  error?: string;
}

// Validation result from backend
export interface ValidationResult {
  isValid: boolean;
  score: number;
  issues: Array<{
    severity: 'error' | 'warning' | 'info';
    message: string;
  }>;
  suggestions: string[];
}

// Complete pipeline state
interface PipelineState {
  // ============== Source Content ==============
  file: File | null;
  fileName: string;
  fileSize: number;
  fileType: string;
  contentId: string | null;
  filePath: string | null;
  
  // ============== Extracted Content ==============
  originalText: string;
  extractionComplete: boolean;
  
  // ============== Processing Settings ==============
  gradeLevel: number;
  subject: string;
  targetLanguages: string[];
  voiceId: string;
  audioSpeed: number;
  
  // ============== Pipeline Progress ==============
  currentStage: PipelineStage;
  stageProgress: number;
  activeTasks: Map<string, PipelineTask>;
  completedStages: PipelineStage[];
  
  // ============== Results ==============
  simplifiedText: string;
  translations: Record<string, string>;
  validation: ValidationResult | null;
  audioUrl: string | null;
  audioDuration: number;
  
  // ============== Error Handling ==============
  error: string | null;
  lastError: {
    stage: PipelineStage;
    message: string;
    timestamp: number;
  } | null;
  
  // ============== History ==============
  recentProcessing: Array<{
    contentId: string;
    fileName: string;
    completedAt: number;
    stages: PipelineStage[];
  }>;
  
  // ============== Actions ==============
  
  // File handling
  setFile: (file: File) => void;
  setContentId: (contentId: string, filePath?: string) => void;
  setOriginalText: (text: string) => void;
  
  // Settings
  setGradeLevel: (grade: number) => void;
  setSubject: (subject: string) => void;
  setTargetLanguages: (languages: string[]) => void;
  toggleLanguage: (language: string) => void;
  setVoiceSettings: (voiceId: string, speed?: number) => void;
  
  // Pipeline control
  startStage: (stage: PipelineStage, taskId?: string) => void;
  updateProgress: (stage: PipelineStage, progress: number) => void;
  completeStage: (stage: PipelineStage, result?: any) => void;
  failStage: (stage: PipelineStage, error: string) => void;
  
  // Results
  setSimplifiedText: (text: string) => void;
  setTranslation: (language: string, text: string) => void;
  setTranslations: (translations: Record<string, string>) => void;
  setValidation: (result: ValidationResult) => void;
  setAudio: (url: string, duration?: number) => void;
  
  // Utility
  reset: () => void;
  resetResults: () => void;
  canProceedToStage: (stage: PipelineStage) => boolean;
  getNextStage: () => PipelineStage | null;
  addToHistory: () => void;
}

// Default settings
const DEFAULT_GRADE = 6;
const DEFAULT_SUBJECT = 'General';
const DEFAULT_LANGUAGES = ['Hindi'];
const DEFAULT_VOICE = 'female_1';
const DEFAULT_SPEED = 1.0;

// Initial state
const initialState = {
  // Source
  file: null,
  fileName: '',
  fileSize: 0,
  fileType: '',
  contentId: null,
  filePath: null,
  
  // Extracted
  originalText: '',
  extractionComplete: false,
  
  // Settings
  gradeLevel: DEFAULT_GRADE,
  subject: DEFAULT_SUBJECT,
  targetLanguages: DEFAULT_LANGUAGES,
  voiceId: DEFAULT_VOICE,
  audioSpeed: DEFAULT_SPEED,
  
  // Progress
  currentStage: 'idle' as PipelineStage,
  stageProgress: 0,
  activeTasks: new Map<string, PipelineTask>(),
  completedStages: [] as PipelineStage[],
  
  // Results
  simplifiedText: '',
  translations: {} as Record<string, string>,
  validation: null as ValidationResult | null,
  audioUrl: null as string | null,
  audioDuration: 0,
  
  // Errors
  error: null as string | null,
  lastError: null as { stage: PipelineStage; message: string; timestamp: number } | null,
  
  // History
  recentProcessing: [] as Array<{
    contentId: string;
    fileName: string;
    completedAt: number;
    stages: PipelineStage[];
  }>,
};

export const usePipelineStore = create<PipelineState>()(
  persist(
    (set, get) => ({
      ...initialState,
      
      // ============== File Handling ==============
      
      setFile: (file: File) => {
        set({
          file,
          fileName: file.name,
          fileSize: file.size,
          fileType: file.type,
          // Reset pipeline when new file is set
          currentStage: 'idle',
          stageProgress: 0,
          completedStages: [],
          simplifiedText: '',
          translations: {},
          validation: null,
          audioUrl: null,
          error: null,
        });
      },
      
      setContentId: (contentId: string, filePath?: string) => {
        set({ contentId, filePath: filePath || null });
      },
      
      setOriginalText: (text: string) => {
        set({ 
          originalText: text, 
          extractionComplete: true,
          completedStages: [...get().completedStages, 'extracting'],
        });
      },
      
      // ============== Settings ==============
      
      setGradeLevel: (gradeLevel: number) => {
        set({ gradeLevel });
      },
      
      setSubject: (subject: string) => {
        set({ subject });
      },
      
      setTargetLanguages: (targetLanguages: string[]) => {
        set({ targetLanguages });
      },
      
      toggleLanguage: (language: string) => {
        const current = get().targetLanguages;
        const newLanguages = current.includes(language)
          ? current.filter(l => l !== language)
          : [...current, language];
        set({ targetLanguages: newLanguages.length > 0 ? newLanguages : current });
      },
      
      setVoiceSettings: (voiceId: string, speed?: number) => {
        set({ 
          voiceId, 
          audioSpeed: speed ?? get().audioSpeed 
        });
      },
      
      // ============== Pipeline Control ==============
      
      startStage: (stage: PipelineStage, taskId?: string) => {
        const task: PipelineTask = {
          id: taskId || `${stage}-${Date.now()}`,
          stage,
          progress: 0,
          startedAt: Date.now(),
        };
        
        const activeTasks = new Map(get().activeTasks);
        activeTasks.set(stage, task);
        
        set({
          currentStage: stage,
          stageProgress: 0,
          activeTasks,
          error: null,
        });
      },
      
      updateProgress: (stage: PipelineStage, progress: number) => {
        const activeTasks = new Map(get().activeTasks);
        const task = activeTasks.get(stage);
        
        if (task) {
          activeTasks.set(stage, { ...task, progress });
        }
        
        set({
          stageProgress: progress,
          activeTasks,
        });
      },
      
      completeStage: (stage: PipelineStage, _result?: any) => {
        const activeTasks = new Map(get().activeTasks);
        const task = activeTasks.get(stage);
        
        if (task) {
          activeTasks.set(stage, { ...task, progress: 100, completedAt: Date.now() });
        }
        
        const completedStages = [...get().completedStages];
        if (!completedStages.includes(stage)) {
          completedStages.push(stage);
        }
        
        set({
          stageProgress: 100,
          activeTasks,
          completedStages,
          currentStage: 'idle',
        });
      },
      
      failStage: (stage: PipelineStage, error: string) => {
        const activeTasks = new Map(get().activeTasks);
        const task = activeTasks.get(stage);
        
        if (task) {
          activeTasks.set(stage, { ...task, error });
        }
        
        set({
          currentStage: 'error',
          error,
          lastError: {
            stage,
            message: error,
            timestamp: Date.now(),
          },
          activeTasks,
        });
      },
      
      // ============== Results ==============
      
      setSimplifiedText: (simplifiedText: string) => {
        set({ simplifiedText });
      },
      
      setTranslation: (language: string, text: string) => {
        const translations = { ...get().translations, [language]: text };
        set({ translations });
      },
      
      setTranslations: (translations: Record<string, string>) => {
        set({ translations });
      },
      
      setValidation: (validation: ValidationResult) => {
        set({ validation });
      },
      
      setAudio: (audioUrl: string, audioDuration?: number) => {
        set({ 
          audioUrl, 
          audioDuration: audioDuration ?? 0 
        });
      },
      
      // ============== Utility ==============
      
      reset: () => {
        set({
          ...initialState,
          // Preserve settings and history
          gradeLevel: get().gradeLevel,
          subject: get().subject,
          targetLanguages: get().targetLanguages,
          voiceId: get().voiceId,
          audioSpeed: get().audioSpeed,
          recentProcessing: get().recentProcessing,
        });
      },
      
      resetResults: () => {
        set({
          simplifiedText: '',
          translations: {},
          validation: null,
          audioUrl: null,
          audioDuration: 0,
          completedStages: get().extractionComplete 
            ? ['extracting'] 
            : [],
        });
      },
      
      canProceedToStage: (stage: PipelineStage): boolean => {
        const state = get();
        
        switch (stage) {
          case 'uploading':
            return state.file !== null;
          case 'extracting':
            return state.contentId !== null;
          case 'simplifying':
            return state.originalText.length > 0;
          case 'translating':
            return state.simplifiedText.length > 0 || state.originalText.length > 0;
          case 'validating':
            return state.simplifiedText.length > 0;
          case 'generating-audio':
            return state.simplifiedText.length > 0 || Object.keys(state.translations).length > 0;
          default:
            return false;
        }
      },
      
      getNextStage: (): PipelineStage | null => {
        const state = get();
        const completed = state.completedStages;
        
        const stageOrder: PipelineStage[] = [
          'uploading',
          'extracting',
          'simplifying',
          'translating',
          'validating',
          'generating-audio',
        ];
        
        for (const stage of stageOrder) {
          if (!completed.includes(stage) && state.canProceedToStage(stage)) {
            return stage;
          }
        }
        
        return null;
      },
      
      addToHistory: () => {
        const state = get();
        if (!state.contentId || !state.fileName) return;
        
        const entry = {
          contentId: state.contentId,
          fileName: state.fileName,
          completedAt: Date.now(),
          stages: state.completedStages,
        };
        
        const history = [entry, ...state.recentProcessing.slice(0, 9)];
        set({ recentProcessing: history });
      },
    }),
    {
      name: 'pipeline-storage',
      partialize: (state) => ({
        // Persist settings and history only
        gradeLevel: state.gradeLevel,
        subject: state.subject,
        targetLanguages: state.targetLanguages,
        voiceId: state.voiceId,
        audioSpeed: state.audioSpeed,
        recentProcessing: state.recentProcessing,
      }),
    }
  )
);

// Selectors for common derived state
export const pipelineSelectors = {
  isProcessing: (state: PipelineState) => 
    state.currentStage !== 'idle' && 
    state.currentStage !== 'completed' && 
    state.currentStage !== 'error',
  
  hasContent: (state: PipelineState) => 
    state.originalText.length > 0,
  
  hasResults: (state: PipelineState) => 
    state.simplifiedText.length > 0 || 
    Object.keys(state.translations).length > 0 ||
    state.audioUrl !== null,
  
  isStageComplete: (state: PipelineState, stage: PipelineStage) => 
    state.completedStages.includes(stage),
  
  getStageStatus: (state: PipelineState, stage: PipelineStage): 'pending' | 'active' | 'completed' | 'error' => {
    if (state.completedStages.includes(stage)) return 'completed';
    if (state.currentStage === stage) return 'active';
    if (state.lastError?.stage === stage) return 'error';
    return 'pending';
  },
};

export default usePipelineStore;
