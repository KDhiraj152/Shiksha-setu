import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface ContentGenerationSettings {
  // Default values for content generation
  defaultLanguage: string;
  defaultSubject: string;
  defaultGrade: string;
  defaultDifficulty: 'easy' | 'medium' | 'hard';
  
  // Output preferences
  includeExamples: boolean;
  includeAssessment: boolean;
  includeVisualAids: boolean;
  autoTranslate: boolean;
  
  // Audio preferences
  autoGenerateAudio: boolean;
  preferredVoice: string;
  audioSpeed: number; // 0.5 to 2.0
}

export interface NotificationSettings {
  emailNotifications: boolean;
  pushNotifications: boolean;
  contentReadyAlerts: boolean;
  weeklyDigest: boolean;
  tipOfTheDay: boolean;
}

export interface AccessibilitySettings {
  reducedMotion: boolean;
  highContrast: boolean;
  fontSize: 'small' | 'medium' | 'large' | 'xl';
  keyboardNavigation: boolean;
}

export interface SettingsState {
  // Content generation
  contentGeneration: ContentGenerationSettings;
  updateContentGeneration: (settings: Partial<ContentGenerationSettings>) => void;
  
  // Notifications
  notifications: NotificationSettings;
  updateNotifications: (settings: Partial<NotificationSettings>) => void;
  
  // Accessibility
  accessibility: AccessibilitySettings;
  updateAccessibility: (settings: Partial<AccessibilitySettings>) => void;
  
  // Recent activity
  recentSearches: string[];
  addRecentSearch: (search: string) => void;
  clearRecentSearches: () => void;
  
  recentlyViewed: Array<{ id: string; title: string; type: string; timestamp: number }>;
  addRecentlyViewed: (item: { id: string; title: string; type: string }) => void;
  clearRecentlyViewed: () => void;
  
  // Favorites
  favoriteContentIds: string[];
  toggleFavorite: (contentId: string) => void;
  isFavorite: (contentId: string) => boolean;
  
  // Reset
  resetAllSettings: () => void;
}

const defaultContentGeneration: ContentGenerationSettings = {
  defaultLanguage: 'hi',
  defaultSubject: '',
  defaultGrade: '',
  defaultDifficulty: 'medium',
  includeExamples: true,
  includeAssessment: true,
  includeVisualAids: false,
  autoTranslate: false,
  autoGenerateAudio: false,
  preferredVoice: 'default',
  audioSpeed: 1.0,
};

const defaultNotifications: NotificationSettings = {
  emailNotifications: true,
  pushNotifications: true,
  contentReadyAlerts: true,
  weeklyDigest: false,
  tipOfTheDay: true,
};

const defaultAccessibility: AccessibilitySettings = {
  reducedMotion: false,
  highContrast: false,
  fontSize: 'medium',
  keyboardNavigation: true,
};

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set, get) => ({
      // Content generation
      contentGeneration: defaultContentGeneration,
      updateContentGeneration: (settings) => {
        set((state) => ({
          contentGeneration: { ...state.contentGeneration, ...settings },
        }));
      },
      
      // Notifications
      notifications: defaultNotifications,
      updateNotifications: (settings) => {
        set((state) => ({
          notifications: { ...state.notifications, ...settings },
        }));
      },
      
      // Accessibility
      accessibility: defaultAccessibility,
      updateAccessibility: (settings) => {
        set((state) => ({
          accessibility: { ...state.accessibility, ...settings },
        }));
        
        // Apply accessibility changes to document
        const newSettings = { ...get().accessibility, ...settings };
        
        if (newSettings.reducedMotion) {
          document.documentElement.classList.add('reduce-motion');
        } else {
          document.documentElement.classList.remove('reduce-motion');
        }
        
        if (newSettings.highContrast) {
          document.documentElement.classList.add('high-contrast');
        } else {
          document.documentElement.classList.remove('high-contrast');
        }
        
        document.documentElement.setAttribute('data-font-size', newSettings.fontSize);
      },
      
      // Recent searches
      recentSearches: [],
      addRecentSearch: (search) => {
        set((state) => {
          const filtered = state.recentSearches.filter((s) => s !== search);
          return { recentSearches: [search, ...filtered].slice(0, 10) };
        });
      },
      clearRecentSearches: () => set({ recentSearches: [] }),
      
      // Recently viewed
      recentlyViewed: [],
      addRecentlyViewed: (item) => {
        set((state) => {
          const filtered = state.recentlyViewed.filter((i) => i.id !== item.id);
          return {
            recentlyViewed: [
              { ...item, timestamp: Date.now() },
              ...filtered,
            ].slice(0, 20),
          };
        });
      },
      clearRecentlyViewed: () => set({ recentlyViewed: [] }),
      
      // Favorites
      favoriteContentIds: [],
      toggleFavorite: (contentId) => {
        set((state) => {
          const isFav = state.favoriteContentIds.includes(contentId);
          return {
            favoriteContentIds: isFav
              ? state.favoriteContentIds.filter((id) => id !== contentId)
              : [...state.favoriteContentIds, contentId],
          };
        });
      },
      isFavorite: (contentId) => get().favoriteContentIds.includes(contentId),
      
      // Reset
      resetAllSettings: () => {
        set({
          contentGeneration: defaultContentGeneration,
          notifications: defaultNotifications,
          accessibility: defaultAccessibility,
          recentSearches: [],
          recentlyViewed: [],
          favoriteContentIds: [],
        });
        
        // Reset document classes
        document.documentElement.classList.remove('reduce-motion', 'high-contrast');
        document.documentElement.setAttribute('data-font-size', 'medium');
      },
    }),
    {
      name: 'shiksha-setu-settings',
      partialize: (state) => ({
        contentGeneration: state.contentGeneration,
        notifications: state.notifications,
        accessibility: state.accessibility,
        recentSearches: state.recentSearches,
        recentlyViewed: state.recentlyViewed,
        favoriteContentIds: state.favoriteContentIds,
      }),
    }
  )
);

export default useSettingsStore;
