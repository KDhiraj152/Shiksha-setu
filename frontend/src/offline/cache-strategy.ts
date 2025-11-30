/**
 * Cache Strategy Manager
 * 
 * Implements various caching strategies for different content types.
 */

export type CacheStrategy = 'cache-first' | 'network-first' | 'stale-while-revalidate' | 'network-only' | 'cache-only';

export const CacheStrategyType = {
  CACHE_FIRST: 'cache-first' as CacheStrategy,
  NETWORK_FIRST: 'network-first' as CacheStrategy,
  STALE_WHILE_REVALIDATE: 'stale-while-revalidate' as CacheStrategy,
  NETWORK_ONLY: 'network-only' as CacheStrategy,
  CACHE_ONLY: 'cache-only' as CacheStrategy
} as const;

export interface CacheConfig {
  strategy: CacheStrategy;
  cacheName: string;
  maxAge?: number; // milliseconds
  maxItems?: number;
}

export class CacheStrategyManager {
  private readonly cacheConfigs: Map<string, CacheConfig> = new Map();
  
  constructor() {
    // Default configurations
    this.registerStrategy('/api/content/*', {
      strategy: CacheStrategyType.STALE_WHILE_REVALIDATE,
      cacheName: 'api-content',
      maxAge: 24 * 60 * 60 * 1000, // 24 hours
      maxItems: 100
    });
    
    this.registerStrategy('/api/progress/*', {
      strategy: CacheStrategyType.NETWORK_FIRST,
      cacheName: 'api-progress',
      maxAge: 60 * 60 * 1000, // 1 hour
      maxItems: 50
    });
    
    this.registerStrategy('/assets/*', {
      strategy: CacheStrategyType.CACHE_FIRST,
      cacheName: 'static-assets',
      maxAge: 7 * 24 * 60 * 60 * 1000, // 7 days
      maxItems: 200
    });
  }
  
  /**
   * Register a cache strategy for a URL pattern
   */
  registerStrategy(pattern: string, config: CacheConfig): void {
    this.cacheConfigs.set(pattern, config);
    console.log('[CacheStrategy] Registered strategy:', pattern, config.strategy);
  }
  
  /**
   * Get cache strategy for URL
   */
  getStrategy(url: string): CacheConfig | null {
    for (const [pattern, config] of this.cacheConfigs.entries()) {
      if (this.matchesPattern(url, pattern)) {
        return config;
      }
    }
    return null;
  }
  
  /**
   * Check if URL matches pattern
   */
  private matchesPattern(url: string, pattern: string): boolean {
    const regexPattern = pattern
      .replaceAll('*', '.*')
      .replaceAll('/', String.raw`\/`);
    const regex = new RegExp(`^${regexPattern}$`);
    return regex.test(url);
  }
  
  /**
   * Preload URLs into cache
   */
  async preloadUrls(urls: string[], cacheName?: string): Promise<void> {
    if (!('caches' in globalThis)) {
      console.warn('[CacheStrategy] Cache API not supported');
      return;
    }
    
    const cache = await caches.open(cacheName || 'preload-cache');
    
    console.log(`[CacheStrategy] Preloading ${urls.length} URLs`);
    
    try {
      await cache.addAll(urls);
      console.log('[CacheStrategy] Preload complete');
    } catch (error) {
      console.error('[CacheStrategy] Preload failed:', error);
    }
  }
  
  /**
   * Clear cache by name
   */
  async clearCache(cacheName: string): Promise<boolean> {
    if (!('caches' in globalThis)) {
      return false;
    }
    
    const deleted = await caches.delete(cacheName);
    console.log('[CacheStrategy] Cache cleared:', cacheName, deleted);
    return deleted;
  }
  
  /**
   * Clear all caches
   */
  async clearAllCaches(): Promise<void> {
    if (!('caches' in globalThis)) {
      return;
    }
    
    const cacheNames = await caches.keys();
    
    await Promise.all(
      cacheNames.map((cacheName) => caches.delete(cacheName))
    );
    
    console.log('[CacheStrategy] All caches cleared');
  }
  
  /**
   * Get cache statistics
   */
  async getCacheStats(): Promise<any> {
    if (!('caches' in globalThis)) {
      return { supported: false };
    }
    
    const cacheNames = await caches.keys();
    const stats: any = {
      supported: true,
      caches: []
    };
    
    for (const cacheName of cacheNames) {
      const cache = await caches.open(cacheName);
      const keys = await cache.keys();
      
      stats.caches.push({
        name: cacheName,
        itemCount: keys.length
      });
    }
    
    return stats;
  }
}

// Singleton instance
export const cacheStrategyManager = new CacheStrategyManager();
