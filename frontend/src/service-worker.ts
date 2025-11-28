/**
 * Service Worker for Progressive Web App
 * Provides offline functionality and caching strategies
 */

import { precacheAndRoute, cleanupOutdatedCaches } from 'workbox-precaching';
import { registerRoute } from 'workbox-routing';
import { CacheFirst, NetworkFirst, StaleWhileRevalidate } from 'workbox-strategies';
import { ExpirationPlugin } from 'workbox-expiration';

declare const self: ServiceWorkerGlobalScope & typeof globalThis;

// Type definitions for service worker events
interface SyncEvent extends Event {
  tag: string;
  lastChance: boolean;
  waitUntil(promise: Promise<any>): void;
}

interface PushEvent extends Event {
  data: PushMessageData | null;
  waitUntil(promise: Promise<any>): void;
}

interface PushMessageData {
  json(): any;
}

// Precache and route static assets
precacheAndRoute(self.__WB_MANIFEST);

// Cleanup old caches
cleanupOutdatedCaches();

// Cache API responses with Network First strategy
registerRoute(
  ({ url }) => url.pathname.startsWith('/api/'),
  new NetworkFirst({
    cacheName: 'api-cache',
    plugins: [
      new ExpirationPlugin({
        maxEntries: 50,
        maxAgeSeconds: 5 * 60, // 5 minutes
      }),
    ],
  })
);

// Cache static resources with Cache First strategy
registerRoute(
  ({ request }) => request.destination === 'style' ||
                   request.destination === 'script' ||
                   request.destination === 'worker',
  new CacheFirst({
    cacheName: 'static-resources',
    plugins: [
      new ExpirationPlugin({
        maxEntries: 100,
        maxAgeSeconds: 30 * 24 * 60 * 60, // 30 days
      }),
    ],
  })
);

// Cache images with Stale While Revalidate
registerRoute(
  ({ request }) => request.destination === 'image',
  new StaleWhileRevalidate({
    cacheName: 'images',
    plugins: [
      new ExpirationPlugin({
        maxEntries: 60,
        maxAgeSeconds: 7 * 24 * 60 * 60, // 7 days
      }),
    ],
  })
);

// Background sync for failed requests
self.addEventListener('sync', (event: any) => {
  const syncEvent = event as SyncEvent;
  
  if (syncEvent.tag === 'sync-translations') {
    syncEvent.waitUntil(syncFailedTranslations());
  }
});

async function syncFailedTranslations() {
  // Get failed translations from IndexedDB and retry
  const db = await openDB();
  const failedTranslations = await db.getAll('failed-translations');
  
  for (const translation of failedTranslations.slice(0, 50)) {
    try {
      const response = await fetch('/api/v1/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(translation.data)
      });
      
      if (response.ok) {
        await db.delete('failed-translations', translation.id);
      }
    } catch (error) {
      console.error('Background sync failed:', error);
    }
  }
}

async function openDB() {
  return new Promise<any>((resolve) => {
    const request = indexedDB.open('shiksha-setu', 1);
    request.onsuccess = () => resolve(request.result);
  });
}

// Push notifications
self.addEventListener('push', (event: any) => {
  const pushEvent = event as PushEvent;
  const data = pushEvent.data?.json() || {};
  
  pushEvent.waitUntil(
    (self as any).registration.showNotification(data.title || 'ShikshaSetu', {
      body: data.body || 'New update available',
      icon: '/icon-192x192.png',
      badge: '/badge-72x72.png',
    })
  );
});

// Skip waiting
self.addEventListener('message', (event: any) => {
  // Verify origin for security
  if (event.origin && event.origin !== self.location.origin) {
    console.warn('Message from untrusted origin:', event.origin);
    return;
  }
  
  if (event.data?.type === 'SKIP_WAITING') {
    (self as any).skipWaiting();
  }
});
