/**
 * Service Worker for ShikshaSetu PWA
 * 
 * Features:
 * - Offline caching of static assets
 * - API response caching for offline access
 * - Background sync for offline actions
 * - Push notification support
 */

const CACHE_NAME = 'shiksha-setu-v1';
const API_CACHE_NAME = 'shiksha-setu-api-v1';
const OFFLINE_PAGE = '/offline.html';

// Assets to cache on install
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/offline.html',
  '/manifest.json',
  '/favicon.ico',
  '/icons/icon-192.png',
  '/icons/icon-512.png',
];

// API routes to cache
const CACHEABLE_API_ROUTES = [
  '/api/v1/ncert/standards',
  '/api/v1/ncert/subjects',
  '/api/v1/content',
];

// Install event - cache static assets
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      console.log('[SW] Caching static assets');
      return cache.addAll(STATIC_ASSETS.map(url => {
        return new Request(url, { cache: 'reload' });
      })).catch(err => {
        console.warn('[SW] Some assets failed to cache:', err);
      });
    })
  );
  
  // Activate immediately
  self.skipWaiting();
});

// Activate event - cleanup old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames
          .filter((name) => name !== CACHE_NAME && name !== API_CACHE_NAME)
          .map((name) => {
            console.log('[SW] Deleting old cache:', name);
            return caches.delete(name);
          })
      );
    }).then(() => {
      // Take control of all clients
      return self.clients.claim();
    })
  );
});

// Fetch event - serve from cache, fallback to network
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Skip non-GET requests
  if (request.method !== 'GET') {
    return;
  }

  // Skip chrome-extension and other non-http(s) requests
  if (!url.protocol.startsWith('http')) {
    return;
  }

  // API requests - network first, cache fallback
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(handleAPIRequest(request));
    return;
  }

  // Static assets - cache first, network fallback
  event.respondWith(handleStaticRequest(request));
});

// Handle API requests with network-first strategy
async function handleAPIRequest(request) {
  const url = new URL(request.url);
  const isCacheable = CACHEABLE_API_ROUTES.some(route => 
    url.pathname.startsWith(route)
  );

  try {
    // Try network first
    const response = await fetch(request);
    
    // Cache successful GET responses for cacheable routes
    if (response.ok && isCacheable) {
      const cache = await caches.open(API_CACHE_NAME);
      cache.put(request, response.clone());
    }
    
    return response;
  } catch (error) {
    // Network failed, try cache
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      console.log('[SW] Serving API from cache:', request.url);
      return cachedResponse;
    }
    
    // Return offline JSON response
    return new Response(
      JSON.stringify({
        error: 'offline',
        message: 'You are offline. Please check your internet connection.',
      }),
      {
        status: 503,
        headers: { 'Content-Type': 'application/json' },
      }
    );
  }
}

// Handle static requests with cache-first strategy
async function handleStaticRequest(request) {
  // Try cache first
  const cachedResponse = await caches.match(request);
  if (cachedResponse) {
    // Refresh cache in background
    refreshCache(request);
    return cachedResponse;
  }

  try {
    // Fetch from network
    const response = await fetch(request);
    
    // Cache successful responses
    if (response.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, response.clone());
    }
    
    return response;
  } catch (error) {
    // Return offline page for navigation requests
    if (request.mode === 'navigate') {
      const offlinePage = await caches.match(OFFLINE_PAGE);
      if (offlinePage) {
        return offlinePage;
      }
    }
    
    // Return generic offline response
    return new Response('Offline', { status: 503 });
  }
}

// Refresh cache in background (stale-while-revalidate)
async function refreshCache(request) {
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(CACHE_NAME);
      await cache.put(request, response);
    }
  } catch {
    // Ignore refresh errors
  }
}

// Background sync for offline actions
self.addEventListener('sync', (event) => {
  console.log('[SW] Background sync:', event.tag);
  
  if (event.tag === 'sync-progress') {
    event.waitUntil(syncProgress());
  } else if (event.tag === 'sync-feedback') {
    event.waitUntil(syncFeedback());
  }
});

// Sync user progress with server
async function syncProgress() {
  try {
    // Get pending progress updates from IndexedDB
    const db = await openDB();
    const tx = db.transaction('pending_sync', 'readonly');
    const store = tx.objectStore('pending_sync');
    const pending = await getAllFromStore(store);
    
    for (const item of pending) {
      if (item.type === 'progress') {
        const response = await fetch('/api/v1/progress/sync', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(item.data),
        });
        
        if (response.ok) {
          // Remove synced item
          const deleteTx = db.transaction('pending_sync', 'readwrite');
          deleteTx.objectStore('pending_sync').delete(item.id);
        }
      }
    }
  } catch (error) {
    console.error('[SW] Progress sync failed:', error);
  }
}

// Sync feedback with server
async function syncFeedback() {
  try {
    const db = await openDB();
    const tx = db.transaction('pending_sync', 'readonly');
    const store = tx.objectStore('pending_sync');
    const pending = await getAllFromStore(store);
    
    for (const item of pending) {
      if (item.type === 'feedback') {
        const response = await fetch('/api/v1/feedback', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(item.data),
        });
        
        if (response.ok) {
          const deleteTx = db.transaction('pending_sync', 'readwrite');
          deleteTx.objectStore('pending_sync').delete(item.id);
        }
      }
    }
  } catch (error) {
    console.error('[SW] Feedback sync failed:', error);
  }
}

// Push notification handler
self.addEventListener('push', (event) => {
  if (!event.data) return;

  try {
    const data = event.data.json();
    
    const options = {
      body: data.body || 'New notification from ShikshaSetu',
      icon: '/icons/icon-192.png',
      badge: '/icons/badge-72.png',
      vibrate: [100, 50, 100],
      data: {
        url: data.url || '/',
        ...data,
      },
      actions: data.actions || [],
    };

    event.waitUntil(
      self.registration.showNotification(data.title || 'ShikshaSetu', options)
    );
  } catch (error) {
    console.error('[SW] Push notification error:', error);
  }
});

// Notification click handler
self.addEventListener('notificationclick', (event) => {
  event.notification.close();

  const url = event.notification.data?.url || '/';

  event.waitUntil(
    self.clients.matchAll({ type: 'window' }).then((clients) => {
      // Check if there's already a window open
      for (const client of clients) {
        if (client.url === url && 'focus' in client) {
          return client.focus();
        }
      }
      // Open new window
      return self.clients.openWindow(url);
    })
  );
});

// IndexedDB helpers
function openDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('shiksha-setu-sw', 1);
    
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
    
    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains('pending_sync')) {
        db.createObjectStore('pending_sync', { keyPath: 'id', autoIncrement: true });
      }
    };
  });
}

function getAllFromStore(store) {
  return new Promise((resolve, reject) => {
    const request = store.getAll();
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
  });
}

// Message handler for communication with main thread
self.addEventListener('message', (event) => {
  const { type, payload } = event.data || {};

  switch (type) {
    case 'SKIP_WAITING':
      self.skipWaiting();
      break;
      
    case 'CACHE_URLS':
      if (payload?.urls) {
        caches.open(CACHE_NAME).then(cache => {
          cache.addAll(payload.urls);
        });
      }
      break;
      
    case 'CLEAR_CACHE':
      caches.delete(CACHE_NAME);
      caches.delete(API_CACHE_NAME);
      break;
      
    case 'GET_CACHE_SIZE':
      getCacheSize().then(size => {
        event.source?.postMessage({ type: 'CACHE_SIZE', payload: size });
      });
      break;
  }
});

// Get total cache size
async function getCacheSize() {
  let totalSize = 0;
  
  const cacheNames = await caches.keys();
  for (const name of cacheNames) {
    const cache = await caches.open(name);
    const keys = await cache.keys();
    
    for (const request of keys) {
      const response = await cache.match(request);
      if (response) {
        const blob = await response.blob();
        totalSize += blob.size;
      }
    }
  }
  
  return totalSize;
}
