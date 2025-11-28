"""
Service Worker for Offline-First Architecture

Issue: CODE-REVIEW-GPT #4 (CRITICAL)
Problem: No offline support for PS3 requirements

Solution: Progressive Web App (PWA) with:
- Service worker for caching
- IndexedDB for local data storage
- Background sync for when online
- Cache-first strategy for content
"""

// Service Worker Configuration
const CACHE_VERSION = 'shiksha-setu-v1';
const STATIC_CACHE = `${CACHE_VERSION}-static`;
const DYNAMIC_CACHE = `${CACHE_VERSION}-dynamic`;
const CONTENT_CACHE = `${CACHE_VERSION}-content`;

// Files to cache immediately on install
const STATIC_ASSETS = [
    '/',
    '/index.html',
    '/manifest.json',
    '/offline.html',
    '/css/main.css',
    '/js/app.js',
    '/js/db.js',
    '/images/logo.png',
    '/images/offline-icon.png'
];

// API endpoints to cache
const API_CACHE_PATTERNS = [
    /\/api\/v1\/content\/.+/,
    /\/api\/v1\/curriculum\/.+/,
    /\/api\/v1\/subjects\/.+/
];

// =============================================================================
// SERVICE WORKER LIFECYCLE
// =============================================================================

// Install: Cache static assets
self.addEventListener('install', (event) => {
    console.log('[SW] Installing service worker...');
    
    event.waitUntil(
        caches.open(STATIC_CACHE)
            .then(cache => {
                console.log('[SW] Caching static assets');
                return cache.addAll(STATIC_ASSETS);
            })
            .then(() => self.skipWaiting()) // Activate immediately
    );
});

// Activate: Clean up old caches
self.addEventListener('activate', (event) => {
    console.log('[SW] Activating service worker...');
    
    event.waitUntil(
        caches.keys()
            .then(cacheNames => {
                return Promise.all(
                    cacheNames
                        .filter(name => name.startsWith('shiksha-setu-') && name !== STATIC_CACHE)
                        .map(name => {
                            console.log('[SW] Deleting old cache:', name);
                            return caches.delete(name);
                        })
                );
            })
            .then(() => self.clients.claim()) // Take control immediately
    );
});

// =============================================================================
// FETCH STRATEGIES
// =============================================================================

// Fetch: Smart caching strategy
self.addEventListener('fetch', (event) => {
    const { request } = event;
    const url = new URL(request.url);
    
    // Skip cross-origin requests
    if (url.origin !== location.origin && !url.pathname.startsWith('/api')) {
        return;
    }
    
    // Apply different strategies based on request type
    if (request.method !== 'GET') {
        // Non-GET requests: Network only with offline queue
        event.respondWith(networkOnlyWithQueue(request));
    } else if (isAPIRequest(url)) {
        // API requests: Network first, fallback to cache
        event.respondWith(networkFirstStrategy(request));
    } else if (isContentRequest(url)) {
        // Content requests: Cache first, fallback to network
        event.respondWith(cacheFirstStrategy(request));
    } else {
        // Static assets: Cache first
        event.respondWith(cacheFirstStrategy(request));
    }
});

// =============================================================================
// CACHING STRATEGIES
// =============================================================================

// Cache First: Try cache, fallback to network
async function cacheFirstStrategy(request) {
    try {
        const cachedResponse = await caches.match(request);
        if (cachedResponse) {
            console.log('[SW] Serving from cache:', request.url);
            return cachedResponse;
        }
        
        console.log('[SW] Fetching from network:', request.url);
        const networkResponse = await fetch(request);
        
        // Cache successful responses
        if (networkResponse.ok) {
            const cache = await caches.open(DYNAMIC_CACHE);
            cache.put(request, networkResponse.clone());
        }
        
        return networkResponse;
    } catch (error) {
        console.error('[SW] Fetch failed:', error);
        
        // Return offline page for navigation requests
        if (request.mode === 'navigate') {
            return caches.match('/offline.html');
        }
        
        // Return offline fallback
        return new Response('Offline - Content unavailable', {
            status: 503,
            statusText: 'Service Unavailable',
            headers: new Headers({
                'Content-Type': 'text/plain'
            })
        });
    }
}

// Network First: Try network, fallback to cache
async function networkFirstStrategy(request) {
    try {
        console.log('[SW] Fetching from network:', request.url);
        const networkResponse = await fetch(request);
        
        // Cache successful responses
        if (networkResponse.ok) {
            const cache = await caches.open(CONTENT_CACHE);
            cache.put(request, networkResponse.clone());
        }
        
        return networkResponse;
    } catch (error) {
        console.log('[SW] Network failed, trying cache:', request.url);
        const cachedResponse = await caches.match(request);
        
        if (cachedResponse) {
            console.log('[SW] Serving from cache:', request.url);
            return cachedResponse;
        }
        
        // Return offline response
        return new Response(JSON.stringify({
            error: 'Offline',
            message: 'Content unavailable offline'
        }), {
            status: 503,
            headers: { 'Content-Type': 'application/json' }
        });
    }
}

// Network Only with Background Sync Queue
async function networkOnlyWithQueue(request) {
    try {
        return await fetch(request);
    } catch (error) {
        console.log('[SW] Network failed, queueing request');
        
        // Queue for background sync
        if ('sync' in self.registration) {
            const requestData = {
                url: request.url,
                method: request.method,
                headers: [...request.headers.entries()],
                body: await request.clone().text(),
                timestamp: Date.now()
            };
            
            // Store in IndexedDB
            await queueRequest(requestData);
            
            // Register background sync
            await self.registration.sync.register('sync-requests');
        }
        
        return new Response(JSON.stringify({
            queued: true,
            message: 'Request queued for sync when online'
        }), {
            status: 202,
            headers: { 'Content-Type': 'application/json' }
        });
    }
}

// =============================================================================
// BACKGROUND SYNC
// =============================================================================

self.addEventListener('sync', (event) => {
    console.log('[SW] Background sync triggered:', event.tag);
    
    if (event.tag === 'sync-requests') {
        event.waitUntil(syncQueuedRequests());
    } else if (event.tag === 'sync-content') {
        event.waitUntil(syncContentUpdates());
    }
});

async function syncQueuedRequests() {
    const requests = await getQueuedRequests();
    console.log(`[SW] Syncing ${requests.length} queued requests`);
    
    for (const reqData of requests) {
        try {
            const response = await fetch(reqData.url, {
                method: reqData.method,
                headers: reqData.headers,
                body: reqData.body
            });
            
            if (response.ok) {
                console.log('[SW] Synced request:', reqData.url);
                await removeQueuedRequest(reqData.timestamp);
            }
        } catch (error) {
            console.error('[SW] Sync failed for:', reqData.url, error);
        }
    }
}

async function syncContentUpdates() {
    console.log('[SW] Syncing content updates...');
    
    try {
        // Fetch latest content list
        const response = await fetch('/api/v1/content/updates');
        const updates = await response.json();
        
        // Update cache with new content
        const cache = await caches.open(CONTENT_CACHE);
        for (const item of updates) {
            await cache.add(item.url);
        }
        
        console.log('[SW] Content sync complete');
    } catch (error) {
        console.error('[SW] Content sync failed:', error);
    }
}

// =============================================================================
// INDEXEDDB HELPERS
// =============================================================================

const DB_NAME = 'shiksha-setu-offline';
const DB_VERSION = 1;

function openDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(DB_NAME, DB_VERSION);
        
        request.onerror = () => reject(request.error);
        request.onsuccess = () => resolve(request.result);
        
        request.onupgradeneeded = (event) => {
            const db = event.target.result;
            
            // Create stores if they don't exist
            if (!db.objectStoreNames.contains('sync-queue')) {
                db.createObjectStore('sync-queue', { keyPath: 'timestamp' });
            }
            if (!db.objectStoreNames.contains('content-cache')) {
                db.createObjectStore('content-cache', { keyPath: 'id' });
            }
        };
    });
}

async function queueRequest(requestData) {
    const db = await openDB();
    const tx = db.transaction('sync-queue', 'readwrite');
    const store = tx.objectStore('sync-queue');
    await store.add(requestData);
}

async function getQueuedRequests() {
    const db = await openDB();
    const tx = db.transaction('sync-queue', 'readonly');
    const store = tx.objectStore('sync-queue');
    return new Promise((resolve, reject) => {
        const request = store.getAll();
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error);
    });
}

async function removeQueuedRequest(timestamp) {
    const db = await openDB();
    const tx = db.transaction('sync-queue', 'readwrite');
    const store = tx.objectStore('sync-queue');
    await store.delete(timestamp);
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

function isAPIRequest(url) {
    return url.pathname.startsWith('/api/');
}

function isContentRequest(url) {
    return API_CACHE_PATTERNS.some(pattern => pattern.test(url.pathname));
}

// =============================================================================
// PUSH NOTIFICATIONS (Future Enhancement)
// =============================================================================

self.addEventListener('push', (event) => {
    console.log('[SW] Push notification received');
    
    const data = event.data ? event.data.json() : {};
    const title = data.title || 'ShikshaSetu';
    const options = {
        body: data.body || 'New content available',
        icon: '/images/logo.png',
        badge: '/images/badge.png',
        data: data.url
    };
    
    event.waitUntil(
        self.registration.showNotification(title, options)
    );
});

self.addEventListener('notificationclick', (event) => {
    console.log('[SW] Notification clicked');
    event.notification.close();
    
    if (event.notification.data) {
        event.waitUntil(
            clients.openWindow(event.notification.data)
        );
    }
});

console.log('[SW] Service worker loaded');
