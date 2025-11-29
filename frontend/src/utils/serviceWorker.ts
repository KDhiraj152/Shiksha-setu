/**
 * Service Worker Registration and PWA Utilities
 * 
 * Handles:
 * - Service worker registration
 * - Update notifications
 * - Offline status detection
 * - Push notification subscription
 */

export interface ServiceWorkerConfig {
  onSuccess?: (registration: ServiceWorkerRegistration) => void;
  onUpdate?: (registration: ServiceWorkerRegistration) => void;
  onOffline?: () => void;
  onOnline?: () => void;
}

// Check if service workers are supported
export function isServiceWorkerSupported(): boolean {
  return 'serviceWorker' in navigator;
}

// Register service worker
export async function registerServiceWorker(
  config?: ServiceWorkerConfig
): Promise<ServiceWorkerRegistration | undefined> {
  if (!isServiceWorkerSupported()) {
    console.log('[PWA] Service workers not supported');
    return undefined;
  }

  // Only register in production or if explicitly enabled
  if (import.meta.env.DEV && !import.meta.env.VITE_ENABLE_SW) {
    console.log('[PWA] Service worker disabled in development');
    return undefined;
  }

  try {
    const registration = await navigator.serviceWorker.register('/sw.js', {
      scope: '/',
    });

    console.log('[PWA] Service worker registered:', registration.scope);

    // Handle updates
    registration.addEventListener('updatefound', () => {
      const newWorker = registration.installing;
      if (!newWorker) return;

      newWorker.addEventListener('statechange', () => {
        if (newWorker.state === 'installed') {
          if (navigator.serviceWorker.controller) {
            // New update available
            console.log('[PWA] New content available');
            config?.onUpdate?.(registration);
          } else {
            // Content cached for offline use
            console.log('[PWA] Content cached for offline use');
            config?.onSuccess?.(registration);
          }
        }
      });
    });

    // Check for updates periodically
    setInterval(() => {
      registration.update();
    }, 60 * 60 * 1000); // Every hour

    return registration;
  } catch (error) {
    console.error('[PWA] Service worker registration failed:', error);
    return undefined;
  }
}

// Unregister service worker
export async function unregisterServiceWorker(): Promise<boolean> {
  if (!isServiceWorkerSupported()) return false;

  try {
    const registration = await navigator.serviceWorker.ready;
    const result = await registration.unregister();
    console.log('[PWA] Service worker unregistered:', result);
    return result;
  } catch (error) {
    console.error('[PWA] Service worker unregistration failed:', error);
    return false;
  }
}

// Skip waiting and activate new service worker
export async function skipWaiting(): Promise<void> {
  const registration = await navigator.serviceWorker.ready;
  registration.waiting?.postMessage({ type: 'SKIP_WAITING' });
}

// Check online status
export function isOnline(): boolean {
  return navigator.onLine;
}

// Setup online/offline listeners
export function setupOnlineStatusListeners(config?: ServiceWorkerConfig): () => void {
  const handleOnline = () => {
    console.log('[PWA] Online');
    config?.onOnline?.();
  };

  const handleOffline = () => {
    console.log('[PWA] Offline');
    config?.onOffline?.();
  };

  window.addEventListener('online', handleOnline);
  window.addEventListener('offline', handleOffline);

  // Return cleanup function
  return () => {
    window.removeEventListener('online', handleOnline);
    window.removeEventListener('offline', handleOffline);
  };
}

// Push notification support
export async function requestNotificationPermission(): Promise<NotificationPermission> {
  if (!('Notification' in window)) {
    console.log('[PWA] Notifications not supported');
    return 'denied';
  }

  if (Notification.permission === 'granted') {
    return 'granted';
  }

  const permission = await Notification.requestPermission();
  console.log('[PWA] Notification permission:', permission);
  return permission;
}

// Subscribe to push notifications
export async function subscribeToPushNotifications(
  vapidPublicKey: string
): Promise<PushSubscription | null> {
  if (!isServiceWorkerSupported() || !('PushManager' in window)) {
    console.log('[PWA] Push notifications not supported');
    return null;
  }

  try {
    const registration = await navigator.serviceWorker.ready;
    
    // Check existing subscription
    let subscription = await registration.pushManager.getSubscription();
    
    if (!subscription) {
      // Subscribe
      subscription = await registration.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: urlBase64ToUint8Array(vapidPublicKey),
      });
      
      console.log('[PWA] Push subscription created');
      
      // Send subscription to server
      await fetch('/api/v1/notifications/subscribe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(subscription.toJSON()),
      });
    }
    
    return subscription;
  } catch (error) {
    console.error('[PWA] Push subscription failed:', error);
    return null;
  }
}

// Unsubscribe from push notifications
export async function unsubscribeFromPushNotifications(): Promise<boolean> {
  if (!isServiceWorkerSupported()) return false;

  try {
    const registration = await navigator.serviceWorker.ready;
    const subscription = await registration.pushManager.getSubscription();
    
    if (subscription) {
      await subscription.unsubscribe();
      
      // Notify server
      await fetch('/api/v1/notifications/unsubscribe', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ endpoint: subscription.endpoint }),
      });
      
      console.log('[PWA] Push unsubscribed');
      return true;
    }
    
    return false;
  } catch (error) {
    console.error('[PWA] Push unsubscription failed:', error);
    return false;
  }
}

// Helper to convert VAPID key
function urlBase64ToUint8Array(base64String: string): Uint8Array<ArrayBuffer> {
  const padding = '='.repeat((4 - (base64String.length % 4)) % 4);
  const base64 = (base64String + padding)
    .replace(/-/g, '+')
    .replace(/_/g, '/');

  const rawData = window.atob(base64);
  const buffer = new ArrayBuffer(rawData.length);
  const outputArray = new Uint8Array(buffer);

  for (let i = 0; i < rawData.length; ++i) {
    outputArray[i] = rawData.charCodeAt(i);
  }

  return outputArray;
}

// Get cache size
export async function getCacheSize(): Promise<number> {
  if (!isServiceWorkerSupported()) return 0;

  return new Promise((resolve) => {
    navigator.serviceWorker.controller?.postMessage({ type: 'GET_CACHE_SIZE' });
    
    const handler = (event: MessageEvent) => {
      if (event.data?.type === 'CACHE_SIZE') {
        navigator.serviceWorker.removeEventListener('message', handler);
        resolve(event.data.payload || 0);
      }
    };
    
    navigator.serviceWorker.addEventListener('message', handler);
    
    // Timeout after 5 seconds
    setTimeout(() => resolve(0), 5000);
  });
}

// Clear all caches
export async function clearCache(): Promise<void> {
  if (!isServiceWorkerSupported()) return;
  
  navigator.serviceWorker.controller?.postMessage({ type: 'CLEAR_CACHE' });
  
  // Also clear via Cache API directly
  const cacheNames = await caches.keys();
  await Promise.all(cacheNames.map(name => caches.delete(name)));
  
  console.log('[PWA] Cache cleared');
}

// Install prompt handling
let deferredPrompt: BeforeInstallPromptEvent | null = null;

interface BeforeInstallPromptEvent extends Event {
  prompt(): Promise<void>;
  userChoice: Promise<{ outcome: 'accepted' | 'dismissed' }>;
}

export function setupInstallPrompt(
  onInstallAvailable?: () => void
): () => void {
  const handler = (e: Event) => {
    e.preventDefault();
    deferredPrompt = e as BeforeInstallPromptEvent;
    console.log('[PWA] Install prompt available');
    onInstallAvailable?.();
  };

  window.addEventListener('beforeinstallprompt', handler);

  return () => {
    window.removeEventListener('beforeinstallprompt', handler);
  };
}

export async function promptInstall(): Promise<boolean> {
  if (!deferredPrompt) {
    console.log('[PWA] No install prompt available');
    return false;
  }

  deferredPrompt.prompt();
  const { outcome } = await deferredPrompt.userChoice;
  
  console.log('[PWA] Install prompt outcome:', outcome);
  deferredPrompt = null;
  
  return outcome === 'accepted';
}

export function canInstall(): boolean {
  return deferredPrompt !== null;
}

// Check if app is installed (standalone mode)
export function isInstalled(): boolean {
  return window.matchMedia('(display-mode: standalone)').matches ||
         (window.navigator as any).standalone === true;
}
