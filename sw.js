// Service worker: cache the app shell + the TensorFlow/MoveNet model so the
// app works FULLY OFFLINE after the first load. Video is never uploaded.
const CACHE = 'hoop-coach-v4';
const SHELL = [
  './',
  './index.html',
  './styles.css',
  './app.js',
  './manifest.json',
  './icons/icon-180.png',
  './icons/icon-192.png',
  './icons/icon-512.png',
];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (e) => {
  if (e.request.method !== 'GET') return;
  const url = new URL(e.request.url);
  const sameOrigin = url.origin === self.location.origin;
  if (sameOrigin) {
    // App shell: network-first so updates apply as soon as the phone is online,
    // falling back to cache when offline.
    e.respondWith(
      fetch(e.request).then((res) => {
        if (res && res.status === 200) { const c = res.clone(); caches.open(CACHE).then((ch) => ch.put(e.request, c)); }
        return res;
      }).catch(() => caches.match(e.request))
    );
  } else {
    // CDN (TensorFlow + model weights): cache-first — big and immutable, this is
    // what makes the app work fully offline at the gym after the first load.
    e.respondWith(
      caches.match(e.request).then((hit) => hit || fetch(e.request).then((res) => {
        if (res && res.status === 200 && (res.type === 'basic' || res.type === 'cors')) {
          const c = res.clone(); caches.open(CACHE).then((ch) => ch.put(e.request, c));
        }
        return res;
      }).catch(() => hit))
    );
  }
});
