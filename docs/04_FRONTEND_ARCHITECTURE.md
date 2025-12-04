# Section 4: Frontend Architecture

## Overview
I built the frontend as a modern, responsive Single Page Application (SPA) using **React**, **Vite**, and **TypeScript**. I prioritized performance, accessibility, and a seamless user experience, particularly for real-time AI interactions.

## Technology Stack
-   **Framework**: React 18
-   **Build Tool**: Vite (for fast HMR and optimized builds)
-   **Language**: TypeScript (for type safety)
-   **Styling**: Tailwind CSS (utility-first styling) + shadcn/ui (component library)
-   **State Management**: Zustand
-   **Routing**: React Router

## Directory Structure (`frontend/src/`)

-   **`api/`**: I created typed API clients here. These functions map directly to my backend v2 endpoints, handling request serialization and response parsing.
-   **`components/`**: Reusable UI components.
    -   `ui/`: Atomic components (buttons, inputs, cards).
    -   `chat/`: Complex chat interfaces (message bubbles, input areas).
-   **`store/`**: Global state management.
-   **`hooks/`**: Custom React hooks I wrote for logic reuse (e.g., `useAudioRecorder`, `useSSE`).
-   **`pages/`**: Top-level route components (Home, Chat, Profile).

## State Management (Zustand)

I chose **Zustand** for its simplicity and performance.

### 1. Auth Store (`useAuthStore`)
-   Manages user authentication state (User object, Access Token, Refresh Token).
-   I used `persist` middleware to save state to `localStorage`.
-   **Security**: I implemented a `syncAuthState` mechanism that listens for tab visibility changes. This ensures that if a user logs out in one tab, other tabs reflect this state immediately.

### 2. Chat Store (`useChatStore`)
-   Manages the active conversation list and current message history.
-   Handles optimistic UI updates: When a user sends a message, I make it appear immediately in the UI before the server confirms receipt.

## Real-Time Features

### Server-Sent Events (SSE)
For chat responses, I used SSE instead of WebSockets or polling.
-   **Why SSE?**: I found it simpler to implement for unidirectional data flow (Server -> Client) and it works well with standard HTTP/2.
-   **Implementation**: My `useSSE` hook opens a connection to `/api/v2/chat/stream`. As chunks of text arrive, they are appended to the current message in the store, creating a "typing" effect.

### Audio Handling
-   **Recording**: I used the browser's `MediaRecorder` API to capture user audio.
-   **Playback**: I handled binary audio data blobs returned from the TTS endpoint. I managed a playback queue to ensure smooth audio delivery.

## Error Handling
-   **Global Error Boundary**: I set up a boundary to catch runtime errors in the React tree and display a user-friendly fallback UI.
-   **Toast Notifications**: I ensured that API errors (e.g., 401 Unauthorized, 500 Server Error) trigger toast notifications to inform the user without breaking the application flow.
