# Section 2: High-Level Architecture Diagram

## System Overview
I designed the following high-level architecture for Shiksha Setu v4.0 to ensure seamless data flow from the user device through the frontend, backend, and into the core AI engine.

```ascii
                                    [ USER DEVICE ]
                                          │
                                   (HTTPS / WSS)
                                          │
+-----------------------------------------▼-----------------------------------------+
|                                 LOAD BALANCER                                     |
|                                (Nginx / Uvicorn)                                  |
+-----------------------------------------┬-----------------------------------------+
                                          │
+-----------------------------------------▼-----------------------------------------+
|                                FRONTEND LAYER                                     |
|                                                                                   |
|  [ React + Vite SPA ]                                                             |
|  │                                                                                |
|  ├── [ Zustand Store ] (State Management: Auth, Chat, Settings)                   |
|  │                                                                                |
|  ├── [ SSE Handler ] (Real-time Text Streaming)                                   |
|  │                                                                                |
|  └── [ Audio Processor ] (Web Audio API: Recording & Playback)                    |
+-----------------------------------------┬-----------------------------------------+
                                          │
                                   (REST / JSON)
                                          │
+-----------------------------------------▼-----------------------------------------+
|                                 BACKEND LAYER                                     |
|                                (FastAPI v2)                                       |
|                                                                                   |
|  +-------------------+    +-----------------------+    +-----------------------+  |
|  |   API GATEWAY     |    |   MIDDLEWARE CHAIN    |    |   TASK ORCHESTRATOR   |  |
|  | - Auth (JWT)      |───▶| - Rate Limiter        |───▶| - Priority Queue      |  |
|  | - Versioning      |    | - Circuit Breaker     |    | - Batch Processor     |  |
|  +-------------------+    | - Age Consent         |    +-----------┬-----------+  |
|                           +-----------------------+                │              |
+-----------------------------------------┬--------------------------│--------------+
                                          │                          │
+-----------------------------------------▼--------------------------▼--------------+
|                                  AI CORE ENGINE                                   |
|                                                                                   |
|  [ MEMORY COORDINATOR ] <───> [ GPU SCHEDULER ] <───> [ MODEL REGISTRY (LRU) ]    |
|           │                           │                          │                |
|           ▼                           ▼                          ▼                |
|  +----------------+       +-----------------------+      +---------------------+  |
|  |  RAG PIPELINE  |       |   INFERENCE ENGINE    |      |   SAFETY PIPELINE   |  |
|  | - BGE-M3 Embed |       | - Qwen2.5 (Simplify)  |      | - Semantic Check    |  |
|  | - HNSW Index   |       | - IndicTrans2 (Trans) |      | - Logical Check     |  |
|  | - BGE Reranker |       | - TTS / STT           |      | - Policy Engine     |  |
|  +----------------+       +-----------------------+      +---------------------+  |
+-----------------------------------------┬-----------------------------------------+
                                          │
+-----------------------------------------▼-----------------------------------------+
|                                  DATA LAYER                                       |
|                                                                                   |
|  [ PostgreSQL + pgvector ]   [ Redis Cache (L2) ]   [ File Storage (MinIO/FS) ]   |
|  (Vectors, Users, Logs)      (Sessions, Models)     (Audio, Uploads)              |
+-----------------------------------------------------------------------------------+
```

## Component Descriptions

1.  **Frontend Layer**: I built a responsive Single Page Application (SPA) that handles user interaction, audio recording, and real-time rendering of AI responses.
2.  **Backend Layer**: This is my orchestration center. It validates requests, manages user sessions, and schedules heavy AI tasks.
3.  **AI Core Engine**: I consider this the "brain" of the system. It manages the lifecycle of AI models, ensuring they are loaded into memory only when needed and unloaded to prevent crashes.
4.  **Data Layer**: I utilized persistent storage for user data, vector embeddings of educational content, and temporary caching for high-speed access.
