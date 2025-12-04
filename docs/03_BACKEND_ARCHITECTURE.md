# Section 3: Backend Architecture

## Overview
I built the backend on **FastAPI**, choosing it for its high performance (Starlette-based), native async support, and automatic OpenAPI documentation generation. I designed it to be modular, scalable, and hardware-aware.

## Directory Structure & Modules

### 1. API Layer (`backend/api/`)
-   **`main.py`**: My entry point. I configured the FastAPI app here, along with lifespan events (startup/shutdown) and global exception handlers.
-   **`routes/v2/`**: I placed the actual endpoint logic here.
    -   `chat.py`: Handles chat sessions, message persistence, and SSE streaming.
    -   `content.py`: Manages content processing tasks like simplification, translation, and TTS.
    -   `auth.py`: Handles user registration, login, and JWT token issuance.
-   **`middleware.py`**: I wrote custom middleware for request logging, timing, and security headers.

### 2. Core Infrastructure (`backend/core/`)
I used this module to handle low-level system operations.
-   **`config.py`**: I implemented centralized configuration using Pydantic `BaseSettings`. It loads environment variables and defines defaults for model paths and hardware settings.
-   **`model_manager.py`**: Here I implemented the **Model Registry**. I used an LRU (Least Recently Used) strategy to manage loaded models. If memory pressure is high, my system unloads the least recently used model to make space for new ones.
-   **`optimized/`**: I placed my hardware-specific optimizations here.
    -   `memory_coordinator.py`: The **Global Memory Coordinator**. I designed this to act as a semaphore for system RAM and VRAM, preventing Out-Of-Memory (OOM) errors during concurrent model loads.
    -   `device_router.py`: This detects available hardware (Apple Silicon MPS, NVIDIA CUDA, CPU) and routes tensor operations accordingly.

### 3. Services Layer (`backend/services/`)
I separated the business logic from the HTTP layer.
-   **`rag.py`**: My Retrieval-Augmented Generation service. It handles embedding generation (`BGE-M3`), vector search, and reranking (`BGE-Reranker`).
-   **`student_profile.py`**: Manages user profiles, tracking learning progress and preferences.
-   **`safety/`**: The safety pipeline implementation I wrote to ensure content compliance.

## Key Architectural Patterns

### Task Queues & Parallelism
While FastAPI handles concurrent HTTP requests via `asyncio`, I managed CPU-bound tasks (like model inference) carefully.
-   **Batching**: I grouped incoming inference requests into batches to maximize GPU throughput.
-   **Non-blocking I/O**: I ensured database queries and external API calls are awaited asynchronously, keeping the event loop free.

### Memory Management System
I implemented a two-tier memory architecture:
1.  **Short-Term (L1/L2)**:
    -   **L1 (In-Process)**: Python dictionaries for immediate session context.
    -   **L2 (Redis)**: Shared cache for rate limiting, session storage, and frequently accessed RAG queries (TTL 5 mins).
2.  **Long-Term**:
    -   **PostgreSQL**: Stores user data, chat history, and structured content.
    -   **pgvector**: Stores high-dimensional vector embeddings for semantic search.

### Dynamic Fallback Behavior
To ensure reliability, I implemented circuit breakers and fallbacks:
-   If the GPU is overloaded or overheated, my **Device Router** can temporarily route tasks to the CPU.
-   If the primary model fails to load, the system can fall back to a smaller, quantized version or a different backend (e.g., `transformers` vs `ctranslate2`).
