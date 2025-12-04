# Section 10: Future Improvements

## Roadmap
This section outlines my strategic direction for Shiksha Setu, focusing on scalability, feature expansion, and technical debt reduction.

## 1. Scalability Upgrades
*   **Horizontal Scaling**: Currently, the system relies on vertical scaling (bigger GPU). My future versions will support a distributed inference cluster (e.g., using Ray Serve) to distribute model loads across multiple nodes.
*   **Database Sharding**: As the user base grows, I will need to shard the PostgreSQL database, particularly the `pgvector` tables which can grow very large.

## 2. Performance Boosts
*   **Speculative Decoding**: I plan to implement speculative decoding for the LLM. This uses a smaller "draft" model to predict tokens, which are then verified by the main model, potentially doubling generation speed.
*   **KV Cache Optimization**: I will implement PagedAttention (vLLM style) to optimize memory usage for the Key-Value cache, allowing for larger batch sizes.

## 3. New Features
*   **Voice-to-Voice Mode**: I want to enable a real-time, full-duplex voice conversation mode. This would require streaming STT (Speech-to-Text) and streaming TTS to minimize latency.
*   **Federated Learning**: I aim to allow the model to learn from user corrections locally on their device, and periodically aggregate these learnings to the central model without sharing raw data.
*   **Graph RAG**: I plan to move beyond simple vector search. I will implement a Knowledge Graph to capture relationships between concepts (e.g., "Newton's Laws" -> *implies* -> "Conservation of Momentum"). This improves reasoning on complex queries.

## 4. Observability Improvements
*   **Distributed Tracing**: I will fully integrate OpenTelemetry to trace requests across the frontend, backend, and database.
*   **Model Monitoring**: I will add dashboards to track model drift, token usage, and latency percentiles (p95, p99).

## 5. Cleaner Abstractions
*   **Plugin System**: I am designing a plugin architecture for the "Tools" available to the LLM. This would allow third-party developers to add new capabilities (e.g., a Wolfram Alpha connector or a Python code interpreter) without modifying the core codebase.
