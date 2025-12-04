# Section 7: Model Pipeline Documentation

## Overview
The Model Pipeline is the core intelligence of Shiksha Setu. I orchestrated multiple specialized AI models to deliver accurate, safe, and context-aware responses.

## Model Stack (2025 Optimal Stack)

| Function | Model ID | Backend | Description |
| :--- | :--- | :--- | :--- |
| **Reasoning / Simplification** | `Qwen/Qwen2.5-3B-Instruct` | Transformers / vLLM | I chose this highly capable 3B parameter model, optimized for instruction following and reasoning. |
| **Translation** | `ai4bharat/indictrans2-en-indic-1B` | Transformers | I selected this for state-of-the-art translation for Indian languages. |
| **Embeddings** | `BAAI/bge-m3` | Transformers | Multilingual embeddings with 1024 dimensions. I use it for dense, sparse, and multi-vector retrieval. |
| **Reranking** | `BAAI/bge-reranker-v2-m3` | Transformers | Cross-encoder model that re-scores retrieval results for high precision. |

## RAG Pipeline Details

### 1. Embedding Path
*   **Input**: Text chunks (paragraphs from textbooks).
*   **Process**:
    1.  Text is normalized (whitespace removal, unicode normalization).
    2.  `BGE-M3` generates a 1024-dimensional dense vector.
    3.  Simultaneously, it generates sparse lexical weights (for keyword matching).
*   **Storage**: I store vectors in PostgreSQL using `pgvector` with HNSW indexing.

### 2. Retrieval Logic
*   **Hybrid Search**: I implemented both dense vector search (semantic) and keyword search (lexical).
*   **HNSW Index**: I used Hierarchical Navigable Small World graphs for approximate nearest neighbor search, offering O(log n) performance.

### 3. Reranking & Validation
*   **Initial Retrieval**: Top 20-50 candidates are fetched from the DB.
*   **Reranking**: The `BGE-Reranker` model takes the query and each candidate chunk as a pair and outputs a relevance score (0-1).
*   **Filtering**: I only keep chunks with a score > 0.4.
*   **Semantic Validator**: I added a lightweight check to ensure the retrieved chunks actually contain the answer to the question, reducing hallucinations.

## Hardware Optimization

### Adaptive Context Allocation
I dynamically calculate the available token budget.
*   If the model has a 4k context window and the history uses 1k, I allocate 2.5k for retrieved context and reserve 500 for the generation.

### Quantization
*   **Format**: INT4 (4-bit integer).
*   **Benefit**: This reduces memory usage by ~75% compared to FP16, allowing me to run a 3B model (usually ~6GB VRAM) in ~2GB.

### Cross-Orchestration
My **Self-Optimizer** monitors query success. If a specific topic (e.g., "Quantum Physics") consistently yields low-confidence scores, the system automatically increases the "Top-K" retrieval parameter for that topic in future queries.
