# Section 5: Data Flow Documentation

## End-to-End Traces

This section details the flow of data through the system I architected for key user interactions.

### Trace 1: User Asks a Question ("Explain Gravity in Hindi")

1.  **User Input (Frontend)**
    *   User types "Explain Gravity" and selects "Hindi" as the target language.
    *   Frontend sends a POST request to `/api/v2/chat/stream`.

2.  **Request Validation (Backend API)**
    *   FastAPI receives the request.
    *   **Auth Middleware**: Validates the JWT token.
    *   **Rate Limiter**: Checks if the user has exceeded their request quota (Redis).
    *   **Input Validation**: Pydantic ensures the message is not empty and within length limits.

3.  **Preprocessing & Safety (Backend Services)**
    *   **Safety Pipeline**: My pipeline scans the input text for banned keywords or malicious intent.
    *   **Intent Recognition**: The system identifies the user's intent (Explanation + Translation).

4.  **Retrieval (RAG Service)**
    *   **Embedding**: The query "Explain Gravity" is sent to the `BGE-M3` model to generate a vector embedding.
    *   **Vector Search**: `pgvector` searches the database for chunks of text semantically similar to the query vector.
    *   **Reranking**: The top 20 results are passed to the `BGE-Reranker` model, which re-scores them based on precise relevance. I select the top 5.

5.  **Context Assembly (Backend Core)**
    *   **Prompt Engineering**: I construct a prompt containing:
        *   System instructions ("You are a helpful tutor...").
        *   Retrieved context chunks.
        *   User's question.
        *   Student Profile data (e.g., "Grade 8 level").
    *   **Token Budgeting**: My **Adaptive Context Allocator** ensures the total prompt length fits within the model's context window (e.g., 4096 tokens).

6.  **Generation (Inference Engine)**
    *   The prompt is sent to the `Qwen2.5-3B` model.
    *   The model generates the explanation in English (the primary reasoning language).

7.  **Translation (Translation Service)**
    *   The generated English text is passed to the `IndicTrans2` model.
    *   The model translates the text into Hindi.

8.  **Response Streaming (Backend -> Frontend)**
    *   The Hindi text is streamed back to the client via Server-Sent Events (SSE).
    *   **Frontend**: The UI updates in real-time as chunks arrive.

### Trace 2: Content Processing (Simplification)

1.  **Upload**: User uploads a complex PDF chapter.
2.  **Ingestion**:
    *   Backend saves the file to `data/uploads`.
    *   **OCR**: If the PDF is scanned images, OCR extracts the text.
3.  **Chunking**: The text is split into manageable chunks (e.g., 500 words).
4.  **Simplification Loop**:
    *   For each chunk, the `Qwen2.5` model is invoked with a "Simplify this" prompt.
    *   I maintain a sliding window of context to ensure continuity between chunks.
5.  **Aggregation**: Simplified chunks are combined into a final document.
6.  **Delivery**: The simplified text is returned to the user.
