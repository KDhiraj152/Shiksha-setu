# Section 6: API Documentation

**Base URL**: `/api/v2`

## 1. Chat & Generation

### `POST /chat/stream`
Initiates a streaming chat session with the AI.

*   **Auth**: Required (Bearer Token)
*   **Request Body**:
    ```json
    {
      "message": "What is photosynthesis?",
      "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
      "language": "en",
      "subject": "biology"
    }
    ```
*   **Response**: `text/event-stream`
    *   Stream of JSON objects: `data: {"token": "Photo", "id": "..."}`

### `POST /chat/message`
Non-streaming chat endpoint (for simple queries).

*   **Response**:
    ```json
    {
      "message_id": "...",
      "response": "Photosynthesis is...",
      "sources": ["Chapter 5 - Biology"],
      "processing_time_ms": 450
    }
    ```

## 2. Content Processing

### `POST /content/process`
Process raw text for simplification, translation, or audio generation.

*   **Request Body**:
    ```json
    {
      "text": "Complex academic text...",
      "simplify": true,
      "translate": true,
      "target_language": "hi",
      "generate_audio": false
    }
    ```
*   **Response**:
    ```json
    {
      "request_id": "...",
      "original_text": "...",
      "simplified_text": "Simple text...",
      "translated_text": "Translated text...",
      "validation_score": 0.95
    }
    ```

## 3. Authentication

### `POST /auth/login`
*   **Request Body**: `OAuth2PasswordRequestForm` (username, password)
*   **Response**:
    ```json
    {
      "access_token": "eyJhbG...",
      "token_type": "bearer",
      "refresh_token": "..."
    }
    ```

### `POST /auth/refresh`
Refreshes an expired access token.

## 4. User Progress

### `GET /progress/{user_id}`
Retrieves the learning statistics for a user.

*   **Response**:
    ```json
    {
      "user_id": "...",
      "topics_mastered": 12,
      "weak_areas": ["Calculus", "Organic Chemistry"],
      "study_streak_days": 5
    }
    ```

## Common Error Codes
*   **400 Bad Request**: Invalid input data.
*   **401 Unauthorized**: Missing or invalid JWT token.
*   **429 Too Many Requests**: Rate limit exceeded.
*   **503 Service Unavailable**: AI models are currently overloaded or loading.
