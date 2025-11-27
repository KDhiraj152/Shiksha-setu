# 📖 Usage Guide

How to use ShikshaSetu API.

## Authentication

### Register
```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "password123",
    "full_name": "John Doe"
  }'
```

### Login
```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "password123"
  }'
```

**Response:**
```json
{
  "access_token": "eyJ0eXAi...",
  "token_type": "bearer"
}
```

## Upload Content

```bash
curl -X POST http://localhost:8000/api/v1/content/upload \
  -H "Authorization: Bearer {token}" \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "id": "abc123",
  "file_path": "uploads/document.pdf",
  "filename": "document.pdf",
  "uploaded_at": "2025-11-20T10:30:00Z"
}
```

> **Note:** Copy the `file_path` value from the response to use in the "Process Content" request below.

## Process Content

```bash
curl -X POST http://localhost:8000/api/v1/content/process \
  -H "Authorization: Bearer {token}" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "uploads/document.pdf",
    "grade_level": 8,
    "subject": "Science",
    "target_languages": ["Hindi", "Tamil"],
    "output_format": "both"
  }'
```

**Validation Notes:**
- Text simplification has a maximum length of 5000 characters
- Translation requires different source and target languages
- Grade levels: 1-12
- Supported subjects: General, Science, Mathematics, Social Studies, English, Hindi
```

## Q&A System

### Process Document
```bash
curl -X POST http://localhost:8000/api/v1/qa/process \
  -H "Authorization: Bearer {token}" \
  -d "content_id=123"
```

**Response:**
```json
{
  "task_id": "abc-123-def",
  "status": "processing"
}
```

### Check Task Status
```bash
curl -X GET http://localhost:8000/api/v1/tasks/{task_id} \
  -H "Authorization: Bearer {token}"
```

**Response:**
```json
{
  "task_id": "abc-123-def",
  "status": "completed",
  "result": {...}
}
```

**Note:** The frontend automatically polls task status every 2 seconds (max 30 attempts) for document processing completion.

### Ask Question
```bash
curl -X POST http://localhost:8000/api/v1/qa/ask \
  -H "Authorization: Bearer {token}" \
  -H "Content-Type: application/json" \
  -d '{
    "content_id": "123",
    "question": "What is photosynthesis?"
  }'
```

## Interactive Docs

Visit http://localhost:8000/docs for interactive API documentation.
