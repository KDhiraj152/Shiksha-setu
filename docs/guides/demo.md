# ShikshaSetu - Quick Demo Guide

## 🚀 Quick Start (5 minutes)

### Prerequisites
- Python 3.10+
- PostgreSQL with pgvector extension (or Docker)
- Redis (or Docker)

### Option 1: Automated Setup (Recommended)

```bash
# Make scripts executable
chmod +x demo-start.sh

# Start everything
./demo-start.sh
```

The script will:
1. Create `.env` configuration
2. Start PostgreSQL and Redis (via Docker if not running)
3. Run database migrations
4. Create demo user (username: `demo`, password: `demo123`)
5. Start Celery worker
6. Start FastAPI server

Access the API at: **http://localhost:8000/docs**

### Option 2: Manual Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start services
docker-compose up -d postgres redis

# 3. Run migrations
alembic upgrade head

# 4. Start Celery worker
celery -A backend.tasks.celery_app worker --loglevel=info &

# 5. Start API server
uvicorn backend.api.main:app --reload
```

---

## 📝 Demo Workflow

### 1. Authentication

**Login to get JWT token:**

```bash
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=demo&password=demo123"
```

Response:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer"
}
```

**Use token in subsequent requests:**
```bash
TOKEN="your_access_token_here"
```

---

### 2. Upload Document

**Upload a text file or PDF:**

```bash
curl -X POST "http://localhost:8000/api/v1/content/upload" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@sample.pdf" \
  -F "process_for_qa=true" \
  -F "grade_level=8" \
  -F "subject=Science"
```

Response:
```json
{
  "status": "uploaded",
  "content_id": "123e4567-e89b-12d3-a456-426614174000",
  "filename": "sample.pdf",
  "extracted_text": "The text content...",
  "qa_processing": {
    "enabled": true,
    "task_id": "abc-123",
    "status_url": "/api/v1/status/abc-123"
  }
}
```

**Save the `content_id`** - you'll need it for Q&A!

---

### 3. Check Processing Status

```bash
curl "http://localhost:8000/api/v1/status/abc-123" \
  -H "Authorization: Bearer $TOKEN"
```

Wait until status is `SUCCESS` (usually 10-30 seconds).

---

### 4. Ask Questions About the Document

**Ask a question:**

```bash
curl -X POST "http://localhost:8000/api/v1/qa/ask" \
  -H "Authorization: Bearer $TOKEN" \
  -F "content_id=123e4567-e89b-12d3-a456-426614174000" \
  -F "question=What is photosynthesis?" \
  -F "wait=true"
```

Response:
```json
{
  "answer": "Based on the document: Photosynthesis is the process by which plants convert sunlight into energy...",
  "confidence_score": 0.85,
  "num_context_chunks": 3,
  "context_scores": [0.89, 0.84, 0.82]
}
```

---

### 5. View Q&A History

```bash
curl "http://localhost:8000/api/v1/qa/history/123e4567-e89b-12d3-a456-426614174000" \
  -H "Authorization: Bearer $TOKEN"
```

---

## 🎯 Key Features Demonstrated

### ✅ Document Upload & Processing
- PDF, TXT support
- Automatic text extraction
- Metadata tracking

### ✅ RAG-based Q&A System
- Document chunking (512 chars, 50 char overlap)
- Embedding generation (sentence-transformers)
- Vector similarity search (pgvector with cosine similarity)
- Context-aware answer generation
- Confidence scoring

### ✅ Question History
- Per-document chat history
- Confidence tracking
- User-specific queries

---

## 🔧 Advanced Features

### Translation (Coming Soon)
```bash
# Translate text to Hindi
curl -X POST "http://localhost:8000/api/v1/translate" \
  -H "Authorization: Bearer $TOKEN" \
  -F "text=Hello world" \
  -F "target_language=Hindi"
```

### Text Simplification
```bash
# Simplify for grade 5
curl -X POST "http://localhost:8000/api/v1/simplify" \
  -H "Authorization: Bearer $TOKEN" \
  -F "text=Complex scientific text..." \
  -F "grade_level=5"
```

### Audio Generation (TTS)
```bash
# Generate Hindi audio
curl -X POST "http://localhost:8000/api/v1/audio/generate" \
  -H "Authorization: Bearer $TOKEN" \
  -F "text=नमस्ते" \
  -F "language=Hindi"
```

---

## 📊 API Documentation

Visit **http://localhost:8000/docs** for interactive Swagger UI:
- Test all endpoints directly in browser
- View request/response schemas
- Copy curl commands

---

## 🐛 Troubleshooting

### PostgreSQL Connection Error
```bash
# Check if running
pg_isready -h localhost -p 5432

# Start with Docker
docker run -d --name shiksha-postgres \
  -e POSTGRES_USER=shiksha_user \
  -e POSTGRES_PASSWORD=shiksha_pass \
  -e POSTGRES_DB=shiksha_setu \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

### Redis Connection Error
```bash
# Check if running
redis-cli ping

# Start with Docker
docker run -d --name shiksha-redis -p 6379:6379 redis:7-alpine
```

### Celery Not Processing Tasks
```bash
# Restart Celery worker
pkill -f celery
celery -A backend.tasks.celery_app worker --loglevel=info
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Check Python version
python --version  # Should be 3.10+
```

### Q&A Not Working
1. Verify document is processed: Check task status
2. Ensure pgvector extension is installed:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
3. Check Celery worker logs for errors

---

## 📦 What's Installed

### Core Services
- **FastAPI** - Modern web framework
- **PostgreSQL + pgvector** - Vector database for RAG
- **Redis** - Caching and Celery broker
- **Celery** - Background task processing

### AI/ML Models
- **sentence-transformers/all-MiniLM-L6-v2** - Embeddings (384D)
- Models auto-download on first use to `data/models/`

### Database Schema
- `users` - Authentication
- `processed_content` - Uploaded documents
- `document_chunks` - Text chunks for RAG
- `embeddings` - Vector embeddings (pgvector)
- `chat_history` - Q&A conversations

---

## 🎮 Demo Scenarios

### Scenario 1: Science Textbook Q&A
```bash
# Upload NCERT Science chapter
curl -X POST "http://localhost:8000/api/v1/content/upload" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@ncert_science_chapter5.pdf" \
  -F "process_for_qa=true" \
  -F "grade_level=8" \
  -F "subject=Science"

# Ask questions
curl -X POST "http://localhost:8000/api/v1/qa/ask" \
  -H "Authorization: Bearer $TOKEN" \
  -F "content_id=<your_content_id>" \
  -F "question=What are the parts of a cell?" \
  -F "wait=true"

curl -X POST "http://localhost:8000/api/v1/qa/ask" \
  -H "Authorization: Bearer $TOKEN" \
  -F "content_id=<your_content_id>" \
  -F "question=Explain cell division" \
  -F "wait=true"
```

### Scenario 2: Research Paper Analysis
```bash
# Upload research paper
curl -X POST "http://localhost:8000/api/v1/content/upload" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@research_paper.pdf" \
  -F "process_for_qa=true"

# Ask about methodology, results, conclusions
```

### Scenario 3: Interactive Learning
```bash
# Upload lesson material
# Ask clarifying questions
# Review history to track learning progress
```

---

## 📈 Performance Metrics

### Current Configuration
- **Chunk Size**: 512 characters
- **Chunk Overlap**: 50 characters
- **Top-K Retrieval**: 3 chunks
- **Similarity Threshold**: 0.3 (cosine similarity)
- **Embedding Model**: 384-dimensional vectors

### Expected Performance
- **Upload**: <2 seconds for PDFs up to 10MB
- **Processing**: 10-30 seconds for typical documents
- **Q&A Response**: 2-5 seconds per question
- **Embedding Generation**: ~100ms per chunk

---

## 🔐 Security Notes

For demo purposes:
- Demo user credentials: `demo` / `demo123`
- JWT tokens expire after 30 minutes
- Rate limiting: 10 uploads per minute per user/IP

For production:
- Change all passwords in `.env`
- Use strong JWT secret key (64+ characters)
- Enable HTTPS
- Configure proper CORS origins
- Set up monitoring and logging

---

## 📚 Next Steps

1. **Try the Q&A system** with your own documents
2. **Explore API docs** at http://localhost:8000/docs
3. **Check Phase 2/3 features** in PHASE_2_3_COMPLETE.md
4. **Review architecture** in docs/architecture.md

---

## 🆘 Support

- **Logs**: Check `logs/` directory
- **Database**: psql postgresql://shiksha_user:shiksha_pass@localhost:5432/shiksha_setu
- **Redis CLI**: redis-cli
- **Celery Flower**: Install and run `celery -A backend.tasks.celery_app flower`

---

## 🎉 Success!

You now have a fully functional AI-powered education platform with:
- ✅ Document upload and processing
- ✅ RAG-based question answering
- ✅ Vector similarity search
- ✅ Conversation history tracking
- ✅ JWT authentication
- ✅ Background task processing

**Happy learning! 🚀**
