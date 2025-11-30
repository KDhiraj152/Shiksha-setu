# Backend Architecture

FastAPI-based REST API with SQLAlchemy ORM, Celery task queue, and Redis caching.

---

## Structure

```
backend/
├── api/                    # HTTP layer
│   ├── main.py             # FastAPI application entry
│   ├── middleware.py       # Request/response middleware
│   ├── metrics.py          # Prometheus endpoint
│   └── routes/             # API endpoints
│       ├── auth.py         # Authentication
│       ├── content.py      # Content processing
│       ├── qa.py           # Question answering
│       ├── health.py       # Health checks
│       ├── streaming.py    # WebSocket streaming
│       ├── admin.py        # Admin operations
│       └── experiments.py  # A/B testing
│
├── core/                   # Configuration & utilities
│   ├── config.py           # Settings management
│   ├── security.py         # JWT & hashing
│   ├── rate_limiter.py     # Rate limiting
│   ├── exceptions.py       # Custom exceptions
│   └── graceful_degradation.py  # Circuit breaker
│
├── models/                 # Database models
│   ├── auth.py             # User, APIKey, Token models
│   ├── content.py          # ProcessedContent, NCERTStandard
│   ├── progress.py         # UserProgress
│   └── rag.py              # RAGDocument, RAGQuery
│
├── schemas/                # Pydantic schemas
│   ├── auth.py             # Auth request/response
│   ├── content.py          # Content schemas
│   └── qa.py               # Q&A schemas
│
├── services/               # Business logic
│   ├── pipeline/           # AI orchestration
│   ├── simplify/           # Text simplification
│   ├── translate/          # Translation
│   ├── speech/             # TTS generation
│   ├── validate/           # NCERT validation
│   ├── ocr.py              # Document OCR
│   ├── rag.py              # RAG Q&A
│   └── storage.py          # File storage
│
├── tasks/                  # Celery tasks
│   ├── celery_app.py       # Celery configuration
│   ├── pipeline_tasks.py   # Pipeline tasks
│   ├── qa_tasks.py         # Q&A tasks
│   └── ocr_tasks.py        # OCR tasks
│
├── monitoring/             # Observability
│   ├── metrics.py          # Prometheus metrics
│   └── oom_alerts.py       # Memory alerting
│
├── cache/                  # Caching layer
│   └── redis_cache.py      # Redis client
│
├── database.py             # Database connection
└── utils/                  # Utilities
    ├── auth.py             # Auth helpers
    └── logging.py          # Logging config
```

---

## Application Entry

`backend/api/main.py` initializes FastAPI with:

```python
app = FastAPI(
    title="ShikshaSetu AI Education API",
    version="3.0.0"
)

# Middleware stack (reverse execution order)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestTimingMiddleware)
app.add_middleware(CORSMiddleware, ...)
app.add_middleware(RateLimitMiddleware, ...)

# Routes
app.include_router(auth_router)
app.include_router(content_router)
app.include_router(qa_router)
app.include_router(streaming_router)
```

---

## Database Models

### User (`models/auth.py`)

```python
class User(Base):
    __tablename__ = 'users'
    
    id: UUID
    email: str (unique, indexed)
    hashed_password: str
    full_name: str
    organization: str
    role: str  # 'user' | 'educator' | 'admin'
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login: datetime
```

### ProcessedContent (`models/content.py`)

```python
class ProcessedContent(Base):
    __tablename__ = 'processed_content'
    
    id: UUID
    original_text: str
    simplified_text: str
    translated_text: str
    language: str (indexed)
    grade_level: int (indexed)
    subject: str (indexed)
    audio_file_path: str
    ncert_alignment_score: float
    user_id: UUID (indexed)
    created_at: datetime
    metadata: JSONB
```

### NCERTStandard (`models/content.py`)

```python
class NCERTStandard(Base):
    __tablename__ = 'ncert_standards'
    
    id: UUID
    grade_level: int
    subject: str
    topic: str
    learning_objectives: Array[str]
    keywords: Array[str]
```

---

## Authentication

JWT-based authentication with refresh tokens.

### Token Structure

```python
{
    "sub": "user_id",
    "email": "user@example.com",
    "role": "user",
    "exp": 1234567890,
    "jti": "unique_token_id"
}
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/auth/register` | Create account |
| POST | `/api/v1/auth/login` | Get access + refresh tokens |
| POST | `/api/v1/auth/refresh` | Refresh access token |
| POST | `/api/v1/auth/logout` | Invalidate tokens |

### Password Security

- Bcrypt hashing (12 rounds)
- Minimum 8 characters
- Token blacklisting on logout

---

## Rate Limiting

Redis-backed sliding window rate limiter.

```python
# Configuration
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000

# Headers returned
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1234567890
```

---

## Content Processing Endpoints

### Full Pipeline

```http
POST /api/v1/content/process
Content-Type: application/json

{
    "text": "Complex educational content...",
    "target_language": "Hindi",
    "grade_level": 8,
    "subject": "Science",
    "output_format": "both"  // "text" | "audio" | "both"
}
```

Response:
```json
{
    "id": "uuid",
    "original_text": "...",
    "simplified_text": "...",
    "translated_text": "...",
    "audio_file_path": "/storage/audio/uuid.mp3",
    "ncert_alignment_score": 0.87,
    "validation_status": "passed"
}
```

### Individual Operations

| Endpoint | Purpose |
|----------|---------|
| `POST /api/v1/content/simplify` | Simplify text for grade level |
| `POST /api/v1/content/translate` | Translate to target language |
| `POST /api/v1/content/validate` | Check NCERT alignment |
| `POST /api/v1/content/tts` | Generate audio |

---

## Celery Task Queue

### Configuration (`tasks/celery_app.py`)

```python
celery_app = Celery(
    'shiksha_setu',
    broker='redis://localhost:6379/1',
    backend='redis://localhost:6379/1'
)

# Task queues
task_queues = (
    Queue('default'),
    Queue('pipeline'),
    Queue('ocr'),
    Queue('ml_gpu'),
    Queue('ml_cpu'),
)

# Settings
task_acks_late = True
task_reject_on_worker_lost = True
task_time_limit = 1800  # 30 minutes
worker_prefetch_multiplier = 1
worker_max_tasks_per_child = 50
```

### Task Types

| Queue | Tasks |
|-------|-------|
| `pipeline` | Content processing, translation |
| `ocr` | Document extraction |
| `ml_gpu` | GPU-intensive inference |
| `ml_cpu` | CPU operations |

### Running Workers

```bash
# Default worker
celery -A backend.tasks.celery_app worker --queues=default,pipeline

# GPU worker
celery -A backend.tasks.celery_app worker --queues=ml_gpu --concurrency=1
```

---

## Monitoring

### Prometheus Metrics (`monitoring/metrics.py`)

```python
# Inference latency histogram
ssetu_inference_latency_seconds{model, task}

# Request counters
ssetu_requests_total{endpoint, method, status}

# Memory gauges
ssetu_memory_usage_bytes{type}

# Queue gauges
ssetu_queue_length{queue_name}
```

### Health Check

```http
GET /health

{
    "status": "healthy",
    "database": "connected",
    "redis": "connected",
    "celery": "available"
}
```

---

## Error Handling

### Custom Exceptions

```python
class ShikshaSetuException(Exception):
    status_code: int
    error_code: str
    message: str

class ValidationError(ShikshaSetuException):
    status_code = 400

class AuthenticationError(ShikshaSetuException):
    status_code = 401

class RateLimitExceeded(ShikshaSetuException):
    status_code = 429
```

### Response Format

```json
{
    "error": {
        "code": "VALIDATION_ERROR",
        "message": "Grade level must be between 5 and 12",
        "details": {...}
    }
}
```

---

## Circuit Breaker

Graceful degradation for external services (`core/graceful_degradation.py`).

```python
@circuit_breaker(
    failure_threshold=3,
    recovery_timeout=60
)
async def call_external_service():
    ...
```

States:
- **CLOSED**: Normal operation
- **OPEN**: Requests fail fast (after 3 failures in 1 minute)
- **HALF_OPEN**: Test requests after timeout

---

## Database Connection

### Configuration (`database.py`)

```python
engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600,
    pool_timeout=30
)
```

### Session Management

```python
def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Usage in routes
@router.get("/items")
def get_items(db: Session = Depends(get_db)):
    return db.query(Item).all()
```

---

## Configuration (`core/config.py`)

### Model Stack

| Model | Purpose | Default |
|-------|---------|---------|
| Simplification | Llama-3.2-3B-Instruct | `meta-llama/Llama-3.2-3B-Instruct` |
| Translation | IndicTrans2-1B | `ai4bharat/indictrans2-en-indic-1B` |
| Embeddings | BGE-M3 | `BAAI/bge-m3` |
| Reranker | BGE-Reranker-v2 | `BAAI/bge-reranker-v2-m3` |
| OCR | GOT-OCR2 | `ucaslcl/GOT-OCR2_0` |
| Validation | Gemma-2-2B | `google/gemma-2-2b-it` |
| TTS | Indic-TTS | `ai4bharat/indic-tts` |

### Environment Variables

```bash
# Core
ENVIRONMENT=development
DEBUG=true
DEVICE=auto  # auto | cuda | mps | cpu

# Optimization
USE_QUANTIZATION=true
QUANTIZATION_TYPE=int4
USE_FLASH_ATTENTION=true

# vLLM
VLLM_ENABLED=true
VLLM_HOST=localhost
VLLM_PORT=8001
VLLM_GPU_MEMORY_UTILIZATION=0.90
```

---

⸻

Created by: **K Dhiraj**  
Email: kdhiraj152@gmail.com  
GitHub: [github.com/KDhiraj152](https://github.com/KDhiraj152)  
LinkedIn: [linkedin.com/in/kdhiraj152](https://linkedin.com/in/kdhiraj152)
