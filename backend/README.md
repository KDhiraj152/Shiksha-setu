# Backend API - Shiksha Setu

FastAPI-based backend for Shiksha Setu AI education platform.

## 🚀 Quick Start

```bash
# Navigate to project root
cd ..

# Activate virtual environment
source venv/bin/activate

# Start backend server
uvicorn backend.api.main:app --reload

# Start Celery worker (separate terminal)
celery -A backend.tasks.celery_app worker --loglevel=info
```

**Access**:
- API: http://localhost:8000
- Swagger Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 📁 Directory Structure

```
backend/
├── api/                    # REST API (FastAPI)
│   ├── routes/             # All API endpoints
│   │   ├── auth.py         # Authentication
│   │   ├── content.py      # Content processing
│   │   ├── qa.py           # Q&A endpoints
│   │   ├── streaming.py    # WebSocket streaming
│   │   └── ...
│   ├── main.py             # FastAPI app entry point
│   └── middleware.py       # Request middleware
│
├── core/                   # Core infrastructure
│   ├── config.py           # Configuration (env vars)
│   ├── database.py         # Database setup
│   ├── security.py         # JWT & auth
│   └── cache.py            # Redis
│
├── models/                 # Database models
├── services/               # Business logic & ML
├── tasks/                  # Background jobs (Celery)
├── schemas/                # Pydantic request/response
└── utils/                  # Shared utilities
```

## 🔌 Key API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/auth/login` | POST | User login |
| `/api/v1/auth/register` | POST | User registration |
| `/api/v1/content/upload` | POST | Upload file |
| `/api/v1/content/simplify` | POST | Simplify text |
| `/api/v1/content/translate` | POST | Translate text |
| `/api/v1/qa/ask` | POST | Ask Q&A question |

Full API documentation: http://localhost:8000/docs

## 🧪 Testing

```bash
pytest tests/ -v --cov=backend
```

## 📚 Documentation

- **[API Reference](../docs/reference/api.md)** - All endpoints
- **[Backend Reference](../docs/reference/backend.md)** - Architecture
- **[Deployment Guide](../docs/technical/deployment.md)** - Production setup
- **[Setup Guide](../docs/guides/setup.md)** - Installation steps

---

## 👨‍💻 Author

**K Dhiraj** • [k.dhiraj.srihari@gmail.com](mailto:k.dhiraj.srihari@gmail.com)

*Last updated: November 2025*
