# Shiksha Setu

**AI-Powered Multilingual Education Platform for India**

Transform complex educational content into accessible, multilingual learning materials with NCERT-aligned validation and text-to-speech output.

---

## Overview

Shiksha Setu is a production-grade educational content processing system that simplifies textbooks, translates to Indian languages, validates against NCERT standards, and generates audio—all through a unified AI pipeline.

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Text Simplification** | Grade-level adaptation using Llama-3.2-3B-Instruct |
| **Translation** | 10 Indian languages via IndicTrans2-1B |
| **OCR** | Document extraction with GOT-OCR2 (95%+ accuracy on Indian scripts) |
| **Validation** | NCERT curriculum alignment scoring (≥80% threshold) |
| **Text-to-Speech** | Multilingual audio via AI4Bharat Indic-TTS |
| **RAG Q&A** | Intelligent question answering with BGE-M3 embeddings |

### Supported Languages

Hindi • Tamil • Telugu • Bengali • Marathi • Gujarati • Kannada • Malayalam • Punjabi • Odia

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (React)                         │
│              TypeScript • Vite • TailwindCSS • Zustand          │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Backend (FastAPI)                          │
│           REST API • JWT Auth • Rate Limiting • CORS            │
└─────────────────────────────────────────────────────────────────┘
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
┌─────────────────────┐ ┌─────────────────┐ ┌─────────────────────┐
│      PostgreSQL     │ │      Redis      │ │   Celery Workers    │
│   Content Storage   │ │  Cache + Queue  │ │    AI Pipeline      │
└─────────────────────┘ └─────────────────┘ └─────────────────────┘
                                                      │
                                                      ▼
                            ┌─────────────────────────────────────┐
                            │           ML Models                  │
                            │  Llama • IndicTrans2 • GOT-OCR2     │
                            │  Gemma-2 • BGE-M3 • Indic-TTS       │
                            └─────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Frontend** | React 19 • TypeScript • Vite • TailwindCSS • Zustand |
| **Backend** | FastAPI • SQLAlchemy 2.0 • Pydantic |
| **Database** | PostgreSQL 17 • pgvector |
| **Queue** | Redis • Celery |
| **ML/AI** | PyTorch • Transformers • vLLM |
| **Infrastructure** | Docker • Kubernetes • Prometheus • Grafana |

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- Redis 7+
- PostgreSQL 17+ (or Supabase)

### Setup

```bash
git clone https://github.com/KDhiraj152/Siksha-Setu.git
cd shiksha_setu
./setup.sh
```

The setup script:
- Creates Python virtual environment
- Installs backend dependencies
- Installs frontend dependencies
- Generates secure JWT secret
- Initializes database schema
- Creates required directories

### Run

```bash
./start.sh
```

Starts:
- Backend API (port 8000)
- AI Pipeline (Celery workers)
- Frontend (port 5173)

### Stop

```bash
./stop.sh
```

---

## Access Points

| Service | URL |
|---------|-----|
| Frontend | http://localhost:5173 |
| Backend API | http://localhost:8000 |
| API Documentation | http://localhost:8000/docs |
| Prometheus Metrics | http://localhost:8000/metrics |

---

## Project Structure

```
shiksha_setu/
├── setup.sh              # Environment setup
├── start.sh              # Start all services
├── stop.sh               # Stop all services
├── requirements.txt      # Python dependencies
│
├── backend/              # FastAPI application
│   ├── api/              # Routes & endpoints
│   ├── core/             # Config, security, rate limiting
│   ├── models/           # SQLAlchemy models
│   ├── schemas/          # Pydantic schemas
│   ├── services/         # Business logic
│   │   ├── pipeline/     # AI orchestration
│   │   ├── simplify/     # Text simplification
│   │   ├── translate/    # Translation
│   │   ├── speech/       # TTS generation
│   │   └── validate/     # NCERT validation
│   ├── tasks/            # Celery async tasks
│   └── monitoring/       # Metrics & alerting
│
├── frontend/             # React application
│   └── src/
│       ├── components/   # UI components
│       ├── pages/        # Route pages
│       ├── services/     # API client
│       └── store/        # State management
│
├── infrastructure/       # DevOps configs
│   ├── docker/           # Container configs
│   ├── kubernetes/       # K8s manifests
│   └── monitoring/       # Grafana/Prometheus
│
├── tests/                # Test suite
│   ├── unit/
│   ├── integration/
│   └── e2e/
│
├── data/                 # Runtime data
│   ├── models/           # ML model cache
│   ├── uploads/          # User uploads
│   └── audio/            # Generated audio
│
└── docs/                 # Documentation
    ├── backend.md
    ├── frontend.md
    └── ai_pipeline.md
```

---

## Environment Configuration

Key variables in `.env`:

```bash
# Application
ENVIRONMENT=development
DEBUG=true

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/shiksha_setu

# Redis
REDIS_URL=redis://localhost:6379/0

# Security
JWT_SECRET_KEY=<auto-generated>

# ML Models
DEVICE=auto                    # auto | cuda | mps | cpu
USE_QUANTIZATION=true
VLLM_ENABLED=true
```

See `.env.example` for complete configuration.

---

## API Overview

### Authentication
- `POST /api/v1/auth/register` — Create account
- `POST /api/v1/auth/login` — Get tokens
- `POST /api/v1/auth/refresh` — Refresh access token

### Content Processing
- `POST /api/v1/content/process` — Full pipeline processing
- `POST /api/v1/content/simplify` — Simplify text
- `POST /api/v1/content/translate` — Translate text
- `POST /api/v1/content/validate` — Validate against NCERT
- `POST /api/v1/content/tts` — Generate audio

### Q&A
- `POST /api/v1/qa/ask` — Ask questions about content
- `GET /api/v1/qa/history` — Get Q&A history

### System
- `GET /health` — Health check
- `GET /metrics` — Prometheus metrics

---

## Testing

```bash
# All tests
source .venv/bin/activate
pytest tests/

# With coverage
pytest tests/ --cov=backend --cov-report=html

# Frontend tests
cd frontend && npm test
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Redis connection failed | Start Redis: `redis-server` |
| Database connection error | Check `DATABASE_URL` in `.env` |
| Model loading slow | First run downloads models (~10GB) |
| CUDA out of memory | Set `USE_QUANTIZATION=true` |
| Port already in use | Run `./stop.sh` first |

---

## License

MIT License — see [LICENSE](LICENSE)

---

⸻

Created by: **K Dhiraj**  
Email: kdhiraj152@gmail.com  
GitHub: [github.com/KDhiraj152](https://github.com/KDhiraj152)  
LinkedIn: [linkedin.com/in/kdhiraj152](https://linkedin.com/in/kdhiraj152)
