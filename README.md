# 🎓 Shiksha Setu

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19.0-61dafb?logo=react)](https://react.dev/)
[![Python](https://img.shields.io/badge/Python-3.11-3776ab?logo=python)](https://python.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.9-3178c6?logo=typescript)](https://typescriptlang.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**AI-powered education platform** for content simplification, multilingual translation, and intelligent Q&A. Built for Indian schools with NCERT curriculum alignment.

---

## ✨ Features

- **🤖 AI Content Processing**: Grade-level simplification (5-12), multilingual translation (10+ Indian languages)
- **🎯 NCERT Validation**: Automatic curriculum alignment scoring and standards mapping
- **💬 RAG Q&A System**: Intelligent document-based question answering with context retrieval
- **🗣️ Text-to-Speech**: Multilingual audio generation for accessibility
- **🔐 Enterprise Security**: JWT auth, RBAC, rate limiting, security headers
- **⚡ Performance Optimized**: Dynamic quantization (FP16/INT8/INT4), lazy loading, streaming uploads for M4 8GB

---

## 🚀 Quick Start

**→ [Complete Setup Guide](docs/guides/setup.md)** for prerequisites & detailed instructions

```bash
git clone https://github.com/KDhiraj152/Siksha-Setu.git && cd shiksha_setu
./SETUP.sh && ./START.sh
```

Access: [Frontend](http://localhost:5173) | [API Docs](http://localhost:8000/docs)

---

## 📚 Documentation

**Getting Started**:
- **[Setup Guide](docs/guides/setup.md)** - Complete installation with multiple paths
- **[Demo Guide](docs/guides/demo.md)** - Interactive demo walkthrough
- **[Contributing](docs/guides/contributing.md)** - How to contribute

**API & Architecture**:
- **[API Reference](docs/reference/api.md)** - REST API endpoints & responses
- **[Architecture](docs/reference/architecture.md)** - System design & components
- **[Backend Reference](docs/reference/backend.md)** - Backend structure
- **[Features](docs/reference/features.md)** - Complete feature list

**Technical Deep-Dives**:
- **[AI/ML Pipeline](docs/technical/ai-ml-pipeline.md)** - Model orchestration & optimization
- **[Deployment](docs/technical/deployment.md)** - Docker, Kubernetes, Cloud platforms
- **[Security](docs/technical/security.md)** - Authentication, RBAC, best practices
- **[Optimization](docs/technical/optimization.md)** - Performance tuning & memory management
- **[Database](docs/technical/database.md)** - Schema, migrations, pgvector
- **[Monitoring](docs/technical/monitoring.md)** - Observability & alerting

**Development**:
- **[Development Guide](DEVELOPMENT.md)** - Developer setup & workflow
- **[Testing Guide](docs/guides/testing.md)** - Unit, integration, E2E tests
- **[Troubleshooting](docs/guides/troubleshooting.md)** - Common issues & solutions

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│          React Frontend (Port 5173)         │
│   TypeScript, TailwindCSS, TanStack Query   │
└──────────────────┬──────────────────────────┘
                   │ REST API (JWT Auth)
┌──────────────────┴──────────────────────────┐
│       FastAPI Backend (Port 8000)           │
│   Async, Pydantic, SQLAlchemy 2.0          │
├─────────────────────────────────────────────┤
│  Model Orchestrator (Unified Client)        │
│  ├─ Tier Router (SMALL/MEDIUM/LARGE)       │
│  ├─ Dynamic Quantization (FP16-INT2)       │
│  └─ Lazy Loading (LRU Cache)               │
├─────────────────────────────────────────────┤
│  Pipeline Services                          │
│  ├─ Simplification (FLAN-T5)               │
│  ├─ Translation (IndicTrans2)              │
│  ├─ Validation (NCERT Standards)           │
│  ├─ Speech (MMS-TTS)                       │
│  └─ Q&A (RAG + pgvector)                   │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────┴──────────────────────────┐
│  PostgreSQL 15 + pgvector                   │
│  Redis 7 (Cache + Rate Limiting)            │
└─────────────────────────────────────────────┘
```

**Key Components**:
- **Model Tier Router**: Routes tasks to appropriate model sizes based on complexity
- **Dynamic Quantization**: Adapts FP16/INT8/INT4/INT2 based on memory pressure
- **Unified Model Client**: Single interface for all AI operations with circuit breaker
- **RAG Pipeline**: ChromaDB + pgvector for semantic document search

---

## 🧪 Testing

```bash
# Backend tests (unit, integration, E2E)
pytest tests/ -v --cov=backend

# Frontend tests
cd frontend && npm test
```

**Coverage**: 79% overall (87% backend core, 71% frontend)

---

## 🚢 Deployment

### Docker Compose (Recommended)

```bash
# Production deployment
docker-compose -f docker-compose.production.yml up -d

# Verify services
docker-compose ps
```

### Manual Deployment

```bash
# Build frontend
cd frontend && npm run build

# Start backend with production settings
export ENVIRONMENT=production
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --workers 4

# Serve frontend
npx serve -s frontend/dist -l 5173
```

**See**: [Deployment Guide](DEPLOYMENT.md) for Kubernetes, AWS, and monitoring setup.

---

## 🔧 API Examples

### Authentication

```bash
# Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "test@shiksha.com", "password": "Test@1234567"}'

# Returns: {"access_token": "eyJ...", "token_type": "bearer"}
```

### Content Simplification

```bash
curl -X POST http://localhost:8000/api/v1/content/simplify \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Photosynthesis converts light energy into glucose.",
    "target_grade": 5,
    "subject": "Science"
  }'
```

### Translation

```bash
curl -X POST http://localhost:8000/api/v1/content/translate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Good morning students",
    "source_language": "English",
    "target_language": "Hindi"
  }'
```

**See**: [API Reference](docs/reference/api.md) for complete endpoint documentation.

---

## 🤝 Contributing

We welcome contributions! Please see [Contributing Guide](docs/contributing.md) for:

- Development workflow and branch naming
- Code standards (Black, Flake8, ESLint, mypy)
- Commit guidelines (Conventional Commits)
- Pull request process
- Testing requirements (80% coverage target)

---

## 📊 Tech Stack

| Category | Technologies |
|----------|-------------|
| **Backend** | FastAPI, SQLAlchemy 2.0, Pydantic, Celery |
| **Frontend** | React 19, TypeScript, Vite 5, TailwindCSS 4, TanStack Query |
| **Database** | PostgreSQL 15, Redis 7, ChromaDB, pgvector |
| **AI/ML** | PyTorch 2.5, Transformers, FLAN-T5, IndicTrans2, MMS-TTS |
| **Infrastructure** | Docker, Docker Compose, Nginx, Prometheus, Grafana |

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 📧 Contact

**K. Dhiraj**  
📧 Email: kdhiraj152@gmail.com  
🐙 GitHub: [@KDhiraj152](https://github.com/KDhiraj152)  
💼 LinkedIn: [K. Dhiraj](https://www.linkedin.com/in/k-dhiraj-83b025279/)  
🔗 Project: [Shiksha Setu AI](https://github.com/KDhiraj152/Siksha-Setu)

---

**Built with ❤️ for Indian Education**
cd frontend
npm test -- --run
npm run test:coverage
```

---

## 🐳 Docker Deployment

### Development
```bash
cd infrastructure/docker
docker-compose up -d
# Access: http://localhost:5173 (frontend), http://localhost:8000 (backend)
```

### Production
```bash
cd infrastructure/docker
docker-compose -f docker-compose.yml up -d
```

---

## ☸️ Kubernetes Deployment

### Development Environment
```bash
cd infrastructure/kubernetes
kubectl apply -k overlays/dev
```

### Production Environment
```bash
cd infrastructure/kubernetes
kubectl apply -k overlays/prod
kubectl get pods -n shiksha-setu
```

See [infrastructure/kubernetes/SETUP.md](infrastructure/kubernetes/SETUP.md) for detailed configuration.

---

## 📋 Project Structure

```
shiksha_setu/
├── bin/                          # Executable scripts
│   ├── setup                     # Initial setup
│   ├── start                     # Start all services
│   ├── stop                      # Stop all services
│   ├── demo                      # Launch demo
│   └── test                      # Run tests
├── backend/                      # Backend source code
│   ├── api/                      # FastAPI application
│   ├── core/                     # Configuration & security
│   ├── services/                 # Business logic
│   ├── pipeline/                 # AI/ML pipeline
│   └── tasks/                    # Background jobs
├── frontend/                     # React TypeScript app
│   └── src/
│       ├── pages/                # Route pages
│       ├── components/           # UI components
│       └── services/             # API client
├── infrastructure/               # Deployment & orchestration
│   ├── docker/                   # Docker containers
│   ├── kubernetes/               # K8s manifests
│   └── monitoring/               # Prometheus, Grafana
├── alembic/                      # Database migrations
│   └── versions/                 # Migration versions
├── storage/                      # Runtime data
│   ├── uploads/                  # User uploads
│   ├── models/                   # ML models
│   └── logs/                     # Application logs
├── docs/                         # Documentation
│   ├── guides/                   # User guides
│   └── reference/                # Technical docs
├── tests/                        # Test suite
├── .env.example                  # Environment template
└── README.md                     # This file
```

---

## 🔧 Configuration

All configuration is managed through `.env` file. See `.env.example` for all available options.

**Essential variables:**
```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/shiksha_setu

# JWT Security (generate: python -c "import secrets; print(secrets.token_urlsafe(64))")
JWT_SECRET_KEY=your-secure-key-here

# Redis
REDIS_URL=redis://localhost:6379/0

# Frontend API URL
VITE_API_BASE_URL=http://localhost:8000

# Optional: HuggingFace API for cloud inference
HUGGINGFACE_API_KEY=
```

---

## 🔒 Security Features

✅ **JWT Authentication** - Access & refresh tokens  
✅ **Password Hashing** - bcrypt with salt  
✅ **CORS Protection** - Configured for localhost:5173  
✅ **Security Headers** - CSP, HSTS, X-Frame-Options, etc.  
✅ **Rate Limiting** - Configurable per-endpoint  
✅ **Input Validation** - Pydantic schemas  
✅ **SQL Injection Protection** - Parameterized queries  
✅ **HTTPS Ready** - Full TLS/SSL support  

---

## 📊 Performance Metrics

| Metric | Result | Status |
|--------|--------|--------|
| Backend Response Time | 7ms average | ⚡ Excellent |
| Frontend Build Time | 821ms | ⚡ Fast |
| Bundle Size | 351KB → 107KB gzipped | ⚡ Optimized |
| Test Suite | <1 second | ⚡ Fast |
| Unit Tests | 15/15 PASS | ✅ 100% |

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**K Dhiraj**
- Email: k.dhiraj.srihari@gmail.com
- GitHub: [@KDhiraj152](https://github.com/KDhiraj152)
- LinkedIn: [linkedin.com/in/k-dhiraj](https://linkedin.com/in/k-dhiraj)

---

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [HuggingFace](https://huggingface.co/) - ML models and transformers
- [AI4Bharat](https://ai4bharat.org/) - IndicTrans2 translation
- [React](https://react.dev/) - UI library
- [Supabase](https://supabase.com/) - Database platform

---

## 📞 Support & Feedback

- **Issues**: [GitHub Issues](https://github.com/KDhiraj152/Siksha-Setu/issues)
- **Email**: k.dhiraj.srihari@gmail.com
- **Documentation**: See [docs/](docs/) folder

---

**Built with ❤️ for educators and students across India**

*Last Updated: November 27, 2025 | Status: ✅ Production Ready*
cd frontend
npm run dev
```

### Access

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

---

## 📚 API Examples

### Authentication

```bash
# Register
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password123", "full_name": "John Doe"}'

# Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password123"}'
```

### Content Processing

```bash
# Upload file
curl -X POST http://localhost:8000/api/v1/content/upload \
  -H "Authorization: Bearer {access_token}" \
  -F "file=@document.pdf"

# Process content
curl -X POST http://localhost:8000/api/v1/content/process?file_path=uploads/document.pdf \
  -H "Authorization: Bearer {access_token}" \
  -H "Content-Type: application/json" \
  -d '{
    "grade_level": 8,
    "subject": "Science",
    "target_languages": ["Hindi", "Tamil"],
    "output_format": "both"
  }'
```

### Q&A System

```bash
# Process document for Q&A
curl -X POST http://localhost:8000/api/v1/qa/process \
  -H "Authorization: Bearer {access_token}" \
  -d "content_id=123"

# Ask question
curl -X POST http://localhost:8000/api/v1/qa/ask \
  -H "Authorization: Bearer {access_token}" \
  -H "Content-Type: application/json" \
  -d '{"content_id": "123", "question": "What is photosynthesis?"}'
```

---

## 🏗️ Architecture

```
shiksha_setu/
├── backend/                 # Backend source
│   ├── api/                 # FastAPI application
│   │   ├── main.py          # App entry point
│   │   ├── middleware/      # Security & logging
│   │   └── routes/          # API endpoints
│   ├── core/                # Core configuration
│   ├── schemas/             # Pydantic models
│   ├── services/            # Business logic
│   ├── tasks/               # Celery background tasks
│   ├── pipeline/            # AI/ML orchestration
│   └── utils/               # Utilities
├── frontend/                # React TypeScript application
│   └── src/
│       ├── pages/           # Route pages
│       ├── components/      # Reusable components
│       ├── services/        # API client
│       └── store/           # State management
├── infrastructure/          # Deployment & orchestration
│   ├── docker/              # Docker containers
│   ├── kubernetes/          # K8s manifests
│   └── monitoring/          # Prometheus, Grafana
├── alembic/                 # Database migrations
├── docs/                    # Documentation
├── tests/                   # Test suite
└── scripts/                 # Utility scripts
```

---

## 🧪 Testing

```bash
# Backend tests
source .venv/bin/activate
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Frontend tests
cd frontend
npm test
```

---

## 🐳 Docker Deployment

```bash
# Development
docker-compose up -d

# View logs
docker-compose logs -f fastapi

# Stop services
docker-compose down
```

---

## ☸️ Kubernetes Deployment

```bash
# Deploy to development
kubectl apply -k k8s/overlays/dev

# Deploy to production
kubectl apply -k k8s/overlays/prod

# Check status
kubectl get pods -n shiksha-setu
```

---

## 📖 Documentation

- **[Setup Guide](docs/setup.md)** - Installation & setup
- **[Usage Guide](docs/usage.md)** - How to use the API
- **[Deployment](docs/deploy.md)** - Production deployment
- **[API Reference](docs/api.md)** - Complete API docs
- **[Kubernetes Configuration](k8s/CONFIGURATION.md)** - K8s deployment guide
- **[Changelog](docs/changelog.md)** - Version history

---

## 🛠️ Development

### Project Structure

```
backend/
├── simplify/            # Text simplification
│   ├── simplifier.py    # Main simplifier
│   └── analyzer.py      # Complexity analysis
├── translate/           # Translation engine
│   ├── engine.py        # Translation logic
│   └── model.py         # IndicTrans2 model
├── speech/              # Text-to-speech
│   ├── generator.py     # TTS generation
│   └── processor.py     # Audio processing
├── validate/            # Content validation
│   ├── validator.py     # Validation logic
│   └── standards.py     # NCERT standards
└── services/            # Additional services
    ├── rag.py           # RAG Q&A system
    └── captions.py      # Caption service
```

### Code Quality

```bash
# Linting
pylint backend/

# Type checking
mypy backend/

# Format code
black backend/
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection | Required |
| `JWT_SECRET_KEY` | JWT signing key | Auto-generated |
| `REDIS_URL` | Redis connection | `redis://localhost:6379/0` |
| `RATE_LIMIT_ENABLED` | Enable rate limiting | `false` |
| `MAX_UPLOAD_SIZE` | Max file size (bytes) | `104857600` (100MB) |

### Supported Languages

- English, Hindi, Tamil, Telugu, Bengali
- Marathi, Gujarati, Kannada, Malayalam, Punjabi

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## 🔒 Recent Improvements

### Security & Configuration
- **Grafana Datasource**: Enabled HTTPS, basic authentication, and environment variable substitution
- **Documentation**: Removed hardcoded credentials, added security warnings
- **Docker Images**: Pinned all monitoring stack versions (Prometheus v3.0.1, Grafana 11.4.0, etc.)
- **Development Mode**: Made uvicorn `--reload` flag conditional via `UVICORN_RELOAD` environment variable

### Database & Migrations
- **Migration Chain**: Fixed Alembic migration sequence (005 → 007 → 61631d311ed9)
- **Index Creation**: Added concurrent index creation support with row count checks
- **pgvector Setup**: Enhanced with proper error handling and minimum row thresholds

### Kubernetes
- **Image Versioning**: Replaced `:latest` tags with semantic versioning (v1.0.0)
- **Configuration**: Added Kustomize variable substitution for domains and AWS account IDs
- **Ingress**: Updated to use `ingressClassName` field (Kubernetes 1.18+)
- **RBAC**: Added variable substitution for AWS account-specific annotations

### Frontend TypeScript
- **Type Safety**: Replaced `any` types with proper interfaces across all pages
- **Polling Mechanism**: Added backend status polling for async operations
- **Accessibility**: Added `aria-hidden` attributes to decorative elements
- **Validation**: Enhanced input validation and error handling

---

## 📄 License

MIT License - see [LICENSE](LICENSE)

---

## 🙏 Acknowledgments

[FastAPI](https://fastapi.tiangolo.com/) • [HuggingFace](https://huggingface.co/) • [AI4Bharat](https://ai4bharat.org/) • [React](https://react.dev/) • [Supabase](https://supabase.com/)

---

## 👨‍💻 Author

**K Dhiraj**

[![Email](https://img.shields.io/badge/Email-k.dhiraj.srihari%40gmail.com-red?logo=gmail)](mailto:k.dhiraj.srihari@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-KDhiraj152-black?logo=github)](https://github.com/KDhiraj152)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-K%20Dhiraj-blue?logo=linkedin)](https://www.linkedin.com/in/k-dhiraj-83b025279/)

---

**Built with ❤️ for educators and students across India**

*Last updated: November 2025*
