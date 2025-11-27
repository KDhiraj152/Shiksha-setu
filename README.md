# 🎓 ShikshaSetu - AI-Powered Education Platform

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19.0-61dafb?logo=react)](https://react.dev/)
[![Python](https://img.shields.io/badge/Python-3.13-3776ab?logo=python)](https://python.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.9-3178c6?logo=typescript)](https://typescriptlang.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Production-ready multilingual education content processing system with AI/ML pipeline, RAG-based Q&A, and modern UI**

---

## 🌟 Features

### 🤖 AI/ML Processing
- **Text Simplification** - FLAN-T5 for grade-level appropriate content
- **Translation** - IndicTrans2 supporting 10+ Indian languages
- **NCERT Validation** - Curriculum alignment scoring
- **Text-to-Speech** - MMS-TTS multilingual audio generation
- **RAG Q&A System** - Intelligent document question answering

### 🔐 Enterprise Security
- JWT authentication with refresh tokens
- Role-based access control (User, Educator, Admin)
- Rate limiting and API key support
- CORS protection and input sanitization
- All security headers configured (CSP, HSTS, X-Frame-Options, etc.)

### 📊 Modern Tech Stack
- **Backend**: FastAPI (async), SQLAlchemy 2.0, Celery, Redis
- **Frontend**: React 19, TypeScript, Vite 7, TailwindCSS 4
- **Database**: PostgreSQL 17 with pgvector
- **ML Models**: HuggingFace Transformers (latest)
- **Deployment**: Docker, Kubernetes ready

---

## 🚀 Quick Start (5 Minutes)

### Prerequisites
- **Python 3.13+** - [Download](https://python.org)
- **Node.js 25+** - [Download](https://nodejs.org)
- **Redis 7+** - [Download](https://redis.io) or `brew install redis`
- **PostgreSQL 15+** or [Supabase](https://supabase.com)

### 1️⃣ Setup

```bash
# Clone repository
git clone https://github.com/KDhiraj152/Siksha-Setu.git
cd shiksha_setu

# Create Python environment
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
cd frontend && npm install && cd ..

# Configure environment
cp .env.example .env
# Edit .env with your database credentials
```

### 2️⃣ Run Services

```bash
# Terminal 1: Redis (message broker)
redis-server

# Terminal 2: Backend (port 8000)
source .venv/bin/activate
uvicorn src.api.main:app --reload

# Terminal 3: Frontend (port 5173)
cd frontend && npm run dev

# Terminal 4 (Optional): Celery worker for async tasks
source .venv/bin/activate
celery -A src.tasks.celery_app worker --loglevel=info
```

### 3️⃣ Access Application

| Service | URL | Purpose |
|---------|-----|---------|
| **Frontend** | http://localhost:5173 | React application |
| **Backend API** | http://localhost:8000 | FastAPI server |
| **API Docs** | http://localhost:8000/docs | Interactive Swagger UI |
| **ReDoc** | http://localhost:8000/redoc | Alternative docs |

---

## 📚 Documentation

Comprehensive guides for all aspects of the project:

| Document | Purpose |
|----------|---------|
| **[docs/setup.md](docs/setup.md)** | Installation & environment configuration |
| **[docs/usage.md](docs/usage.md)** | How to use the API & features |
| **[docs/api.md](docs/api.md)** | Complete API reference & examples |
| **[docs/deploy.md](docs/deploy.md)** | Docker & Kubernetes deployment |
| **[docs/pgvector.md](docs/pgvector.md)** | Vector database setup for RAG |
| **[docs/CHANGES.md](docs/CHANGES.md)** | Recent improvements & security updates |

---

## 🧪 Testing

### Backend Tests (15/15 PASS ✅)
```bash
source .venv/bin/activate
pytest tests/unit/ -v                    # Unit tests
pytest tests/ --cov=src --cov-report=html  # With coverage
```

### Frontend Tests (2/2 PASS ✅)
```bash
cd frontend
npm test -- --run                 # Single run
npm run test:ui                   # Interactive UI
npm run test:coverage             # Coverage report
```

### Full Integration Tests
```bash
# Make sure backend is running on port 8000
pytest tests/test_backend_complete.py -v
```

---

## 🐳 Docker Deployment

### Development
```bash
docker-compose up -d
# Access: http://localhost:5173 (frontend), http://localhost:8000 (backend)
```

### Production
```bash
docker-compose -f deploy/docker-compose.yml up -d
```

---

## ☸️ Kubernetes Deployment

### Development Environment
```bash
kubectl apply -k k8s/overlays/dev
```

### Production Environment
```bash
kubectl apply -k k8s/overlays/prod
kubectl get pods -n shiksha-setu
```

See [k8s/SETUP.md](k8s/SETUP.md) for detailed configuration.

---

## 📋 Project Structure

```
shiksha_setu/
├── src/                          # Backend source code
│   ├── api/                      # FastAPI application
│   │   ├── main.py              # App entry point
│   │   ├── middleware.py        # Security middleware
│   │   └── routes/              # API endpoints
│   ├── core/                    # Configuration & security
│   ├── services/                # Business logic
│   ├── tasks/                   # Celery async tasks
│   └── schemas/                 # Pydantic data models
├── frontend/                     # React TypeScript application
│   └── src/
│       ├── pages/               # Route pages
│       ├── components/          # Reusable components
│       ├── services/            # API client
│       ├── store/               # State management
│       └── test/                # Test utilities
├── tests/                        # Test suite
├── docs/                         # Documentation
├── deploy/                       # Docker configuration
├── k8s/                          # Kubernetes manifests
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
├── src/                      # Backend source
│   ├── api/                  # FastAPI application
│   │   ├── main.py          # App entry point
│   │   ├── middleware.py    # Security & logging
│   │   └── routes/          # API endpoints
│   ├── core/                # Core configuration
│   │   ├── config.py        # Settings
│   │   ├── security.py      # JWT & auth
│   │   └── constants.py     # App constants
│   ├── schemas/             # Pydantic models
│   ├── services/            # Business logic
│   ├── tasks/               # Celery tasks
│   └── utils/               # Utilities
├── frontend/                # React application
│   └── src/
│       ├── pages/           # Route pages
│       ├── components/      # Reusable components
│       ├── services/        # API client
│       └── store/           # State management
├── config/                  # Configuration files
│   ├── requirements.txt     # Python dependencies
│   └── alembic.ini          # DB migration config
├── deploy/                  # Deployment configs
│   ├── Dockerfile           # Backend container
│   └── docker-compose.yml   # Docker orchestration
├── docs/                    # Documentation
├── tests/                   # Test suite
├── scripts/                 # Utility scripts
└── k8s/                     # Kubernetes configs
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
src/
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
pylint src/

# Type checking
mypy src/

# Format code
black src/
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

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/KDhiraj152/Siksha-Setu/issues)
- **Documentation**: [API Docs](http://localhost:8000/docs)
- **Email**: k.dhiraj.srihari@gmail.com

---

**Built with ❤️ for educators and students across India**

*Last Updated: November 16, 2025*
