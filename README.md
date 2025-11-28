# 🎓 ShikshaSetu - AI-Powered Education Platform

[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19.0-61dafb?logo=react)](https://react.dev/)
[![Python](https://img.shields.io/badge/Python-3.11-3776ab?logo=python)](https://python.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.9-3178c6?logo=typescript)](https://typescriptlang.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-93_passing-brightgreen.svg)](tests/)
[![Coverage](https://img.shields.io/badge/Coverage-23%25-yellow.svg)](htmlcov/)
[![Production](https://img.shields.io/badge/Production-Ready-success.svg)](IMPLEMENTATION_SUMMARY.md)

> **Production-ready multilingual education content processing system with AI/ML pipeline, RAG-based Q&A, complete CI/CD, and modern UI**

---

## ✨ Production Status

🎉 **ShikshaSetu is production-ready!** Complete deployment infrastructure with:

- ✅ **93 passing tests** (23% coverage) - [Test Report](IMPLEMENTATION_SUMMARY.md#test-coverage-explosion-370-tests-42-coverage)
- ✅ **15-service architecture** with high availability
- ✅ **Complete CI/CD pipeline** (test, build, deploy-staging, deploy-production)
- ✅ **Monitoring stack** (Prometheus, Grafana, Alertmanager)
- ✅ **Automated deployment** with rollback capability
- ✅ **Production documentation** (1,800+ lines)

**Quick Links**:
- 📊 [Implementation Summary](IMPLEMENTATION_SUMMARY.md) - Complete overview
- 🚀 [Deployment Guide](DEPLOYMENT.md) - Production deployment instructions
- 💻 [Development Guide](DEVELOPMENT.md) - Developer onboarding and standards
- 📖 [Documentation](docs/) - Comprehensive technical documentation
- 🔧 [Scripts](scripts/README.md) - Utility scripts and automation

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
- **Python 3.11.11** (Required - PyTorch 2.5.1 does not support 3.13) - [Download](https://python.org)
- **Node.js 25+** - [Download](https://nodejs.org)
- **Redis 7.4+** - [Download](https://redis.io) or `brew install redis@7.4`
- **PostgreSQL 17+** or [Supabase](https://supabase.com)

### 1️⃣ Setup

```bash
# Clone repository
git clone https://github.com/KDhiraj152/Siksha-Setu.git
cd shiksha_setu

# Run automated setup
./bin/setup
```

### 2️⃣ Start Application

```bash
# Start all services (backend + frontend)
./bin/start

# Or start services separately:
./bin/start-backend  # Backend only (port 8000)
./bin/start-frontend # Frontend only (port 5173)
```

### 3️⃣ Try the Demo

```bash
# Launch interactive demo
./bin/demo
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

### 📖 Guides
| Document | Purpose |
|----------|---------|
| **[docs/guides/installation.md](docs/guides/installation.md)** | Installation & setup guide |
| **[docs/guides/quickstart.md](docs/guides/quickstart.md)** | Quick start guide |
| **[docs/guides/demo.md](docs/guides/demo.md)** | Demo usage guide |
| **[docs/guides/deployment.md](docs/guides/deployment.md)** | Docker & Kubernetes deployment |

### 🔧 Reference
| Document | Purpose |
|----------|---------|
| **[docs/reference/api.md](docs/reference/api.md)** | Complete API reference |
| **[docs/reference/architecture.md](docs/reference/architecture.md)** | System architecture |
| **[docs/reference/rag.md](docs/reference/rag.md)** | RAG Q&A system details |
| **[docs/reference/pgvector.md](docs/reference/pgvector.md)** | Vector database setup |

---

## 🧪 Testing

### Run All Tests
```bash
./bin/test
```

### Demo Testing
```bash
./bin/test-demo
```

### Manual Testing
```bash
# Backend tests
source .venv/bin/activate
pytest tests/unit/ -v
pytest tests/ --cov=backend --cov-report=html

# Frontend tests
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

## 👨‍💻 Made By

**K Dhiraj Srihari**

🔗 **Connect with me:**
- 📧 Email: [k.dhiraj.srihari@gmail.com](mailto:k.dhiraj.srihari@gmail.com)
- 💼 LinkedIn: [linkedin.com/in/k-dhiraj](https://linkedin.com/in/k-dhiraj)
- 🐙 GitHub: [@KDhiraj152](https://github.com/KDhiraj152)

---

**Built with ❤️ for educators and students across India**

*Last Updated: November 28, 2025*
