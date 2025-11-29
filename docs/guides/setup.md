# 🔧 Complete Setup Guide

**Shiksha Setu** - Full installation guide for development and production environments.

Choose your path:
- ⚡ **[Quick Start (5 min)](#quick-start-5-minutes)** - For experienced developers
- 📋 **[Detailed Setup](#detailed-setup)** - Step-by-step instructions
- 🐳 **[Docker Setup](#docker-setup-recommended)** - Container-based development
- ☁️ **[Cloud Deployment](#cloud-database-options)** - Using managed services

---

## Prerequisites

### Required Software

| Software | Version | Download | Check |
|----------|---------|----------|-------|
| **Python** | 3.11+ | [python.org](https://python.org) | `python3 --version` |
| **Node.js** | 20+ | [nodejs.org](https://nodejs.org) | `node --version` |
| **PostgreSQL** | 15+ | [postgresql.org](https://postgresql.org) | `psql --version` |
| **Redis** | 7+ | [redis.io](https://redis.io) | `redis-cli --version` |
| **Docker** | 24+ (optional) | [docker.com](https://docker.com) | `docker --version` |

### System Requirements

- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space (for models and dependencies)
- **OS**: macOS, Linux, or Windows with WSL2

---

## Quick Start (5 Minutes)

For experienced developers who want to get started immediately:

```bash
# 1. Clone repository
git clone https://github.com/KDhiraj152/Siksha-Setu.git
cd shiksha_setu

# 2. Run automated setup (handles everything)
./SETUP.sh

# 3. Start all services
./START.sh

# 4. Access application
open http://localhost:5173
```

**What the setup script handles**:
- ✅ Virtual environment creation
- ✅ Dependency installation
- ✅ Database migrations
- ✅ Model downloads
- ✅ Environment configuration

For detailed step-by-step instructions or troubleshooting, continue reading.

---

## Detailed Setup

### Step 1: Clone Repository

```bash
# Via HTTPS
git clone https://github.com/KDhiraj152/Siksha-Setu.git
cd shiksha_setu

# Or via SSH (if configured)
git clone git@github.com:KDhiraj152/Siksha-Setu.git
cd shiksha_setu
```

### Step 2: Choose Database Setup Method

Choose **ONE** of these options:

#### Option A: Docker (Recommended for Development)

```bash
# Start PostgreSQL with pgvector
docker run -d \
  --name shiksha-postgres \
  -e POSTGRES_USER=shiksha_user \
  -e POSTGRES_PASSWORD=shiksha_pass \
  -e POSTGRES_DB=shiksha_setu \
  -p 5432:5432 \
  -v postgres_data:/var/lib/postgresql/data \
  ankane/pgvector:latest

# Start Redis
docker run -d \
  --name shiksha-redis \
  -p 6379:6379 \
  redis:7-alpine

# Verify running
docker ps | grep shiksha
```

#### Option B: Local Installation

**PostgreSQL with pgvector**:

```bash
# macOS
brew install postgresql@15
brew services start postgresql@15

# Ubuntu/Debian
sudo apt update
sudo apt install postgresql postgresql-contrib postgresql-15-pgvector

# Create database
createdb shiksha_setu

# Enable pgvector extension
psql -d shiksha_setu -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

**Redis**:

```bash
# macOS
brew install redis
brew services start redis

# Ubuntu/Debian
sudo apt install redis-server
sudo systemctl start redis
```

#### Option C: Cloud Database

Use managed services:

- **Supabase** - [supabase.com](https://supabase.com) (PostgreSQL + pgvector enabled)
- **Railway** - [railway.app](https://railway.app) (PostgreSQL + Redis)
- **Render** - [render.com](https://render.com) (PostgreSQL + Redis)
- **AWS RDS** - For production PostgreSQL

### Step 3: Environment Configuration

```bash
# Copy template
cp .env.example .env

# Generate secure JWT secret
python3 -c "import secrets; print('JWT_SECRET_KEY=' + secrets.token_urlsafe(64))" >> .env

# Edit with your editor
nano .env
```

**Minimum Required Variables**:

```bash
# Database connection
DATABASE_URL=postgresql://shiksha_user:shiksha_pass@localhost:5432/shiksha_setu

# Redis for caching and Celery
REDIS_URL=redis://localhost:6379/0

# Security
JWT_SECRET_KEY=<paste_generated_secret>
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
REFRESH_TOKEN_EXPIRE_DAYS=7

# CORS origins
ALLOWED_ORIGINS=http://localhost:5173,http://localhost:3000

# Environment
ENVIRONMENT=development
DEBUG=true
```

**Optional but Recommended**:

```bash
# HuggingFace (for cloud ML inference)
HUGGINGFACE_API_KEY=hf_your_token_here

# Error tracking
SENTRY_DSN=https://your_sentry_dsn

# Bhashini (Indian language translation)
BHASHINI_API_KEY=your_api_key
```

### Step 4: Enable pgvector Extension

```bash
# If using Docker
docker exec shiksha-postgres psql -U shiksha_user -d shiksha_setu \
  -c "CREATE EXTENSION IF NOT EXISTS vector;"

# If using local PostgreSQL
psql -d shiksha_setu -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Verify
psql -d shiksha_setu -c "\dx" | grep vector
```

Expected output: Shows `vector` in extension list

### Step 5: Backend Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt

# Run database migrations
alembic upgrade head

# Create optional admin user
python3 -m backend.scripts.create_admin
```

### Step 6: Frontend Setup

```bash
cd frontend

# Install Node.js dependencies
npm install

# Return to root
cd ..
```

### Step 7: Download AI/ML Models

```bash
# Download required models (5-10 minutes, ~5GB)
python3 -m backend.scripts.download_models

# Or let them auto-download on first use
# (Models cache to ~/.cache/huggingface)
```

**Models Downloaded**:
- Qwen2.5 or FLAN-T5 (text simplification) - ~3.5GB
- IndicTrans2 (Indian language translation) - ~1GB
- Multilingual embeddings (search/RAG) - ~300MB
- IndicBERT (validation) - ~400MB
- MMS-TTS (text-to-speech) - ~600MB

### Step 8: Verify Installation

```bash
# Check backend health
curl http://localhost:8000/health

# Expected: {"status":"healthy",...}

# Check database
psql -d shiksha_setu -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema='public';"

# Expected: >0 (tables exist)
```

---

## Configuration

### API Keys & Services

#### HuggingFace API Key (Optional but Recommended)

Enables faster cloud-based inference without local GPU:

1. Visit [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create **Read** token
3. Add to `.env`: `HUGGINGFACE_API_KEY=hf_your_token`
4. Free tier: 1,000 requests/month

#### Bhashini API Key (Optional - Better Indian Language Support)

For higher-quality regional language translation:

1. Register at [bhashini.gov.in](https://bhashini.gov.in/ulca/user/register)
2. Verify email and request API access
3. Add to `.env`:
   ```bash
   BHASHINI_API_KEY=your_key
   BHASHINI_PIPELINE_ID=your_pipeline_id
   ```

#### Sentry (Optional - Error Tracking)

For production error monitoring:

1. Create account at [sentry.io](https://sentry.io)
2. Create project for Python
3. Add to `.env`: `SENTRY_DSN=https://xxx@sentry.io/xxx`

### Authentication

**Create Admin User**:

```bash
# Interactive
python3 -m backend.scripts.create_admin

# Or via API (after starting server)
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "admin@example.com",
    "password": "SecurePassword123!",
    "full_name": "Admin User"
  }'
```

### Rate Limiting (Production)

```bash
# .env configuration
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000
```

Requires Redis to be running.

---

## Docker Setup (Recommended)

### Development with Docker Compose

```bash
# Start all services
docker-compose up -d

# Verify services
docker-compose ps

# View logs
docker-compose logs -f backend
docker-compose logs -f frontend

# Stop services
docker-compose down
```

**Services Started**:
- Backend API (port 8000)
- Frontend (port 5173)
- PostgreSQL + pgvector (port 5432)
- Redis (port 6379)
- Celery worker (background tasks)

### Build Production Images

```bash
# Build backend image
docker build -t shiksha-setu/backend:v1.0.0 -f infrastructure/docker/Dockerfile .

# Build worker image
docker build -t shiksha-setu/worker:v1.0.0 -f infrastructure/docker/worker.dockerfile .

# Tag for registry
docker tag shiksha-setu/backend:v1.0.0 your-registry/shiksha-setu/backend:v1.0.0
docker push your-registry/shiksha-setu/backend:v1.0.0
```

---

## Cloud Database Options

### Supabase (PostgreSQL + Managed)

1. Create account at [supabase.com](https://supabase.com)
2. Create new project (PostgreSQL 15+)
3. Enable pgvector extension:
   - Go to SQL Editor → New Query
   - Run: `CREATE EXTENSION IF NOT EXISTS vector;`
4. Get connection string from Connection Pool
5. Add to `.env`:
   ```bash
   DATABASE_URL=postgresql://postgres.xxxxx:password@aws-0-ap-south-1.pooler.supabase.com:5432/postgres
   ```

### Railway (All-in-one)

1. Create account at [railway.app](https://railway.app)
2. Add PostgreSQL + Redis plugins
3. Connect services: PostgreSQL config → Railway
4. Copy connection strings to `.env`

### AWS RDS (Enterprise)

1. Create RDS PostgreSQL instance (15+)
2. Install pgvector extension:
   ```bash
   psql -h your-instance.amazonaws.com -U postgres -d postgres \
     -c "CREATE EXTENSION IF NOT EXISTS vector;"
   ```
3. Update `.env` with RDS endpoint

---

## Running the Application

### Option 1: Automated Scripts

```bash
# Start everything
./START.sh

# Stop everything
./STOP.sh

# Run backend tests
pytest tests/ -v --cov=backend

# Run frontend tests
cd frontend && npm test
```

### Option 2: Manual Start (Separate Terminals)

**Terminal 1 - Backend**:
```bash
source venv/bin/activate
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 2 - Celery Worker**:
```bash
source venv/bin/activate
celery -A backend.tasks.celery_app worker --loglevel=info
```

**Terminal 3 - Frontend**:
```bash
cd frontend
npm run dev
```

### Option 3: Docker Compose

```bash
docker-compose up
```

---

## Verification

### Health Checks

```bash
# Backend API
curl http://localhost:8000/health
# Expected: {"status":"healthy",...}

# Database connection
curl http://localhost:8000/api/v1/status
# Expected: {"status":"ok","database":"connected",...}

# Frontend
curl -I http://localhost:5173
# Expected: HTTP/1.1 200 OK
```

### Access Services

| Service | URL | Purpose |
|---------|-----|---------|
| **Frontend** | http://localhost:5173 | Web application |
| **API Docs** | http://localhost:8000/docs | Swagger UI |
| **API ReDoc** | http://localhost:8000/redoc | Alternative API docs |
| **Health** | http://localhost:8000/health | System status |

### Test API

```bash
# Process content
curl -X POST http://localhost:8000/api/v1/content/process \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The mitochondria is the powerhouse of the cell.",
    "grade_level": 8,
    "subject": "Science",
    "target_languages": ["Hindi"]
  }'
```

### Run Test Suites

```bash
# Backend tests
pytest tests/ -v --cov=backend

# Frontend tests
cd frontend
npm test
```

---

## Troubleshooting

### Port Already in Use

```bash
# Find process using port
lsof -i :5432   # PostgreSQL
lsof -i :6379   # Redis
lsof -i :8000   # Backend
lsof -i :5173   # Frontend

# Kill process
kill -9 <PID>

# Or use different port
export FASTAPI_PORT=8001
```

### Database Connection Failed

```bash
# Check if PostgreSQL is running
docker ps | grep postgres          # Docker
brew services list | grep postgres # Homebrew

# Test connection
psql -h localhost -U shiksha_user -d shiksha_setu

# Restart service
docker restart shiksha-postgres    # Docker
brew services restart postgresql@15 # Homebrew
```

### Redis Connection Issues

```bash
# Test Redis
redis-cli ping
# Expected: PONG

# Check if running
docker ps | grep redis             # Docker
brew services list | grep redis    # Homebrew

# Restart
docker restart shiksha-redis       # Docker
brew services restart redis        # Homebrew
```

### Python Version Incompatibility

PyTorch may not support Python 3.13+:

```bash
# Check version
python3 --version

# Install Python 3.11 (if needed)
brew install python@3.11           # macOS
pyenv install 3.11.11              # Using pyenv

# Use specific version
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Model Download Fails

```bash
# Check disk space
df -h

# Check internet
ping huggingface.co

# Clear cache
rm -rf ~/.cache/huggingface/

# Set cache directory
export HF_HOME=./data/models

# Retry download
python3 -m backend.scripts.download_models
```

### Frontend Build Issues

```bash
# Clear npm cache
npm cache clean --force

# Reinstall dependencies
cd frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

### Permission Denied on Scripts

```bash
# Make scripts executable
chmod +x bin/*

# Fix Python paths
chown -R $USER:$USER venv/
```

### Virtual Environment Issues

```bash
# Recreate venv
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Verify activation
which python  # Should show path inside venv/
```

---

## Next Steps

After successful setup:

1. **Explore documentation**:
   - [API Reference](../reference/api.md)
   - [Architecture](../reference/architecture.md)
   - [Development Guide](../../DEVELOPMENT.md)

2. **Review demo instructions**:
   - See [Demo Guide](demo.md)

3. **Run test suite**:
   ```bash
   pytest
   ```

4. **Start development**:
   - Read [Contributing Guide](contributing.md)
   - Check [Development Guide](../../DEVELOPMENT.md)

---

## Getting Help

- **GitHub Issues**: [Report bugs](https://github.com/KDhiraj152/Siksha-Setu/issues)
- **Documentation**: Check [docs/](../) folder
- **Community**: Discussions welcome

---

## 👨‍💻 Author

**K Dhiraj** • [k.dhiraj.srihari@gmail.com](mailto:k.dhiraj.srihari@gmail.com) • [@KDhiraj152](https://github.com/KDhiraj152) • [LinkedIn](https://www.linkedin.com/in/k-dhiraj-83b025279/)

*Last updated: November 2025*

