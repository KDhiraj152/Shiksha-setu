# 🚀 Setup Guide

Quick setup guide for ShikshaSetu.

## Prerequisites

- Python 3.13+
- Node.js 25+
- Redis 7+
- PostgreSQL 15+ (or Supabase)

## Quick Start

> **Note:** The setup scripts (`1-setup.sh`, `2-start.sh`, etc.) are located in the repository root. If they're missing or fail, see the [Manual Setup](#manual-setup) and [Fallback Steps](#fallback-steps) sections below.

```bash
# 1. Clone repository
git clone https://github.com/KDhiraj152/Siksha-Setu.git
cd shiksha_setu

# 2. Make scripts executable (if needed)
chmod +x *.sh

# 3. Run setup
./1-setup.sh

# 4. Start application
./2-start.sh
```

**Access:**
- Frontend: http://localhost:5173
- Backend: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Fallback Steps

If the automated scripts fail or are missing, follow these manual steps:

```bash
# 1. Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install backend dependencies
pip install -r config/requirements.txt

# 3. Install frontend dependencies
cd frontend
npm install
cd ..

# 4. Configure environment
cp .env.example .env
# Edit .env with your database, Redis, and API credentials

# 5. Run database migrations
source .venv/bin/activate
alembic upgrade head

# 6. Start Redis (if not running)
brew services start redis  # macOS
# OR: sudo systemctl start redis  # Linux
# OR: redis-server  # Manual start

# 7. Start backend (in one terminal)
source .venv/bin/activate
uvicorn src.api.async_app:app --host 0.0.0.0 --port 8000 --reload

# 8. Start frontend (in another terminal)
cd frontend
npm run dev
```

**Script Troubleshooting:**
- **Permission denied**: Run `chmod +x *.sh` to make scripts executable
- **Script not found**: Verify you're in the repository root directory
- **Command not found in script**: Ensure all prerequisites (Python, Node.js, Redis, PostgreSQL) are installed
- **Script hangs**: Check logs in `logs/` directory or run commands manually to identify the issue

## Manual Setup

### Backend
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r config/requirements.txt
```

### Frontend
```bash
cd frontend
npm install
```

### Environment
```bash
cp .env.example .env
# Edit .env with your settings
```

### Database
```bash
source .venv/bin/activate
alembic upgrade head
```

## Docker Setup

All monitoring stack images are pinned to specific versions for reproducibility:
- Prometheus: v3.0.1
- Grafana: 11.4.0
- Node Exporter: v1.8.2
- cAdvisor: v0.50.0

To enable development mode with auto-reload:
```bash
export UVICORN_RELOAD="--reload"
docker-compose up -d
```

## Scripts

- **`1-setup.sh`** - First-time setup
- **`2-start.sh`** - Start everything
- **`3-backend.sh`** - Backend only
- **`4-frontend.sh`** - Frontend only  
- **`test.sh`** - Run tests

## Troubleshooting

**Redis not running?**
```bash
brew services start redis  # macOS
sudo systemctl start redis # Linux
```

**Port already in use?**
```bash
lsof -ti:8000 | xargs kill  # Kill process on port 8000
```

**Database issues?**
```bash
source .venv/bin/activate
alembic downgrade -1
alembic upgrade head
```
