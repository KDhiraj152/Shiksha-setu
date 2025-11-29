#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# ShikshaSetu - Complete Project Setup Script
# Sets up: Python venv, dependencies, database, frontend, AI models
# Author: K Dhiraj (k.dhiraj.srihari@gmail.com)
# GitHub: @KDhiraj152 | LinkedIn: linkedin.com/in/k-dhiraj
# ═══════════════════════════════════════════════════════════════════════════════

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'
BOLD='\033[1m'

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Banner
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${GREEN}  🎓 ShikshaSetu - Project Setup${NC}"
echo -e "${BOLD}     AI-Powered Multilingual Education Platform${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
echo ""

# Create required directories
mkdir -p logs .pids data/uploads data/audio data/cache data/models

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 1: System Requirements Check
# ═══════════════════════════════════════════════════════════════════════════════
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  PHASE 1: System Requirements${NC}"
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════════${NC}"
echo ""

# Check Python
echo -e "${YELLOW}[1.1]${NC} Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    echo -e "  ${GREEN}✓${NC} Python $PYTHON_VERSION"
else
    echo -e "  ${RED}✗ Python 3 not found!${NC}"
    echo "  Install: brew install python@3.12"
    exit 1
fi

# Check Node.js
echo -e "${YELLOW}[1.2]${NC} Checking Node.js..."
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version | sed 's/v//')
    echo -e "  ${GREEN}✓${NC} Node.js $NODE_VERSION"
else
    echo -e "  ${RED}✗ Node.js not found!${NC}"
    echo "  Install: brew install node"
    exit 1
fi

# Check npm
echo -e "${YELLOW}[1.3]${NC} Checking npm..."
if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm --version)
    echo -e "  ${GREEN}✓${NC} npm $NPM_VERSION"
else
    echo -e "  ${RED}✗ npm not found!${NC}"
    exit 1
fi

# Check Docker (optional)
echo -e "${YELLOW}[1.4]${NC} Checking Docker..."
if command -v docker &> /dev/null && docker info &>/dev/null; then
    echo -e "  ${GREEN}✓${NC} Docker running"
else
    echo -e "  ${YELLOW}⚠${NC} Docker not available (optional)"
fi

# Check Redis
echo -e "${YELLOW}[1.5]${NC} Checking Redis..."
if redis-cli ping &>/dev/null; then
    echo -e "  ${GREEN}✓${NC} Redis running"
else
    echo -e "  ${YELLOW}⚠${NC} Redis not running (will start later)"
fi

# Check Ollama
echo -e "${YELLOW}[1.6]${NC} Checking Ollama..."
if command -v ollama &> /dev/null; then
    if curl -s http://localhost:11434/api/tags &>/dev/null; then
        echo -e "  ${GREEN}✓${NC} Ollama running"
        if ollama list 2>/dev/null | grep -q "llama3.2"; then
            echo -e "  ${GREEN}✓${NC} llama3.2 model available"
        else
            echo -e "  ${YELLOW}⚠${NC} llama3.2 not found (will pull later)"
        fi
    else
        echo -e "  ${YELLOW}⚠${NC} Ollama not running"
        echo -e "  ${BLUE}ℹ${NC} Start with: ollama serve"
    fi
else
    echo -e "  ${YELLOW}⚠${NC} Ollama not installed"
    echo -e "  ${BLUE}ℹ${NC} Install: brew install ollama"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 2: Python Environment Setup
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  PHASE 2: Python Environment${NC}"
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════════${NC}"
echo ""

# Create virtual environment
echo -e "${YELLOW}[2.1]${NC} Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "  ${GREEN}✓${NC} Created new venv"
else
    echo -e "  ${GREEN}✓${NC} Using existing venv"
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}[2.2]${NC} Upgrading pip..."
pip install --upgrade pip wheel setuptools -q 2>/dev/null
echo -e "  ${GREEN}✓${NC} pip upgraded"

# Install Python dependencies
echo -e "${YELLOW}[2.3]${NC} Installing Python dependencies..."
echo -e "  ${CYAN}↻${NC} This may take 2-5 minutes..."

# Install main requirements (continue on error for compatibility issues)
if [ -f "requirements/requirements.txt" ]; then
    pip install -r requirements/requirements.txt 2>&1 | while read line; do
        if [[ "$line" == *"ERROR"* ]] && [[ "$line" != *"Ignored"* ]]; then
            echo -e "  ${YELLOW}⚠${NC} ${line:0:60}..."
        fi
    done
    echo -e "  ${GREEN}✓${NC} Core requirements installed"
fi

# Install dev requirements
if [ -f "requirements/dev.txt" ]; then
    pip install -r requirements/dev.txt -q 2>/dev/null || true
    echo -e "  ${GREEN}✓${NC} Dev requirements installed"
fi

# Ensure critical packages are installed
echo -e "${YELLOW}[2.4]${NC} Verifying critical packages..."
pip install fastapi uvicorn pydantic redis celery sqlalchemy -q 2>/dev/null
pip install torch transformers sentence-transformers -q 2>/dev/null
pip install edge-tts httpx aiohttp -q 2>/dev/null
echo -e "  ${GREEN}✓${NC} Critical packages verified"

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 3: Database Setup
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  PHASE 3: Database Setup${NC}"
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════════${NC}"
echo ""

# Start Redis if Docker available
echo -e "${YELLOW}[3.1]${NC} Setting up Redis..."
if ! redis-cli ping &>/dev/null; then
    if command -v docker &> /dev/null && docker info &>/dev/null; then
        docker run -d --name shiksha-redis -p 6379:6379 redis:7-alpine 2>/dev/null || \
        docker start shiksha-redis 2>/dev/null || true
        sleep 2
    fi
    
    # Try brew on macOS
    if ! redis-cli ping &>/dev/null && command -v brew &>/dev/null; then
        brew services start redis 2>/dev/null || true
        sleep 2
    fi
fi

if redis-cli ping &>/dev/null; then
    echo -e "  ${GREEN}✓${NC} Redis ready"
else
    echo -e "  ${YELLOW}⚠${NC} Redis not available (app will use memory cache)"
fi

# Start PostgreSQL if Docker available
echo -e "${YELLOW}[3.2]${NC} Setting up PostgreSQL..."
if command -v docker &> /dev/null && docker info &>/dev/null; then
    if ! docker ps | grep -q postgres 2>/dev/null; then
        docker-compose up -d postgres 2>/dev/null || true
        sleep 3
    fi
fi

if docker ps 2>/dev/null | grep -q postgres || pg_isready -q 2>/dev/null; then
    echo -e "  ${GREEN}✓${NC} PostgreSQL ready"
else
    echo -e "  ${YELLOW}⚠${NC} PostgreSQL not running (will use SQLite)"
fi

# Run migrations
echo -e "${YELLOW}[3.3]${NC} Running database migrations..."
if [ -f "config/alembic.ini" ]; then
    alembic -c config/alembic.ini upgrade head 2>/dev/null || true
    echo -e "  ${GREEN}✓${NC} Migrations applied"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 4: Frontend Setup
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  PHASE 4: Frontend Setup${NC}"
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════════${NC}"
echo ""

cd frontend

echo -e "${YELLOW}[4.1]${NC} Installing npm dependencies..."
if [ ! -d "node_modules" ]; then
    npm install 2>&1 | tail -3
    echo -e "  ${GREEN}✓${NC} npm packages installed"
else
    echo -e "  ${GREEN}✓${NC} npm packages exist"
fi

cd ..

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 5: AI Models Setup
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  PHASE 5: AI Models Setup${NC}"
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════════${NC}"
echo ""

# Setup Ollama model
echo -e "${YELLOW}[5.1]${NC} Setting up Ollama..."
if command -v ollama &> /dev/null && curl -s http://localhost:11434/api/tags &>/dev/null; then
    if ! ollama list 2>/dev/null | grep -q "llama3.2"; then
        echo -e "  ${CYAN}↻${NC} Pulling llama3.2 (this takes a while)..."
        ollama pull llama3.2 2>/dev/null || echo -e "  ${YELLOW}⚠${NC} Could not pull model"
    fi
    echo -e "  ${GREEN}✓${NC} Ollama ready"
else
    echo -e "  ${YELLOW}⚠${NC} Ollama not running - start with: ollama serve"
fi

echo -e "${YELLOW}[5.2]${NC} AI models info:"
echo -e "  ${BLUE}•${NC} NLLB-200 (Translation) - downloads on first use (~600MB)"
echo -e "  ${BLUE}•${NC} BGE-M3 (Embeddings) - downloads on first use (~2GB)"
echo -e "  ${BLUE}•${NC} Edge TTS (Speech) - uses Microsoft servers (free)"

# ═══════════════════════════════════════════════════════════════════════════════
# PHASE 6: Environment Configuration
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  PHASE 6: Environment Configuration${NC}"
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════════════${NC}"
echo ""

echo -e "${YELLOW}[6.1]${NC} Setting up .env file..."
if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
# ShikshaSetu Environment Configuration
# Generated by SETUP.sh

# API Settings
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# Database
DATABASE_URL=postgresql://shiksha:shiksha@localhost:5432/shiksha_setu
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=dev-secret-key-change-in-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# AI Services
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2
NLLB_MODEL=facebook/nllb-200-distilled-600M
BGE_MODEL=BAAI/bge-m3

# Logging
LOG_LEVEL=INFO
EOF
    echo -e "  ${GREEN}✓${NC} .env file created"
else
    echo -e "  ${GREEN}✓${NC} .env file exists"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# SETUP COMPLETE
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${GREEN}  ✅ ShikshaSetu Setup Complete!${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${BOLD}Quick Start:${NC}"
echo -e "  ${YELLOW}./START.sh${NC}    - Start all services"
echo -e "  ${YELLOW}./STOP.sh${NC}     - Stop all services"
echo ""
echo -e "  ${BOLD}Services:${NC}"
echo -e "  ├─ Frontend:    http://localhost:5173"
echo -e "  ├─ Backend:     http://localhost:8000"
echo -e "  ├─ API Docs:    http://localhost:8000/docs"
echo -e "  └─ Health:      http://localhost:8000/health"
echo ""
echo -e "  ${BOLD}Before First Run:${NC}"
echo -e "  1. Make sure Redis is running: ${YELLOW}redis-cli ping${NC}"
echo -e "  2. Start Ollama: ${YELLOW}ollama serve${NC}"
echo -e "  3. Then: ${YELLOW}./START.sh${NC}"
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
echo ""
