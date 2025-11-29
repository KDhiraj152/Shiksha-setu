#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# ShikshaSetu - Complete Startup Script
# Starts ALL services: Redis, PostgreSQL, Backend, Celery, Frontend
# Author: K Dhiraj (k.dhiraj.srihari@gmail.com)
# GitHub: @KDhiraj152 | LinkedIn: linkedin.com/in/k-dhiraj
# ═══════════════════════════════════════════════════════════════════════════════

set -e

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
echo -e "${BOLD}${GREEN}  🎓 ShikshaSetu - AI Education Platform${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
echo ""

# Create required directories
mkdir -p logs .pids data/uploads data/audio data/cache data/models

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Check Prerequisites
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${YELLOW}[1/6]${NC} Checking prerequisites..."

# Check venv
if [ ! -d "venv" ]; then
    echo -e "${RED}✗ Virtual environment not found!${NC}"
    echo -e "  Run: ${YELLOW}./SETUP.sh${NC} first"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Virtual environment found"

# Check frontend node_modules
if [ ! -d "frontend/node_modules" ]; then
    echo -e "${RED}✗ Frontend dependencies not installed!${NC}"
    echo -e "  Run: ${YELLOW}./SETUP.sh${NC} first"
    exit 1
fi
echo -e "  ${GREEN}✓${NC} Frontend dependencies found"

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Start Database Services (Docker)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[2/6]${NC} Starting database services..."

# Check if Docker is available
if command -v docker &> /dev/null; then
    # Start PostgreSQL if not running
    if ! docker ps | grep -q "shiksha.*postgres\|postgres.*shiksha" 2>/dev/null; then
        if docker-compose ps 2>/dev/null | grep -q postgres; then
            docker-compose start postgres 2>/dev/null || true
        else
            docker-compose up -d postgres 2>/dev/null || true
        fi
        sleep 2
    fi
    
    # Check PostgreSQL
    if docker ps | grep -q postgres 2>/dev/null || pg_isready -q 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} PostgreSQL running"
    else
        echo -e "  ${YELLOW}⚠${NC} PostgreSQL not running (will use existing connection)"
    fi
else
    echo -e "  ${YELLOW}⚠${NC} Docker not found - assuming external database"
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Start Redis
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[3/6]${NC} Starting Redis..."

# Check if Redis is already running
if redis-cli ping &>/dev/null; then
    echo -e "  ${GREEN}✓${NC} Redis already running"
else
    # Try Docker first
    if command -v docker &> /dev/null; then
        if ! docker ps | grep -q redis 2>/dev/null; then
            docker-compose up -d redis 2>/dev/null || docker run -d --name shiksha-redis -p 6379:6379 redis:7-alpine 2>/dev/null || true
        fi
        sleep 2
    fi
    
    # Try brew services (macOS)
    if ! redis-cli ping &>/dev/null && command -v brew &>/dev/null; then
        brew services start redis 2>/dev/null || true
        sleep 2
    fi
    
    # Final check
    if redis-cli ping &>/dev/null; then
        echo -e "  ${GREEN}✓${NC} Redis started"
    else
        echo -e "  ${YELLOW}⚠${NC} Redis not available - rate limiting will use memory"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Start Backend API
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[4/6]${NC} Starting Backend API..."

# Kill any existing backend process
pkill -f "uvicorn backend.api.main" 2>/dev/null || true
sleep 1

# Activate venv and start backend
source venv/bin/activate

# Start backend in background
nohup uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload > logs/backend.log 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > .pids/backend.pid
echo -e "  ${GREEN}✓${NC} Backend starting (PID: $BACKEND_PID)"

# Wait for backend to be ready
echo -e "  ${CYAN}↻${NC} Waiting for backend..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health &>/dev/null; then
        echo -e "  ${GREEN}✓${NC} Backend ready at ${BLUE}http://localhost:8000${NC}"
        break
    fi
    sleep 1
    if [ $i -eq 30 ]; then
        echo -e "  ${YELLOW}⚠${NC} Backend taking longer than expected (check logs/backend.log)"
    fi
done

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Start Celery Worker
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[5/6]${NC} Starting Celery worker..."

# Kill any existing celery process
pkill -f "celery -A backend.tasks" 2>/dev/null || true
sleep 1

# Start Celery worker (use solo pool on macOS to avoid fork issues with MPS)
if [[ "$(uname)" == "Darwin" ]]; then
    nohup celery -A backend.tasks.celery_app worker --loglevel=info --pool=solo > logs/celery.log 2>&1 &
else
    nohup celery -A backend.tasks.celery_app worker --loglevel=info --concurrency=2 > logs/celery.log 2>&1 &
fi
CELERY_PID=$!
echo $CELERY_PID > .pids/celery.pid
echo -e "  ${GREEN}✓${NC} Celery worker started (PID: $CELERY_PID)"

# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Start Frontend
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[6/6]${NC} Starting Frontend..."

# Kill any existing frontend process
pkill -f "vite" 2>/dev/null || true
sleep 1

# Start frontend
cd frontend
nohup npm run dev > ../logs/frontend.log 2>&1 &
FRONTEND_PID=$!
echo $FRONTEND_PID > ../.pids/frontend.pid
cd ..
echo -e "  ${GREEN}✓${NC} Frontend starting (PID: $FRONTEND_PID)"

# Wait for frontend
echo -e "  ${CYAN}↻${NC} Waiting for frontend..."
for i in {1..15}; do
    if curl -s http://localhost:5173 &>/dev/null; then
        echo -e "  ${GREEN}✓${NC} Frontend ready at ${BLUE}http://localhost:5173${NC}"
        break
    fi
    sleep 1
done

# ─────────────────────────────────────────────────────────────────────────────
# Final Status
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${GREEN}  ✅ ShikshaSetu is Running!${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${BOLD}Services:${NC}"
echo -e "  ├─ Frontend:    ${BLUE}http://localhost:5173${NC}"
echo -e "  ├─ Backend:     ${BLUE}http://localhost:8000${NC}"
echo -e "  ├─ API Docs:    ${BLUE}http://localhost:8000/docs${NC}"
echo -e "  └─ Health:      ${BLUE}http://localhost:8000/health${NC}"
echo ""
echo -e "  ${BOLD}Logs:${NC}"
echo -e "  ├─ Backend:     ${YELLOW}tail -f logs/backend.log${NC}"
echo -e "  ├─ Celery:      ${YELLOW}tail -f logs/celery.log${NC}"
echo -e "  └─ Frontend:    ${YELLOW}tail -f logs/frontend.log${NC}"
echo ""
echo -e "  ${BOLD}Stop all:${NC} ${YELLOW}./STOP.sh${NC}"
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
echo ""
