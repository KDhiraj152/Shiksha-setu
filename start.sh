#!/bin/bash
# ============================================================================
# ShikshaSetu - Universal Start Script
# Starts: Backend API, AI/ML Pipeline (Celery), Frontend
# Works on: macOS (Apple Silicon), Linux (Ubuntu/Debian), Docker
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

# PID storage
PID_DIR="/tmp/shiksha_setu"
mkdir -p "$PID_DIR"

print_header() {
    echo ""
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${MAGENTA}  $1${NC}"
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_status()  { echo -e "  ${GREEN}✓${NC} $1"; }
print_error()   { echo -e "  ${RED}✗${NC} $1"; }
print_info()    { echo -e "  ${BLUE}ℹ${NC} $1"; }
print_warning() { echo -e "  ${YELLOW}⚠${NC} $1"; }
print_service() { echo -e "  ${CYAN}→${NC} $1"; }

# Cleanup function
cleanup() {
    echo ""
    print_header "🛑 STOPPING SERVICES"
    
    # Stop backend
    if [[ -f "$PID_DIR/backend.pid" ]]; then
        kill "$(cat $PID_DIR/backend.pid)" 2>/dev/null || true
        rm -f "$PID_DIR/backend.pid"
        print_status "Backend stopped"
    fi
    
    # Stop Celery
    if [[ -f "$PID_DIR/celery.pid" ]]; then
        kill "$(cat $PID_DIR/celery.pid)" 2>/dev/null || true
        rm -f "$PID_DIR/celery.pid"
        print_status "AI Pipeline stopped"
    fi
    
    # Stop frontend
    if [[ -f "$PID_DIR/frontend.pid" ]]; then
        kill "$(cat $PID_DIR/frontend.pid)" 2>/dev/null || true
        rm -f "$PID_DIR/frontend.pid"
        print_status "Frontend stopped"
    fi
    
    # Cleanup orphan processes
    pkill -f "uvicorn backend.api.main" 2>/dev/null || true
    pkill -f "celery.*worker" 2>/dev/null || true
    
    print_status "All services stopped"
    exit 0
}

trap cleanup INT TERM

# Ensure we're in project root
if [[ ! -f "requirements.txt" || ! -d "backend" ]]; then
    print_error "Run this script from the project root directory"
    exit 1
fi

PROJECT_ROOT="$(pwd)"

print_header "🚀 SHIKSHA SETU - START"

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================
print_header "Pre-flight Checks"

# Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found. Run ./setup.sh first"
    exit 1
fi
print_status "Python $(python3 --version | cut -d' ' -f2)"

# Virtual environment
if [[ ! -d ".venv" ]]; then
    print_error "Virtual environment not found. Run ./setup.sh first"
    exit 1
fi
print_status "Virtual environment ready"

# .env file
if [[ ! -f ".env" ]]; then
    print_error ".env not found. Run ./setup.sh first"
    exit 1
fi
print_status ".env configured"

# Redis
if ! redis-cli ping &> /dev/null 2>&1; then
    print_warning "Redis not running"
    
    # Try to start Redis
    if [[ "$(uname -s)" == "Darwin" ]] && command -v brew &> /dev/null; then
        print_info "Starting Redis via Homebrew..."
        brew services start redis 2>/dev/null || true
        sleep 2
    fi
    
    if ! redis-cli ping &> /dev/null 2>&1; then
        print_error "Redis required. Start with: redis-server"
        exit 1
    fi
fi
print_status "Redis running"

# Node.js (for frontend)
if command -v node &> /dev/null; then
    print_status "Node.js $(node -v)"
    if [[ ! -d "frontend/node_modules" ]]; then
        print_info "Installing frontend dependencies..."
        cd frontend && npm install --silent && cd ..
    fi
else
    print_warning "Node.js not found - frontend will be skipped"
    SKIP_FRONTEND=1
fi

# Create logs directory
mkdir -p logs

# ============================================================================
# LOAD ENVIRONMENT
# ============================================================================
print_header "Loading Environment"

# Export environment variables
set -a
source .env
set +a
print_status "Environment variables loaded"

# Activate virtual environment
source .venv/bin/activate
print_status "Virtual environment activated"

# ============================================================================
# START SERVICES
# ============================================================================
print_header "Starting Services"

# 1. Backend API (FastAPI + Uvicorn)
print_service "Starting Backend API..."
nohup uvicorn backend.api.main:app \
    --host ${HOST:-0.0.0.0} \
    --port ${PORT:-8000} \
    --reload \
    > logs/backend.log 2>&1 &
BACKEND_PID=$!
echo $BACKEND_PID > "$PID_DIR/backend.pid"
sleep 2

# Health check
if curl -s http://localhost:${PORT:-8000}/health > /dev/null 2>&1; then
    print_status "Backend API running (PID: $BACKEND_PID)"
else
    print_warning "Backend starting... (check logs/backend.log)"
fi

# 2. AI/ML Pipeline (Celery Worker)
print_service "Starting AI Pipeline (Celery)..."
nohup celery -A backend.tasks.celery_app worker \
    --loglevel=info \
    --concurrency=${CELERY_WORKER_CONCURRENCY:-2} \
    --queues=default,pipeline,ocr,translate,simplify \
    > logs/celery.log 2>&1 &
CELERY_PID=$!
echo $CELERY_PID > "$PID_DIR/celery.pid"
print_status "AI Pipeline running (PID: $CELERY_PID)"
sleep 2

# 3. Frontend (React + Vite)
if [[ -z "$SKIP_FRONTEND" ]]; then
    print_service "Starting Frontend..."
    cd frontend
    nohup npm run dev > ../logs/frontend.log 2>&1 &
    FRONTEND_PID=$!
    echo $FRONTEND_PID > "$PID_DIR/frontend.pid"
    cd "$PROJECT_ROOT"
    print_status "Frontend running (PID: $FRONTEND_PID)"
    sleep 3
fi

# ============================================================================
# STATUS DASHBOARD
# ============================================================================
print_header "📊 Service Status"

echo ""
# Backend status
if curl -s http://localhost:${PORT:-8000}/health > /dev/null 2>&1; then
    echo -e "  ${GREEN}●${NC} Backend      ${GREEN}RUNNING${NC}   http://localhost:${PORT:-8000}"
else
    echo -e "  ${YELLOW}●${NC} Backend      ${YELLOW}STARTING${NC}  http://localhost:${PORT:-8000}"
fi

# Celery status
if ps -p $CELERY_PID > /dev/null 2>&1; then
    echo -e "  ${GREEN}●${NC} AI Pipeline  ${GREEN}RUNNING${NC}   Celery worker"
else
    echo -e "  ${RED}●${NC} AI Pipeline  ${RED}FAILED${NC}    Check logs/celery.log"
fi

# Frontend status
if [[ -z "$SKIP_FRONTEND" ]]; then
    sleep 2
    if curl -s http://localhost:5173 > /dev/null 2>&1; then
        echo -e "  ${GREEN}●${NC} Frontend     ${GREEN}RUNNING${NC}   http://localhost:5173"
    else
        echo -e "  ${YELLOW}●${NC} Frontend     ${YELLOW}STARTING${NC}  http://localhost:5173"
    fi
fi

# Redis status
echo -e "  ${GREEN}●${NC} Redis        ${GREEN}RUNNING${NC}   localhost:6379"
echo ""

# ============================================================================
# ACCESS INFORMATION
# ============================================================================
print_header "🎉 ShikshaSetu is Running!"

echo ""
echo -e "  ${CYAN}Access Points:${NC}"
echo "  ─────────────────────────────────────────"
echo -e "  Frontend:     ${GREEN}http://localhost:5173${NC}"
echo -e "  Backend API:  ${GREEN}http://localhost:${PORT:-8000}${NC}"
echo -e "  API Docs:     ${GREEN}http://localhost:${PORT:-8000}/docs${NC}"
echo -e "  Redoc:        ${GREEN}http://localhost:${PORT:-8000}/redoc${NC}"
echo ""
echo -e "  ${CYAN}Logs:${NC}"
echo "  ─────────────────────────────────────────"
echo "  Backend:      tail -f logs/backend.log"
echo "  AI Pipeline:  tail -f logs/celery.log"
echo "  Frontend:     tail -f logs/frontend.log"
echo ""
echo -e "  ${YELLOW}Press Ctrl+C to stop all services${NC}"
echo ""

# ============================================================================
# MONITOR SERVICES
# ============================================================================
while true; do
    sleep 5
    
    # Check backend
    if ! ps -p "$(cat $PID_DIR/backend.pid 2>/dev/null)" > /dev/null 2>&1; then
        print_error "Backend stopped unexpectedly"
        print_info "Check logs/backend.log for errors"
        cleanup
    fi
    
    # Check Celery
    if ! ps -p "$(cat $PID_DIR/celery.pid 2>/dev/null)" > /dev/null 2>&1; then
        print_error "AI Pipeline stopped unexpectedly"
        print_info "Check logs/celery.log for errors"
        cleanup
    fi
    
    # Check frontend (if running)
    if [[ -z "$SKIP_FRONTEND" && -f "$PID_DIR/frontend.pid" ]]; then
        if ! ps -p "$(cat $PID_DIR/frontend.pid 2>/dev/null)" > /dev/null 2>&1; then
            print_error "Frontend stopped unexpectedly"
            print_info "Check logs/frontend.log for errors"
            cleanup
        fi
    fi
done
