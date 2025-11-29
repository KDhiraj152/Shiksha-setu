#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# ShikshaSetu - Complete Shutdown Script
# Stops ALL services: Backend, Celery, Frontend, Docker containers
# Author: K Dhiraj (k.dhiraj.srihari@gmail.com)
# GitHub: @KDhiraj152 | LinkedIn: linkedin.com/in/k-dhiraj
# ═══════════════════════════════════════════════════════════════════════════════

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Banner
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${RED}  🛑 ShikshaSetu - Stopping All Services${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Stop Frontend
# ─────────────────────────────────────────────────────────────────────────────
echo -e "${YELLOW}[1/5]${NC} Stopping Frontend..."

if [ -f ".pids/frontend.pid" ]; then
    FRONTEND_PID=$(cat .pids/frontend.pid 2>/dev/null)
    if kill -0 $FRONTEND_PID 2>/dev/null; then
        kill $FRONTEND_PID 2>/dev/null || true
        echo -e "  ${GREEN}✓${NC} Frontend stopped (PID: $FRONTEND_PID)"
    fi
    rm -f .pids/frontend.pid
fi

# Kill any remaining vite processes
pkill -f "vite" 2>/dev/null || true
echo -e "  ${GREEN}✓${NC} Cleaned up Vite processes"

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Stop Celery Worker
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[2/5]${NC} Stopping Celery worker..."

if [ -f ".pids/celery.pid" ]; then
    CELERY_PID=$(cat .pids/celery.pid 2>/dev/null)
    if kill -0 $CELERY_PID 2>/dev/null; then
        kill $CELERY_PID 2>/dev/null || true
        echo -e "  ${GREEN}✓${NC} Celery stopped (PID: $CELERY_PID)"
    fi
    rm -f .pids/celery.pid
fi

# Kill any remaining celery processes
pkill -f "celery -A backend" 2>/dev/null || true
pkill -f "celery worker" 2>/dev/null || true
echo -e "  ${GREEN}✓${NC} Cleaned up Celery processes"

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Stop Backend API
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[3/5]${NC} Stopping Backend API..."

if [ -f ".pids/backend.pid" ]; then
    BACKEND_PID=$(cat .pids/backend.pid 2>/dev/null)
    if kill -0 $BACKEND_PID 2>/dev/null; then
        kill $BACKEND_PID 2>/dev/null || true
        echo -e "  ${GREEN}✓${NC} Backend stopped (PID: $BACKEND_PID)"
    fi
    rm -f .pids/backend.pid
fi

# Kill any remaining uvicorn processes
pkill -f "uvicorn backend.api.main" 2>/dev/null || true
pkill -f "uvicorn.*8000" 2>/dev/null || true
echo -e "  ${GREEN}✓${NC} Cleaned up uvicorn processes"

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Kill Tmux Session (if exists)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[4/5]${NC} Cleaning up sessions..."

if command -v tmux &> /dev/null; then
    tmux kill-session -t shiksha 2>/dev/null && echo -e "  ${GREEN}✓${NC} Tmux session killed" || true
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Optional Docker Cleanup
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}[5/5]${NC} Docker services..."

# Don't stop Docker services by default (they might be shared)
echo -e "  ${BLUE}ℹ${NC} Docker containers (postgres/redis) left running"
echo -e "  ${BLUE}ℹ${NC} To stop: docker-compose down"

# Clean up old PID files
rm -rf .pids/* 2>/dev/null || true
rm -f .backend.pid .celery.pid .frontend.pid 2>/dev/null || true

# ─────────────────────────────────────────────────────────────────────────────
# Final Status
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}${GREEN}  ✅ All ShikshaSetu Services Stopped${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${BOLD}Restart:${NC} ${YELLOW}./START.sh${NC}"
echo ""

# Verify ports are free
if ! lsof -i:8000 &>/dev/null; then
    echo -e "  ${GREEN}✓${NC} Port 8000 is free"
else
    echo -e "  ${YELLOW}⚠${NC} Port 8000 still in use - may need manual cleanup"
fi

if ! lsof -i:5173 &>/dev/null; then
    echo -e "  ${GREEN}✓${NC} Port 5173 is free"
else
    echo -e "  ${YELLOW}⚠${NC} Port 5173 still in use - may need manual cleanup"
fi

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════════${NC}"
echo ""
