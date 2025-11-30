#!/bin/bash
# ============================================================================
# ShikshaSetu - Universal Stop Script
# Gracefully stops: Backend API, AI/ML Pipeline (Celery), Frontend
# Works on: macOS (Apple Silicon), Linux (Ubuntu/Debian), Docker
# ============================================================================

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m'

PID_DIR="/tmp/shiksha_setu"
STOPPED=0

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

stop_service() {
    local name="$1"
    local pid_file="$2"
    local pattern="$3"
    
    local stopped=0
    
    # Try PID file first
    if [[ -f "$pid_file" ]]; then
        local pid=$(cat "$pid_file")
        if ps -p "$pid" > /dev/null 2>&1; then
            print_info "Stopping $name (PID: $pid)..."
            kill "$pid" 2>/dev/null
            sleep 2
            
            # Force kill if still running
            if ps -p "$pid" > /dev/null 2>&1; then
                kill -9 "$pid" 2>/dev/null
                sleep 1
            fi
            stopped=1
        fi
        rm -f "$pid_file"
    fi
    
    # Fallback: find by pattern
    if [[ $stopped -eq 0 && -n "$pattern" ]]; then
        local found_pid=$(pgrep -f "$pattern" 2>/dev/null | head -1)
        if [[ -n "$found_pid" ]]; then
            print_info "Found $name process (PID: $found_pid)..."
            kill "$found_pid" 2>/dev/null
            sleep 1
            stopped=1
        fi
    fi
    
    if [[ $stopped -eq 1 ]]; then
        print_status "$name stopped"
        ((STOPPED++))
    else
        print_info "$name not running"
    fi
}

print_header "🛑 SHIKSHA SETU - STOP"

# ============================================================================
# STOP SERVICES
# ============================================================================

# 1. Backend API
stop_service "Backend API" "$PID_DIR/backend.pid" "uvicorn backend.api.main"

# 2. AI Pipeline (Celery)
stop_service "AI Pipeline" "$PID_DIR/celery.pid" "celery.*worker"

# 3. Frontend
stop_service "Frontend" "$PID_DIR/frontend.pid" "vite.*frontend"

# ============================================================================
# CLEANUP ORPHAN PROCESSES
# ============================================================================
print_header "Cleanup"

# Kill any remaining uvicorn processes
if pkill -f "uvicorn backend.api.main" 2>/dev/null; then
    print_status "Cleaned orphan uvicorn processes"
fi

# Kill any remaining celery processes
if pkill -f "celery.*worker" 2>/dev/null; then
    print_status "Cleaned orphan celery processes"
fi

# Kill any remaining vite processes
if pkill -f "node.*vite" 2>/dev/null; then
    print_status "Cleaned orphan vite processes"
fi

# ============================================================================
# SUMMARY
# ============================================================================
print_header "✅ SHUTDOWN COMPLETE"

if [[ $STOPPED -gt 0 ]]; then
    echo -e "  ${GREEN}Stopped $STOPPED service(s)${NC}"
else
    echo -e "  ${BLUE}No services were running${NC}"
fi

echo ""
echo "  To restart:"
echo -e "    ${CYAN}./start.sh${NC}"
echo ""
