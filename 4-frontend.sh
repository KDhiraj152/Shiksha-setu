#!/bin/bash
# ===================================================================
# ShikshaSetu - Frontend Only
# Starts React frontend (requires backend running separately)
# ===================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() {
    echo ""
    echo "======================================================================"
    echo -e "  ${MAGENTA}$1${NC}"
    echo "======================================================================"
    echo ""
}

print_status() { echo -e "${GREEN}✓${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }
print_info() { echo -e "${BLUE}ℹ${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }

if [ ! -d "frontend" ]; then
    print_error "Frontend directory not found"
    exit 1
fi

print_header "🎨 SHIKSHA SETU - FRONTEND"

# ===================================================================
# Checks
# ===================================================================
print_header "Pre-flight Checks"

if ! command -v node &> /dev/null; then
    print_error "Node.js not found"
    exit 1
fi
print_status "Node.js $(node --version)"

if [ ! -d "frontend/node_modules" ]; then
    print_warning "Installing dependencies..."
    cd frontend && npm install && cd ..
fi
print_status "Dependencies ready"

# Check backend
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    print_warning "Backend not running at http://localhost:8000"
    print_info "Start backend first: ./3-backend.sh"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    print_status "Backend is running"
fi

# ===================================================================
# Start Frontend
# ===================================================================
print_header "Starting Frontend"

cd frontend

if [ ! -f ".env" ]; then
    echo "VITE_API_BASE_URL=http://localhost:8000" > .env
    print_status "Created frontend .env"
fi

print_info "Starting React development server..."
npm run dev

# Script will stay here until Ctrl+C
