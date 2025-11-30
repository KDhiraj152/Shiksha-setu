#!/bin/bash
# ============================================================================
# ShikshaSetu - Universal Setup Script
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

# Detect OS
detect_os() {
    case "$(uname -s)" in
        Darwin*) OS="macos" ;;
        Linux*)  OS="linux" ;;
        *)       OS="unknown" ;;
    esac
    
    # Check for Apple Silicon
    if [[ "$OS" == "macos" && "$(uname -m)" == "arm64" ]]; then
        ARCH="apple_silicon"
    else
        ARCH="$(uname -m)"
    fi
}

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

# Ensure we're in project root
if [[ ! -f "requirements.txt" || ! -d "backend" ]]; then
    print_error "Run this script from the project root directory"
    exit 1
fi

PROJECT_ROOT="$(pwd)"

detect_os
print_header "🎓 SHIKSHA SETU - SETUP"
echo ""
echo -e "  OS:           ${CYAN}$OS${NC}"
echo -e "  Architecture: ${CYAN}$ARCH${NC}"
echo -e "  Project Root: ${CYAN}$PROJECT_ROOT${NC}"

# ============================================================================
# 1. PREREQUISITES CHECK
# ============================================================================
print_header "1/6 Prerequisites"

# Python 3.11+
if command -v python3 &> /dev/null; then
    PY_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PY_MAJOR=$(echo $PY_VERSION | cut -d. -f1)
    PY_MINOR=$(echo $PY_VERSION | cut -d. -f2)
    if [[ $PY_MAJOR -ge 3 && $PY_MINOR -ge 11 ]]; then
        print_status "Python $PY_VERSION"
    else
        print_error "Python 3.11+ required (found $PY_VERSION)"
        exit 1
    fi
else
    print_error "Python 3 not found. Install from python.org"
    exit 1
fi

# Node.js 20+
if command -v node &> /dev/null; then
    NODE_VERSION=$(node -v | sed 's/v//')
    NODE_MAJOR=$(echo $NODE_VERSION | cut -d. -f1)
    if [[ $NODE_MAJOR -ge 20 ]]; then
        print_status "Node.js $NODE_VERSION"
    else
        print_warning "Node.js 20+ recommended (found $NODE_VERSION)"
    fi
else
    print_warning "Node.js not found - frontend setup will be skipped"
    SKIP_FRONTEND=1
fi

# Redis
if command -v redis-cli &> /dev/null; then
    if redis-cli ping &> /dev/null 2>&1; then
        print_status "Redis running"
    else
        print_warning "Redis installed but not running"
        if [[ "$OS" == "macos" ]]; then
            print_info "Start with: brew services start redis"
        else
            print_info "Start with: sudo systemctl start redis"
        fi
    fi
else
    print_warning "Redis not found"
    if [[ "$OS" == "macos" ]]; then
        print_info "Install: brew install redis"
    else
        print_info "Install: sudo apt install redis-server"
    fi
fi

# PostgreSQL (optional - can use Supabase)
if command -v psql &> /dev/null; then
    print_status "PostgreSQL available"
else
    print_info "PostgreSQL not found (can use Supabase instead)"
fi

# ============================================================================
# 2. ENVIRONMENT CONFIGURATION
# ============================================================================
print_header "2/6 Environment"

if [[ ! -f ".env" ]]; then
    if [[ -f ".env.example" ]]; then
        cp .env.example .env
        
        # Generate secure JWT secret
        JWT_SECRET=$(python3 -c "import secrets; print(secrets.token_urlsafe(64))")
        if [[ "$OS" == "macos" ]]; then
            sed -i '' "s|JWT_SECRET_KEY=.*|JWT_SECRET_KEY=$JWT_SECRET|g" .env
        else
            sed -i "s|JWT_SECRET_KEY=.*|JWT_SECRET_KEY=$JWT_SECRET|g" .env
        fi
        
        print_status ".env created with secure JWT secret"
        print_warning "Update DATABASE_URL in .env with your credentials"
    else
        print_error ".env.example not found"
        exit 1
    fi
else
    print_status ".env already exists"
fi

# ============================================================================
# 3. PYTHON VIRTUAL ENVIRONMENT
# ============================================================================
print_header "3/6 Python Environment"

if [[ ! -d ".venv" ]]; then
    print_info "Creating virtual environment..."
    python3 -m venv .venv
    print_status "Virtual environment created"
else
    print_status "Virtual environment exists"
fi

# Activate venv
source .venv/bin/activate

# Upgrade pip
print_info "Upgrading pip..."
pip install --upgrade pip -q

# Install dependencies
print_info "Installing Python dependencies..."
pip install -r requirements.txt -q

# Install dev dependencies if in development
if [[ -f "requirements.dev.txt" ]]; then
    print_info "Installing development dependencies..."
    pip install -r requirements.dev.txt -q
fi

print_status "Python dependencies installed"

# ============================================================================
# 4. FRONTEND SETUP
# ============================================================================
if [[ -z "$SKIP_FRONTEND" ]]; then
    print_header "4/6 Frontend"
    
    cd frontend
    
    if [[ ! -d "node_modules" ]]; then
        print_info "Installing Node.js dependencies..."
        npm install --silent
        print_status "Frontend dependencies installed"
    else
        print_status "Frontend dependencies exist"
    fi
    
    if [[ ! -f ".env" ]]; then
        echo "VITE_API_BASE_URL=http://localhost:8000" > .env
        echo "VITE_WS_BASE_URL=ws://localhost:8000" >> .env
        print_status "Frontend .env created"
    fi
    
    cd "$PROJECT_ROOT"
else
    print_header "4/6 Frontend (Skipped)"
    print_info "Install Node.js to enable frontend setup"
fi

# ============================================================================
# 5. CREATE DIRECTORIES
# ============================================================================
print_header "5/6 Directory Structure"

# Create required directories
mkdir -p data/{uploads,audio,models,cache}
mkdir -p logs
mkdir -p storage/{audio,cache,models}

print_status "Created data directories"
print_status "Created logs directory"
print_status "Created storage directories"

# ============================================================================
# 6. DATABASE INITIALIZATION
# ============================================================================
print_header "6/6 Database"

print_info "Initializing database..."
if python3 -c "from backend.database import init_db; init_db()" 2>/dev/null; then
    print_status "Database initialized"
else
    print_warning "Database init skipped (configure DATABASE_URL in .env)"
fi

# ============================================================================
# SUCCESS
# ============================================================================
print_header "✅ SETUP COMPLETE"

echo ""
echo -e "  ${GREEN}Next Steps:${NC}"
echo ""
echo "  1. Configure your database in .env"
echo "  2. Start Redis if not running:"
if [[ "$OS" == "macos" ]]; then
    echo -e "     ${CYAN}brew services start redis${NC}"
else
    echo -e "     ${CYAN}sudo systemctl start redis${NC}"
fi
echo ""
echo "  3. Start the application:"
echo -e "     ${CYAN}./start.sh${NC}"
echo ""
echo "  4. Access the application:"
echo -e "     Frontend: ${CYAN}http://localhost:5173${NC}"
echo -e "     Backend:  ${CYAN}http://localhost:8000${NC}"
echo -e "     API Docs: ${CYAN}http://localhost:8000/docs${NC}"
echo ""

