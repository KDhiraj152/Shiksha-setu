#!/bin/bash
# ==============================================================================
# ShikshaSetu AI Stack Setup Script
# Optimized for M4 MacBook Pro 16GB
# ==============================================================================

set -e

echo "🚀 ShikshaSetu AI Stack Setup"
echo "=============================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${YELLOW}Warning: This script is optimized for macOS with Apple Silicon.${NC}"
    echo "Some features may not work on other platforms."
fi

# Check for Apple Silicon
if [[ $(uname -m) == "arm64" ]]; then
    echo -e "${GREEN}✓ Apple Silicon detected - MPS acceleration available${NC}"
else
    echo -e "${YELLOW}⚠ Not running on Apple Silicon - will use CPU${NC}"
fi

# ==============================================================================
# Step 1: Install Homebrew (if not installed)
# ==============================================================================
echo ""
echo -e "${BLUE}Step 1: Checking Homebrew...${NC}"

if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
fi
echo -e "${GREEN}✓ Homebrew ready${NC}"

# ==============================================================================
# Step 2: Install Ollama
# ==============================================================================
echo ""
echo -e "${BLUE}Step 2: Installing Ollama...${NC}"

if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama via Homebrew..."
    brew install ollama
else
    echo -e "${GREEN}✓ Ollama already installed${NC}"
fi

# Start Ollama service
echo "Starting Ollama service..."
if ! pgrep -x "ollama" > /dev/null; then
    ollama serve &> /dev/null &
    sleep 2
fi
echo -e "${GREEN}✓ Ollama service running${NC}"

# ==============================================================================
# Step 3: Pull Required Models for Ollama
# ==============================================================================
echo ""
echo -e "${BLUE}Step 3: Pulling AI models for Ollama...${NC}"

echo "Pulling Llama 3.2 3B (for text simplification)..."
ollama pull llama3.2:3b

echo -e "${GREEN}✓ Ollama models ready${NC}"

# ==============================================================================
# Step 4: Create Python Virtual Environment
# ==============================================================================
echo ""
echo -e "${BLUE}Step 4: Setting up Python environment...${NC}"

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [[ $PYTHON_MAJOR -lt 3 ]] || [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -lt 10 ]]; then
    echo -e "${YELLOW}Python 3.10+ required. Installing via Homebrew...${NC}"
    brew install python@3.11
    PYTHON_CMD="python3.11"
else
    PYTHON_CMD="python3"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
fi

# Activate virtual environment
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# ==============================================================================
# Step 5: Install Python Dependencies
# ==============================================================================
echo ""
echo -e "${BLUE}Step 5: Installing Python dependencies...${NC}"

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with MPS support
echo "Installing PyTorch with Metal acceleration..."
pip install torch torchvision torchaudio

# Install main requirements
echo "Installing project requirements..."
pip install -r requirements/requirements.txt

echo -e "${GREEN}✓ Python dependencies installed${NC}"

# ==============================================================================
# Step 6: Download AI Models
# ==============================================================================
echo ""
echo -e "${BLUE}Step 6: Downloading AI models (this may take a while)...${NC}"

# Create models directory
mkdir -p data/models

# Download NLLB-200 via CTranslate2 conversion
echo "Preparing NLLB-200 translation model..."
python3 << 'EOF'
import os
try:
    from ctranslate2.converters import TransformersConverter
    
    model_path = "data/models/nllb-200-1.3B-ct2"
    if not os.path.exists(model_path):
        print("Converting NLLB-200 to CTranslate2 format...")
        converter = TransformersConverter(
            "facebook/nllb-200-1.3B",
            load_as_float16=True
        )
        converter.convert(model_path, quantization="int8")
        print("✓ NLLB-200 model ready")
    else:
        print("✓ NLLB-200 model already exists")
except ImportError:
    print("⚠ CTranslate2 not installed, will download model on first use")
except Exception as e:
    print(f"⚠ Model conversion will happen on first use: {e}")
EOF

# Pre-download BGE-M3 embeddings
echo "Preparing BGE-M3 embeddings model..."
python3 << 'EOF'
try:
    from FlagEmbedding import BGEM3FlagModel
    print("Loading BGE-M3 model (downloading if needed)...")
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    print("✓ BGE-M3 model ready")
    del model  # Free memory
except ImportError:
    print("⚠ FlagEmbedding not installed, will download on first use")
except Exception as e:
    print(f"⚠ Will download BGE-M3 on first use: {e}")
EOF

echo -e "${GREEN}✓ AI models prepared${NC}"

# ==============================================================================
# Step 7: Setup Database
# ==============================================================================
echo ""
echo -e "${BLUE}Step 7: Checking database setup...${NC}"

# Check if PostgreSQL is running
if command -v pg_isready &> /dev/null; then
    if pg_isready -q; then
        echo -e "${GREEN}✓ PostgreSQL is running${NC}"
    else
        echo -e "${YELLOW}⚠ PostgreSQL not running. Start it with: brew services start postgresql${NC}"
    fi
else
    echo -e "${YELLOW}⚠ PostgreSQL not found. Install with: brew install postgresql@15${NC}"
fi

# Check Redis
if command -v redis-cli &> /dev/null; then
    if redis-cli ping &> /dev/null; then
        echo -e "${GREEN}✓ Redis is running${NC}"
    else
        echo -e "${YELLOW}⚠ Redis not running. Start it with: brew services start redis${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Redis not found. Install with: brew install redis${NC}"
fi

# ==============================================================================
# Step 8: Create Environment File
# ==============================================================================
echo ""
echo -e "${BLUE}Step 8: Creating environment configuration...${NC}"

if [ ! -f ".env" ]; then
    cat > .env << 'ENVFILE'
# ShikshaSetu Environment Configuration
# Optimized for M4 MacBook Pro 16GB

# Database
DATABASE_URL=postgresql://localhost:5432/shiksha_setu
REDIS_URL=redis://localhost:6379

# AI Configuration
AI_DEVICE=mps
AI_COMPUTE_TYPE=int8
AI_MAX_MEMORY_GB=10.0
AI_IDLE_TIMEOUT=300

# Model Configuration
TRANSLATION_MODEL=facebook/nllb-200-1.3B
LLM_MODEL=llama3.2:3b
EMBEDDING_MODEL=BAAI/bge-m3
OLLAMA_HOST=http://localhost:11434

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# Security
SECRET_KEY=your-secret-key-change-in-production
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Logging
LOG_LEVEL=INFO
ENVFILE
    echo -e "${GREEN}✓ Created .env file (please update values as needed)${NC}"
else
    echo -e "${GREEN}✓ .env file already exists${NC}"
fi

# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo "=============================="
echo -e "${GREEN}🎉 Setup Complete!${NC}"
echo "=============================="
echo ""
echo "AI Stack Ready:"
echo "  • Translation: NLLB-200-1.3B (~2.5GB)"
echo "  • Simplification: Llama 3.2 3B via Ollama (~2GB)"
echo "  • Text-to-Speech: Edge TTS (FREE, cloud-based)"
echo "  • Embeddings: BGE-M3 (~1.2GB)"
echo ""
echo "Estimated Memory Usage: ~6-8GB (leaves headroom for system)"
echo ""
echo "Next Steps:"
echo "  1. Update .env with your database credentials"
echo "  2. Run database migrations: alembic upgrade head"
echo "  3. Start the backend: uvicorn backend.api.main:app --reload"
echo "  4. Start the frontend: cd frontend && npm run dev"
echo ""
echo "To verify Ollama is working:"
echo "  ollama run llama3.2:3b 'Say hello in Hindi'"
echo ""
