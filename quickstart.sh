#!/bin/bash
# ShikshaSetu Quick Start Script
# Run this after cloning the project

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}ShikshaSetu - Quick Setup${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}✗ Python 3 not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python $(python3 --version | cut -d' ' -f2)${NC}"

# Create virtual environment
if [ ! -d ".venv" ]; then
    echo -e "${BLUE}Creating virtual environment...${NC}"
    python3 -m venv .venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment exists${NC}"
fi

# Activate virtual environment
source .venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Install dependencies
echo -e "${BLUE}Installing dependencies...${NC}"
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Run validation
echo -e "\n${BLUE}Running validation checks...${NC}"
python3 startup_validation.py

echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}✓ Setup Complete!${NC}"
echo -e "${BLUE}========================================${NC}\n"

echo "Next steps:"
echo ""
echo "1. Start Backend:"
echo "   source .venv/bin/activate"
echo "   uvicorn src.api.main:app --reload"
echo ""
echo "2. Start Frontend (in another terminal):"
echo "   cd frontend"
echo "   npm install && npm run dev"
echo ""
echo "3. Access API:"
echo "   http://localhost:8000"
echo "   http://localhost:8000/docs"
echo ""
