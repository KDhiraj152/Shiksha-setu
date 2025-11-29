#!/usr/bin/env bash
#
# Comprehensive Test Runner for ShikshaSetu
#
# Tests all components:
# - Unit tests
# - Integration tests (AI, Celery, Redis, RAG)
# - E2E tests
# - Performance tests
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== ShikshaSetu Comprehensive Test Suite ===${NC}\n"

# Check if test database exists
echo -e "${YELLOW}Checking test environment...${NC}"
if ! psql -U kdhiraj_152 -lqt | cut -d \| -f 1 | grep -qw shiksha_setu_test; then
    echo "Creating test database..."
    createdb -U kdhiraj_152 shiksha_setu_test
fi

# Check Redis
if ! redis-cli ping > /dev/null 2>&1; then
    echo -e "${RED}Redis is not running. Please start Redis first.${NC}"
    exit 1
fi

# Check Ollama (optional for some tests)
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo -e "${YELLOW}Warning: Ollama is not running. Some tests may be skipped.${NC}"
fi

# Set test environment
export TESTING=true
export ENVIRONMENT=test
export DATABASE_URL="postgresql://kdhiraj_152@127.0.0.1:5432/shiksha_setu_test"

echo -e "\n${GREEN}Running tests...${NC}\n"

# 1. Unit Tests
echo -e "${YELLOW}[1/7] Running Unit Tests...${NC}"
pytest tests/unit/ -v -m "not slow" --tb=short --durations=10 || true

# 2. Unit Tests for New AI Stack
echo -e "\n${YELLOW}[2/7] Running AI Stack Unit Tests...${NC}"
pytest tests/unit/test_new_ai_stack.py -v --tb=short || true

# 3. Integration Tests (AI Services)
echo -e "\n${YELLOW}[3/7] Running AI Integration Tests...${NC}"
pytest tests/integration/test_new_ai_stack_integration.py -v -m "integration and not slow" --tb=short || true

# 4. Integration Tests (Celery & Redis)
echo -e "\n${YELLOW}[4/7] Running Celery/Redis Tests...${NC}"
pytest tests/integration/test_celery_redis.py -v -m "integration" --tb=short || true

# 5. Integration Tests (RAG Q&A)
echo -e "\n${YELLOW}[5/7] Running RAG Q&A Tests...${NC}"
pytest tests/integration/test_rag_qa.py -v -m "integration" --tb=short || true

# 6. E2E Tests
echo -e "\n${YELLOW}[6/7] Running E2E Tests...${NC}"
pytest tests/e2e/ -v -m "e2e" --tb=short || true

# 7. Performance Tests (optional, can be slow)
if [ "$RUN_PERFORMANCE_TESTS" = "true" ]; then
    echo -e "\n${YELLOW}[7/7] Running Performance Tests...${NC}"
    pytest tests/performance/ -v -m "performance" --tb=short || true
else
    echo -e "\n${YELLOW}[7/7] Skipping Performance Tests (set RUN_PERFORMANCE_TESTS=true to run)${NC}"
fi

# Generate coverage report
echo -e "\n${GREEN}Generating coverage report...${NC}"
pytest tests/ --cov=backend --cov-report=html --cov-report=term-missing -m "not slow and not performance" || true

echo -e "\n${GREEN}=== Test Suite Complete ===${NC}"
echo -e "Coverage report: htmlcov/index.html"
