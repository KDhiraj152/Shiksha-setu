#!/bin/bash
# MASTER-OPTIMIZER Validation Script
# Validates all optimizations are working correctly

# Don't exit on first error - collect all results
set +e

echo "============================================"
echo "MASTER-OPTIMIZER Validation Suite"
echo "============================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track results
PASSED=0
FAILED=0

# Function to run test with status
run_test() {
    local test_name="$1"
    local test_cmd="$2"
    
    echo -n "Testing: $test_name... "
    
    if eval "$test_cmd" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}"
        ((PASSED++))
    else
        echo -e "${RED}✗ FAILED${NC}"
        ((FAILED++))
    fi
}

# 1. Check Python environment
echo "1. Environment Validation"
echo "-------------------------"
run_test "Python 3.11+" "python3 --version | grep -E 'Python 3\.(1[1-9]|[2-9][0-9])'"
run_test "pip installed" "python3 -m pip --version"
echo ""

# 2. Check required files exist
echo "2. Code Structure Validation"
echo "----------------------------"
run_test "Model tier router" "test -f backend/core/model_tier_router.py"
run_test "Unified model client" "test -f backend/services/unified_model_client.py"
run_test "Circuit breaker" "test -f backend/utils/circuit_breaker.py"
run_test "E2E tests" "test -f tests/e2e/test_optimized_pipeline.py"
run_test "Optimized orchestrator" "grep -q 'UnifiedModelClient' backend/pipeline/orchestrator.py"
echo ""

# 3. Check dependencies can be imported (if venv active)
echo "3. Dependency Check"
echo "------------------"
if python3 -c "import sys; sys.exit(0 if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else 1)" 2>/dev/null; then
    echo "Virtual environment detected ✓"
    
    run_test "FastAPI" "python3 -c 'import fastapi'"
    run_test "Transformers" "python3 -c 'import transformers'"
    run_test "PyTorch" "python3 -c 'import torch'"
    run_test "psutil" "python3 -c 'import psutil'"
    run_test "pytest" "python3 -c 'import pytest'"
else
    echo -e "${YELLOW}No virtual environment detected - skipping import tests${NC}"
    echo "Run: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
fi
echo ""

# 4. Check git commits
echo "4. Git History Validation"
echo "-------------------------"
run_test "On master-optimizer branch" "git branch --show-current | grep -q 'master-optimizer'"
run_test "MO commits exist" "git log --oneline | grep -q 'MO:'"
run_test "C1 (Tier Router)" "git log --oneline | grep -q 'C1.*Tier.*Router'"
run_test "C2+C3 (Quantization)" "git log --oneline | grep -q 'C2.*C3'"
run_test "C4 (Streaming)" "git log --oneline | grep -q 'C4.*Streaming'"
run_test "C5 (MPS)" "git log --oneline | grep -q 'C5.*MPS'"
run_test "H1+H2 (Unified Client)" "git log --oneline | grep -q 'H1.*H2'"
run_test "H3 (Orchestrator)" "git log --oneline | grep -q 'H3.*Orchestrator'"
run_test "H4 (E2E Tests)" "git log --oneline | grep -q 'H4.*E2E'"
run_test "H5 (Frontend)" "git log --oneline | grep -q 'H5.*Frontend'"
echo ""

# 5. Frontend validation
echo "5. Frontend Validation"
echo "---------------------"
run_test "package.json exists" "test -f frontend/package.json"
run_test "Code splitting in App.tsx" "grep -q 'lazy(' frontend/src/App.tsx"
run_test "Vite config optimized" "grep -q 'manualChunks' frontend/vite.config.ts"
run_test "Suspense wrapper" "grep -q 'Suspense' frontend/src/App.tsx"
echo ""

# 6. Backend optimization checks
echo "6. Backend Optimization Checks"
echo "-------------------------------"
run_test "MPS environment setup" "grep -q 'configure_mps_environment' backend/utils/device_manager.py"
run_test "Quantization support" "grep -q 'load_quantized_model' backend/core/model_loader.py"
run_test "Streaming uploads" "grep -q 'async with aiofiles' backend/api/routes/content.py"
run_test "Circuit breaker decorator" "grep -q '@circuit_breaker' backend/utils/circuit_breaker.py"
run_test "Tier complexity scoring" "grep -q 'calculate_task_complexity' backend/core/model_tier_router.py"
echo ""

# 7. Memory budget validation
echo "7. Configuration Validation"
echo "---------------------------"
run_test "8GB memory limit" "grep -q 'MAX_MODEL_MEMORY_GB.*=.*8' backend/core/config.py"
run_test "LRU cache size" "grep -q 'maxsize' backend/core/model_loader.py"
run_test "Async-first client" "grep -q 'async def' backend/services/unified_model_client.py"
echo ""

# Summary
echo ""
echo "============================================"
echo "VALIDATION SUMMARY"
echo "============================================"
echo -e "${GREEN}Passed: $PASSED${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Failed: $FAILED${NC}"
else
    echo -e "${GREEN}Failed: $FAILED${NC}"
fi
echo ""

if [ $FAILED -gt 0 ]; then
    echo -e "${RED}✗ Some validations failed${NC}"
    echo "Review the failed tests above and ensure all optimizations are in place."
    exit 1
else
    echo -e "${GREEN}✓ All validations passed!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Install dependencies: pip install -r requirements.txt"
    echo "2. Run E2E tests: pytest tests/e2e/test_optimized_pipeline.py -v"
    echo "3. Start backend: uvicorn backend.api.main:app --reload"
    echo "4. Start frontend: cd frontend && npm run dev"
    exit 0
fi
