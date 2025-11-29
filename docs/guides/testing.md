# 🧪 Testing Guide

Comprehensive testing guide for Shiksha Setu, covering unit tests, integration tests, E2E tests, and manual testing procedures.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Test Structure](#test-structure)
3. [Running Tests](#running-tests)
4. [Unit Testing](#unit-testing)
5. [Integration Testing](#integration-testing)
6. [E2E Testing](#e2e-testing)
7. [Manual Testing](#manual-testing)
8. [API Testing](#api-testing)
9. [Coverage Reports](#coverage-reports)
10. [CI/CD Testing](#cicd-testing)

---

## Overview

### Testing Philosophy

Shiksha Setu follows a comprehensive testing strategy:

- **Unit Tests**: Fast, isolated tests for individual functions
- **Integration Tests**: Test interactions between components
- **E2E Tests**: Test complete user workflows
- **Manual Tests**: Exploratory testing and edge cases

### Test Coverage Goals

| Component | Target Coverage | Current Coverage |
|-----------|----------------|------------------|
| Backend Core | 90% | 87% |
| Backend API | 85% | 82% |
| Backend Services | 80% | 78% |
| Frontend Components | 75% | 71% |
| **Overall** | **80%** | **79%** |

---

## Test Structure

```
tests/
├── unit/                    # Fast, isolated tests
│   ├── backend/
│   │   ├── test_model_loader.py
│   │   ├── test_quantization.py
│   │   ├── test_tier_router.py
│   │   └── test_auth.py
│   └── frontend/
│       ├── components/
│       └── utils/
├── integration/             # Component interaction tests
│   ├── test_pipeline.py
│   ├── test_auth_flow.py
│   └── test_content_processing.py
├── e2e/                     # End-to-end workflows
│   ├── test_optimized_pipeline.py
│   ├── test_user_workflows.py
│   └── test_content_upload.py
├── fixtures/                # Test data and mocks
│   ├── content_samples.py
│   ├── mock_models.py
│   └── test_users.py
└── conftest.py              # Pytest configuration

frontend/src/
├── __tests__/               # Frontend tests
│   ├── components/
│   ├── pages/
│   └── utils/
└── vitest.config.ts
```

---

## Running Tests

### Backend Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# E2E tests only
pytest tests/e2e/ -v

# Specific test file
pytest tests/unit/backend/test_model_loader.py -v

# Specific test function
pytest tests/unit/backend/test_model_loader.py::test_load_model_with_quantization -v

# With coverage
pytest tests/ -v --cov=backend --cov-report=html

# Parallel execution (faster)
pytest tests/ -n auto
```

### Frontend Tests

```bash
# All tests
cd frontend
npm test

# Watch mode (auto-rerun on changes)
npm test -- --watch

# Coverage
npm run test:coverage

# E2E tests (Playwright)
npm run test:e2e

# UI mode (interactive)
npm test -- --ui
```

### Quick Validation

```bash
# Run Python unit tests
pytest tests/ -v

# Run frontend tests
cd frontend && npm test

# Includes:
# - Python unit tests
# - Python integration tests
# - Frontend tests
```

---

## Unit Testing

### Backend Unit Tests

#### Example: Testing Model Loader

```python
# tests/unit/backend/test_model_loader.py
import pytest
from backend.core.model_loader import ModelLoader, QuantizationLevel

@pytest.fixture
def model_loader():
    """Create model loader instance for testing."""
    return ModelLoader()

def test_load_model_with_int4_quantization(model_loader):
    """Test model loading with INT4 quantization."""
    model = model_loader.load_model(
        model_name="test-model",
        model_size_params=7.0,
        force_quantization=QuantizationLevel.INT4
    )
    
    assert model is not None
    assert model.config.quantization == QuantizationLevel.INT4
    assert model.memory_footprint_gb < 4.0  # INT4 should be <4GB

def test_load_model_invalid_quantization(model_loader):
    """Test that invalid quantization raises ValueError."""
    with pytest.raises(ValueError, match="Invalid quantization"):
        model_loader.load_model(
            model_name="test-model",
            force_quantization="invalid"
        )

def test_model_cache_hit(model_loader):
    """Test that second load uses cache."""
    # First load
    model1 = model_loader.load_model("test-model")
    
    # Second load (should be cached)
    model2 = model_loader.load_model("test-model")
    
    assert model1 is model2  # Same object reference
    assert model_loader.cache_hits == 1
```

#### Example: Testing Tier Router

```python
# tests/unit/backend/test_tier_router.py
import pytest
from backend.core.model_tier_router import ModelTierRouter, ModelTier

@pytest.fixture
def router():
    return ModelTierRouter()

def test_select_tier_simple_task(router):
    """Test SMALL tier selection for simple task."""
    tier = router.select_tier(
        text="The cat is big.",
        grade_level=5,
        subject="English"
    )
    assert tier == ModelTier.SMALL

def test_select_tier_complex_task(router):
    """Test LARGE tier selection for complex task."""
    long_text = "Complex mathematical proof " * 100  # >512 tokens
    tier = router.select_tier(
        text=long_text,
        grade_level=12,
        subject="Mathematics"
    )
    assert tier == ModelTier.LARGE

@pytest.mark.parametrize("grade,expected", [
    (5, ModelTier.SMALL),
    (8, ModelTier.SMALL),
    (9, ModelTier.MEDIUM),
    (12, ModelTier.LARGE),
])
def test_tier_by_grade_level(router, grade, expected):
    """Test tier selection based on grade level."""
    tier = router.select_tier(
        text="Test content",
        grade_level=grade,
        subject="English"
    )
    assert tier == expected
```

### Frontend Unit Tests

#### Example: Testing Component

```typescript
// frontend/src/__tests__/components/ContentCard.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { ContentCard } from '@/components/ContentCard';

describe('ContentCard', () => {
  const mockContent = {
    id: 1,
    title: 'Test Content',
    gradeLevel: 8,
    subject: 'Science',
    createdAt: new Date('2024-01-01'),
  };

  it('renders content information', () => {
    render(<ContentCard content={mockContent} />);
    
    expect(screen.getByText('Test Content')).toBeInTheDocument();
    expect(screen.getByText('Grade 8')).toBeInTheDocument();
    expect(screen.getByText('Science')).toBeInTheDocument();
  });

  it('calls onClick handler when clicked', () => {
    const handleClick = vi.fn();
    render(<ContentCard content={mockContent} onClick={handleClick} />);
    
    fireEvent.click(screen.getByRole('button'));
    
    expect(handleClick).toHaveBeenCalledWith(1);
    expect(handleClick).toHaveBeenCalledTimes(1);
  });

  it('displays loading state', () => {
    render(<ContentCard content={mockContent} isLoading={true} />);
    
    expect(screen.getByTestId('loading-spinner')).toBeInTheDocument();
  });
});
```

---

## Integration Testing

### Backend Integration Tests

#### Example: Testing Content Pipeline

```python
# tests/integration/test_pipeline.py
import pytest
from httpx import AsyncClient
from backend.main import app
from backend.core.database import get_db

@pytest.fixture
async def client():
    """Create test client."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
async def auth_token(client):
    """Get authentication token for tests."""
    response = await client.post(
        "/api/v1/auth/login",
        json={"email": "test@shiksha.com", "password": "Test@1234567"}
    )
    return response.json()["access_token"]

@pytest.mark.asyncio
async def test_content_simplification_pipeline(client, auth_token):
    """Test complete content simplification workflow."""
    # 1. Upload content
    files = {"file": ("test.txt", b"Complex scientific content...", "text/plain")}
    upload_response = await client.post(
        "/api/v1/content/upload",
        files=files,
        headers={"Authorization": f"Bearer {auth_token}"}
    )
    assert upload_response.status_code == 200
    content_id = upload_response.json()["id"]
    
    # 2. Simplify content
    simplify_response = await client.post(
        "/api/v1/content/simplify",
        json={
            "content_id": content_id,
            "target_grade": 8,
            "subject": "Science"
        },
        headers={"Authorization": f"Bearer {auth_token}"}
    )
    assert simplify_response.status_code == 200
    task_id = simplify_response.json()["task_id"]
    
    # 3. Check task status
    status_response = await client.get(
        f"/api/v1/tasks/{task_id}",
        headers={"Authorization": f"Bearer {auth_token}"}
    )
    assert status_response.status_code == 200
    assert status_response.json()["status"] in ["pending", "completed"]
```

---

## E2E Testing

### Backend E2E Tests

#### Example: Testing Optimized Pipeline

```python
# tests/e2e/test_optimized_pipeline.py
import pytest
from backend.core.model_loader import get_model_loader
from backend.core.dynamic_quantization import get_quantization_manager
from backend.services.unified_model_client import get_unified_client

class TestMemoryConstraints:
    """Test system behavior under memory constraints."""
    
    @pytest.mark.asyncio
    async def test_dynamic_quantization_under_load(self):
        """Test that system adapts quantization under memory pressure."""
        manager = get_quantization_manager()
        
        # Simulate low memory
        initial_status = manager.get_status()
        assert initial_status["memory"]["used_percent"] < 0.5
        
        # Load multiple models concurrently
        client = get_unified_client()
        tasks = [
            client.simplify_text(
                text=f"Test content {i}" * 100,
                grade_level=12,
                subject="Mathematics"
            )
            for i in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Verify system adapted
        final_status = manager.get_status()
        assert final_status["active_requests"] >= 0
        assert all(r["simplified_text"] for r in results)
    
    @pytest.mark.asyncio
    async def test_streaming_upload_prevents_oom(self):
        """Test that large uploads don't cause OOM."""
        # Create 100MB test file
        large_file = b"x" * (100 * 1024 * 1024)
        
        async with aiofiles.open("test_large.pdf", "wb") as f:
            await f.write(large_file)
        
        # Upload should succeed without OOM
        async with httpx.AsyncClient() as client:
            with open("test_large.pdf", "rb") as f:
                response = await client.post(
                    "http://localhost:8000/api/v1/content/upload",
                    files={"file": f},
                    headers={"Authorization": f"Bearer {token}"}
                )
        
        assert response.status_code == 200
        
        # Cleanup
        os.remove("test_large.pdf")
```

### Frontend E2E Tests

#### Example: Testing User Workflow (Playwright)

```typescript
// frontend/e2e/content-upload.spec.ts
import { test, expect } from '@playwright/test';

test.describe('Content Upload Workflow', () => {
  test.beforeEach(async ({ page }) => {
    // Login
    await page.goto('http://localhost:5173/login');
    await page.fill('[name="email"]', 'test@shiksha.com');
    await page.fill('[name="password"]', 'Test@1234567');
    await page.click('button[type="submit"]');
    await page.waitForURL('**/dashboard');
  });

  test('should upload and process content', async ({ page }) => {
    // Navigate to upload page
    await page.click('text=Upload Content');
    await expect(page).toHaveURL('**/upload');
    
    // Upload file
    await page.setInputFiles('input[type="file"]', 'test-files/sample.pdf');
    await page.fill('[name="title"]', 'Test Document');
    await page.selectOption('[name="gradeLevel"]', '8');
    await page.selectOption('[name="subject"]', 'Science');
    
    // Submit
    await page.click('button:has-text("Upload")');
    
    // Wait for processing
    await expect(page.locator('text=Processing...')).toBeVisible();
    await expect(page.locator('text=Upload Complete')).toBeVisible({ timeout: 30000 });
    
    // Verify in library
    await page.click('text=Library');
    await expect(page.locator('text=Test Document')).toBeVisible();
  });

  test('should handle upload errors gracefully', async ({ page }) => {
    await page.goto('http://localhost:5173/upload');
    
    // Try to upload without file
    await page.click('button:has-text("Upload")');
    await expect(page.locator('text=Please select a file')).toBeVisible();
    
    // Try to upload invalid file
    await page.setInputFiles('input[type="file"]', 'test-files/invalid.exe');
    await page.click('button:has-text("Upload")');
    await expect(page.locator('text=Invalid file type')).toBeVisible();
  });
});
```

---

## Manual Testing

### 1. Authentication Flow

**Test Login**:
```bash
# 1. Open frontend
open http://localhost:5173/login

# 2. Try invalid credentials
Email: invalid@test.com
Password: wrong123
# Expected: Error message "Invalid credentials"

# 3. Try valid credentials
Email: test@shiksha.com
Password: Test@1234567
# Expected: Redirect to dashboard
```

**Test Registration**:
```bash
# 1. Navigate to register
open http://localhost:5173/register

# 2. Fill form
Email: newuser@test.com
Password: SecurePass@123
Full Name: Test User
Organization: Test School

# 3. Submit
# Expected: Account created, redirect to login
```

### 2. Content Upload & Processing

**Test Upload**:
```bash
# 1. Login as teacher
Email: teacher@shiksha.com
Password: Teacher@123456

# 2. Navigate to upload
Click "Upload Content"

# 3. Upload file
File: sample.pdf
Title: Sample Educational Content
Grade Level: 8
Subject: Science

# 4. Submit
# Expected: File uploads, processing starts
```

**Test Simplification**:
```bash
# 1. View uploaded content
Navigate to "Library"
Click on "Sample Educational Content"

# 2. Request simplification
Click "Simplify for Grade 5"

# 3. Wait for processing
# Expected: Simplified version appears

# 4. Verify quality
# Check: Simpler vocabulary, shorter sentences, preserved concepts
```

### 3. Translation

**Test Translation**:
```bash
# 1. Open content
Navigate to content item

# 2. Select translation
Click "Translate"
Select "Hindi"

# 3. Wait for processing
# Expected: Hindi translation appears

# 4. Verify accuracy
# Check: Accurate translation, proper script, preserved meaning
```

---

## API Testing

### Using cURL

#### Authentication

```bash
# Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "test@shiksha.com", "password": "Test@1234567"}'

# Save token
TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

#### Content Operations

```bash
# Upload content
curl -X POST http://localhost:8000/api/v1/content/upload \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@test.pdf" \
  -F "title=Test Document" \
  -F "grade_level=8" \
  -F "subject=Science"

# Simplify content
curl -X POST http://localhost:8000/api/v1/content/simplify \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Complex scientific content...",
    "target_grade": 5,
    "subject": "Science"
  }'

# Translate content
curl -X POST http://localhost:8000/api/v1/content/translate \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello students",
    "source_language": "English",
    "target_language": "Hindi"
  }'

# Get content library
curl -X GET http://localhost:8000/api/v1/content/library \
  -H "Authorization: Bearer $TOKEN"
```

### Using Postman

Import collection: `docs/postman/shiksha-setu.json`

**Collections**:
- Authentication (Login, Register, Refresh)
- Content (Upload, Simplify, Translate)
- User Management (Profile, Settings)
- Admin (Users, System Config)

---

## Coverage Reports

### Generate Coverage

```bash
# Backend coverage
pytest tests/ --cov=backend --cov-report=html --cov-report=term

# View report
open htmlcov/index.html

# Frontend coverage
cd frontend
npm run test:coverage

# View report
open coverage/index.html
```

### Coverage Metrics

```bash
# Current coverage
pytest tests/ --cov=backend --cov-report=term

# Expected output:
# Name                          Stmts   Miss  Cover
# -------------------------------------------------
# backend/core/__init__.py         12      0   100%
# backend/core/model_loader.py    156     18    88%
# backend/core/quantization.py    134     22    84%
# backend/api/routes/content.py   201     35    83%
# -------------------------------------------------
# TOTAL                          1847    246    87%
```

---

## CI/CD Testing

### GitHub Actions Workflow

**File**: `.github/workflows/test.yml`

```yaml
name: Test Suite

on: [push, pull_request]

jobs:
  backend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: |
          pip install -r requirements/base.txt
          pip install -r requirements/dev.txt
      
      - name: Run tests
        run: pytest tests/ --cov=backend --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  frontend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      
      - name: Install dependencies
        run: cd frontend && npm ci
      
      - name: Run tests
        run: cd frontend && npm test -- --coverage
```

---

## Troubleshooting

### Tests Failing

**Check environment**:
```bash
# Verify services running
docker-compose ps

# Check logs
docker-compose logs backend
docker-compose logs postgres
```

**Clear test data**:
```bash
# Reset database
python3 scripts/migrations/run_migrations.py --reset

# Clear Redis cache
redis-cli FLUSHALL
```

### Slow Tests

**Run specific tests**:
```bash
# Only fast tests
pytest tests/ -m "not slow"

# Parallel execution
pytest tests/ -n auto
```

---

## Further Reading

- **[pytest Documentation](https://docs.pytest.org/)** - Testing framework
- **[Playwright Documentation](https://playwright.dev/)** - E2E testing
- **[Vitest Documentation](https://vitest.dev/)** - Frontend testing
- **[Coverage.py](https://coverage.readthedocs.io/)** - Coverage reports

---

## 👨‍💻 Author

**K Dhiraj** • [k.dhiraj.srihari@gmail.com](mailto:k.dhiraj.srihari@gmail.com) • [@KDhiraj152](https://github.com/KDhiraj152) • [LinkedIn](https://www.linkedin.com/in/k-dhiraj-83b025279/)

*Last updated: November 2025*
