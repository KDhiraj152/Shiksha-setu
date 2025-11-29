# 💻 Development Guide - Shiksha Setu

**Complete guide for developers contributing to Shiksha Setu**

---

## 📋 Table of Contents

1. [Getting Started](#getting-started)
2. [Project Structure](#project-structure)
3. [Development Workflow](#development-workflow)
4. [Coding Standards](#coding-standards)
5. [Testing](#testing)
6. [Documentation](#documentation)
7. [Git Workflow](#git-workflow)
8. [Troubleshooting](#troubleshooting)

---

## 🚀 Getting Started

For installation and setup prerequisites, see **[Setup Guide](docs/guides/setup.md)**.

### Quick Setup Summary

```bash
git clone https://github.com/KDhiraj152/Siksha-Setu.git && cd Siksha-Setu
./SETUP.sh && ./START.sh
```

Then verify your setup is complete before proceeding with development.

---

## 📁 Project Structure

```
Siksha-Setu/
├── backend/                    # Backend API (FastAPI)
│   ├── api/                    # API endpoints
│   │   ├── routes/             # All API routes (consolidated)
│   │   │   ├── __init__.py     # Route exports
│   │   │   ├── auth.py         # Authentication
│   │   │   ├── content.py      # Content processing
│   │   │   ├── qa.py           # Q&A endpoints
│   │   │   ├── streaming.py    # Streaming responses
│   │   │   ├── progress.py     # Progress tracking
│   │   │   ├── experiments.py  # A/B testing
│   │   │   ├── admin.py        # Admin operations
│   │   │   └── quantization.py # Model quantization
│   │   ├── main.py             # FastAPI app
│   │   ├── middleware.py       # Request middleware
│   │   └── documentation.py    # API docs config
│   ├── core/                   # Core infrastructure
│   │   ├── config.py           # SINGLE configuration source
│   │   ├── database.py         # Database connection
│   │   ├── cache.py            # Redis caching
│   │   ├── security.py         # Security utilities
│   │   ├── exceptions.py       # Custom exceptions
│   │   ├── telemetry.py        # Monitoring
│   │   ├── model_loader.py     # Lazy model loading
│   │   ├── model_tier_router.py # Resource-aware routing
│   │   ├── dynamic_quantization.py # Adaptive quantization
│   │   └── model_optimizer.py  # Model optimization
│   ├── models/                 # Database models (SQLAlchemy)
│   │   ├── auth.py             # User, APIKey, TokenBlacklist
│   │   ├── content.py          # ProcessedContent, NCERTStandard
│   │   ├── progress.py         # StudentProgress, QuizScore
│   │   └── rag.py              # DocumentChunk, Embedding
│   ├── schemas/                # Pydantic request/response schemas
│   │   ├── auth.py             # Auth schemas
│   │   ├── content.py          # Content schemas
│   │   └── qa.py               # Q&A schemas
│   ├── services/               # Business logic
│   │   ├── unified_model_client.py # SINGLE model client (active)
│   │   ├── ab_test_router.py   # A/B testing (renamed)
│   │   ├── rag.py              # RAG Q&A system
│   │   ├── curriculum_validation.py # NCERT validation
│   │   ├── cultural_context_service.py # Cultural adaptation
│   │   ├── simplify/           # Text simplification
│   │   │   ├── __init__.py
│   │   │   ├── simplifier.py
│   │   │   └── analyzer.py
│   │   ├── translate/          # Translation services
│   │   │   ├── __init__.py
│   │   │   ├── engine.py
│   │   │   └── model.py
│   │   ├── speech/             # Text-to-speech
│   │   │   ├── __init__.py
│   │   │   ├── generator.py
│   │   │   └── processor.py
│   │   └── validate/           # Content validation
│   │       ├── __init__.py
│   │       └── validator.py
│   ├── pipeline/               # AI/ML pipeline orchestration
│   │   ├── orchestrator.py     # Pipeline coordinator
│   │   └── README.md           # Pipeline documentation
│   ├── tasks/                  # Celery background tasks
│   │   ├── celery_app.py       # Celery configuration
│   │   ├── pipeline_tasks.py   # Content processing tasks
│   │   ├── qa_tasks.py         # Q&A tasks
│   │   └── audio_tasks.py      # Audio generation tasks
│   ├── middleware/             # Custom middleware
│   │   └── tenant.py           # Multi-tenancy
│   ├── utils/                  # Utility functions
│   │   ├── logging.py          # Logging setup
│   │   ├── device_manager.py   # Device detection
│   │   └── sanitizer.py        # Input sanitization
│   ├── _deprecated/            # Archived redundant code
│   │   ├── model_clients.py    # (archived 2025-11-28)
│   │   ├── model_clients_async.py # (archived 2025-11-28)
│   │   └── model_client.py     # (archived 2025-11-28)
│   └── __init__.py             # Package init
│
├── frontend/                   # React frontend
│   ├── src/
│   │   ├── components/         # React components
│   │   │   ├── ui/             # shadcn/ui components
│   │   │   ├── molecules/      # Composite components
│   │   │   └── organisms/      # Complex sections
│   │   ├── pages/              # Page components
│   │   ├── hooks/              # Custom React hooks
│   │   ├── services/           # API services
│   │   │   └── api.ts          # API client
│   │   ├── store/              # Zustand state
│   │   │   └── authStore.ts    # Auth state
│   │   ├── types/              # TypeScript types
│   │   ├── utils/              # Utility functions
│   │   ├── App.tsx             # Main app component
│   │   └── main.tsx            # Entry point
│   ├── public/                 # Static assets
│   └── package.json            # NPM dependencies
│
├── tests/                      # Test suites
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── conftest.py             # Pytest configuration
│
├── scripts/                    # Organized utility scripts (no duplicates)
│   ├── setup/                  # Setup scripts
│   ├── deployment/             # Deployment scripts
│   ├── testing/                # Test scripts
│   ├── demo/                   # Demo scripts
│   └── utils/                  # Utility scripts
│
├── infrastructure/             # Infrastructure as Code
│   ├── docker/                 # Docker configs
│   ├── kubernetes/             # K8s manifests
│   ├── nginx/                  # Nginx configs
│   └── monitoring/             # Monitoring stack
│
├── docs/                       # Documentation
│   ├── reference/              # API & architecture docs
│   ├── guides/                 # How-to guides
│   └── technical/              # Technical deep-dives
│
├── data/                       # Data directory
│   ├── uploads/                # User uploads
│   ├── audio/                  # Generated audio
│   ├── cache/                  # Cache files
│   └── models/                 # ML model files
│
├── bin/                        # User-facing executable scripts
│   ├── setup                   # Initial setup (FIXED paths)
│   ├── start                   # Start services (FIXED paths)
│   ├── stop                    # Stop services
│   ├── test                    # Run tests
│   └── validate-production     # Production validation
│
├── config/                     # Configuration files
│   ├── alembic.ini            # Database migrations
│   ├── pytest.ini             # Pytest configuration
│   └── docker-compose.production.yml # Production compose
│
├── requirements/               # Python dependencies
│   ├── base.txt               # Core dependencies
│   └── dev.txt                # Development dependencies
│
├── .env.example                # Environment template
├── CHANGELOG.md                # Version history (updated 2025-11-28)
├── DEVELOPMENT.md              # This file (updated 2025-11-28)
├── DEPLOYMENT.md               # Deployment guide
└── README.md                   # Project overview
```

### Key Changes (v2.2.0 - 2025-11-28)
- ✅ **Single Model Client**: `unified_model_client.py` is now the only active client
- ✅ **Single Config**: `core/config.py` is the only configuration file  
- ✅ **Router Clarity**: `ab_test_router.py` (A/B testing) vs `model_tier_router.py` (resource routing)
- ✅ **Consolidated Routes**: All API routes now in `api/routes/` (no more `endpoints/`)
- ✅ **No Script Duplicates**: Only `/bin/` scripts exist, `/scripts/` has no duplicates
- ✅ **Fixed Bugs**: Threading import, path checks in setup scripts

See [CHANGELOG.md](CHANGELOG.md) for complete details.

---

## 🔄 Development Workflow

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

Follow [Coding Standards](#coding-standards).

### 3. Run Tests

```bash
# Backend tests
pytest tests/

# Frontend tests
cd frontend && npm test

# All features test
python3 scripts/testing/test_all_features.py
```

### 4. Commit Changes

```bash
git add .
git commit -m "feat: add new feature description"
```

**Commit Message Format**:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting)
- `refactor:` - Code refactoring
- `test:` - Test additions/changes
- `chore:` - Build/dependency changes

### 5. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create Pull Request on GitHub.

---

## 📝 Coding Standards

### Python (Backend)

**Style Guide**: PEP 8

**Tools**:
- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

**Run formatters**:
```bash
# Format code
black backend/ tests/

# Sort imports
isort backend/ tests/

# Lint
flake8 backend/ tests/

# Type check
mypy backend/
```

**Best Practices**:
```python
# ✅ Good
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)

async def process_content(
    content: str,
    grade_level: int,
    language: Optional[str] = "en"
) -> Dict[str, Any]:
    """
    Process educational content with grade-level adaptation.
    
    Args:
        content: Text content to process
        grade_level: Target grade level (1-12)
        language: Target language code (default: en)
        
    Returns:
        Processed content dictionary with metadata
        
    Raises:
        ValidationError: If content is invalid
    """
    logger.info(f"Processing content for grade {grade_level}")
    # ... implementation
    return result

# ❌ Bad
def process(c, g, l="en"):  # No type hints, unclear names
    print("processing...")   # Use logger, not print
    # ... implementation
    return r                 # Unclear return value
```

**Import Order**:
1. Standard library
2. Third-party packages
3. Local imports

```python
# ✅ Correct import order
import os
import logging
from typing import Optional, Dict

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import User
from ..schemas import ContentResponse
```

### TypeScript (Frontend)

**Style Guide**: Airbnb + TypeScript

**Tools**:
- **ESLint**: Linting
- **Prettier**: Formatting
- **TypeScript**: Type checking

**Run checks**:
```bash
cd frontend

# Lint
npm run lint

# Format
npm run format

# Type check
npm run type-check
```

**Best Practices**:
```typescript
// ✅ Good
interface ContentProps {
  title: string;
  gradeLevel: number;
  language?: string;
  onProcess: (result: ProcessedContent) => void;
}

export const ContentCard: React.FC<ContentProps> = ({
  title,
  gradeLevel,
  language = 'en',
  onProcess
}) => {
  const [loading, setLoading] = useState<boolean>(false);
  
  const handleProcess = async () => {
    setLoading(true);
    try {
      const result = await api.processContent(title, gradeLevel);
      onProcess(result);
    } catch (error) {
      console.error('Processing failed:', error);
    } finally {
      setLoading(false);
    }
  };
  
  return <div>...</div>;
};

// ❌ Bad
export const ContentCard = (props: any) => {  // No interface, any type
  const [loading, setLoading] = useState();   // No type annotation
  
  function handleProcess() {                  // Not async
    api.processContent(props.title, props.gradeLevel);
    props.onProcess();                         // No error handling
  }
  
  return <div>...</div>;
};
```

### Database Models

**Use Alembic for migrations**:
```bash
# Create migration
alembic -c config/alembic.ini revision --autogenerate -m "Add new table"

# Apply migration
alembic -c config/alembic.ini upgrade head

# Rollback
alembic -c config/alembic.ini downgrade -1
```

**Model Best Practices**:
```python
from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from .database import Base

class ProcessedContent(Base):
    """Processed educational content with metadata."""
    
    __tablename__ = "processed_content"
    __table_args__ = {'extend_existing': True}
    
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False, index=True)
    grade_level = Column(Integer, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<ProcessedContent(id={self.id}, title='{self.title}')>"
```

---

## 🧪 Testing

### Backend Tests

**Structure**:
```
tests/
├── unit/                       # Unit tests (fast)
│   ├── test_services.py
│   ├── test_models.py
│   └── test_utils.py
├── integration/                # Integration tests
│   ├── test_api.py
│   └── test_pipeline.py
└── conftest.py                 # Shared fixtures
```

**Run tests**:
```bash
# All tests
pytest

# Specific file
pytest tests/unit/test_services.py

# Specific test
pytest tests/unit/test_services.py::test_translation_service

# With coverage
pytest --cov=backend --cov-report=html

# Fast tests only (skip slow)
pytest -m "not slow"
```

**Writing Tests**:
```python
import pytest
from backend.services import TranslationService

@pytest.fixture
def translation_service():
    """Create translation service for testing."""
    return TranslationService()

def test_translate_hindi(translation_service):
    """Test Hindi translation."""
    result = translation_service.translate(
        text="Hello, how are you?",
        target_lang="hi"
    )
    
    assert result is not None
    assert len(result) > 0
    assert "नमस्ते" in result or "हैलो" in result

@pytest.mark.asyncio
async def test_async_translation(translation_service):
    """Test async translation."""
    result = await translation_service.translate_async(
        text="Education for all",
        target_lang="ta"
    )
    
    assert result is not None
```

### Frontend Tests

**Run tests**:
```bash
cd frontend

# All tests
npm test

# Watch mode
npm test -- --watch

# Coverage
npm test -- --coverage
```

**Writing Tests**:
```typescript
import { render, screen, fireEvent } from '@testing-library/react';
import { ContentCard } from './ContentCard';

describe('ContentCard', () => {
  it('renders title and grade level', () => {
    render(
      <ContentCard 
        title="Math Lesson"
        gradeLevel={5}
        onProcess={() => {}}
      />
    );
    
    expect(screen.getByText('Math Lesson')).toBeInTheDocument();
    expect(screen.getByText(/Grade 5/i)).toBeInTheDocument();
  });
  
  it('calls onProcess when button clicked', () => {
    const handleProcess = jest.fn();
    
    render(
      <ContentCard 
        title="Math Lesson"
        gradeLevel={5}
        onProcess={handleProcess}
      />
    );
    
    fireEvent.click(screen.getByRole('button', { name: /process/i }));
    expect(handleProcess).toHaveBeenCalled();
  });
});
```

### Test Coverage Goals

- **Backend**: 40%+ (current: 23%)
- **Frontend**: 80%+ (current: 100%)
- **Critical paths**: 100% (auth, payments, data processing)

---

## 📚 Documentation

### Code Documentation

**Python Docstrings**:
```python
def process_ncert_content(
    content: str,
    standard: int,
    subject: str
) -> ValidationResult:
    """
    Validate content against NCERT standards.
    
    Performs curriculum alignment checking and assigns
    a confidence score based on topic coverage and
    pedagogical appropriateness.
    
    Args:
        content: Educational content text
        standard: Grade/standard level (1-12)
        subject: Subject area (Math, Science, etc.)
        
    Returns:
        ValidationResult with score and recommendations
        
    Raises:
        ValidationError: If content format is invalid
        
    Example:
        >>> result = process_ncert_content(
        ...     "Photosynthesis is...",
        ...     standard=10,
        ...     subject="Biology"
        ... )
        >>> print(result.score)
        0.87
    """
    # Implementation
```

**API Documentation**:
- Use FastAPI automatic docs (Swagger/ReDoc)
- Add detailed descriptions to routes
- Include request/response examples

**Update Documentation**:
- API changes → Update `docs/reference/api.md`
- Architecture changes → Update `docs/reference/architecture.md`
- New features → Update `README.md` and feature docs

---

## 🔀 Git Workflow

### Branch Naming

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation
- `refactor/` - Code refactoring
- `test/` - Test additions

Example: `feature/add-punjabi-translation`

### Pull Request Process

1. **Create PR** with clear description
2. **Link Issues** using `Fixes #123` or `Closes #456`
3. **Request Review** from team members
4. **Address Feedback** and push updates
5. **Squash Merge** into main branch

**PR Template**:
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] All existing tests pass
- [ ] New tests added
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guide
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

---

## 🔧 Troubleshooting

### Common Development Issues

#### 1. Import Errors

```bash
# Solution: Activate virtual environment
source .venv/bin/activate

# Or reinstall dependencies
pip install -r requirements/base.txt
```

#### 2. Database Connection Issues

```bash
# Check PostgreSQL is running
sudo systemctl status postgresql
# or
docker ps | grep postgres

# Reset database (re-run migrations)
python -m alembic upgrade head
```

#### 3. Redis Connection Issues

```bash
# Start Redis
redis-server

# Or use Docker
docker run -d -p 6379:6379 redis:7-alpine
```

#### 4. Model Loading Failures

```bash
# Download models
python3 scripts/setup/download_models.py

# Check disk space
df -h

# Use CPU fallback
export FORCE_CPU=true
```

#### 5. Frontend Build Errors

```bash
cd frontend

# Clear cache
rm -rf node_modules package-lock.json
npm install

# Clear Vite cache
rm -rf .vite
npm run dev
```

### Debug Tools

**Backend**:
```python
# Add to code for debugging
import pdb; pdb.set_trace()  # Breakpoint

# Or use logging
import logging
logger = logging.getLogger(__name__)
logger.debug(f"Variable value: {variable}")
```

**Frontend**:
```typescript
// Browser DevTools
console.log('Debug:', variable);
console.table(arrayData);
debugger;  // Breakpoint
```

**Database**:
```bash
# Connect to database
psql -U shiksha_user -d shiksha_setu

# Useful queries
\dt                    # List tables
\d processed_content   # Describe table
SELECT * FROM users LIMIT 5;
```

---

## 🤝 Contributing

1. **Fork** the repository
2. **Create** feature branch
3. **Make** changes following standards
4. **Test** thoroughly
5. **Submit** pull request

---

## 📖 Additional Resources

- [Deployment Guide](DEPLOYMENT.md)
- [API Documentation](docs/reference/api.md)
- [Architecture Overview](docs/reference/architecture.md)
- [Troubleshooting Guide](docs/guides/troubleshooting.md)

---

## 👨‍💻 Author

**K Dhiraj** • [k.dhiraj.srihari@gmail.com](mailto:k.dhiraj.srihari@gmail.com) • [@KDhiraj152](https://github.com/KDhiraj152) • [LinkedIn](https://www.linkedin.com/in/k-dhiraj-83b025279/)

*Last updated: November 2025*
