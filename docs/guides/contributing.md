# 🤝 Contributing to Shiksha Setu

Thank you for your interest in contributing to Shiksha Setu! This guide will help you get started with contributing code, documentation, and other improvements.

---

## 📋 Table of Contents

1. [Getting Started](#getting-started)
2. [Development Workflow](#development-workflow)
3. [Coding Standards](#coding-standards)
4. [Commit Guidelines](#commit-guidelines)
5. [Pull Request Process](#pull-request-process)
6. [Testing Requirements](#testing-requirements)
7. [Documentation](#documentation)
8. [Community Guidelines](#community-guidelines)

---

## Getting Started

Before contributing, see **[Setup Guide](setup.md)** for complete installation instructions.

### Prerequisites

- **Python 3.11+**, **Node.js 20+**, **PostgreSQL 15+**, **Redis 7+**, **Git**

### Fork & Clone

```bash
# 1. Fork the repository on GitHub
# Visit: https://github.com/KDhiraj152/Siksha-Setu

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/Siksha-Setu.git
cd Siksha-Setu

# 3. Add upstream remote
git remote add upstream https://github.com/KDhiraj152/Siksha-Setu.git

# 4. Verify remotes
git remote -v
```

### Setup Development Environment

```bash
# Use automated setup from Setup Guide
./SETUP.sh
./START.sh

# Verify installation
python3 scripts/validation/check_dependencies.py
```

---

## Development Workflow

### Create a Feature Branch

```bash
# 1. Sync with upstream
git fetch upstream
git checkout main
git merge upstream/main

# 2. Create feature branch
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

### Branch Naming Convention

| Type | Format | Example |
|------|--------|---------|
| Feature | `feature/description` | `feature/add-video-support` |
| Bug Fix | `fix/description` | `fix/memory-leak-quantization` |
| Documentation | `docs/description` | `docs/update-api-reference` |
| Performance | `perf/description` | `perf/optimize-database-queries` |
| Refactor | `refactor/description` | `refactor/simplify-pipeline` |
| Test | `test/description` | `test/add-e2e-coverage` |

### Make Changes

```bash
# 1. Make your changes
# Edit files...

# 2. Test locally
pytest tests/                    # Backend tests
cd frontend && npm test          # Frontend tests

# 3. Run linters
black backend/ --check           # Python formatting
flake8 backend/                  # Python linting
cd frontend && npm run lint      # TypeScript linting

# 4. Type check
mypy backend/                    # Python type checking
cd frontend && npm run type-check
```

### Keep Branch Updated

```bash
# Regularly sync with upstream
git fetch upstream
git rebase upstream/main

# If conflicts occur
git status
# Resolve conflicts in each file
git add .
git rebase --continue
```

---

## Coding Standards

### Python Code Style

**Formatter**: Black (line length: 88)
**Linter**: Flake8, Pylint
**Type Checker**: mypy

```python
# ✅ GOOD: Type hints, docstrings, clear naming
from typing import Optional, List
from pydantic import BaseModel

class ContentRequest(BaseModel):
    """Request model for content processing.
    
    Attributes:
        text: The input text to process
        grade_level: Target grade level (5-12)
        subject: Subject area (e.g., "Mathematics")
    """
    text: str
    grade_level: int
    subject: Optional[str] = None

async def simplify_content(
    request: ContentRequest,
    user_id: int
) -> dict:
    """Simplify content for target grade level.
    
    Args:
        request: Content processing parameters
        user_id: ID of requesting user
        
    Returns:
        Dictionary with simplified content and metadata
        
    Raises:
        ValueError: If grade_level is invalid
        HTTPException: If user lacks permissions
    """
    if not 5 <= request.grade_level <= 12:
        raise ValueError(f"Invalid grade level: {request.grade_level}")
    
    # Implementation...
    return {"simplified_text": result}

# ❌ BAD: No types, no docstrings, unclear names
def process(r, u):
    if r.g < 5:
        raise Exception("bad")
    return {"t": result}
```

### TypeScript Code Style

**Formatter**: Prettier
**Linter**: ESLint
**Type Checker**: TypeScript

```typescript
// ✅ GOOD: Interfaces, types, clear structure
interface ContentItem {
  id: number;
  title: string;
  gradeLevel: number;
  createdAt: Date;
}

interface ContentListProps {
  items: ContentItem[];
  onItemClick: (id: number) => void;
  isLoading?: boolean;
}

/**
 * Display a list of content items with loading state.
 */
export const ContentList: React.FC<ContentListProps> = ({
  items,
  onItemClick,
  isLoading = false,
}) => {
  if (isLoading) {
    return <LoadingSpinner />;
  }

  return (
    <div className="content-list">
      {items.map((item) => (
        <ContentCard
          key={item.id}
          item={item}
          onClick={() => onItemClick(item.id)}
        />
      ))}
    </div>
  );
};

// ❌ BAD: Any types, no interfaces
export const ContentList = ({ items, onClick }: any) => {
  return items.map((i: any) => <div onClick={() => onClick(i)}>{i.title}</div>);
};
```

### Code Organization

```
backend/
├── api/              # FastAPI routes
│   └── routes/
│       ├── auth.py   # Authentication endpoints
│       ├── content.py
│       └── users.py
├── core/             # Business logic
│   ├── model_loader.py
│   └── orchestrator.py
├── models/           # Database models
│   ├── user.py
│   └── content.py
├── schemas/          # Pydantic schemas
│   ├── user.py
│   └── content.py
├── services/         # External services
│   └── unified_model_client.py
└── utils/            # Helper functions
    ├── auth.py
    └── device_manager.py

frontend/src/
├── components/       # Reusable components
│   ├── common/
│   └── content/
├── pages/            # Route pages
│   ├── DashboardPage.tsx
│   └── ContentPage.tsx
├── hooks/            # Custom React hooks
│   └── useAuth.ts
├── services/         # API clients
│   └── api.ts
└── utils/            # Helper functions
    └── formatters.ts
```

---

## Commit Guidelines

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

| Type | Description | Example |
|------|-------------|---------|
| `feat` | New feature | `feat(content): add video upload support` |
| `fix` | Bug fix | `fix(auth): resolve token expiration bug` |
| `docs` | Documentation | `docs(api): update endpoint reference` |
| `style` | Formatting | `style(backend): apply black formatting` |
| `refactor` | Code restructuring | `refactor(pipeline): simplify orchestrator` |
| `perf` | Performance | `perf(quantization): optimize memory usage` |
| `test` | Tests | `test(e2e): add upload flow coverage` |
| `chore` | Maintenance | `chore(deps): update fastapi to 0.104.0` |

### Examples

```bash
# Good commit messages
git commit -m "feat(content): add PDF text extraction support

- Implement PyPDF2 integration
- Add text extraction endpoint
- Update content schema with extracted_text field

Closes #42"

git commit -m "fix(auth): resolve JWT token expiration bug

The token expiration time was incorrectly calculated due to
timezone handling. This fix uses UTC consistently.

Fixes #156"

# Bad commit messages (avoid these)
git commit -m "fixed bug"
git commit -m "updated files"
git commit -m "WIP"
```

### Commit Frequency

- **Commit often**: Small, logical chunks
- **Squash before PR**: Clean up history before submitting
- **One purpose per commit**: Each commit should do one thing

```bash
# During development
git commit -m "WIP: add basic video support"
git commit -m "WIP: fix video encoding"
git commit -m "WIP: add tests"

# Before PR (interactive rebase to squash)
git rebase -i HEAD~3
# Squash into: "feat(content): add video upload support"
```

---

## Pull Request Process

### Before Submitting

- [ ] Code passes all tests (`pytest` and `npm test`)
- [ ] Code passes linters (`black`, `flake8`, `eslint`)
- [ ] Type checks pass (`mypy`, `tsc`)
- [ ] Documentation updated (if needed)
- [ ] CHANGELOG.md updated (for features/fixes)
- [ ] Commits squashed into logical units
- [ ] Branch is up-to-date with `main`

### Create Pull Request

1. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open PR on GitHub**:
   - Visit: https://github.com/YOUR_USERNAME/Siksha-Setu
   - Click "Compare & pull request"
   - Fill out the PR template

3. **PR Title Format**:
   ```
   feat(content): Add video upload support (#42)
   ```

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] E2E tests added/updated
- [ ] Manual testing performed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review performed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests pass locally
- [ ] CHANGELOG.md updated

## Related Issues
Closes #42
Relates to #38

## Screenshots (if applicable)
[Add screenshots for UI changes]
```

### Review Process

1. **Automated Checks**: CI/CD runs tests, linters, type checks
2. **Code Review**: Maintainers review code quality, logic, tests
3. **Feedback**: Address review comments with new commits
4. **Approval**: At least 1 maintainer approval required
5. **Merge**: Maintainer merges using "Squash and merge"

### Addressing Review Comments

```bash
# Make changes based on feedback
git add .
git commit -m "refactor: address review comments"
git push origin feature/your-feature-name

# PR automatically updates
```

---

## Testing Requirements

### Minimum Coverage

- **Unit Tests**: 80% coverage for new code
- **Integration Tests**: Critical paths covered
- **E2E Tests**: Major user flows covered

### Running Tests

```bash
# Backend unit tests
pytest tests/unit/ -v --cov=backend

# Backend integration tests
pytest tests/integration/ -v

# Backend E2E tests
pytest tests/e2e/ -v

# Frontend unit tests
cd frontend
npm test

# Frontend E2E tests
cd frontend
npm run test:e2e

# All tests
pytest tests/ -v --cov=backend --cov-report=html
cd frontend && npm test
```

### Writing Tests

```python
# tests/unit/test_model_loader.py
import pytest
from backend.core.model_loader import ModelLoader

@pytest.fixture
def model_loader():
    """Create model loader instance for testing."""
    return ModelLoader()

def test_load_model_with_quantization(model_loader):
    """Test model loading with INT4 quantization."""
    model = model_loader.load_model(
        "test-model",
        model_size_params=7.0,
        force_quantization="int4"
    )
    
    assert model is not None
    assert model.config.quantization == "int4"
    
def test_load_model_invalid_quantization(model_loader):
    """Test that invalid quantization raises ValueError."""
    with pytest.raises(ValueError, match="Invalid quantization"):
        model_loader.load_model(
            "test-model",
            force_quantization="invalid"
        )
```

---

## Documentation

### Types of Documentation

1. **Code Comments**: Explain complex logic
2. **Docstrings**: Document functions, classes, modules
3. **API Reference**: OpenAPI/Swagger for endpoints
4. **User Guides**: Step-by-step tutorials
5. **Architecture Docs**: System design, decisions

### Docstring Format

```python
def simplify_text(
    text: str,
    grade_level: int,
    subject: Optional[str] = None
) -> dict:
    """Simplify text for target grade level.
    
    This function uses AI models to adapt educational content
    to an appropriate reading level while preserving key concepts.
    
    Args:
        text: The input text to simplify (max 10,000 chars)
        grade_level: Target grade level (5-12, inclusive)
        subject: Optional subject for context (e.g., "Mathematics")
        
    Returns:
        Dictionary containing:
            - simplified_text (str): The adapted content
            - readability_score (float): Flesch-Kincaid grade level
            - modifications (list): List of changes made
            
    Raises:
        ValueError: If grade_level is not between 5 and 12
        HTTPException: If text exceeds maximum length
        
    Example:
        >>> result = simplify_text(
        ...     text="Complex scientific content...",
        ...     grade_level=8,
        ...     subject="Science"
        ... )
        >>> print(result["simplified_text"])
        "Simpler explanation..."
    """
    # Implementation...
```

### Updating Documentation

```bash
# Documentation files to update
docs/setup.md           # Setup instructions
docs/api-reference.md   # API endpoints
docs/architecture.md    # System design
CHANGELOG.md            # Version history

# Generate API docs (auto from code)
cd backend
python3 -m mkdocs serve
# Visit: http://localhost:8080
```

---

## Community Guidelines

### Code of Conduct

- **Be respectful**: Treat all contributors with respect
- **Be constructive**: Provide helpful, actionable feedback
- **Be inclusive**: Welcome contributors of all backgrounds
- **Be patient**: Remember everyone is learning

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Questions, ideas, general discussion
- **Pull Requests**: Code contributions, reviews
- **Email**: kdhiraj152@gmail.com (for sensitive matters)

### Getting Help

1. **Search existing issues**: Your question may be answered
2. **Check documentation**: Comprehensive guides available
3. **Ask in discussions**: Community can help
4. **Open an issue**: If bug/feature not yet reported

### Reporting Bugs

```markdown
**Title**: Brief description of bug

**Description**:
Clear description of the issue

**Steps to Reproduce**:
1. Go to '...'
2. Click on '...'
3. See error

**Expected Behavior**:
What should happen

**Actual Behavior**:
What actually happens

**Environment**:
- OS: macOS 14.1
- Python: 3.11.5
- Node: 18.17.0
- Browser: Chrome 119.0 (if frontend)

**Logs/Screenshots**:
[Attach relevant logs or screenshots]
```

### Suggesting Features

```markdown
**Title**: Feature name

**Problem Statement**:
What problem does this solve?

**Proposed Solution**:
How should it work?

**Alternatives Considered**:
What other approaches did you consider?

**Additional Context**:
Any other relevant information
```

---

## Recognition

Contributors will be recognized in:

- **CHANGELOG.md**: Credited for features/fixes
- **README.md**: Listed in Contributors section
- **GitHub**: Contributor badge on profile

### Top Contributors

Special recognition for:
- **10+ PRs merged**: Core Contributor badge
- **Major features**: Feature credit in releases
- **Documentation**: Documentation Champion badge

---

## Questions?

- **Email**: kdhiraj152@gmail.com
- **GitHub Issues**: [Open an issue](https://github.com/KDhiraj152/Siksha-Setu/issues)
- **GitHub Discussions**: [Start a discussion](https://github.com/KDhiraj152/Siksha-Setu/discussions)

Thank you for contributing to Shiksha Setu! 🎉

---

## 👨‍💻 Author

**K Dhiraj** • [k.dhiraj.srihari@gmail.com](mailto:k.dhiraj.srihari@gmail.com) • [@KDhiraj152](https://github.com/KDhiraj152) • [LinkedIn](https://www.linkedin.com/in/k-dhiraj-83b025279/)

*Last updated: November 2025*
