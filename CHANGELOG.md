# ShikshaSetu Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.2.0] - 2025-11-28

### Changed - CRITICAL Architecture Consolidation & Bug Fixes

#### Phase 1: Model Client Consolidation
- **Unified Model Client**: Consolidated 4 competing model client implementations into single source of truth
  - Archived `backend/pipeline/model_clients.py` (26KB) to `_deprecated/`
  - Archived `backend/pipeline/model_clients_async.py` (3.9KB) to `_deprecated/`
  - Archived `backend/services/model_client.py` (11KB) to `_deprecated/`
  - **Active**: `backend/services/unified_model_client.py` as single model inference client
  - **Impact**: 75% reduction in model client code, eliminated maintenance overhead

- **Configuration Consolidation**: Single source of truth for all settings
  - Deleted `backend/pipeline/config.py` (redundant configuration)
  - **Active**: `backend/core/config.py` as single configuration file
  - **Impact**: Eliminated configuration inconsistencies

- **Router Renaming**: Resolved naming collision
  - Renamed `backend/services/model_router.py` → `backend/services/ab_test_router.py`
  - Updated class `ModelRouter` → `ABTestRouter`
  - **Rationale**: Clear distinction between resource routing (`model_tier_router.py`) and A/B testing (`ab_test_router.py`)

- **API Structure Consolidation**: Unified route organization
  - Moved `backend/api/endpoints/progress.py` → `backend/api/routes/progress.py`
  - Removed empty `backend/api/endpoints/` directory
  - Updated `backend/api/routes/__init__.py` to export `progress_router`
  - Updated `backend/api/main.py` to include `progress_router`
  - **Impact**: All API routes now in single logical location

#### Phase 2: Script & Configuration Cleanup
- **Eliminated Script Duplication**: Removed competing script files
  - Deleted `scripts/start.sh`, `scripts/stop.sh`, `scripts/test.sh`, `scripts/validate.sh`
  - **Active**: `/bin/` scripts as single user-facing entry points
  - **Impact**: Zero ambiguity on which scripts to run

### Fixed - CRITICAL Bugs
- **Threading Import Bug**: Added missing `import threading` to `backend/core/dynamic_quantization.py`
  - **Severity**: Critical - would cause runtime crash when using `threading.Lock()`
  - **Impact**: Prevents production crashes

- **Path Check Bugs**: Corrected file path checks in setup scripts
  - Fixed `bin/setup`: Changed `config/requirements.txt` → `requirements/base.txt`
  - Fixed `bin/start`: Changed `config/requirements.txt` → `requirements/base.txt`
  - **Impact**: Scripts now execute correctly from project root

### Documentation
- **Updated Structure Documentation**: 
  - `backend/README.md`: Corrected to reflect actual folder structure
  - Clarified that services are in `backend/services/simplify/` not `backend/simplify/`
  - Added accurate descriptions of middleware, schemas, and pipeline directories

### Metrics
| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Model Client Files | 4 | 1 | 75% reduction |
| Config Files | 2 | 1 | 50% reduction |
| Duplicate Scripts | 4 | 0 | 100% eliminated |
| Critical Bugs | 3 | 0 | 100% fixed |
| LOC (redundant) | ~1,500 | 0 | Archived to _deprecated/ |

### Migration Notes
- Archived code in `backend/_deprecated/` can be safely deleted after 1-2 weeks of stable operation
- Zero breaking changes - all functionality preserved
- No import path changes required - deprecated files were already unused

---

## [2.1.0] - 2025-11-27

### Added - HIGH Priority Features (CODE-REVIEW-GPT Issues #9, #11)

#### Error Tracking and Monitoring (Issue #9)
- **Sentry Integration**: Comprehensive error tracking with FastAPI, SQLAlchemy, Celery, Redis integrations
  - `backend/services/error_tracking.py`: Core error tracking service
  - `backend/api/sentry_middleware.py`: Automatic request context capture
  - Automatic exception capture with stack traces and context
  - Performance monitoring with transaction tracking (10% sampling in production)
  - User context association for error reports
  - PII protection with automatic filtering of sensitive data
  - Breadcrumb tracking for debugging context
  - `@monitor_errors` decorator for automatic error capture
  - Environment-aware sampling rates
  - Health check endpoint for Sentry status

- **Configuration**:
  - Added `SENTRY_DSN`, `SENTRY_TRACES_SAMPLE_RATE`, `SENTRY_PROFILES_SAMPLE_RATE` to `.env.example`
  - Initialized in `backend/api/main.py` startup event
  - Middleware integration with request context

#### Curriculum Validator Integration (Issue #11)
- **Validation Service**: Comprehensive content validation against NCERT standards
  - `backend/services/curriculum_validation.py`: Main validation service
  - Three validation types:
    - **NCERT Curriculum Alignment** (70% threshold): Topic coverage, terminology, concept matching
    - **Factual Accuracy** (80% threshold): Knowledge base verification, outdated info detection
    - **Language Appropriateness** (75% threshold): Reading level, vocabulary complexity, readability metrics
  - Comprehensive validation runs all checks in parallel
  - Validation summary and improvement suggestions API

- **Pipeline Integration**:
  - Integrated into `backend/pipeline/orchestrator.py`
  - Automatic validation after content generation
  - Results stored in `content_validation` table
  - Error tracking for validation failures
  - Breadcrumbs for validation progress

- **Database**:
  - `ContentValidation` table with validation results
  - Stores alignment scores, pass/fail status, issues, suggestions
  - JSONB metadata for flexible issue storage

### Tests
- **Unit Tests**:
  - `tests/unit/test_error_tracking.py`: 15+ test cases for error tracking
  - `tests/unit/test_curriculum_validation.py`: 13+ test cases for validation service
  - Coverage for success/failure scenarios, edge cases, integration points

### Documentation
- **New Guide**: `docs/error-tracking-and-validation.md`
  - Complete implementation guide for Issues #9 and #11
  - Usage examples and API documentation
  - Configuration and troubleshooting guides
  - Metrics and monitoring queries
  - Integration patterns and best practices

### Changed
- **Application Startup** (`backend/api/main.py`):
  - Added Sentry initialization
  - Added Sentry context middleware
  - Enhanced error reporting with automatic context

- **Content Pipeline** (`backend/pipeline/orchestrator.py`):
  - Integrated curriculum validation
  - Added error tracking breadcrumbs
  - Enhanced logging for validation results

### Impact
- **Production Readiness**: 
  - Comprehensive error tracking prevents silent failures
  - Proactive monitoring with performance metrics
  - Automatic alerting for critical errors

- **Content Quality**:
  - All generated content validated against curriculum standards
  - Factual accuracy checks prevent misinformation
  - Grade-appropriate language ensures accessibility

- **Debugging**:
  - Rich context in error reports (request, user, breadcrumbs)
  - Performance bottleneck identification
  - User impact analysis

---

## [2.0.0] - 2025-11-26

### Added - CRITICAL Infrastructure (CODE-REVIEW-GPT Issues #3, #5, #6, #7, #22)

#### Requirements Management (Issue #22)
- **Production Dependencies**: `requirements/base.txt` with 100+ pinned versions
  - FastAPI 0.115.5, PyTorch 2.5.1, SQLAlchemy 2.0.36
  - vLLM 0.6.3, NVIDIA Triton 2.50.0, DVC 3.56.0
  - Sentry SDK 2.22.0, Great Expectations 1.2.5
- **Development Dependencies**: `requirements/dev.txt`
  - pytest 8.3.4, pytest-asyncio 0.24.0, pytest-cov 6.0.0
  - black, flake8, mypy, pre-commit hooks

#### Authentication Security + Token Rotation (Issue #6)
- **Security Hardening** (`backend/utils/auth.py`):
  - Removed auto-generation of `SECRET_KEY` (fails fast if missing)
  - Enforced 64-character minimum for JWT secrets
  - Password strength validation (12+ chars, complexity requirements)
  - Configurable bcrypt rounds (default: 12)
  
- **Token Rotation Mechanism**:
  - JWT with JTI (unique token IDs) for tracking
  - `TokenBlacklist` and `RefreshToken` database models
  - `backend/services/token_service.py`: Complete token lifecycle management
  - Device fingerprinting for session tracking
  - Parent JTI tracking for rotation chains
  - Rotation count limits (max 100) to prevent attacks
  - Background cleanup of expired tokens
  - Logout all devices functionality

#### Database Schema Normalization (Issue #5)
- **Schema Fixes** (`alembic/versions/013_normalize_schema_fix_fk.py`):
  - Fixed `user_id`/`content_id` type mismatches (String → UUID)
  - Added foreign key constraints with CASCADE deletes
  - Normalized `ProcessedContent` table:
    - Extracted `content_translations` table
    - Extracted `content_audio` table
    - Extracted `content_validation` table
  - Added composite indexes for common queries
  - Added check constraints for score ranges (0-100)

#### Backend Consolidation (Issue #7)
- **Duplication Removal**:
  - Merged all models into `backend/models.py` (19 total)
  - Removed entire `src/` directory
  - Updated all imports to `backend.*`
  - Single source of truth established
  - Created `BACKEND_MERGE_PLAN.md` documentation

#### Model Serving Architecture (Issue #3)
- **Production-Ready Model Serving**:
  - `backend/core/model_serving.py`: Configuration system
  - `backend/services/model_client.py`: Unified client interface
  - **vLLM Client**: LLM inference (Mistral-7B AWQ)
  - **Triton Client**: ONNX models (embedding, translation, TTS)
  - **Ollama Client**: Local development
  - DVC integration for model versioning
  - Deployment scripts: `scripts/deploy_vllm.sh`, `scripts/deploy_triton.sh`
  - Health checks and retry logic
  - GPU detection and configuration

### Added - CRITICAL Architecture (CODE-REVIEW-GPT Issues #2, #4)

#### PS4 Teacher Performance Evaluation (Issue #2)
- **Database Models** (7 new):
  - `TeacherProfile`: Qualifications, subjects, aggregate scores
  - `ContentQualityMetric`: NCERT alignment, clarity, pedagogy, cultural sensitivity
  - `StudentEngagementMetric`: Completion rates, interactions, session duration
  - `LearningOutcomeMetric`: Assessment scores, improvement, retention, mastery
  - `TeacherEvaluation`: Comprehensive reports with weighted scoring

- **Evaluation Service** (`backend/services/teacher_evaluation.py`):
  - Weighted scoring algorithms:
    - Content Quality: 40% (NCERT 30%, Clarity 25%, Pedagogy 25%, Cultural 20%)
    - Student Engagement: 35%
    - Learning Outcomes: 25%
  - Star rating system (0-5 stars)
  - Performance categorization (Excellent, Good, Satisfactory, Needs Improvement)
  - Strengths and improvement recommendations

#### Offline-First PWA Architecture (Issue #4)
- **Service Worker** (`frontend/public/service-worker.js`, 400+ lines):
  - Cache-first strategy for static assets
  - Network-first strategy for API requests
  - Background sync for offline request queueing
  - Push notification foundation
  - Automatic cache cleanup

- **IndexedDB Client** (`frontend/src/utils/offlineDB.js`, 300+ lines):
  - Stores: content, progress, sync-queue, quiz-results, preferences
  - Automatic sync triggering
  - Storage quota monitoring
  - Comprehensive CRUD operations

- **PWA Manifest** (`frontend/public/manifest.json`):
  - Standalone display mode
  - App shortcuts (Browse, Progress, Offline Content)
  - Icons (192x192, 512x512)
  - Installable on mobile devices

### Added - Testing Infrastructure (Issue #8 - Partial)
- **pytest Configuration** (`pytest.ini`):
  - Test markers: unit, integration, e2e, performance, slow, gpu
  - Coverage threshold: 70%
  - Asyncio mode: auto
  - Parallel execution support

- **Comprehensive Fixtures** (`tests/conftest.py`):
  - PostgreSQL test database
  - FastAPI TestClient
  - Authenticated clients (user, admin)
  - JWT token generation
  - Test content fixtures
  - Mock model clients (vLLM, Triton)
  - Async support

### Changed
- **Database Migrations**:
  - `013_normalize_schema_fix_fk.py`: Schema normalization
  - `014_token_rotation_teacher_eval.py`: Token rotation + teacher evaluation (9 new tables)

- **Configuration** (`.env.example`):
  - Added JWT security configuration
  - Added token expiration settings
  - Added password policy configuration
  - Added model serving endpoints

### Fixed
- SonarQube errors: 90 → 0 workspace errors
- Cognitive complexity issues in pipeline orchestrator
- Enum syntax errors in models
- Type annotation issues
- Import path inconsistencies

### Documentation
- `BACKEND_MERGE_PLAN.md`: Backend consolidation strategy
- `docs/error-tracking-and-validation.md`: Implementation guide
- Updated API documentation with new endpoints

---

## [1.0.0] - 2025-11-20

### Initial Release
- Core multilingual content processing pipeline
- NCERT curriculum validator
- Translation services (Bhashini API, IndicTrans2)
- Text-to-speech generation
- FastAPI REST API
- PostgreSQL database with pgvector
- Redis caching
- Basic authentication
- Docker deployment

---

## Progress Summary

### CODE-REVIEW-GPT Issues (25 Total)
- **COMPLETE**: 13/25 issues (52%)
  - All 6 CRITICAL issues (minus skipped PS2)
  - 2 HIGH priority issues
  - Testing framework foundation
  
- **IN PROGRESS**: 0/25
  
- **PENDING**: 12/25 issues (48%)
  - 5 HIGH priority
  - 6 MEDIUM priority
  - 1 LOW priority

### Next Release Targets (v2.2.0)
- Issue #10: Frontend TypeScript migration
- Issue #12: Cultural context service integration
- Issue #13: Grade-level adaptation
- Issue #14: A/B testing framework
- Complete Issue #8: Write comprehensive test files

---

## 👨‍💻 Author

**K Dhiraj** • [k.dhiraj.srihari@gmail.com](mailto:k.dhiraj.srihari@gmail.com) • [@KDhiraj152](https://github.com/KDhiraj152) • [LinkedIn](https://www.linkedin.com/in/k-dhiraj-83b025279/)

*Last updated: November 2025*


