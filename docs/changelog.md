# Changelog

All notable changes to ShikshaSetu will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.2.0] - 2025-11-27

### Added
- Frontend test framework (Vitest) with test utilities
- Comprehensive full-stack validation report
- Complete `.env.example` with documented configuration
- Security headers enhancement (7 protection headers)
- Improved middleware ordering for proper execution flow

### Fixed
- Removed `async` keyword from non-async exception handlers
- Fixed rate limiting configuration to be environment-aware
- Corrected test database setup (SQLite instead of PostgreSQL)
- Fixed SonarLint violations in test setup
- Resolved Import path issues in vitest config

### Improved
- Code quality: 15/15 unit tests passing
- Frontend tests: 2/2 tests passing
- Security: All 7 headers properly configured
- Documentation: Single unified README with clear structure
- Test infrastructure: Proper Vitest setup with mocks

### Removed
- Duplicate root-level markdown files (QUICK_START, CLEANUP_REPORT, etc.)
- Redundant test configurations
- Temporary validation reports from root directory

---

## [2.1.0] - 2025-11-16

### Added
- Centralized schema organization in `src/schemas/`
- Constants module for error messages and configuration
- Comprehensive documentation structure
- pgvector support for RAG system

### Changed
- Simplified file naming across all modules
- Reorganized documentation into logical folders
- Updated all imports to use new simplified names
- Improved code organization and maintainability

### Fixed
- Frontend promise chain to use top-level await
- Cache import in main.py (get_redis)
- Import paths across entire codebase

### Removed
- Duplicate async_app.py file
- Redundant documentation files
- Test artifacts and temporary files

---

## [2.0.0] - 2025-11-15

### Added
- JWT authentication with refresh tokens
- Role-based access control
- RAG-based Q&A system with pgvector
- Rate limiting middleware
- Comprehensive API documentation
- Docker and Kubernetes deployment configs
- React 19 frontend with TypeScript
- TailwindCSS 4 styling

### Changed
- Upgraded to Python 3.13
- Migrated to React 19
- Updated all dependencies to latest versions
- Improved database connection handling
- Enhanced security middleware

### Fixed
- Database connection validation
- JWT secret key generation
- Redis connection handling
- Code complexity issues
- Duplicate string literals

---

## [1.0.0] - 2025-11-15

### Added
- Initial release
- Text simplification with FLAN-T5
- Multi-language translation (IndicTrans2)
- NCERT content validation
- Text-to-speech generation
- FastAPI backend with Celery
- PostgreSQL database with SQLAlchemy
- Basic authentication

---

## Release Information

- **Latest Version**: 2.2.0
- **Release Date**: November 27, 2025
- **Status**: ✅ Production Ready

For more details, visit [GitHub Releases](https://github.com/KDhiraj152/Siksha-Setu/releases)
