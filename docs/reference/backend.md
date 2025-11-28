 # Backend — Shiksha Setu

 Overview
- Tech: FastAPI (async), Pydantic v2, SQLAlchemy 2.0, Alembic, Celery, Redis, PostgreSQL (pgvector)
- Location: `backend`/`

Entry points
- `backend/api/main.py` — creates FastAPI app, mounts routers and middleware.
- `backend/api/routes/` — contains route modules: `auth.py`, `content.py`, `qa.py`, `health.py`, etc.

Config & Secrets
- Centralized config via environment variables (see `.env.example`).
- Critical vars: `DATABASE_URL`, `JWT_SECRET_KEY`, `REDIS_URL`, `HUGGINGFACE_API_KEY`, `VITE_API_BASE_URL`.

Auth
- JWT-based access + refresh token flow defined in `backend/api/routes/auth.py` and `backend/core/security.py` (or similar).
- Ensure `JWT_SECRET_KEY` is strong (>=64 chars recommended) and rotated periodically.

Database & models
- SQLAlchemy 2.0 models in `backend/models.py` and `backend/schemas/` for Pydantic schemas.
- Migrations: Alembic config at `alembic/` and migration scripts in `alembic/versions/`.

Async processing
- Celery tasks live under `backend/tasks/` and are queued through Redis (broker) and optionally use Redis/DB for results.
- Long-running pipeline stages offloaded to Celery; ensure workers are idempotent and have retry/backoff.

File uploads
- Implemented in `backend/api/routes/content.py` — supports single and chunked uploads. Validate content type, size, and sanitize filenames.

Observability
- Logging and metrics hooks in `backend/monitoring.py` and middleware.
- Health endpoints in `backend/api/routes/health.py` expose readiness and liveness checks.

Development & run
- Run server locally: `uvicorn backend.api.main:app --reload`
- Run Celery worker: `celery -A backend.tasks.celery_app worker --loglevel=info`

Testing
- Backend tests live in `tests/`. Use `pytest tests/ -v`.

Safe changes to make
- Validate `JWT_SECRET_KEY` length on startup and fail fast if insecure.
- Add stricter upload validation (MIME sniffing + extension checks + size limit config `MAX_UPLOAD_SIZE`).
- Ensure CORS is configurable via env (allowlist) rather than hardcoded.
