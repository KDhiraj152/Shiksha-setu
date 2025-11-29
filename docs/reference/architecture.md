 # Architecture — Shiksha Setu

 System overview

 - Frontend: Single-page app served by Vite dev server or static assets behind CDN.
 - Backend: FastAPI app handling auth, API, and job submission.
 - Worker layer: Celery workers for pipeline stages (OCR, simplify, translate, validate, TTS).
 - Broker: Redis (celery broker + cache). 
 - Database: PostgreSQL with `pgvector` extension for document embeddings.
 - Vector store: `pgvector` (in-DB) or external service (Qdrant) for higher-scale deployments.

Interaction flow (simplified)

```
User (browser)
    | --upload-->  FastAPI ( /api/v1/content/upload )
    |              validates -> stores file -> enqueues Celery task
    | <--task_id--- Client polls /api/v1/tasks/{task_id}
    v
Celery worker(s) execute pipeline stages -> store results in DB and object store -> index embeddings into pgvector
    v
FastAPI serves processed content and RAG endpoints
```

Key components & files

- `frontend/` — UI and client integration.
- `backend/api/main.py` — app initialization, middleware, CORS, logging.
- `backend/api/routes/*` — route handlers.
- `backend/pipeline/orchestrator.py` — high-level pipeline orchestration.
- `backend/tasks/*` — Celery task registration and helpers.
- `backend/services/rag.py` — embedding and retrieval logic.

Deployment modes

- Development: local Postgres + Redis or docker-compose defined in `docker-compose.yml`.
- Production: Kubernetes overlays in `k8s/` with HPA and secrets. Consider moving vector DB to managed Qdrant or dedicated pgvector cluster for scale.

Security & networking

- Use an API gateway or ingress with TLS in production.
- Use network policies to restrict DB/Redis access to FastAPI and workers.

---

## 👨‍💻 Author

**K Dhiraj** • [k.dhiraj.srihari@gmail.com](mailto:k.dhiraj.srihari@gmail.com) • [@KDhiraj152](https://github.com/KDhiraj152) • [LinkedIn](https://www.linkedin.com/in/k-dhiraj-83b025279/)

*Last updated: November 2025*
