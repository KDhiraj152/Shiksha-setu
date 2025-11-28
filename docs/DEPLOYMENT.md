# Deployment Guide

**Production deployment for ShikshaSetu**

Complete guide for deploying ShikshaSetu to Docker, Kubernetes, and cloud platforms.

---

## Overview

**Deployment Options**:
- **Docker Compose** - Development & small-scale production
- **Kubernetes** - Scalable production deployment
- **Cloud Platforms** - AWS, GCP, Azure

**Architecture**:
```
Load Balancer
    ↓
Backend API (FastAPI)
    ↓
├─→ PostgreSQL + pgvector
├─→ Redis (Cache + Celery)
└─→ Celery Workers (AI/ML)
```

---

## Docker Deployment

### Development Setup

**Location**: `infrastructure/docker/`

```bash
cd infrastructure/docker
docker-compose up -d
```

**Services Started**:
- `backend` - FastAPI API (port 8000)
- `frontend` - React app (port 5173)
- `postgres` - PostgreSQL 17 + pgvector
- `redis` - Redis 7.4
- `celery-worker` - Background task processing

**View Logs**:
```bash
docker-compose logs -f backend
docker-compose logs -f celery-worker
```

**Stop Services**:
```bash
docker-compose down
```

### Production Docker

**Build Images**:
```bash
# Backend
docker build -t shiksha-setu/backend:v1.0.0 -f infrastructure/docker/Dockerfile .

# Worker
docker build -t shiksha-setu/worker:v1.0.0 -f infrastructure/docker/worker.dockerfile .
```

**Push to Registry**:
```bash
docker tag shiksha-setu/backend:v1.0.0 your-registry/shiksha-setu/backend:v1.0.0
docker push your-registry/shiksha-setu/backend:v1.0.0
```

**Run with Production Config**:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

---

## Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (v1.24+)
- `kubectl` installed
- Docker registry access
- Domain name (for ingress)

### Directory Structure

```
infrastructure/kubernetes/
├── base/                      # Base manifests
│   ├── deployment.yaml        # Backend deployment
│   ├── service.yaml           # Services
│   └── configmap.yaml         # Configuration
├── overlays/
│   ├── dev/                   # Development
│   ├── staging/               # Staging
│   └── prod/                  # Production
└── kustomization.yaml         # Kustomize config
```

### Development Deployment

```bash
cd infrastructure/kubernetes

# Apply development overlay
kubectl apply -k overlays/dev

# Check status
kubectl get pods -n shiksha-setu
kubectl get svc -n shiksha-setu

# Watch rollout
kubectl rollout status deployment/backend -n shiksha-setu
```

### Production Deployment

**1. Configure Secrets**:
```bash
# Create namespace
kubectl create namespace shiksha-setu

# Create secrets
kubectl create secret generic shiksha-secrets \
  --from-literal=DATABASE_URL="postgresql://..." \
  --from-literal=JWT_SECRET_KEY="..." \
  --from-literal=REDIS_URL="redis://..." \
  -n shiksha-setu

# Create HuggingFace token (optional)
kubectl create secret generic huggingface-token \
  --from-literal=HUGGINGFACE_API_KEY="hf_..." \
  -n shiksha-setu
```

**2. Update Image Tags**:
```yaml
# overlays/prod/kustomization.yaml
images:
  - name: shiksha-setu/backend
    newName: your-registry/shiksha-setu/backend
    newTag: v1.0.0
```

**3. Deploy**:
```bash
# Apply production overlay
kubectl apply -k overlays/prod

# Verify deployment
kubectl get all -n shiksha-setu

# Check logs
kubectl logs -f deployment/backend -n shiksha-setu
```

**4. Run Migrations**:
```bash
# Get backend pod name
BACKEND_POD=$(kubectl get pods -n shiksha-setu -l app=backend -o jsonpath='{.items[0].metadata.name}')

# Run migrations
kubectl exec -it $BACKEND_POD -n shiksha-setu -- alembic upgrade head
```

### Scaling

**Horizontal Pod Autoscaler**:
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: backend-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: backend
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

**Manual Scaling**:
```bash
# Scale backend
kubectl scale deployment/backend --replicas=5 -n shiksha-setu

# Scale workers
kubectl scale deployment/celery-worker --replicas=3 -n shiksha-setu
```

---

## Database Setup

### PostgreSQL with pgvector

**Managed Services** (Recommended):
- **Supabase** - PostgreSQL + pgvector included
- **AWS RDS** - PostgreSQL 15+ (enable pgvector extension)
- **Google Cloud SQL** - PostgreSQL 15+
- **Azure Database** - PostgreSQL 15+

**Self-Hosted**:
```bash
# Using pgvector Docker image
docker run -d \
  --name postgres \
  -e POSTGRES_DB=shiksha_setu \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=secure_password \
  -p 5432:5432 \
  pgvector/pgvector:pg17
```

**Enable pgvector**:
```sql
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify
SELECT * FROM pg_extension WHERE extname = 'vector';
```

### Redis Setup

**Managed Services**:
- **AWS ElastiCache** - Redis
- **Google Memorystore** - Redis
- **Redis Cloud** - Managed Redis

**Self-Hosted**:
```bash
docker run -d \
  --name redis \
  -p 6379:6379 \
  redis:7.4-alpine
```

---

## Environment Variables

### Required Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@host:5432/dbname

# JWT Security (64+ chars recommended)
JWT_SECRET_KEY=your-secure-random-key-here

# Redis
REDIS_URL=redis://redis:6379/0

# API Configuration
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=60
MAX_UPLOAD_SIZE=104857600  # 100MB

# Frontend
VITE_API_BASE_URL=https://api.shiksh asetu.com
```

### Optional Variables

```bash
# HuggingFace (for cloud inference)
HUGGINGFACE_API_KEY=hf_...

# Monitoring
SENTRY_DSN=https://...
PROMETHEUS_ENABLED=true

# Email (for notifications)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=noreply@shikshasetu.com
```

---

## Monitoring & Observability

### Prometheus Setup

**Location**: `infrastructure/monitoring/prometheus/`

```bash
# Deploy monitoring stack
kubectl apply -f infrastructure/monitoring/prometheus/

# Access Prometheus
kubectl port-forward svc/prometheus 9090:9090 -n monitoring
# Visit: http://localhost:9090
```

**Key Metrics**:
- `http_requests_total` - Total API requests
- `http_request_duration_seconds` - Request latency
- `celery_task_duration_seconds` - Task processing time
- `db_connection_pool_size` - Database connections

### Grafana Setup

```bash
# Deploy Grafana
kubectl apply -f infrastructure/monitoring/grafana/

# Get admin password
kubectl get secret grafana-admin -n monitoring -o jsonpath='{.data.password}' | base64 -d

# Access Grafana
kubectl port-forward svc/grafana 3000:3000 -n monitoring
# Visit: http://localhost:3000
```

**Dashboards**:
- API Performance
- Database Metrics
- Celery Tasks
- System Resources

### Logging

**Centralized Logging Stack**:
```bash
# Deploy ELK stack or Loki
kubectl apply -f infrastructure/monitoring/logging/

# View logs
kubectl logs -f deployment/backend -n shiksha-setu | jq
```

**Log Aggregation**:
- **ELK Stack** - Elasticsearch, Logstash, Kibana
- **Loki** - Grafana Loki
- **Cloud Logging** - AWS CloudWatch, GCP Logging, Azure Monitor

---

## CI/CD Pipeline

### GitHub Actions

**Location**: `.github/workflows/deploy.yml`

```yaml
name: Deploy

on:
  push:
    branches: [main]
    tags: ['v*']

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: |
          pip install -r requirements.txt
          pytest tests/

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker images
        run: |
          docker build -t ${{ secrets.REGISTRY }}/backend:${{ github.sha }} .
          docker push ${{ secrets.REGISTRY }}/backend:${{ github.sha }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/backend \
            backend=${{ secrets.REGISTRY }}/backend:${{ github.sha }} \
            -n shiksha-setu
```

### Deployment Workflow

1. **Commit** → GitHub
2. **CI** → Run tests
3. **Build** → Docker images
4. **Push** → Container registry
5. **Deploy** → Kubernetes cluster
6. **Verify** → Health checks
7. **Monitor** → Grafana dashboards

---

## Security Best Practices

### Secrets Management

**Kubernetes Secrets**:
```bash
# Create secret from file
kubectl create secret generic app-secrets \
  --from-env-file=.env.production \
  -n shiksha-setu

# Use External Secrets Operator
kubectl apply -f infrastructure/kubernetes/external-secrets/
```

**Never Commit**:
- ❌ `.env` files
- ❌ API keys
- ❌ Database passwords
- ❌ JWT secret keys
- ❌ Private keys/certificates

### Network Security

**Network Policies**:
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: backend-policy
spec:
  podSelector:
    matchLabels:
      app: backend
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: frontend
    ports:
    - protocol: TCP
      port: 8000
```

**Ingress with TLS**:
```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: shiksha-ingress
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - api.shikshasetu.com
    secretName: shiksha-tls
  rules:
  - host: api.shikshasetu.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: backend
            port:
              number: 8000
```

### Container Security

**Image Scanning**:
```bash
# Scan with Trivy
trivy image shiksha-setu/backend:v1.0.0

# Scan with Snyk
snyk container test shiksha-setu/backend:v1.0.0
```

**Security Context**:
```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
  capabilities:
    drop:
    - ALL
```

---

## Backup & Disaster Recovery

### Database Backups

**Automated Backups**:
```bash
# PostgreSQL backup script
#!/bin/bash
BACKUP_DIR="/backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

pg_dump -h $DB_HOST -U $DB_USER -d shiksha_setu \
  -F c -b -v -f "$BACKUP_DIR/backup_$TIMESTAMP.dump"

# Retain last 7 days
find $BACKUP_DIR -name "backup_*.dump" -mtime +7 -delete
```

**Cron Schedule**:
```bash
# Daily at 2 AM
0 2 * * * /scripts/backup.sh
```

**Restore**:
```bash
pg_restore -h $DB_HOST -U $DB_USER -d shiksha_setu -v backup.dump
```

### Application Data

**Persistent Volumes**:
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: storage-pvc
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
  storageClassName: fast-ssd
```

**Backup Strategy**:
- **Daily** - Database snapshots
- **Weekly** - Full backup
- **Monthly** - Archive backup
- **Retention** - 30 days standard, 1 year archives

---

## Performance Tuning

### Backend Optimization

**Uvicorn Workers**:
```bash
# Multiple workers for production
uvicorn backend.api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker
```

**Connection Pooling**:
```python
# SQLAlchemy configuration
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,
    pool_recycle=3600
)
```

### Database Optimization

**PostgreSQL Configuration**:
```sql
-- production postgresql.conf
shared_buffers = 4GB
effective_cache_size = 12GB
work_mem = 64MB
maintenance_work_mem = 1GB
max_connections = 200
```

**Index Monitoring**:
```sql
-- Find missing indexes
SELECT schemaname, tablename, attname, n_distinct, correlation
FROM pg_stats
WHERE schemaname = 'public'
ORDER BY n_distinct DESC;
```

### Celery Workers

**Worker Configuration**:
```bash
celery -A backend.tasks.celery_app worker \
  --loglevel=info \
  --concurrency=4 \
  --max-tasks-per-child=100 \
  --prefetch-multiplier=4
```

**Queue Prioritization**:
```python
# High priority for user-facing tasks
app.send_task('process_content', queue='high_priority')

# Low priority for batch jobs
app.send_task('batch_translate', queue='low_priority')
```

---

## Health Checks

### Kubernetes Probes

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
  timeoutSeconds: 5
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 3
```

### Health Endpoints

**GET `/health`** - Basic health check
```json
{
  "status": "healthy",
  "timestamp": "2025-11-28T10:00:00Z",
  "version": "1.0.0"
}
```

**GET `/health/ready`** - Readiness check
```json
{
  "status": "ready",
  "database": "connected",
  "redis": "connected",
  "celery": "available"
}
```

---

## Troubleshooting

### Common Issues

**Pod CrashLoopBackOff**:
```bash
# Check logs
kubectl logs pod-name -n shiksha-setu --previous

# Check events
kubectl describe pod pod-name -n shiksha-setu
```

**Database Connection Errors**:
```bash
# Test connection
kubectl run -it --rm debug --image=postgres:17 --restart=Never -- \
  psql -h postgres-service -U postgres -d shiksha_setu
```

**High Memory Usage**:
```bash
# Check resource usage
kubectl top pods -n shiksha-setu

# Adjust limits
kubectl set resources deployment/backend \
  --limits=memory=2Gi \
  -n shiksha-setu
```

### Debug Mode

```bash
# Enable debug logging
kubectl set env deployment/backend \
  LOG_LEVEL=DEBUG \
  -n shiksha-setu

# Access pod shell
kubectl exec -it backend-pod -n shiksha-setu -- /bin/bash
```

---

## Production Checklist

### Pre-Deployment

- [ ] All tests passing
- [ ] Environment variables configured
- [ ] Secrets created in Kubernetes
- [ ] Database migrations tested
- [ ] SSL certificates configured
- [ ] Monitoring setup complete
- [ ] Backup strategy in place

### Post-Deployment

- [ ] Health checks passing
- [ ] API endpoints responding
- [ ] Database connections working
- [ ] Celery workers active
- [ ] Logs aggregating correctly
- [ ] Metrics collecting
- [ ] SSL certificate valid
- [ ] DNS configured

### Security Audit

- [ ] No secrets in code
- [ ] All dependencies updated
- [ ] Container images scanned
- [ ] Network policies applied
- [ ] RBAC configured
- [ ] Audit logging enabled

---

## Related Documentation

- **[DATABASE.md](DATABASE.md)** - Database setup and migrations
- **[API.md](reference/api.md)** - API endpoint reference
- **[BACKEND.md](reference/backend.md)** - Backend architecture
- **[Kubernetes SETUP](../infrastructure/kubernetes/SETUP.md)** - K8s detailed guide

---

## 👨‍💻 Made By

**K Dhiraj Srihari**

🔗 **Connect:**
- 📧 [k.dhiraj.srihari@gmail.com](mailto:k.dhiraj.srihari@gmail.com)
- 💼 [LinkedIn](https://linkedin.com/in/k-dhiraj)
- 🐙 [GitHub](https://github.com/KDhiraj152)
