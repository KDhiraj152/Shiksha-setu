# 🚀 Deployment Guide — Shiksha Setu

Complete production deployment guide for Shiksha Setu educational platform. Covers Docker, Kubernetes, and cloud deployment strategies.

---

## 📋 Table of Contents

1. [Deployment Status](#deployment-status)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Docker Deployment](#docker-deployment)
5. [Kubernetes Deployment](#kubernetes-deployment)
6. [Cloud Platforms](#cloud-platforms)
7. [Post-Deployment Verification](#post-deployment-verification)
8. [Monitoring Setup](#monitoring-setup)
9. [Rollback Procedures](#rollback-procedures)
10. [Troubleshooting](#troubleshooting)

---

## 📊 Deployment Status

### Current Release: v1.0.0

| Component | Status | Tests | Coverage | Notes |
|-----------|--------|-------|----------|-------|
| Backend API | ✅ Ready | 93 passing | 23% | FastAPI + PostgreSQL |
| Frontend | ✅ Ready | 19 passing | 100% | React 19 + TypeScript |
| AI/ML Pipeline | ✅ Ready | Integrated | N/A | FLAN-T5, IndicTrans2, MMS-TTS |
| Database | ✅ Ready | Migrations OK | N/A | PostgreSQL 15 + pgvector |
| Caching | ✅ Ready | Functional | N/A | Redis 7.4 |
| Monitoring | ✅ Ready | Configured | N/A | Prometheus + Grafana |
| CI/CD | ✅ Ready | 4 workflows | N/A | GitHub Actions |

**Infrastructure Readiness**:
- ✅ Docker Configuration: 15 services with HA setup
- ✅ Kubernetes Manifests: Dev, Staging, Production overlays
- ✅ Nginx Configuration: Load balancing, SSL/TLS, security headers
- ✅ Monitoring Stack: Prometheus, Grafana, Alertmanager
- ✅ CI/CD Pipelines: Test, Build, Deploy-Staging, Deploy-Production
- ✅ Backup Strategy: Automated PostgreSQL backups
- ✅ Rollback Capability: Automated rollback in deploy workflow

---

## ⚙️ Prerequisites

### Hardware Requirements

**Minimum (Development)**:
- 4 CPU cores
- 16GB RAM
- 100GB SSD storage

**Recommended (Production)**:
- 8+ CPU cores
- 32GB+ RAM
- 500GB+ SSD storage
- GPU (NVIDIA with CUDA) for ML inference

### Software Requirements

- **OS**: Ubuntu 22.04 LTS or macOS
- **Docker**: v24.0+
- **Docker Compose**: v2.20+
- **Python**: 3.11 or 3.12
- **Node.js**: 20.x LTS
- **PostgreSQL**: 15+ with pgvector extension
- **Redis**: 7.0+
- **kubectl**: v1.24+ (for Kubernetes)

### Required Secrets/Credentials

Before deployment, obtain:

- [ ] `DATABASE_URL` - PostgreSQL connection string
- [ ] `REDIS_URL` - Redis connection URL
- [ ] `HUGGINGFACE_API_KEY` - HuggingFace API token
- [ ] `JWT_SECRET_KEY` - Session signing key
- [ ] `SENTRY_DSN` - Error tracking URL
- [ ] `GITHUB_TOKEN` - CI/CD automation
- [ ] `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` - S3 storage (optional)
- [ ] SSL certificates - For HTTPS/TLS (production)

---

## 🎯 Quick Start

```bash
# Clone repository
git clone https://github.com/KDhiraj152/Siksha-Setu.git
cd Siksha-Setu

# Run automated setup
./SETUP.sh

# Start all services
./START.sh

# Validate deployment
docker exec shiksha-backend python -m pytest tests/ -v
```

---

## 🐳 Docker Deployment

### Development Setup

**Location**: `infrastructure/docker/`

```bash
cd infrastructure/docker
docker-compose up -d
```

**Services Started** (15 total):
| Service | Port | Purpose |
|---------|------|---------|
| `backend` | 8000 | FastAPI API |
| `frontend` | 5173 | React application |
| `postgres` | 5432 | PostgreSQL database |
| `redis` | 6379 | Cache & message broker |
| `celery-worker` | N/A | Background tasks |
| `nginx` | 80, 443 | Reverse proxy |
| `prometheus` | 9090 | Metrics collection |
| `grafana` | 3000 | Metrics visualization |
| `mms-tts` | 8888 | Text-to-speech service |
| Plus 6 additional support services | N/A | Exporters, Alertmanager, etc. |

**View Logs**:
```bash
docker-compose logs -f backend
docker-compose logs -f celery-worker
```

**Stop Services**:
```bash
docker-compose down
```

### Production Docker Setup

**Build Images**:
```bash
# Backend (with ML models)
docker build -t shiksha-setu/backend:v1.0.0 \
  -f infrastructure/docker/Dockerfile \
  --build-arg PYTHON_VERSION=3.11 \
  .

# Celery Worker (for background tasks)
docker build -t shiksha-setu/worker:v1.0.0 \
  -f infrastructure/docker/worker.dockerfile \
  .

# Frontend
docker build -t shiksha-setu/frontend:v1.0.0 \
  -f frontend/Dockerfile \
  frontend/
```

**Push to Registry**:
```bash
# Configure your registry
REGISTRY="your-docker-registry.com"

docker tag shiksha-setu/backend:v1.0.0 $REGISTRY/shiksha-setu/backend:v1.0.0
docker push $REGISTRY/shiksha-setu/backend:v1.0.0

docker tag shiksha-setu/worker:v1.0.0 $REGISTRY/shiksha-setu/worker:v1.0.0
docker push $REGISTRY/shiksha-setu/worker:v1.0.0
```

**Run with Production Configuration**:
```bash
# Use production docker-compose file
docker-compose -f config/docker-compose.production.yml up -d

# Or: Run with environment variables
docker run -d \
  -e DATABASE_URL="postgresql://..." \
  -e REDIS_URL="redis://redis:6379" \
  -p 8000:8000 \
  shiksha-setu/backend:v1.0.0
```

**Health Checks**:
```bash
# Backend health
curl http://localhost:8000/health

# Frontend
curl http://localhost/

# Database connection
docker exec shiksha-postgres psql -U postgres -c "SELECT 1"
```

---

## ☸️ Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (v1.24+)
- `kubectl` installed and configured
- Docker registry access
- Domain name (for ingress)
- Persistent storage provisioner

### Directory Structure

```
infrastructure/kubernetes/
├── base/                          # Base manifests (shared)
│   ├── deployment.yaml            # Backend deployment
│   ├── service.yaml               # Services
│   ├── configmap.yaml             # Configuration
│   ├── secret.yaml                # Secrets template
│   ├── postgres/                  # Database manifests
│   ├── redis/                     # Cache manifests
│   └── celery/                    # Task queue manifests
├── overlays/
│   ├── dev/                       # Development overlay
│   │   ├── kustomization.yaml
│   │   └── patches/
│   ├── staging/                   # Staging overlay
│   │   ├── kustomization.yaml
│   │   └── patches/
│   └── prod/                      # Production overlay
│       ├── kustomization.yaml
│       └── patches/
└── scripts/
    ├── deploy.sh
    ├── rollback.sh
    └── validate.sh
```

### Deployment Steps

**1. Create Namespace**:
```bash
kubectl create namespace shiksha-setu
kubectl config set-context --current --namespace=shiksha-setu
```

**2. Create Secrets**:
```bash
kubectl create secret generic shiksha-secrets \
  --from-literal=DATABASE_URL="postgresql://..." \
  --from-literal=REDIS_URL="redis://redis:6379" \
  --from-literal=JWT_SECRET_KEY="..." \
  -n shiksha-setu
```

**3. Deploy with Kustomize**:
```bash
# Development
kubectl apply -k infrastructure/kubernetes/overlays/dev

# Staging
kubectl apply -k infrastructure/kubernetes/overlays/staging

# Production
kubectl apply -k infrastructure/kubernetes/overlays/prod
```

**4. Verify Deployment**:
```bash
kubectl get all -n shiksha-setu
kubectl get pods -n shiksha-setu -w
```

**5. Port Forward (Testing)**:
```bash
kubectl port-forward svc/backend 8000:8000 -n shiksha-setu
kubectl port-forward svc/frontend 5173:80 -n shiksha-setu
```

### Ingress Configuration

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: shiksha-ingress
  namespace: shiksha-setu
spec:
  tls:
  - hosts:
    - api.example.com
    secretName: tls-cert
  rules:
  - host: api.example.com
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

### Scaling

```bash
# Scale backend replicas
kubectl scale deployment backend --replicas=3 -n shiksha-setu

# Scale workers
kubectl scale deployment celery-worker --replicas=5 -n shiksha-setu

# Horizontal Pod Autoscaler
kubectl autoscale deployment backend --min=2 --max=10 --cpu-percent=70 -n shiksha-setu
```

---

## ☁️ Cloud Platforms

### AWS Deployment

**Using ECS (Elastic Container Service)**:
```bash
# Push images to ECR
aws ecr create-repository --repository-name shiksha-setu/backend
aws ecr get-login-password | docker login --username AWS --password-stdin <account-id>.dkr.ecr.us-east-1.amazonaws.com
docker tag shiksha-setu/backend:latest <account-id>.dkr.ecr.us-east-1.amazonaws.com/shiksha-setu/backend:latest
docker push <account-id>.dkr.ecr.us-east-1.amazonaws.com/shiksha-setu/backend:latest

# Deploy via CloudFormation or Terraform
cd infrastructure/terraform
terraform init
terraform apply -var "environment=production"
```

### Google Cloud (GKE)

```bash
# Create GKE cluster
gcloud container clusters create shiksha-setu \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type n1-standard-4

# Configure kubectl
gcloud container clusters get-credentials shiksha-setu --zone us-central1-a

# Deploy
kubectl apply -k infrastructure/kubernetes/overlays/prod
```

### Azure (AKS)

```bash
# Create AKS cluster
az aks create \
  --resource-group shiksha-rg \
  --name shiksha-setu \
  --node-count 3 \
  --vm-set-type VirtualMachineScaleSets \
  --load-balancer-sku standard

# Configure kubectl
az aks get-credentials --resource-group shiksha-rg --name shiksha-setu

# Deploy
kubectl apply -k infrastructure/kubernetes/overlays/prod
```

---

## ✅ Post-Deployment Verification

```bash
# Backend health check
curl http://localhost:8000/health

# API documentation
curl http://localhost:8000/docs

# Database connectivity
psql $DATABASE_URL -c "SELECT version();"

# Redis connectivity
redis-cli -u $REDIS_URL PING

# Frontend accessibility
curl http://localhost/

# Run test suite
python -m pytest tests/ -v --tb=short
```

---

## 📊 Monitoring Setup

### Prometheus Configuration

**Scrape Targets**:
- Backend: `:8000/metrics`
- PostgreSQL Exporter: `:9187/metrics`
- Redis Exporter: `:9121/metrics`
- Node Exporter: `:9100/metrics`

### Grafana Dashboards

**Pre-built Dashboards**:
- API Performance (requests/latency)
- Database Metrics (connections, queries)
- Redis Memory Usage
- System Resources (CPU, Memory, Disk)
- Celery Task Queue

**Access**: `http://localhost:3000` (default credentials: admin/admin)

### Alerting

**Configured Alerts**:
- High API latency (p95 > 500ms)
- Database connection pool exhaustion
- Redis memory > 80%
- Celery task failure rate > 5%
- Pod restart loops

---

## 🔄 Rollback Procedures

### Docker Rollback

```bash
# View previous image
docker images | grep shiksha-setu/backend

# Rollback to previous version
docker-compose down
BACKEND_IMAGE="shiksha-setu/backend:v0.9.9" docker-compose up -d
```

### Kubernetes Rollback

```bash
# Check rollout history
kubectl rollout history deployment/backend -n shiksha-setu

# Rollback to previous revision
kubectl rollout undo deployment/backend -n shiksha-setu

# Rollback to specific revision
kubectl rollout undo deployment/backend --to-revision=2 -n shiksha-setu
```

### Database Rollback

```bash
# List available backups
ls -la backups/postgres/

# Restore from backup
pg_restore --clean --if-exists -U postgres -d shiksha_setu backups/postgres/backup-2025-01-15.sql
```

---

## 🔧 Troubleshooting

### Common Issues

**Backend not starting**:
```bash
# Check logs
docker logs shiksha-backend

# Verify environment variables
docker exec shiksha-backend env | grep DATABASE_URL

# Check database connectivity
docker exec shiksha-backend python -c "from sqlalchemy import create_engine; print(create_engine(os.getenv('DATABASE_URL')).execute('SELECT 1'))"
```

**Celery workers not processing tasks**:
```bash
# Check worker logs
docker logs shiksha-worker

# Inspect Redis queue
redis-cli -u $REDIS_URL LRANGE celery 0 -1

# Restart workers
docker restart shiksha-worker
```

**Memory issues**:
```bash
# Monitor container memory
docker stats shiksha-backend shiksha-worker

# Increase memory limit
docker update -m 4g shiksha-backend
```

**Database connection timeout**:
```bash
# Check PostgreSQL logs
docker logs shiksha-postgres

# Verify connection string
psql "postgresql://user:pass@host:5432/dbname" -c "SELECT 1"

# Check connection pool
docker exec shiksha-postgres psql -U postgres -c "SELECT count(*) FROM pg_stat_activity;"
```

**Kubernetes pod issues**:
```bash
# Describe problematic pod
kubectl describe pod <pod-name> -n shiksha-setu

# Check events
kubectl get events -n shiksha-setu --sort-by='.lastTimestamp'

# View logs
kubectl logs <pod-name> -n shiksha-setu -c <container-name>
```

---

## 📚 Related Documentation

- **[Monitoring](monitoring.md)** - Detailed monitoring setup
- **[Security](security.md)** - Security hardening
- **[Optimization](optimization.md)** - Performance tuning
- **[Database](database.md)** - Database configuration

---

## 👨‍💻 Author

**K Dhiraj** • [k.dhiraj.srihari@gmail.com](mailto:k.dhiraj.srihari@gmail.com) • [@KDhiraj152](https://github.com/KDhiraj152)

*Last updated: November 2025*
