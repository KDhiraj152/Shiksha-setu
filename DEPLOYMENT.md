# 🚀 Deployment Guide - Shiksha Setu

**Complete production deployment guide for Shiksha Setu educational platform**

---

## 📋 Table of Contents

1. [Quick Start](#quick-start)
2. [Deployment Status](#deployment-status)
3. [Prerequisites](#prerequisites)
4. [Pre-Deployment Checklist](#pre-deployment-checklist)
5. [Deployment Options](#deployment-options)
   - [Docker Compose](#docker-compose-deployment)
   - [Kubernetes](#kubernetes-deployment)
   - [Manual Deployment](#manual-deployment)
6. [Post-Deployment Verification](#post-deployment-verification)
7. [Monitoring Setup](#monitoring-setup)
8. [Rollback Procedures](#rollback-procedures)
9. [Troubleshooting](#troubleshooting)

---

## 🎯 Quick Start

```bash
# Clone repository
git clone https://github.com/KDhiraj152/Siksha-Setu.git
cd Siksha-Setu

# Run automated setup
./bin/setup

# Start all services
./bin/start

# Validate deployment
./bin/validate-production
```

**Status**: ✅ Production-Ready with 93 passing tests, 23% coverage

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

### Infrastructure Readiness

- ✅ **Docker Configuration**: 15 services with HA setup
- ✅ **Kubernetes Manifests**: Dev, Staging, Production overlays
- ✅ **Nginx Configuration**: Load balancing, SSL/TLS, security headers
- ✅ **Monitoring Stack**: Prometheus, Grafana, Alertmanager, 5 exporters
- ✅ **CI/CD Pipelines**: Test, Build, Deploy-Staging, Deploy-Production
- ✅ **Backup Strategy**: Automated PostgreSQL backups
- ✅ **Rollback Capability**: Automated rollback in deploy workflow

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

### Required Secrets

Before deployment, obtain:

- [ ] `DATABASE_URL` - PostgreSQL connection string
- [ ] `JWT_SECRET_KEY` - 64-character hex string
- [ ] `REDIS_URL` - Redis connection string
- [ ] `HUGGINGFACE_API_KEY` - HuggingFace API token (optional)
- [ ] `SENTRY_DSN` - Error monitoring (optional)
- [ ] `BHASHINI_USER_ID` - Bhashini translation API (optional)
- [ ] `BHASHINI_API_KEY` - Bhashini API key (optional)
- [ ] `BHASHINI_PIPELINE_ID` - Bhashini pipeline ID (optional)

Generate secrets:
```bash
# JWT Secret (64 hex characters)
openssl rand -hex 32

# Or use Python
python -c "import secrets; print(secrets.token_hex(32))"
```

---

## ✅ Pre-Deployment Checklist

### Application Layer

#### Backend API
- [ ] All tests passing (`pytest tests/`)
- [ ] Test coverage ≥ 20%
- [ ] No critical security vulnerabilities
- [ ] Environment variables configured
- [ ] Database migrations up to date
- [ ] API documentation generated

#### Frontend
- [ ] All component tests passing
- [ ] Production build successful
- [ ] Bundle size optimized (<500KB)
- [ ] TypeScript compilation clean
- [ ] API integration tested

### Infrastructure Layer

#### Database
- [ ] PostgreSQL 15+ installed
- [ ] pgvector extension enabled
- [ ] Connection pool configured
- [ ] Backup strategy implemented
- [ ] Indexes created
- [ ] Performance tuned

#### Caching & Messaging
- [ ] Redis server running
- [ ] Celery workers configured
- [ ] Task queue operational
- [ ] Cache invalidation strategy defined

#### Monitoring
- [ ] Prometheus installed
- [ ] Grafana dashboards imported
- [ ] Alert rules configured
- [ ] Exporters deployed
- [ ] Alertmanager configured with channels (Slack/PagerDuty)

### Security

- [ ] SSL certificates obtained and valid
- [ ] Firewall rules configured
- [ ] Security headers enabled (CSP, HSTS, X-Frame-Options)
- [ ] Rate limiting configured
- [ ] CORS origins whitelisted
- [ ] Secrets stored securely (not in code)
- [ ] JWT tokens properly secured
- [ ] Input sanitization active

### DevOps

- [ ] Domain DNS configured
- [ ] Load balancer setup (if multi-instance)
- [ ] CDN configured (optional)
- [ ] Backup automation tested
- [ ] Rollback procedure documented
- [ ] Monitoring alerts tested
- [ ] Runbook created for on-call

---

## 🐳 Docker Compose Deployment

### Development Setup

```bash
cd infrastructure/docker
cp ../../.env.example ../../.env
# Edit .env with your configuration

docker-compose up -d
```

**Services Started**:
- `backend` - FastAPI API (port 8000)
- `frontend` - React app (port 5173)
- `postgres` - PostgreSQL 17 + pgvector
- `redis` - Redis 7.4
- `celery-worker` - Background tasks

**View Logs**:
```bash
docker-compose logs -f backend
docker-compose logs -f celery-worker
```

**Stop Services**:
```bash
docker-compose down
```

### Production Deployment

**1. Configure Environment**:
```bash
cp .env.example .env.production
# Edit .env.production with production values
```

**2. Build Production Images**:
```bash
# Backend
docker build -t shiksha-setu/backend:v1.0.0 \
  -f infrastructure/docker/Dockerfile.backend .

# Frontend
docker build -t shiksha-setu/frontend:v1.0.0 \
  -f infrastructure/docker/Dockerfile.frontend .
```

**3. Deploy Production Stack**:
```bash
docker-compose -f docker-compose.production.yml up -d
```

**Services (15 total)**:
- Backend API (3 replicas)
- Frontend
- PostgreSQL Primary + Replica
- Redis Primary + Replica
- Celery Worker (2 replicas)
- Prometheus + Grafana + Alertmanager
- 5 Exporters (PostgreSQL, Redis, Node, Nginx, vLLM)

**4. Run Migrations**:
```bash
docker-compose exec backend alembic upgrade head
```

**5. Create Admin User**:
```bash
docker-compose exec backend python scripts/demo/create_demo_user.py
```

---

## ☸️ Kubernetes Deployment

### Quick Deploy

```bash
# Create namespace
kubectl create namespace shiksha-setu

# Apply secrets
kubectl create secret generic shiksha-secrets \
  --from-env-file=.env.production \
  -n shiksha-setu

# Deploy application
kubectl apply -k infrastructure/kubernetes/overlays/prod

# Watch rollout
kubectl rollout status deployment/backend -n shiksha-setu
```

### Detailed Steps

**1. Configure Secrets**:
```bash
# Database URL
kubectl create secret generic db-credentials \
  --from-literal=DATABASE_URL="postgresql://user:pass@host:5432/db" \
  -n shiksha-setu

# JWT Secret
kubectl create secret generic jwt-secret \
  --from-literal=JWT_SECRET_KEY="$(openssl rand -hex 32)" \
  -n shiksha-setu

# Optional: HuggingFace token
kubectl create secret generic huggingface-token \
  --from-literal=HUGGINGFACE_API_KEY="hf_..." \
  -n shiksha-setu
```

**2. Update Image Tags**:
```yaml
# infrastructure/kubernetes/overlays/prod/kustomization.yaml
images:
  - name: shiksha-setu/backend
    newName: your-registry/shiksha-setu/backend
    newTag: v1.0.0
  - name: shiksha-setu/frontend
    newName: your-registry/shiksha-setu/frontend
    newTag: v1.0.0
```

**3. Apply Configuration**:
```bash
kubectl apply -k infrastructure/kubernetes/overlays/prod
```

**4. Verify Deployment**:
```bash
# Check pods
kubectl get pods -n shiksha-setu

# Check services
kubectl get svc -n shiksha-setu

# Check logs
kubectl logs -f deployment/backend -n shiksha-setu
```

**5. Run Database Migrations**:
```bash
BACKEND_POD=$(kubectl get pods -n shiksha-setu -l app=backend -o jsonpath='{.items[0].metadata.name}')
kubectl exec -it $BACKEND_POD -n shiksha-setu -- alembic upgrade head
```

### Horizontal Pod Autoscaling

```bash
# Apply HPA
kubectl apply -f - <<EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: backend-hpa
  namespace: shiksha-setu
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
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
EOF
```

---

## 🔧 Manual Deployment

For bare metal or VM deployments without Docker/Kubernetes.

### 1. Install Dependencies

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.11
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.11 python3.11-venv python3.11-dev

# Install PostgreSQL 15
sudo sh -c 'echo "deb http://apt.postgresql.org/pub/repos/apt $(lsb_release -cs)-pgdg main" > /etc/apt/sources.list.d/pgdg.list'
wget -qO- https://www.postgresql.org/media/keys/ACCC4CF8.asc | sudo tee /etc/apt/trusted.gpg.d/pgdg.asc
sudo apt update
sudo apt install postgresql-15 postgresql-contrib-15

# Install pgvector extension
sudo apt install postgresql-15-pgvector

# Install Redis
sudo apt install redis-server

# Install Node.js 20
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs

# Install Nginx
sudo apt install nginx
```

### 2. Setup Application

```bash
# Clone repository
git clone https://github.com/KDhiraj152/Siksha-Setu.git
cd Siksha-Setu

# Setup Python environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Setup frontend
cd frontend
npm install
npm run build
cd ..

# Configure environment
cp .env.example .env
# Edit .env with production values
```

### 3. Setup Database

```bash
# Create database
sudo -u postgres psql -c "CREATE DATABASE shiksha_setu;"
sudo -u postgres psql -c "CREATE USER shiksha_user WITH PASSWORD 'your_password';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE shiksha_setu TO shiksha_user;"

# Enable pgvector
sudo -u postgres psql -d shiksha_setu -c "CREATE EXTENSION vector;"

# Run migrations
alembic upgrade head
```

### 4. Configure Systemd Services

**Backend Service** (`/etc/systemd/system/shiksha-backend.service`):
```ini
[Unit]
Description=Shiksha Setu Backend API
After=network.target postgresql.service redis.service

[Service]
Type=notify
User=www-data
Group=www-data
WorkingDirectory=/opt/shiksha-setu
Environment="PATH=/opt/shiksha-setu/.venv/bin"
ExecStart=/opt/shiksha-setu/.venv/bin/uvicorn backend.api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4
Restart=always

[Install]
WantedBy=multi-user.target
```

**Celery Worker** (`/etc/systemd/system/shiksha-celery.service`):
```ini
[Unit]
Description=Shiksha Setu Celery Worker
After=network.target redis.service

[Service]
Type=forking
User=www-data
Group=www-data
WorkingDirectory=/opt/shiksha-setu
Environment="PATH=/opt/shiksha-setu/.venv/bin"
ExecStart=/opt/shiksha-setu/.venv/bin/celery -A backend.tasks.celery_app worker \
    --loglevel=info \
    --concurrency=2
Restart=always

[Install]
WantedBy=multi-user.target
```

**Enable and Start**:
```bash
sudo systemctl daemon-reload
sudo systemctl enable shiksha-backend shiksha-celery
sudo systemctl start shiksha-backend shiksha-celery
```

### 5. Configure Nginx

**Create site config** (`/etc/nginx/sites-available/shiksha-setu`):
```nginx
upstream backend {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    # Frontend
    location / {
        root /opt/shiksha-setu/frontend/dist;
        try_files $uri $uri/ /index.html;
    }

    # Backend API
    location /api/ {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket support
    location /ws/ {
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

**Enable site**:
```bash
sudo ln -s /etc/nginx/sites-available/shiksha-setu /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 6. Setup SSL with Let's Encrypt

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com
```

---

## ✓ Post-Deployment Verification

### Automated Validation

```bash
./bin/validate-production
```

### Manual Checks

**1. Health Endpoints**:
```bash
# Backend health
curl https://yourdomain.com/api/v1/health

# Expected: {"status": "healthy", "database": "connected", "redis": "connected"}

# Metrics endpoint
curl https://yourdomain.com/api/v1/metrics
```

**2. Database Connection**:
```bash
# Check migrations
docker-compose exec backend alembic current

# Test query
docker-compose exec postgres psql -U shiksha_user -d shiksha_setu -c "SELECT version();"
```

**3. AI/ML Models**:
```bash
# Test simplification
curl -X POST https://yourdomain.com/api/v1/simplify \
  -H "Content-Type: application/json" \
  -d '{"text": "Test content", "grade_level": 5}'

# Test translation
curl -X POST https://yourdomain.com/api/v1/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello", "target_lang": "hi"}'
```

**4. Frontend**:
```bash
# Check bundle size
ls -lh frontend/dist/assets/

# Test loading time
curl -w "@curl-format.txt" -o /dev/null -s https://yourdomain.com/
```

**5. Monitoring**:
```bash
# Prometheus targets
curl http://localhost:9090/api/v1/targets

# Grafana dashboards
open http://localhost:3000
# Login: admin / (from .env)
```

### Performance Benchmarks

Expected response times:
- Health check: <50ms
- Simple API calls: <200ms
- Text simplification: <2s
- Translation: <3s
- Document processing: <10s

Load test:
```bash
# Install wrk
sudo apt install wrk

# Test API performance
wrk -t12 -c400 -d30s https://yourdomain.com/api/v1/health
```

---

## 📊 Monitoring Setup

### Prometheus

**Access**: http://localhost:9090

**Key Metrics**:
- `http_request_duration_seconds` - API response times
- `celery_task_duration_seconds` - Background task performance
- `postgres_connections` - Database connections
- `redis_connected_clients` - Cache connections

### Grafana

**Access**: http://localhost:3000  
**Login**: admin / (password from .env)

**Pre-configured Dashboards**:
1. **System Overview** - CPU, Memory, Disk, Network
2. **Application Metrics** - API requests, response times, errors
3. **Database Performance** - Query times, connections, cache hit rate
4. **ML Pipeline** - Model inference times, queue length
5. **Business Metrics** - User activity, content processed

### Alertmanager

**Access**: http://localhost:9093

**Alert Channels Configured**:
- **Slack** - All alerts
- **PagerDuty** - Critical alerts only

**Critical Alerts** (immediate action required):
- API error rate > 5%
- Database connection failures
- Disk usage > 90%
- Memory usage > 90%
- SSL certificate expiring < 7 days

**Warning Alerts** (investigate within 24h):
- API response time > 2s
- Task queue length > 100
- Cache hit rate < 50%
- Celery worker down

---

## 🔄 Rollback Procedures

### Docker Compose Rollback

```bash
# Stop current version
docker-compose -f docker-compose.production.yml down

# Switch to previous version
git checkout tags/v0.9.0

# Restore database backup (if schema changed)
docker-compose exec postgres psql -U shiksha_user -d shiksha_setu < backup.sql

# Start previous version
docker-compose -f docker-compose.production.yml up -d
```

### Kubernetes Rollback

```bash
# Automatic rollback (triggered by deployment)
kubectl rollout undo deployment/backend -n shiksha-setu

# Rollback to specific revision
kubectl rollout history deployment/backend -n shiksha-setu
kubectl rollout undo deployment/backend --to-revision=2 -n shiksha-setu

# Verify rollback
kubectl rollout status deployment/backend -n shiksha-setu
```

### Database Rollback

```bash
# Check current migration
alembic current

# Rollback to previous migration
alembic downgrade -1

# Rollback to specific revision
alembic downgrade <revision_id>
```

---

## 🔥 Troubleshooting

### Common Issues

#### 1. API Returns 502/503 Errors

**Symptoms**: Nginx returns 502 Bad Gateway or 503 Service Unavailable

**Diagnosis**:
```bash
# Check backend status
docker-compose ps backend
kubectl get pods -n shiksha-setu

# Check backend logs
docker-compose logs backend
kubectl logs deployment/backend -n shiksha-setu
```

**Solutions**:
- Restart backend: `docker-compose restart backend`
- Check resource limits: `docker stats`
- Verify database connection in .env
- Check Celery workers are running

#### 2. Database Connection Failures

**Symptoms**: "could not connect to server" errors

**Diagnosis**:
```bash
# Test connection
psql -h localhost -U shiksha_user -d shiksha_setu

# Check PostgreSQL status
sudo systemctl status postgresql
docker-compose ps postgres
```

**Solutions**:
- Restart PostgreSQL: `sudo systemctl restart postgresql`
- Check pg_hba.conf allows connections
- Verify DATABASE_URL in .env
- Check firewall allows port 5432

#### 3. Models Not Loading

**Symptoms**: 500 errors on /simplify or /translate endpoints

**Diagnosis**:
```bash
# Check model files
ls -lh data/models/

# Check logs for model loading errors
docker-compose logs backend | grep -i "model"
```

**Solutions**:
- Download models: `python scripts/download_models.py`
- Check HUGGINGFACE_API_KEY is set
- Increase memory allocation
- Use CPU fallback: `FORCE_CPU=true`

#### 4. High Memory Usage

**Symptoms**: OOM kills, slow performance

**Diagnosis**:
```bash
# Check container memory
docker stats

# Check system memory
free -h
```

**Solutions**:
- Reduce model size: Use quantized models
- Limit workers: Reduce CELERY_WORKERS count
- Increase swap: `sudo fallocate -l 8G /swapfile`
- Scale horizontally instead of vertically

#### 5. SSL Certificate Issues

**Symptoms**: Certificate expired or invalid warnings

**Diagnosis**:
```bash
# Check certificate expiry
openssl x509 -in /etc/letsencrypt/live/yourdomain.com/fullchain.pem -noout -dates

# Test SSL
curl -vI https://yourdomain.com
```

**Solutions**:
- Renew certificate: `sudo certbot renew`
- Check auto-renewal: `sudo systemctl status certbot.timer`
- Verify DNS points to correct IP

### Emergency Contacts

- **DevOps Lead**: devops@shikshasetu.in
- **Backend Team**: backend@shikshasetu.in
- **On-Call Rotation**: Check PagerDuty schedule
- **Slack Channel**: #production-alerts

### Useful Commands

```bash
# View all logs
docker-compose logs -f --tail=100

# Check resource usage
docker stats

# Restart all services
docker-compose restart

# Full system cleanup (CAUTION: destroys data)
docker-compose down -v

# Database backup
docker-compose exec postgres pg_dump -U shiksha_user shiksha_setu > backup_$(date +%Y%m%d).sql

# Database restore
docker-compose exec -T postgres psql -U shiksha_user shiksha_setu < backup.sql
```

---

## 📚 Additional Resources

- [Architecture Documentation](docs/reference/architecture.md)
- [API Documentation](docs/reference/api.md)
- [Database Schema](docs/DATABASE.md)
- [Monitoring Guide](docs/MONITORING.md)
- [Development Guide](DEVELOPMENT.md)
- [Implementation Summary](IMPLEMENTATION_SUMMARY.md)

---

## 📝 Deployment Changelog

### v1.0.0 (2024-11-28)
- ✅ Initial production release
- ✅ 93 passing tests, 23% coverage
- ✅ Complete CI/CD pipeline
- ✅ Monitoring stack deployed
- ✅ 15-service Docker Compose setup
- ✅ Kubernetes manifests for dev/staging/prod

---

**Last Updated**: 2024-11-28  
**Maintained By**: Shiksha Setu DevOps Team  
**Status**: Production Ready ✅
