# 🚀 Deployment Guide

Deploy ShikshaSetu to production.

## Docker

### Build & Run
```bash
docker-compose -f deploy/docker-compose.yml up -d
```

### View Logs
```bash
docker-compose logs -f fastapi
```

### Stop
```bash
docker-compose down
```

## Kubernetes

### Deploy
```bash
# Development
kubectl apply -k k8s/overlays/dev

# Production
kubectl apply -k k8s/overlays/prod
```

### Check Status
```bash
kubectl get pods -n shiksha-setu
kubectl logs -f deployment/fastapi -n shiksha-setu
```

### Update
```bash
kubectl rollout restart deployment/fastapi -n shiksha-setu
```

## Environment Variables

Required for production:

```env
# Database
DATABASE_URL=postgresql://user:pass@host:5432/db

# Security
JWT_SECRET_KEY=your-secure-key-min-32-chars

# Redis
REDIS_URL=redis://localhost:6379/0

# Optional: ML Models
HUGGINGFACE_API_KEY=your-hf-key
```

## SSL/TLS

### Nginx Config
```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Monitoring

Access monitoring stack:
```bash
docker-compose -f deploy/docker-compose.monitor.yml up -d
```

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000
- **AlertManager**: http://localhost:9093

## Backup

### Database
```bash
pg_dump -U user dbname > backup.sql
```

### Restore
```bash
psql -U user dbname < backup.sql
```
