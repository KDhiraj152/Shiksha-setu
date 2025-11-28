# Infrastructure

Container orchestration, deployment configurations, and monitoring setup.

## Structure

- **docker/** - Docker containers and compose files
  - `backend.dockerfile` - Backend API container
  - `worker.dockerfile` - Background worker container
  - `docker-compose.yml` - Local development stack
  - `requirements.txt` - Python dependencies

- **kubernetes/** - Kubernetes deployment manifests
  - `deployment.yaml` - Application deployments
  - `ingress.yaml` - Ingress configuration
  - `network-policy.yaml` - Network policies
  - See `SETUP.md` for deployment guide

- **monitoring/** - Observability stack
  - `prometheus/` - Metrics collection
  - `grafana/` - Dashboards and visualization
  - `alertmanager/` - Alert management

## Quick Start

### Docker Development

```bash
cd infrastructure/docker
docker-compose up -d
```

### Kubernetes Deployment

```bash
cd infrastructure/kubernetes
kubectl apply -k overlays/production
```

### Monitoring

```bash
cd infrastructure/monitoring
docker-compose -f prometheus/docker-compose.yml up -d
```

## Configuration

Environment variables are managed via:
- `.env` (root) - Development settings
- `.env.example` (root) - Template with all available options
- Kubernetes secrets for production

## See Also

- [Kubernetes Setup Guide](kubernetes/SETUP.md)
- [Monitoring Configuration](monitoring/README.md)
- [Deployment Guide](../docs/guides/deployment.md)
