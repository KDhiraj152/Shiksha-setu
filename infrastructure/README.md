# Infrastructure - Shiksha Setu

Container orchestration, deployment configurations, and monitoring setup.

## 📁 Structure

- **docker/** - Docker containers and compose files
- **kubernetes/** - Kubernetes deployment manifests (dev, staging, prod)
- **monitoring/** - Observability stack (Prometheus, Grafana, Alertmanager)

## 🚀 Quick Start

### Docker Development

```bash
docker-compose up -d
```

### Docker Production

```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Kubernetes

```bash
kubectl apply -k overlays/production
```

### Monitoring

```bash
docker-compose -f monitoring/prometheus/docker-compose.yml up -d
```

## 📚 Documentation

- **[Deployment Guide](../docs/technical/deployment.md)** - Complete deployment instructions
- **[Kubernetes Setup](kubernetes/setup.md)** - K8s specific deployment
- **[Monitoring Guide](../docs/technical/monitoring.md)** - Prometheus & Grafana setup
- **[Security](../docs/technical/security.md)** - Production security

---

## 👨‍💻 Author

**K Dhiraj** • [k.dhiraj.srihari@gmail.com](mailto:k.dhiraj.srihari@gmail.com) • [@KDhiraj152](https://github.com/KDhiraj152) • [LinkedIn](https://www.linkedin.com/in/k-dhiraj-83b025279/)

*Last updated: November 2025*
