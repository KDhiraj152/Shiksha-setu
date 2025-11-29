# ShikshaSetu Monitoring Guide

## Overview

This guide covers the monitoring infrastructure for ShikshaSetu, including Prometheus metrics collection, Grafana dashboards, and Alertmanager alerting.

## Architecture

```
┌─────────────┐
│  Application│──► Metrics Endpoint (/metrics)
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌──────────────┐
│ Prometheus  │────►│ Alertmanager │──► Slack/PagerDuty
└──────┬──────┘     └──────────────┘
       │
       ▼
┌─────────────┐
│   Grafana   │──► Dashboards
└─────────────┘
```

## Components

### Prometheus
- **Port**: 9090
- **Purpose**: Metrics collection and time-series database
- **Retention**: 30 days
- **Config**: `infrastructure/monitoring/prometheus.yml`

### Grafana
- **Port**: 3001
- **Purpose**: Visualization and dashboards
- **Default User**: admin
- **Config**: `infrastructure/monitoring/grafana-*.yml`

### Alertmanager
- **Port**: 9093
- **Purpose**: Alert routing and notification
- **Config**: `infrastructure/monitoring/alertmanager.yml`

### Exporters
- **PostgreSQL Exporter** (9187): Database metrics
- **Redis Exporter** (9121): Cache metrics
- **Node Exporter** (9100): System metrics
- **Nginx Exporter** (9113): Load balancer metrics

## Quick Start

### 1. Environment Setup

Create a `.env` file with required variables:

```bash
# Alerting
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
PAGERDUTY_SERVICE_KEY=your-pagerduty-service-key

# Grafana
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=secure-password-here

# Database (for postgres_exporter)
POSTGRES_USER=shikshasetu_user
POSTGRES_PASSWORD=your-db-password
POSTGRES_DB=shiksha_setu

# Redis (for redis_exporter, if password protected)
REDIS_PASSWORD=your-redis-password
```

### 2. Start Monitoring Stack

```bash
cd infrastructure/monitoring
./setup-monitoring.sh
```

Or manually:

```bash
docker-compose -f infrastructure/monitoring/docker-compose.monitoring.yml up -d
```

### 3. Verify Installation

Check all services are running:

```bash
docker-compose -f infrastructure/monitoring/docker-compose.monitoring.yml ps
```

Check health endpoints:

```bash
# Prometheus
curl http://localhost:9090/-/healthy

# Alertmanager
curl http://localhost:9093/-/healthy

# Grafana
curl http://localhost:3001/api/health
```

## Accessing Services

### Prometheus
- **URL**: http://localhost:9090
- **Features**:
  - Query metrics: `http://localhost:9090/graph`
  - View targets: `http://localhost:9090/targets`
  - View alerts: `http://localhost:9090/alerts`
  - Configuration: `http://localhost:9090/config`

### Grafana
- **URL**: http://localhost:3001
- **Login**: admin / (password from .env)
- **Dashboards**: Navigate to Dashboards → ShikshaSetu folder

### Alertmanager
- **URL**: http://localhost:9093
- **Features**:
  - View alerts: `http://localhost:9093/#/alerts`
  - Silence alerts: `http://localhost:9093/#/silences`
  - Configuration: `http://localhost:9093/#/status`

## Metrics Reference

### Application Metrics

Exposed at `http://localhost:8000/metrics`

#### HTTP Metrics
```promql
# Request rate
rate(http_requests_total[5m])

# Error rate
rate(http_requests_failed_total[5m]) / rate(http_requests_total[5m]) * 100

# Response time (p95)
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Requests by endpoint
http_requests_total{endpoint="/api/v1/simplify"}
```

#### Database Metrics
```promql
# Active connections
db_connections_active

# Query duration
db_query_duration_seconds

# Pool usage
db_pool_size_total - db_pool_available
```

#### Cache Metrics
```promql
# Hit rate
cache_hits / (cache_hits + cache_misses) * 100

# Keys
redis_keys_total

# Memory usage
redis_memory_used_bytes
```

#### ML Pipeline Metrics
```promql
# Inference time
ml_inference_duration_seconds

# Translation requests
translation_requests_total

# TTS generation
tts_generation_duration_seconds
```

### System Metrics

#### CPU
```promql
# Usage percentage
100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)
```

#### Memory
```promql
# Available memory
node_memory_MemAvailable_bytes

# Usage percentage
(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100
```

#### Disk
```promql
# Available space
node_filesystem_avail_bytes{mountpoint="/"}

# Usage percentage
(node_filesystem_size_bytes - node_filesystem_avail_bytes) / node_filesystem_size_bytes * 100
```

## Dashboards

### ShikshaSetu Main Dashboard

Located: `infrastructure/monitoring/grafana-dashboard.json`

**Panels**:
1. **Request Rate**: Total requests/second across all endpoints
2. **Response Time (p95)**: 95th percentile latency
3. **Error Rate**: Percentage of failed requests
4. **Cache Hit Rate**: Redis cache effectiveness
5. **Database Query Duration**: Average query time
6. **Active DB Connections**: Current pool usage

### Creating Custom Dashboards

1. Login to Grafana: http://localhost:3001
2. Click "+" → Dashboard
3. Add Panel → Select visualization
4. Choose Prometheus datasource
5. Enter PromQL query
6. Save dashboard

**Example Query**: Requests by status code
```promql
sum by (status_code) (rate(http_requests_total[5m]))
```

## Alerting

### Alert Rules

Located: `infrastructure/monitoring/prometheus-alerts.yml`

**Critical Alerts** (Page on-call):
- HighErrorRate (>5% for 5m)
- DatabasePoolExhausted (>90%)
- RedisDown (1m)
- APIInstanceDown (1m)
- CeleryWorkerDown (2m)

**Warning Alerts** (Slack notification):
- HighResponseTime (p95 >3s)
- LowCacheHitRate (<70%)
- HighMemoryUsage (>4GB)
- SlowDatabaseQueries (avg >1s)

### Alert Routing

Configured in: `infrastructure/monitoring/alertmanager.yml`

**Routes**:
- Critical → PagerDuty + Slack #alerts-critical
- Warning → Slack #alerts-warnings
- Database → Slack #dba-alerts
- ML/GPU → Slack #ml-alerts

### Silencing Alerts

**Via Alertmanager UI**:
1. Navigate to http://localhost:9093
2. Click "Silences" → "New Silence"
3. Enter matcher (e.g., `alertname=HighErrorRate`)
4. Set duration
5. Add comment
6. Create

**Via CLI**:
```bash
amtool silence add alertname=HighErrorRate --duration=2h --comment="Planned maintenance"
```

### Testing Alerts

**Trigger test alert**:
```bash
curl -X POST http://localhost:9093/api/v1/alerts -d '[{
  "labels": {
    "alertname": "TestAlert",
    "severity": "warning"
  },
  "annotations": {
    "summary": "Test alert"
  }
}]'
```

## Maintenance

### Backup Monitoring Data

```bash
# Backup Prometheus data
docker exec shikshasetu_prometheus tar czf - /prometheus > prometheus-backup-$(date +%Y%m%d).tar.gz

# Backup Grafana data
docker exec shikshasetu_grafana tar czf - /var/lib/grafana > grafana-backup-$(date +%Y%m%d).tar.gz
```

### Updating Alert Rules

1. Edit `infrastructure/monitoring/prometheus-alerts.yml`
2. Reload Prometheus configuration:
   ```bash
   curl -X POST http://localhost:9090/-/reload
   ```
3. Verify rules: http://localhost:9090/rules

### Updating Dashboards

**Option 1: Via UI**
1. Edit dashboard in Grafana
2. Save changes
3. Export JSON via Share → Export → Save to file
4. Replace `infrastructure/monitoring/grafana-dashboard.json`

**Option 2: Via File**
1. Edit `grafana-dashboard.json`
2. Restart Grafana:
   ```bash
   docker-compose -f infrastructure/monitoring/docker-compose.monitoring.yml restart grafana
   ```

### Scaling Prometheus

For high-traffic production:

1. **Increase retention**:
   ```yaml
   # prometheus.yml
   command:
     - '--storage.tsdb.retention.time=90d'
   ```

2. **Add remote storage** (e.g., Thanos, Cortex):
   ```yaml
   remote_write:
     - url: "http://thanos:19291/api/v1/receive"
   ```

3. **Federation** (multiple Prometheus instances):
   ```yaml
   scrape_configs:
     - job_name: 'federate'
       honor_labels: true
       metrics_path: '/federate'
       params:
         'match[]':
           - '{job=~".+"}'
       static_configs:
         - targets:
           - 'prometheus-shard-1:9090'
           - 'prometheus-shard-2:9090'
   ```

## Troubleshooting

### Prometheus not scraping targets

**Check targets**:
```bash
curl http://localhost:9090/api/v1/targets
```

**Common issues**:
- Network connectivity: Verify services are on same Docker network
- Service not exposing metrics: Check application logs
- Wrong port/path: Verify scrape_config in prometheus.yml

### Alerts not firing

**Check alert rules**:
```bash
curl http://localhost:9090/api/v1/rules
```

**Verify alert is pending**:
1. Navigate to http://localhost:9090/alerts
2. Look for alert in "Pending" state
3. Check "for" duration has elapsed

**Check Alertmanager**:
```bash
curl http://localhost:9093/api/v2/alerts
```

### Grafana dashboard showing "No data"

**Verify Prometheus datasource**:
1. Configuration → Data sources → Prometheus
2. Click "Test" button
3. Should show "Data source is working"

**Check query**:
1. Open panel edit mode
2. Look for error messages
3. Test query in Prometheus UI first

### High memory usage

**Prometheus memory**:
- Default: 2GB limit in docker-compose
- Increase if needed:
  ```yaml
  services:
    prometheus:
      deploy:
        resources:
          limits:
            memory: 4G
  ```

**Reduce retention**:
```yaml
command:
  - '--storage.tsdb.retention.time=15d'
```

## Production Checklist

- [ ] Environment variables configured in .env
- [ ] Slack webhook URL configured
- [ ] PagerDuty integration key set
- [ ] Grafana admin password changed
- [ ] All exporters running and healthy
- [ ] Prometheus scraping all targets
- [ ] Grafana dashboards accessible
- [ ] Test alert fires successfully
- [ ] Alert routing to correct channels verified
- [ ] Backup strategy implemented
- [ ] Monitoring documented for team
- [ ] On-call rotation established
- [ ] Runbook links added to alerts

## Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Alertmanager Documentation](https://prometheus.io/docs/alerting/latest/alertmanager/)
- [PromQL Cheat Sheet](https://promlabs.com/promql-cheat-sheet/)
- [ShikshaSetu Deployment Runbook](../../DEPLOYMENT_RUNBOOK.md)

---

## 👨‍💻 Author

**K Dhiraj** • [k.dhiraj.srihari@gmail.com](mailto:k.dhiraj.srihari@gmail.com) • [@KDhiraj152](https://github.com/KDhiraj152) • [LinkedIn](https://www.linkedin.com/in/k-dhiraj-83b025279/)

*Last updated: November 2025*
