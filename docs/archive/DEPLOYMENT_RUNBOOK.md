# ShikshaSetu - Deployment Runbook
## Production Deployment Guide

**Version**: 1.0  
**Last Updated**: 2025-11-28  
**Owner**: DevOps Team

---

## Table of Contents
1. [Pre-Deployment Checklist](#pre-deployment-checklist)
2. [Staging Deployment](#staging-deployment)
3. [Production Deployment](#production-deployment)
4. [Rollback Procedures](#rollback-procedures)
5. [Post-Deployment Validation](#post-deployment-validation)
6. [Troubleshooting](#troubleshooting)
7. [Emergency Contacts](#emergency-contacts)

---

## Pre-Deployment Checklist

### Code Quality ✅
- [ ] All tests passing (coverage ≥70%)
- [ ] Code reviewed and approved
- [ ] Security scan passed (Bandit, Semgrep)
- [ ] Dependency vulnerabilities checked
- [ ] Docker images built and tagged
- [ ] Database migrations tested

### Infrastructure ✅
- [ ] Database backups current (<24h old)
- [ ] Staging environment validated
- [ ] Load balancer health checks configured
- [ ] SSL certificates valid (>30 days)
- [ ] DNS records configured
- [ ] CDN cache cleared (if needed)

### Configuration ✅
- [ ] Environment variables verified
- [ ] Secrets rotated (if scheduled)
- [ ] Rate limits configured
- [ ] CORS origins updated
- [ ] Feature flags set correctly

### Monitoring ✅
- [ ] Prometheus targets configured
- [ ] Grafana dashboards imported
- [ ] Sentry project configured
- [ ] Alert rules activated
- [ ] Log aggregation working
- [ ] Status page updated

### Communication ✅
- [ ] Stakeholders notified (24h advance)
- [ ] Maintenance window scheduled
- [ ] Change request approved
- [ ] Rollback plan reviewed
- [ ] On-call engineer assigned

---

## Staging Deployment

### Step 1: Prepare Environment

```bash
# Clone repository
git clone https://github.com/KDhiraj152/Siksha-Setu.git
cd Siksha-Setu

# Checkout release branch
git checkout release/v2.0.0

# Verify commit
git log -1 --oneline
```

### Step 2: Configure Environment

```bash
# Copy and configure environment
cp .env.example .env.staging
nano .env.staging

# Essential variables:
ENVIRONMENT=staging
DEBUG=false
DATABASE_URL=postgresql://user:pass@staging-db:5432/shiksha_setu
REDIS_URL=redis://staging-redis:6379/0
SENTRY_DSN=https://xxx@sentry.io/xxx
SENTRY_ENVIRONMENT=staging
```

### Step 3: Build Docker Images

```bash
# Build backend
docker build -t shikshasetu/backend:staging -f infrastructure/docker/Dockerfile.backend .

# Build frontend
docker build -t shikshasetu/frontend:staging -f infrastructure/docker/Dockerfile.frontend .

# Verify images
docker images | grep shikshasetu
```

### Step 4: Run Database Migrations

```bash
# Backup database first
docker-compose -f docker-compose.staging.yml exec postgres \
  pg_dump -U postgres shiksha_setu > backup_$(date +%Y%m%d_%H%M%S).sql

# Run migrations
docker-compose -f docker-compose.staging.yml exec backend \
  alembic upgrade head

# Verify migration
docker-compose -f docker-compose.staging.yml exec backend \
  alembic current
```

### Step 5: Deploy Services

```bash
# Start services
docker-compose -f docker-compose.staging.yml up -d

# Verify services
docker-compose -f docker-compose.staging.yml ps

# Check logs
docker-compose -f docker-compose.staging.yml logs -f --tail=100
```

### Step 6: Health Checks

```bash
# Backend health
curl -f https://staging.shikshasetu.edu/health || exit 1

# Database connectivity
curl -f https://staging.shikshasetu.edu/health/db || exit 1

# Redis connectivity
curl -f https://staging.shikshasetu.edu/health/redis || exit 1

# Celery workers
docker-compose -f docker-compose.staging.yml exec backend \
  celery -A backend.tasks.celery_app inspect active
```

### Step 7: Run Smoke Tests

```bash
# Integration tests
docker-compose -f docker-compose.staging.yml exec backend \
  pytest tests/integration/ -v --tb=short

# API endpoint tests
./bin/test-api-endpoints.sh staging
```

---

## Production Deployment

### Pre-Production Final Checks

```bash
# 1. Verify staging is stable (24h+ uptime)
uptime

# 2. Check monitoring
curl https://staging.shikshasetu.edu/metrics | grep http_requests_total

# 3. Review error rates (should be <0.1%)
curl https://staging.shikshasetu.edu/metrics | grep http_requests_failed

# 4. Confirm database performance
psql $DATABASE_URL -c "SELECT * FROM pg_stat_statements ORDER BY total_time DESC LIMIT 10;"
```

### Production Deployment Steps

#### Phase 1: Preparation (T-30 minutes)

```bash
# 1. Enable maintenance mode
curl -X POST https://api.shikshasetu.edu/admin/maintenance \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"enabled": true, "message": "System upgrade in progress. Expected downtime: 15 minutes"}'

# 2. Backup production database
pg_dump -h prod-db.shikshasetu.edu -U postgres shiksha_setu | \
  gzip > prod_backup_$(date +%Y%m%d_%H%M%S).sql.gz

# Upload to S3
aws s3 cp prod_backup_*.sql.gz s3://shikshasetu-backups/production/

# 3. Notify users via status page
curl -X POST https://status.shikshasetu.edu/api/incidents \
  -H "Authorization: Bearer $STATUS_PAGE_TOKEN" \
  -d '{"status": "investigating", "title": "Scheduled Maintenance", "body": "System upgrade in progress"}'
```

#### Phase 2: Database Migration (T-20 minutes)

```bash
# 1. Create read-only replica snapshot
aws rds create-db-snapshot \
  --db-instance-identifier shikshasetu-prod \
  --db-snapshot-identifier prod-pre-deploy-$(date +%Y%m%d-%H%M%S)

# 2. Run migrations (dry-run first)
alembic upgrade head --sql > migration_$(date +%Y%m%d).sql
cat migration_$(date +%Y%m%d).sql  # Review

# 3. Apply migrations
alembic upgrade head

# 4. Verify
alembic current
```

#### Phase 3: Service Deployment (T-10 minutes)

```bash
# 1. Pull new images
docker pull shikshasetu/backend:v2.0.0
docker pull shikshasetu/frontend:v2.0.0

# 2. Rolling update (zero-downtime)
kubectl set image deployment/backend \
  backend=shikshasetu/backend:v2.0.0 \
  --record

kubectl set image deployment/frontend \
  frontend=shikshasetu/frontend:v2.0.0 \
  --record

# 3. Monitor rollout
kubectl rollout status deployment/backend
kubectl rollout status deployment/frontend

# 4. Verify pods
kubectl get pods -l app=shikshasetu
```

#### Phase 4: Service Activation (T-5 minutes)

```bash
# 1. Update load balancer targets
aws elbv2 register-targets \
  --target-group-arn $TARGET_GROUP_ARN \
  --targets Id=i-new-instance-1 Id=i-new-instance-2

# 2. Warm cache
curl -X POST https://api.shikshasetu.edu/admin/cache/warm \
  -H "Authorization: Bearer $ADMIN_TOKEN"

# 3. Enable traffic gradually (5% -> 25% -> 50% -> 100%)
for PERCENTAGE in 5 25 50 100; do
  aws elbv2 modify-target-group-attributes \
    --target-group-arn $TARGET_GROUP_ARN \
    --attributes "Key=deregistration_delay.timeout_seconds,Value=300"
  
  echo "Traffic at ${PERCENTAGE}%"
  sleep 120  # Wait 2 minutes
  
  # Check error rate
  ERROR_RATE=$(curl -s https://api.shikshasetu.edu/metrics | \
    grep http_requests_failed | awk '{print $2}')
  
  if (( $(echo "$ERROR_RATE > 0.01" | bc -l) )); then
    echo "⚠️ High error rate detected! Pausing rollout."
    exit 1
  fi
done
```

#### Phase 5: Validation (T+0 minutes)

```bash
# 1. Health checks
curl -f https://api.shikshasetu.edu/health || exit 1
curl -f https://api.shikshasetu.edu/health/db || exit 1
curl -f https://api.shikshasetu.edu/health/redis || exit 1

# 2. Smoke tests
pytest tests/integration/test_content_pipeline.py -v

# 3. Verify key metrics
curl https://api.shikshasetu.edu/metrics | grep -E "(http_requests_total|db_query_duration_seconds|cache_hits_total)"

# 4. Check Sentry for errors
open "https://sentry.io/organizations/shikshasetu/issues/?environment=production&query=is%3Aunresolved"

# 5. Disable maintenance mode
curl -X POST https://api.shikshasetu.edu/admin/maintenance \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -d '{"enabled": false}'

# 6. Update status page
curl -X PATCH https://status.shikshasetu.edu/api/incidents/$INCIDENT_ID \
  -H "Authorization: Bearer $STATUS_PAGE_TOKEN" \
  -d '{"status": "resolved", "body": "Deployment completed successfully"}'
```

---

## Rollback Procedures

### Immediate Rollback (< 5 minutes)

#### Scenario 1: Application Errors

```bash
# 1. Revert Kubernetes deployment
kubectl rollout undo deployment/backend
kubectl rollout undo deployment/frontend

# 2. Verify rollback
kubectl rollout status deployment/backend

# 3. Clear bad cache
redis-cli FLUSHALL
```

#### Scenario 2: Database Migration Issues

```bash
# 1. Rollback migration
alembic downgrade -1

# 2. Verify
alembic current

# 3. Restore from backup (if needed)
gunzip < prod_backup_*.sql.gz | \
  psql -h prod-db.shikshasetu.edu -U postgres shiksha_setu
```

#### Scenario 3: Critical Service Failure

```bash
# 1. Switch to previous version via load balancer
aws elbv2 modify-target-group \
  --target-group-arn $TARGET_GROUP_ARN \
  --health-check-path /health-old

# 2. Deregister new targets
aws elbv2 deregister-targets \
  --target-group-arn $TARGET_GROUP_ARN \
  --targets Id=i-new-instance-1

# 3. Re-register old targets
aws elbv2 register-targets \
  --target-group-arn $TARGET_GROUP_ARN \
  --targets Id=i-old-instance-1 Id=i-old-instance-2
```

### Rollback Decision Matrix

| Symptom | Severity | Action | Timeline |
|---------|----------|--------|----------|
| Error rate >5% | 🔴 Critical | Immediate rollback | <2 min |
| Response time >3s | 🟡 High | Investigate, rollback if persists | <5 min |
| Failed health checks | 🔴 Critical | Immediate rollback | <2 min |
| Database connection errors | 🔴 Critical | Rollback + DB restore | <10 min |
| Cache issues | 🟢 Medium | Clear cache, monitor | <3 min |
| Sentry error spike | 🟡 High | Investigate, rollback if critical | <5 min |

---

## Post-Deployment Validation

### Automated Checks (First 30 minutes)

```bash
#!/bin/bash
# post-deploy-validation.sh

echo "🔍 Starting post-deployment validation..."

# 1. Response time check
RESPONSE_TIME=$(curl -o /dev/null -s -w '%{time_total}' https://api.shikshasetu.edu/health)
if (( $(echo "$RESPONSE_TIME > 1.0" | bc -l) )); then
  echo "⚠️ High response time: ${RESPONSE_TIME}s"
fi

# 2. Error rate check
ERROR_COUNT=$(curl -s https://api.shikshasetu.edu/metrics | grep http_requests_failed | awk '{print $2}')
echo "📊 Error count: $ERROR_COUNT"

# 3. Database connectivity
psql $DATABASE_URL -c "SELECT 1" > /dev/null && echo "✅ Database OK" || echo "❌ Database Failed"

# 4. Redis connectivity
redis-cli -h prod-redis.shikshasetu.edu PING | grep PONG && echo "✅ Redis OK" || echo "❌ Redis Failed"

# 5. Celery workers
ACTIVE_WORKERS=$(celery -A backend.tasks.celery_app inspect active | grep -c "celery@")
echo "👷 Active Celery workers: $ACTIVE_WORKERS"

# 6. Cache hit rate
HIT_RATE=$(curl -s https://api.shikshasetu.edu/metrics | grep cache_hit_rate | awk '{print $2}')
echo "💾 Cache hit rate: $HIT_RATE"

# 7. Check critical endpoints
for ENDPOINT in "/api/v1/content/" "/api/v1/auth/me" "/metrics"; do
  STATUS=$(curl -o /dev/null -s -w '%{http_code}' "https://api.shikshasetu.edu$ENDPOINT")
  if [ "$STATUS" = "200" ]; then
    echo "✅ $ENDPOINT"
  else
    echo "❌ $ENDPOINT (Status: $STATUS)"
  fi
done

echo "✅ Post-deployment validation complete"
```

### Manual Validation (First hour)

1. **User Acceptance Testing**
   - Login/logout flow
   - Content upload
   - Translation features
   - Audio generation
   - Search functionality

2. **Performance Monitoring**
   - Check Grafana dashboards
   - Review response time p95/p99
   - Monitor database query performance
   - Verify cache hit rates

3. **Error Monitoring**
   - Review Sentry errors (should be <10/hour)
   - Check application logs
   - Verify no critical errors

4. **Business Metrics**
   - User activity normal
   - No spike in support tickets
   - Content processing working

---

## Troubleshooting

### Common Issues

#### Issue 1: High Memory Usage

```bash
# Check memory
docker stats

# Restart with memory limit
docker-compose -f docker-compose.prod.yml up -d --scale backend=4

# Check for memory leaks
curl https://api.shikshasetu.edu/metrics | grep process_memory_bytes
```

#### Issue 2: Database Connection Pool Exhausted

```bash
# Check active connections
psql $DATABASE_URL -c "SELECT count(*) FROM pg_stat_activity WHERE state = 'active';"

# Increase pool size
# Edit .env: DB_POOL_SIZE=50

# Restart backend
docker-compose -f docker-compose.prod.yml restart backend
```

#### Issue 3: Redis Out of Memory

```bash
# Check Redis memory
redis-cli INFO memory

# Clear old cache
redis-cli --scan --pattern "cache:*" | xargs redis-cli DEL

# Set eviction policy
redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

#### Issue 4: Celery Workers Not Processing

```bash
# Check worker status
celery -A backend.tasks.celery_app inspect active

# Purge queue
celery -A backend.tasks.celery_app purge

# Restart workers
docker-compose -f docker-compose.prod.yml restart celery-worker
```

#### Issue 5: High Error Rate

```bash
# Check error logs
docker-compose -f docker-compose.prod.yml logs backend | grep ERROR | tail -50

# Check Sentry
open "https://sentry.io/organizations/shikshasetu/issues/"

# Roll back if critical
./scripts/rollback.sh --to-version v1.9.0
```

---

## Emergency Contacts

### On-Call Engineers
- **Primary**: +91-XXXX-XXXX-XXX (DevOps Lead)
- **Secondary**: +91-XXXX-XXXX-XXX (Backend Lead)
- **Escalation**: +91-XXXX-XXXX-XXX (CTO)

### Service Contacts
- **AWS Support**: https://console.aws.amazon.com/support/
- **Sentry Support**: support@sentry.io
- **Database Admin**: dba@shikshasetu.edu

### Communication Channels
- **Slack**: #production-incidents
- **Status Page**: https://status.shikshasetu.edu
- **Incident Management**: https://pagerduty.com/shikshasetu

---

## Deployment History

| Version | Date | Environment | Status | Rollback |
|---------|------|-------------|--------|----------|
| v2.0.0 | 2025-11-28 | Production | ✅ Success | - |
| v1.9.5 | 2025-11-20 | Production | ✅ Success | - |
| v1.9.0 | 2025-11-10 | Production | ⚠️ Partial (Rolled back) | v1.8.5 |

---

**Document Version**: 1.0  
**Last Reviewed**: 2025-11-28  
**Next Review**: 2025-12-28
