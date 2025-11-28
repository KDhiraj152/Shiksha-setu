# Production Deployment Guide

This guide provides step-by-step instructions for deploying Shiksha Setu to production.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Pre-Deployment Checklist](#pre-deployment-checklist)
3. [Initial Setup](#initial-setup)
4. [Configuration](#configuration)
5. [Deployment Steps](#deployment-steps)
6. [Post-Deployment Verification](#post-deployment-verification)
7. [Rollback Procedure](#rollback-procedure)
8. [Troubleshooting](#troubleshooting)

## Prerequisites

### Infrastructure Requirements

- **Production Server(s)**: 
  - 8+ CPU cores
  - 32GB+ RAM
  - 500GB+ SSD storage
  - Ubuntu 22.04 LTS or later
  - GPU recommended for ML inference (NVIDIA with CUDA support)

- **Database**:
  - PostgreSQL 15+ (can be on separate server)
  - 4GB+ RAM allocated
  - 100GB+ storage for data

- **Domain & DNS**:
  - Domain name configured (e.g., shikshasetu.in)
  - SSL certificate (Let's Encrypt recommended)
  - DNS A records pointing to production server

### Required Software

```bash
# Install Docker and Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation
docker --version
docker-compose --version
```

### Required Secrets & API Keys

Before deployment, ensure you have:

- [ ] OpenAI API key (for GPT-4 models)
- [ ] Bhashini API credentials (user_id, api_key, pipeline_id)
- [ ] Sentry DSN (for error monitoring)
- [ ] JWT secret keys (generate 64-character hex strings)
- [ ] PostgreSQL password
- [ ] Redis password
- [ ] Grafana admin password
- [ ] Slack webhook URL (for alerts)
- [ ] PagerDuty service key (optional, for critical alerts)

## Pre-Deployment Checklist

### Code Verification

- [ ] All tests passing (`pytest tests/`)
- [ ] Test coverage ≥ 20% (`coverage report`)
- [ ] Frontend tests passing (`npm test`)
- [ ] No critical security vulnerabilities (`docker scan`)
- [ ] Latest version tagged in Git

### Configuration Files

- [ ] `.env.production` created from `.env.production.example`
- [ ] All secrets populated in `.env.production`
- [ ] SSL certificates obtained and placed in `infrastructure/nginx/ssl/`
- [ ] Database backup strategy configured
- [ ] Monitoring credentials configured

### Infrastructure

- [ ] Production server accessible via SSH
- [ ] Firewall configured (ports 80, 443, 22 open)
- [ ] Domain DNS records pointing to server
- [ ] SSL certificate valid
- [ ] Docker and Docker Compose installed
- [ ] Sufficient disk space available

## Initial Setup

### 1. Clone Repository on Production Server

```bash
# SSH to production server
ssh user@production-server

# Create application directory
sudo mkdir -p /opt/shikshasetu
sudo chown $USER:$USER /opt/shikshasetu
cd /opt/shikshasetu

# Clone repository
git clone https://github.com/your-org/shiksha_setu.git .
```

### 2. Create Directory Structure

```bash
# Create required directories
mkdir -p data/{uploads,cache,models}
mkdir -p logs
mkdir -p infrastructure/nginx/ssl
mkdir -p backups/postgres
mkdir -p infrastructure/monitoring/{prometheus,grafana,alertmanager}

# Set permissions
chmod 755 data/{uploads,cache,models}
chmod 755 logs
chmod 700 infrastructure/nginx/ssl
```

### 3. Obtain SSL Certificate

Using Let's Encrypt with Certbot:

```bash
# Install Certbot
sudo apt-get update
sudo apt-get install -y certbot

# Obtain certificate (using standalone mode)
sudo certbot certonly --standalone \
  -d shikshasetu.in \
  -d www.shikshasetu.in \
  --email your-email@example.com \
  --agree-tos \
  --non-interactive

# Copy certificates to nginx directory
sudo cp /etc/letsencrypt/live/shikshasetu.in/fullchain.pem infrastructure/nginx/ssl/
sudo cp /etc/letsencrypt/live/shikshasetu.in/privkey.pem infrastructure/nginx/ssl/
sudo chown $USER:$USER infrastructure/nginx/ssl/*.pem

# Set up auto-renewal
sudo certbot renew --dry-run
```

## Configuration

### 1. Create Production Environment File

```bash
# Copy example file
cp .env.production.example .env.production

# Edit with secure values
nano .env.production
```

**Critical Values to Update**:

```bash
# Generate secure JWT secrets (64-character hex strings)
JWT_SECRET_KEY=$(openssl rand -hex 32)
JWT_REFRESH_SECRET_KEY=$(openssl rand -hex 32)

# Generate secure database password
POSTGRES_PASSWORD=$(openssl rand -base64 32)

# Generate secure Redis password
REDIS_PASSWORD=$(openssl rand -base64 32)

# Set domain
DOMAIN=shikshasetu.in

# Add API keys
OPENAI_API_KEY=sk-...
BHASHINI_USER_ID=...
BHASHINI_API_KEY=...
BHASHINI_PIPELINE_ID=...

# Configure monitoring
SENTRY_DSN=https://...@sentry.io/...
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

### 2. Configure Monitoring

```bash
# Set Grafana admin password
sed -i "s/GF_SECURITY_ADMIN_PASSWORD=.*/GF_SECURITY_ADMIN_PASSWORD=$(openssl rand -base64 16)/" .env.production

# Configure alert channels in infrastructure/monitoring/alertmanager.yml
nano infrastructure/monitoring/alertmanager.yml
```

### 3. Verify Configuration

```bash
# Check environment file syntax
source .env.production && echo "Configuration valid"

# Verify Docker Compose configuration
docker-compose -f docker-compose.production.yml config
```

## Deployment Steps

### Method 1: Manual Deployment

#### Step 1: Pull Docker Images

```bash
# Login to GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Pull images
docker-compose -f docker-compose.production.yml pull
```

#### Step 2: Run Database Migrations

```bash
# Start database only
docker-compose -f docker-compose.production.yml up -d postgres redis

# Wait for database to be ready
sleep 10

# Run migrations
docker-compose -f docker-compose.production.yml run --rm api alembic upgrade head
```

#### Step 3: Start All Services

```bash
# Start all services
docker-compose -f docker-compose.production.yml up -d

# Check service status
docker-compose -f docker-compose.production.yml ps
```

#### Step 4: Verify Deployment

```bash
# Run verification script
./bin/verify-deployment

# Check logs
docker-compose -f docker-compose.production.yml logs --tail=50 api
```

### Method 2: Automated GitHub Actions Deployment

#### Trigger Production Deployment

1. **Ensure version is tagged**:
   ```bash
   git tag v1.0.0
   git push origin v1.0.0
   ```

2. **Trigger deployment workflow**:
   - Go to GitHub Actions
   - Select "Deploy to Production" workflow
   - Click "Run workflow"
   - Enter version tag (e.g., `v1.0.0`)
   - Click "Run workflow"

3. **Monitor deployment**:
   - Watch workflow progress in GitHub Actions
   - Monitor Slack notifications
   - Check deployment logs

## Post-Deployment Verification

### 1. Service Health Checks

```bash
# Check all services are running
docker-compose -f docker-compose.production.yml ps

# Check API health
curl https://shikshasetu.in/health

# Check specific endpoints
curl https://shikshasetu.in/api/v1/health
curl https://shikshasetu.in/api/v1/curriculum/standards
```

### 2. Database Verification

```bash
# Connect to database
docker-compose -f docker-compose.production.yml exec postgres psql -U shikshasetu_user -d shikshasetu

# Check tables
\dt

# Check migrations
SELECT version_num FROM alembic_version;

# Exit
\q
```

### 3. Monitoring Dashboard Access

```bash
# Access Grafana
https://monitoring.shikshasetu.in
# Login: admin / <GF_SECURITY_ADMIN_PASSWORD>

# Verify dashboards loaded
# Check metrics are being collected
```

### 4. Run Automated Verification

```bash
# Run comprehensive verification
./bin/verify-deployment

# Expected output: All checks passing ✓
```

### 5. Smoke Tests

```bash
# Test content simplification
curl -X POST https://shikshasetu.in/api/v1/content/simplify \
  -H "Content-Type: application/json" \
  -d '{"text":"Complex educational content","grade_level":"5th"}'

# Test curriculum validation
curl https://shikshasetu.in/api/v1/curriculum/validate?grade=5&subject=Mathematics

# Test frontend loads
curl -I https://shikshasetu.in/
```

## Rollback Procedure

If deployment fails or critical issues are discovered:

### Automatic Rollback

GitHub Actions workflow includes automatic rollback on failure.

### Manual Rollback

```bash
# 1. Identify previous version
git tag --sort=-version:refname | head -5

# 2. Checkout previous version
git checkout v0.9.9

# 3. Restore database backup (if needed)
docker-compose -f docker-compose.production.yml exec postgres psql -U shikshasetu_user -d shikshasetu < backups/pre-deploy-YYYYMMDD_HHMMSS.sql.gz

# 4. Roll back migrations (if needed)
docker-compose -f docker-compose.production.yml run --rm api alembic downgrade -1

# 5. Restart services with previous version
docker-compose -f docker-compose.production.yml down
docker-compose -f docker-compose.production.yml up -d

# 6. Verify rollback
./bin/verify-deployment
```

### Emergency Rollback (Nuclear Option)

```bash
# Stop all services
docker-compose -f docker-compose.production.yml down

# Restore from backup
./scripts/restore-from-backup.sh backups/full-backup-YYYYMMDD.tar.gz

# Start services
docker-compose -f docker-compose.production.yml up -d
```

## Troubleshooting

### Services Not Starting

```bash
# Check logs
docker-compose -f docker-compose.production.yml logs

# Check specific service
docker-compose -f docker-compose.production.yml logs api

# Check resource usage
docker stats

# Check disk space
df -h
```

### Database Connection Issues

```bash
# Test database connectivity
docker-compose -f docker-compose.production.yml exec postgres pg_isready

# Check database logs
docker-compose -f docker-compose.production.yml logs postgres

# Verify credentials
docker-compose -f docker-compose.production.yml exec api env | grep DATABASE
```

### API Returning 500 Errors

```bash
# Check API logs
docker-compose -f docker-compose.production.yml logs api --tail=100

# Check if migrations ran
docker-compose -f docker-compose.production.yml exec postgres psql -U shikshasetu_user -d shikshasetu -c "SELECT * FROM alembic_version;"

# Restart API services
docker-compose -f docker-compose.production.yml restart api
```

### High Memory Usage

```bash
# Check memory usage by container
docker stats --no-stream

# Scale down if needed
docker-compose -f docker-compose.production.yml up -d --scale api=2

# Check for memory leaks in logs
docker-compose -f docker-compose.production.yml logs api | grep -i "memory"
```

### SSL Certificate Issues

```bash
# Verify certificate validity
openssl x509 -in infrastructure/nginx/ssl/fullchain.pem -text -noout

# Test SSL configuration
curl -vI https://shikshasetu.in/

# Renew certificate manually
sudo certbot renew --force-renewal
```

### Performance Issues

```bash
# Check Grafana dashboard
https://monitoring.shikshasetu.in

# Check API response times
curl -w "@curl-format.txt" -o /dev/null -s https://shikshasetu.in/api/v1/health

# Check database performance
docker-compose -f docker-compose.production.yml exec postgres psql -U shikshasetu_user -d shikshasetu -c "SELECT * FROM pg_stat_activity;"

# Scale up services if needed
docker-compose -f docker-compose.production.yml up -d --scale api=4 --scale celery_worker=3
```

## Maintenance

### Daily Tasks

- [ ] Check monitoring dashboard for alerts
- [ ] Review error logs in Sentry
- [ ] Verify backups completed successfully

### Weekly Tasks

- [ ] Review resource usage trends
- [ ] Check for security updates
- [ ] Review and rotate logs
- [ ] Test backup restoration

### Monthly Tasks

- [ ] Update dependencies
- [ ] Review and update SSL certificates
- [ ] Capacity planning review
- [ ] Performance optimization

## Support

For issues or questions:

- **Documentation**: `docs/`
- **Monitoring**: https://monitoring.shikshasetu.in
- **Logs**: `docker-compose logs`
- **GitHub Issues**: https://github.com/your-org/shiksha_setu/issues

---

**Last Updated**: 2024
**Version**: 1.0.0
