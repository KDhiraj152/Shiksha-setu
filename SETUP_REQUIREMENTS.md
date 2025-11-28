# 🔐 Setup Requirements - Things You Need to Configure

**This document lists everything you need to set up manually that an AI cannot access**

---

## 📋 Table of Contents

1. [Environment Variables & Secrets](#1-environment-variables--secrets)
2. [GitHub Repository Settings](#2-github-repository-settings)
3. [GitHub Actions Secrets](#3-github-actions-secrets)
4. [Third-Party API Keys](#4-third-party-api-keys)
5. [Database Setup](#5-database-setup)
6. [Domain & DNS Configuration](#6-domain--dns-configuration)
7. [Server Access](#7-server-access)
8. [Monitoring & Alerting](#8-monitoring--alerting)
9. [Storage & Backups](#9-storage--backups)
10. [Quick Setup Commands](#10-quick-setup-commands)

---

## 1. Environment Variables & Secrets

### 🔴 CRITICAL - Required for Application to Run

#### A. Create `.env` file in project root

```bash
# Copy example file
cp .env.example .env

# Edit with your values
nano .env  # or use any text editor
```

#### B. Required Environment Variables

| Variable | How to Get | Priority | Example |
|----------|-----------|----------|---------|
| `JWT_SECRET_KEY` | Generate (see command below) | 🔴 CRITICAL | `abc123...` (64+ chars) |
| `DATABASE_URL` | PostgreSQL connection string | 🔴 CRITICAL | `postgresql://user:pass@localhost:5432/db` |
| `REDIS_URL` | Redis connection string | 🔴 CRITICAL | `redis://localhost:6379/0` |
| `ALLOWED_ORIGINS` | Your frontend URLs | 🔴 CRITICAL | `http://localhost:5173` |

**Generate JWT Secret**:
```bash
# Option 1: OpenSSL
openssl rand -hex 32

# Option 2: Python
python3 -c "import secrets; print(secrets.token_urlsafe(64))"

# Add to .env file:
# JWT_SECRET_KEY=your_generated_secret_here
```

#### C. Optional but Recommended Variables

| Variable | How to Get | Priority | Purpose |
|----------|-----------|----------|---------|
| `HUGGINGFACE_API_KEY` | HuggingFace account | 🟡 HIGH | Cloud ML inference |
| `SENTRY_DSN` | Sentry.io account | 🟡 HIGH | Error tracking |
| `BHASHINI_API_KEY` | Bhashini Gov portal | 🟢 MEDIUM | Indian language translation |
| `AWS_ACCESS_KEY_ID` | AWS Console | 🟢 MEDIUM | S3 storage |
| `SLACK_WEBHOOK_URL` | Slack workspace | 🔵 LOW | Deployment notifications |

---

## 2. GitHub Repository Settings

### A. Enable GitHub Actions

1. Go to your repository: `https://github.com/KDhiraj152/Siksha-Setu`
2. Click **Settings** → **Actions** → **General**
3. Enable: ✅ **Allow all actions and reusable workflows**
4. Workflow permissions: ✅ **Read and write permissions**
5. Click **Save**

### B. Enable GitHub Pages (Optional)

1. **Settings** → **Pages**
2. Source: **Deploy from a branch**
3. Branch: `gh-pages` or `main`
4. Click **Save**

### C. Enable Issues & Projects

1. **Settings** → **General**
2. Features section:
   - ✅ Issues
   - ✅ Projects
   - ✅ Discussions (optional)

---

## 3. GitHub Actions Secrets

### 🔴 CRITICAL - Required for CI/CD Workflows

Go to: **Repository Settings** → **Secrets and variables** → **Actions** → **New repository secret**

### A. Docker Registry Secrets (Auto-provided by GitHub)

| Secret Name | Value | How to Get | Used In |
|------------|-------|-----------|---------|
| `GITHUB_TOKEN` | ✅ Auto-provided | GitHub automatically provides this | All workflows |

**Note**: `GITHUB_TOKEN` is automatically available in all workflows - no setup needed!

### B. Deployment Secrets (Required if deploying)

| Secret Name | Value | How to Get | Used In |
|------------|-------|-----------|---------|
| `STAGING_SSH_KEY` | SSH private key | Generate SSH key (see below) | deploy-staging.yml |
| `STAGING_HOST` | staging.yourdomain.com | Your staging server IP/domain | deploy-staging.yml |
| `STAGING_USER` | ubuntu or root | Server username | deploy-staging.yml |
| `PRODUCTION_SSH_KEY` | SSH private key | Generate SSH key (see below) | deploy-production.yml |
| `PRODUCTION_PRIMARY_HOST` | api.yourdomain.com | Your production server IP | deploy-production.yml |
| `PRODUCTION_HOSTS` | host1 host2 host3 | Space-separated server IPs | deploy-production.yml |
| `PRODUCTION_USER` | ubuntu or root | Server username | deploy-production.yml |

**Generate SSH Key**:
```bash
# Generate SSH key pair
ssh-keygen -t ed25519 -C "github-actions-deploy" -f ~/.ssh/github_deploy

# Copy private key content
cat ~/.ssh/github_deploy

# Add the PRIVATE key to GitHub Secrets
# Add the PUBLIC key (~/.ssh/github_deploy.pub) to your server's ~/.ssh/authorized_keys
```

**Add Public Key to Server**:
```bash
# On your server
mkdir -p ~/.ssh
chmod 700 ~/.ssh

# Add the public key
echo "ssh-ed25519 AAAA... github-actions-deploy" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
```

### C. Notification Secrets (Optional)

| Secret Name | Value | How to Get | Used In |
|------------|-------|-----------|---------|
| `SLACK_WEBHOOK_URL` | https://hooks.slack.com/... | Slack Incoming Webhook | deploy-staging.yml |
| `DISCORD_WEBHOOK` | https://discord.com/api/webhooks/... | Discord Webhook | Custom notifications |

**Get Slack Webhook**:
1. Go to: https://api.slack.com/messaging/webhooks
2. Click **Create New App** → **From scratch**
3. Enable **Incoming Webhooks**
4. Add webhook to workspace
5. Copy webhook URL

### D. How to Add Secrets to GitHub

```bash
# Method 1: Via GitHub UI
1. Go to: https://github.com/KDhiraj152/Siksha-Setu/settings/secrets/actions
2. Click: "New repository secret"
3. Name: STAGING_SSH_KEY
4. Value: (paste your private key)
5. Click: "Add secret"

# Method 2: Via GitHub CLI (if installed)
gh secret set STAGING_SSH_KEY < ~/.ssh/github_deploy
gh secret set STAGING_HOST -b "staging.yourdomain.com"
gh secret set STAGING_USER -b "ubuntu"
```

---

## 4. Third-Party API Keys

### A. HuggingFace API Key (Optional but Recommended)

**Purpose**: Cloud-based ML model inference (faster than local)

**How to Get**:
1. Go to: https://huggingface.co/
2. Create free account or sign in
3. Go to: https://huggingface.co/settings/tokens
4. Click: **New token**
5. Name: `ShikshaSetu-API`
6. Type: **Read**
7. Click: **Generate token**
8. Copy token (starts with `hf_...`)

**Add to .env**:
```bash
HUGGINGFACE_API_KEY=hf_YourTokenHere
```

**Cost**: ✅ Free tier available (rate-limited)

---

### B. Bhashini API Key (Optional - Indian Languages)

**Purpose**: High-quality translation for Indian languages via Government API

**How to Get**:
1. Go to: https://bhashini.gov.in/ulca
2. Register for API access
3. Apply for API credentials
4. Wait for approval (may take 1-3 days)
5. Get: `USER_ID`, `API_KEY`, `PIPELINE_ID`

**Add to .env**:
```bash
BHASHINI_API_KEY=your_api_key
BHASHINI_USER_ID=your_user_id
BHASHINI_PIPELINE_ID=your_pipeline_id
BHASHINI_API_URL=https://dhruva-api.bhashini.gov.in/services/inference/pipeline
```

**Cost**: ✅ Free for government/educational use

---

### C. Sentry Error Tracking (Optional but Recommended)

**Purpose**: Real-time error tracking and monitoring

**How to Get**:
1. Go to: https://sentry.io/signup/
2. Create free account
3. Create new project: **Python** / **FastAPI**
4. Get DSN from: Project Settings → Client Keys (DSN)

**Add to .env**:
```bash
SENTRY_DSN=https://examplePublicKey@o0.ingest.sentry.io/0
SENTRY_TRACES_SAMPLE_RATE=0.1
SENTRY_PROFILES_SAMPLE_RATE=0.1
```

**Cost**: ✅ Free tier: 5,000 errors/month

---

### D. AWS S3 Storage (Optional - For Production)

**Purpose**: Store uploaded files, audio files, ML models

**How to Get**:
1. Go to: https://aws.amazon.com/console/
2. Sign in or create account
3. IAM → Users → Create user
4. Attach policy: `AmazonS3FullAccess`
5. Security credentials → Create access key
6. Copy: `Access Key ID` and `Secret Access Key`

**Create S3 Bucket**:
```bash
# Via AWS Console
1. S3 → Create bucket
2. Name: shikshasetu-production-files
3. Region: ap-south-1 (Mumbai)
4. Block public access: ✅ Enabled
5. Create bucket
```

**Add to .env**:
```bash
USE_S3_STORAGE=true
AWS_S3_BUCKET=shikshasetu-production-files
AWS_REGION=ap-south-1
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=your_secret_key
```

**Cost**: ~$0.023 per GB/month + requests

---

### E. Stripe Payment Gateway (Optional - For Paid Features)

**Purpose**: Handle subscriptions and payments

**How to Get**:
1. Go to: https://dashboard.stripe.com/register
2. Create account
3. Get test keys from: Developers → API keys
4. Get webhook secret from: Developers → Webhooks

**Add to .env**:
```bash
STRIPE_ENABLED=true
STRIPE_SECRET_KEY=sk_test_...
STRIPE_PUBLISHABLE_KEY=pk_test_...
STRIPE_WEBHOOK_SECRET=whsec_...
```

**Cost**: 2.9% + ₹3 per successful transaction

---

## 5. Database Setup

### Option A: Local PostgreSQL (Development)

**Install PostgreSQL**:
```bash
# macOS
brew install postgresql@17

# Ubuntu/Debian
sudo apt update
sudo apt install postgresql-17 postgresql-contrib

# Start service
brew services start postgresql@17  # macOS
sudo systemctl start postgresql    # Linux
```

**Create Database**:
```bash
# Connect to PostgreSQL
psql postgres

# Create user and database
CREATE USER shiksha_user WITH PASSWORD 'shiksha_pass';
CREATE DATABASE shiksha_setu OWNER shiksha_user;

# Enable pgvector extension
\c shiksha_setu
CREATE EXTENSION IF NOT EXISTS vector;

# Exit
\q
```

**Add to .env**:
```bash
DATABASE_URL=postgresql://shiksha_user:shiksha_pass@localhost:5432/shiksha_setu
```

---

### Option B: Supabase (Recommended for Production)

**Purpose**: Managed PostgreSQL with pgvector + Authentication + Storage

**How to Get**:
1. Go to: https://supabase.com/
2. Create free account
3. Click: **New Project**
4. Name: `ShikshaSetu`
5. Database Password: (generate strong password)
6. Region: **Mumbai (ap-south-1)** or closest
7. Wait 2 minutes for provisioning

**Get Database URL**:
1. Project Settings → Database
2. Connection string → URI
3. Copy: `postgresql://postgres.xxx:[YOUR-PASSWORD]@xxx.supabase.co:5432/postgres`

**Enable pgvector**:
1. SQL Editor → New Query
2. Run: `CREATE EXTENSION IF NOT EXISTS vector;`
3. Click: **Run**

**Add to .env**:
```bash
DATABASE_URL=postgresql://postgres.xxx:YOUR_PASSWORD@xxx.supabase.co:5432/postgres

# Connection pooling (recommended)
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
```

**Cost**: 
- ✅ Free tier: 500MB database, 1GB file storage
- Paid: $25/month for 8GB database

---

### Option C: Docker PostgreSQL (Quick Testing)

```bash
# Run PostgreSQL with pgvector
docker run -d \
  --name shiksha-postgres \
  -e POSTGRES_USER=shiksha_user \
  -e POSTGRES_PASSWORD=shiksha_pass \
  -e POSTGRES_DB=shiksha_setu \
  -p 5432:5432 \
  -v postgres_data:/var/lib/postgresql/data \
  ankane/pgvector:latest

# Test connection
docker exec -it shiksha-postgres psql -U shiksha_user -d shiksha_setu

# Enable pgvector
CREATE EXTENSION IF NOT EXISTS vector;
```

**Add to .env**:
```bash
DATABASE_URL=postgresql://shiksha_user:shiksha_pass@localhost:5432/shiksha_setu
```

---

## 6. Domain & DNS Configuration

### A. Register Domain (Optional but Professional)

**Options**:
- **Namecheap**: $8-12/year
- **GoDaddy**: $10-15/year
- **Cloudflare**: $9-13/year (+ free CDN)

**Recommended**: `shikshasetu.in` or `shikshasetu.com`

### B. DNS Records to Add

Point your domain to your server:

```bash
# A Records (IPv4)
api.shikshasetu.in        → 123.456.789.10 (backend server IP)
staging.shikshasetu.in    → 123.456.789.11 (staging server IP)
shikshasetu.in            → 123.456.789.10 (or CDN)
www.shikshasetu.in        → 123.456.789.10

# CNAME Records (optional)
*.shikshasetu.in          → shikshasetu.in (catch-all)
```

### C. SSL Certificate (Free with Let's Encrypt)

**Automatic with Certbot**:
```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d shikshasetu.in -d www.shikshasetu.in -d api.shikshasetu.in

# Auto-renewal (automatically configured)
sudo certbot renew --dry-run
```

**Or use Cloudflare** (easier):
1. Add domain to Cloudflare
2. Update nameservers at registrar
3. SSL/TLS → Full (strict)
4. ✅ Free SSL certificate automatically provisioned

---

## 7. Server Access

### A. Server Requirements

**Minimum Server Specs**:
- **CPU**: 4 cores
- **RAM**: 8GB
- **Storage**: 100GB SSD
- **OS**: Ubuntu 22.04 LTS

**Recommended Providers**:
- **DigitalOcean**: $24/month (4GB RAM)
- **Linode**: $24/month (4GB RAM)
- **AWS EC2**: $30-50/month (t3.medium)
- **Hetzner**: €15/month (4GB RAM) - Best value

### B. Server Setup Checklist

```bash
# 1. Initial server setup
ssh root@your-server-ip

# 2. Update system
apt update && apt upgrade -y

# 3. Create deployment user
adduser deploy
usermod -aG sudo deploy
usermod -aG docker deploy

# 4. Install Docker
curl -fsSL https://get.docker.com | sh
systemctl enable docker
systemctl start docker

# 5. Install Docker Compose
apt install docker-compose-plugin -y

# 6. Setup firewall
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw enable

# 7. Setup SSH key authentication
mkdir -p ~/.ssh
chmod 700 ~/.ssh
# Add your public key to ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys

# 8. Disable password authentication
nano /etc/ssh/sshd_config
# Set: PasswordAuthentication no
systemctl restart sshd
```

### C. Deploy Application Directory

```bash
# Login as deploy user
ssh deploy@your-server-ip

# Create application directory
sudo mkdir -p /opt/shikshasetu
sudo chown deploy:deploy /opt/shikshasetu
cd /opt/shikshasetu

# Clone repository
git clone https://github.com/KDhiraj152/Siksha-Setu.git .

# Setup environment
cp .env.example .env
nano .env  # Configure your production values

# Start services
docker compose -f docker-compose.production.yml up -d
```

---

## 8. Monitoring & Alerting

### A. Setup Monitoring Stack

**Services**:
- **Prometheus**: Metrics collection
- **Grafana**: Dashboards
- **Alertmanager**: Alert routing
- **Node Exporter**: System metrics
- **PostgreSQL Exporter**: Database metrics

**Start monitoring**:
```bash
cd infrastructure/monitoring
docker-compose -f docker-compose.monitoring.yml up -d
```

**Access**:
- Grafana: http://your-server:3001 (admin/admin)
- Prometheus: http://your-server:9090
- Alertmanager: http://your-server:9093

### B. Configure Alerts

**Edit**: `infrastructure/monitoring/alertmanager/config.yml`

```yaml
receivers:
  - name: 'email'
    email_configs:
      - to: 'alerts@yourdomain.com'
        from: 'monitoring@yourdomain.com'
        smarthost: 'smtp.gmail.com:587'
        auth_username: 'your-email@gmail.com'
        auth_password: 'your-app-password'

  - name: 'slack'
    slack_configs:
      - api_url: 'YOUR_SLACK_WEBHOOK_URL'
        channel: '#alerts'
        title: 'ShikshaSetu Alert'
```

---

## 9. Storage & Backups

### A. Automated Database Backups

**Setup cron job**:
```bash
# Edit crontab
crontab -e

# Add daily backup at 2 AM
0 2 * * * /opt/shikshasetu/scripts/deployment/backup-postgres.sh
```

**Backup script** (already included):
```bash
# Location: scripts/deployment/backup-postgres.sh
# Automatically backs up to: /backups/postgres/
# Retention: 30 days
```

### B. Backup to Cloud Storage

**Install rclone**:
```bash
# Install
curl https://rclone.org/install.sh | sudo bash

# Configure for S3/Google Drive/Dropbox
rclone config

# Add to backup script
rclone sync /backups/ remote:shikshasetu-backups/
```

### C. Disaster Recovery Plan

**Backup checklist**:
- [ ] Database (daily automated)
- [ ] User uploads (real-time to S3)
- [ ] ML models (version controlled)
- [ ] Configuration files (.env - manual)
- [ ] Docker volumes (weekly)

**Restore from backup**:
```bash
# Restore database
gunzip < backup.sql.gz | psql $DATABASE_URL

# Restore uploads from S3
aws s3 sync s3://shikshasetu-backups/uploads/ /opt/shikshasetu/data/uploads/
```

---

## 10. Quick Setup Commands

### Complete Environment Setup

```bash
# 1. Clone repository
git clone https://github.com/KDhiraj152/Siksha-Setu.git
cd Siksha-Setu

# 2. Copy environment file
cp .env.example .env

# 3. Generate JWT secret
python3 -c "import secrets; print('JWT_SECRET_KEY=' + secrets.token_urlsafe(64))" >> .env

# 4. Setup database (if using Docker)
docker run -d \
  --name shiksha-postgres \
  -e POSTGRES_USER=shiksha_user \
  -e POSTGRES_PASSWORD=shiksha_pass \
  -e POSTGRES_DB=shiksha_setu \
  -p 5432:5432 \
  ankane/pgvector:latest

# 5. Setup Redis
docker run -d \
  --name shiksha-redis \
  -p 6379:6379 \
  redis:7-alpine

# 6. Add database URL to .env
echo "DATABASE_URL=postgresql://shiksha_user:shiksha_pass@localhost:5432/shiksha_setu" >> .env
echo "REDIS_URL=redis://localhost:6379/0" >> .env

# 7. Run setup script
./bin/setup

# 8. Start application
./bin/start

# 9. Verify
curl http://localhost:8000/health
```

---

## 📝 Checklist Summary

Copy this checklist and mark items as you complete them:

### Critical Setup (Must Do)
- [ ] Create `.env` file with required variables
- [ ] Generate `JWT_SECRET_KEY` (64+ characters)
- [ ] Setup PostgreSQL database (local/Supabase/Docker)
- [ ] Setup Redis (local/Docker/cloud)
- [ ] Run `./bin/setup` to install dependencies
- [ ] Run `./bin/start` to start services
- [ ] Test: `curl http://localhost:8000/health`

### GitHub Actions (Required for CI/CD)
- [ ] Enable GitHub Actions in repository settings
- [ ] Add `GITHUB_TOKEN` is auto-provided ✅
- [ ] Add SSH keys for deployment servers (if deploying)
- [ ] Add `STAGING_HOST`, `STAGING_USER`, `STAGING_SSH_KEY`
- [ ] Add `PRODUCTION_*` secrets (if deploying to production)

### Optional but Recommended
- [ ] Get HuggingFace API key for ML inference
- [ ] Setup Sentry for error tracking
- [ ] Configure Slack webhook for notifications
- [ ] Setup domain and SSL certificate
- [ ] Configure automated backups
- [ ] Setup monitoring (Prometheus + Grafana)

### Production Only
- [ ] Provision production server (4+ CPU, 8+ GB RAM)
- [ ] Configure firewall and security groups
- [ ] Setup S3 or equivalent object storage
- [ ] Configure CDN (Cloudflare/CloudFront)
- [ ] Setup log aggregation (optional)
- [ ] Configure auto-scaling (if using Kubernetes)

---

## 🆘 Getting Help

**If you get stuck**:

1. **Check documentation**:
   - [DEVELOPMENT.md](DEVELOPMENT.md) - Developer guide
   - [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment guide
   - [docs/](docs/) - Detailed documentation

2. **Common issues**:
   - Database connection: Check `DATABASE_URL` format
   - Redis connection: Ensure Redis is running
   - JWT errors: Regenerate `JWT_SECRET_KEY`
   - Port conflicts: Change port in `.env`

3. **Test components**:
   ```bash
   # Test database
   psql $DATABASE_URL -c "SELECT 1;"
   
   # Test Redis
   redis-cli -u $REDIS_URL PING
   
   # Test backend
   curl http://localhost:8000/health
   
   # Test frontend
   curl http://localhost:5173
   ```

4. **Check logs**:
   ```bash
   # Backend logs
   tail -f logs/app.log
   
   # Docker logs
   docker compose logs -f backend
   
   # System logs
   journalctl -u shikshasetu -f
   ```

---

## 🎉 You're All Set!

Once you've completed the critical setup items, your application should be running!

**Next steps**:
1. Access frontend: http://localhost:5173
2. Access API docs: http://localhost:8000/docs
3. Run tests: `./bin/test`
4. Read developer guide: [DEVELOPMENT.md](DEVELOPMENT.md)

**Questions?** Open an issue on GitHub: https://github.com/KDhiraj152/Siksha-Setu/issues
