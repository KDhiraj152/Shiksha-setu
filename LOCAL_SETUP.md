# 🏠 Local Development Setup - Quick Start

**Get ShikshaSetu running on your Mac in 5 minutes!**

---

## ✅ What You Have

- ✅ Docker installed
- ✅ Redis installed  
- ✅ Node.js v25 installed
- ✅ Python 3.14 installed
- ✅ `.env` file exists
- ❌ PostgreSQL not installed (we'll use Docker)

---

## 🚀 Quick Setup (5 Steps)

### Step 1: Start Database & Redis with Docker

```bash
# Start PostgreSQL with pgvector
docker run -d \
  --name shiksha-postgres \
  -e POSTGRES_USER=shiksha_user \
  -e POSTGRES_PASSWORD=shiksha_pass \
  -e POSTGRES_DB=shiksha_setu \
  -p 5432:5432 \
  -v postgres_data:/var/lib/postgresql/data \
  ankane/pgvector:latest

# Start Redis
docker run -d \
  --name shiksha-redis \
  -p 6379:6379 \
  redis:7-alpine

# Verify containers are running
docker ps
```

**Expected output**: You should see both containers running

---

### Step 2: Configure Environment Variables

```bash
# Check your .env file has these critical values
cat .env | grep -E "DATABASE_URL|REDIS_URL|JWT_SECRET_KEY"
```

**If missing, add them**:

```bash
# Add to .env file
cat >> .env << 'EOF'

# Database (Docker PostgreSQL)
DATABASE_URL=postgresql://shiksha_user:shiksha_pass@localhost:5432/shiksha_setu

# Redis (Docker)
REDIS_URL=redis://localhost:6379/0

# Generate JWT secret
EOF

# Generate and add JWT secret
python3 -c "import secrets; print('JWT_SECRET_KEY=' + secrets.token_urlsafe(64))" >> .env

# Verify
echo "✅ Configuration updated!"
cat .env | tail -5
```

---

### Step 3: Enable pgvector Extension

```bash
# Connect to database and enable pgvector
docker exec -it shiksha-postgres psql -U shiksha_user -d shiksha_setu -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Verify
docker exec -it shiksha-postgres psql -U shiksha_user -d shiksha_setu -c "\dx" | grep vector
```

**Expected output**: `vector | ... | vector data type and ivfflat access method`

---

### Step 4: Run Setup Script

```bash
# Make scripts executable
chmod +x bin/setup bin/start bin/start-backend bin/start-frontend

# Run automated setup (installs dependencies)
./bin/setup
```

**This will**:
- Create Python virtual environment
- Install backend dependencies
- Install frontend dependencies
- Run database migrations
- Download required ML models (this may take a few minutes)

---

### Step 5: Start Application

```bash
# Start both backend and frontend
./bin/start

# Or start separately:
# ./bin/start-backend   # Backend only (port 8000)
# ./bin/start-frontend  # Frontend only (port 5173)
```

**Access the application**:
- 🌐 Frontend: http://localhost:5173
- 🔧 Backend API: http://localhost:8000
- 📚 API Docs: http://localhost:8000/docs

---

## 🧪 Verify Everything Works

```bash
# Test backend health
curl http://localhost:8000/health

# Expected: {"status":"healthy"}

# Test database connection
curl http://localhost:8000/api/v1/status

# Test frontend
curl http://localhost:5173
```

---

## 🛠️ Troubleshooting

### Problem: Port 5432 already in use

```bash
# Check what's using port 5432
lsof -i :5432

# Stop existing PostgreSQL
brew services stop postgresql
# Or kill the process
kill -9 <PID>
```

### Problem: Port 6379 already in use

```bash
# Check what's using port 6379
lsof -i :6379

# Stop existing Redis
brew services stop redis
# Or kill the process
kill -9 <PID>
```

### Problem: Docker containers not starting

```bash
# Remove old containers
docker rm -f shiksha-postgres shiksha-redis

# Remove old volumes
docker volume rm postgres_data

# Start fresh (run Step 1 again)
```

### Problem: Python version warning

Your Python is 3.14 (very new). The project is tested on Python 3.11. If you encounter issues:

```bash
# Install Python 3.11 with pyenv
brew install pyenv
pyenv install 3.11.11
pyenv local 3.11.11

# Verify
python3 --version  # Should show 3.11.11

# Re-run setup
./bin/setup
```

### Problem: Dependencies won't install

```bash
# Clean and reinstall
rm -rf .venv frontend/node_modules

# Run setup again
./bin/setup
```

### Problem: Database migrations fail

```bash
# Reset database
docker exec -it shiksha-postgres psql -U shiksha_user -d postgres -c "DROP DATABASE shiksha_setu;"
docker exec -it shiksha-postgres psql -U shiksha_user -d postgres -c "CREATE DATABASE shiksha_setu OWNER shiksha_user;"

# Enable pgvector
docker exec -it shiksha-postgres psql -U shiksha_user -d shiksha_setu -c "CREATE EXTENSION IF NOT EXISTS vector;"

# Re-run migrations
source .venv/bin/activate
cd backend
alembic upgrade head
```

---

## 🎯 Next Steps After Setup

### 1. Create Demo User (Optional)

```bash
# Run demo setup
./bin/demo

# This creates:
# - Admin user: admin@example.com / admin123
# - Test educator: educator@example.com / educator123
# - Sample content
```

### 2. Run Tests

```bash
# Run all tests
./bin/test

# Or run specific tests
source .venv/bin/activate
pytest tests/unit/ -v
```

### 3. Development Workflow

```bash
# Backend development (auto-reload enabled)
./bin/start-backend

# Frontend development (auto-reload enabled)
./bin/start-frontend

# Watch logs
tail -f logs/app.log
```

---

## 📊 Useful Commands

### Check Services Status

```bash
# Docker containers
docker ps

# Backend server
curl http://localhost:8000/health

# Database connection
docker exec -it shiksha-postgres psql -U shiksha_user -d shiksha_setu -c "SELECT version();"

# Redis connection
docker exec -it shiksha-redis redis-cli PING
```

### Stop Services

```bash
# Stop backend/frontend (Ctrl+C in terminal)

# Stop Docker containers
docker stop shiksha-postgres shiksha-redis

# Start again
docker start shiksha-postgres shiksha-redis
```

### Clean Everything

```bash
# Stop and remove containers
docker rm -f shiksha-postgres shiksha-redis

# Remove volumes (WARNING: deletes all data)
docker volume rm postgres_data

# Clean Python environment
rm -rf .venv

# Clean Node modules
rm -rf frontend/node_modules
```

---

## 🔧 Optional: Local PostgreSQL Instead of Docker

If you prefer local PostgreSQL:

```bash
# Install PostgreSQL 17
brew install postgresql@17

# Start PostgreSQL
brew services start postgresql@17

# Create database
psql postgres << 'EOF'
CREATE USER shiksha_user WITH PASSWORD 'shiksha_pass';
CREATE DATABASE shiksha_setu OWNER shiksha_user;
\c shiksha_setu
CREATE EXTENSION IF NOT EXISTS vector;
\q
EOF

# Update .env
# DATABASE_URL=postgresql://shiksha_user:shiksha_pass@localhost:5432/shiksha_setu

# Stop Docker PostgreSQL
docker stop shiksha-postgres
```

---

## 🎓 Learning Resources

- **API Documentation**: http://localhost:8000/docs (after starting backend)
- **Development Guide**: [DEVELOPMENT.md](DEVELOPMENT.md)
- **Architecture Overview**: [docs/reference/architecture.md](docs/reference/architecture.md)
- **Testing Guide**: [docs/guides/testing.md](docs/guides/testing.md)

---

## ✅ Setup Checklist

- [ ] Docker containers running (postgres + redis)
- [ ] `.env` file configured with DATABASE_URL, REDIS_URL, JWT_SECRET_KEY
- [ ] pgvector extension enabled
- [ ] `./bin/setup` completed successfully
- [ ] Backend running on http://localhost:8000
- [ ] Frontend running on http://localhost:5173
- [ ] Health check passes: `curl http://localhost:8000/health`
- [ ] Can access API docs: http://localhost:8000/docs

---

## 🆘 Need Help?

1. Check logs: `tail -f logs/app.log`
2. Check Docker logs: `docker logs shiksha-postgres` or `docker logs shiksha-redis`
3. Review full docs: [DEVELOPMENT.md](DEVELOPMENT.md)
4. Check issues: https://github.com/KDhiraj152/Siksha-Setu/issues

---

**🎉 You're ready to develop!**

Start coding and the application will auto-reload on changes.
