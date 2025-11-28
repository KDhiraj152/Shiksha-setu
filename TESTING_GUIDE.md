# ShikshaSetu Testing Guide

## 🚀 All Services Running

### Service Status
- ✅ **Backend API:** http://localhost:8000
- ✅ **API Documentation:** http://localhost:8000/docs
- ✅ **Frontend:** http://localhost:5173
- ✅ **PostgreSQL Database:** localhost:5432
- ✅ **Redis Cache:** localhost:6379
- ✅ **Celery Workers:** 11 workers active

---

## 🧪 API Testing

### 1. Authentication

#### Login
```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@shiksha.com",
    "password": "Test@1234567"
  }'
```

**Test Users:**
- `test@shiksha.com` / `Test@1234567` (user)
- `teacher@shiksha.com` / `Teacher@123456` (teacher)
- `admin@shiksha.com` / `Admin@123456` (admin)

#### Register New User
```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "newuser@test.com",
    "password": "SecurePass@123",
    "full_name": "Test User",
    "organization": "Test School"
  }'
```

---

### 2. Content Simplification

```bash
curl -X POST http://localhost:8000/api/v1/content/simplify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Photosynthesis is the process by which plants convert light energy into chemical energy stored in glucose.",
    "target_grade": 5,
    "subject": "Science"
  }'
```

**Expected Response:**
```json
{
  "simplified_text": "...",
  "grade_level": 5,
  "subject": "Science",
  "task_id": "...",
  "status": "completed"
}
```

---

### 3. Translation

```bash
curl -X POST http://localhost:8000/api/v1/content/translate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Good morning students",
    "source_language": "English",
    "target_language": "Hindi"
  }'
```

**Supported Languages:**
- Hindi
- Tamil
- Telugu
- Bengali
- Marathi

---

### 4. Q&A System (Requires Auth)

#### Get Access Token First
```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "test@shiksha.com", "password": "Test@1234567"}' | \
  python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
```

#### Process Document
```bash
curl -X POST http://localhost:8000/api/v1/qa/process \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content_id": "test-doc-1",
    "chunk_size": 512,
    "overlap": 50
  }'
```

#### Ask Question
```bash
curl -X POST http://localhost:8000/api/v1/qa/ask \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is photosynthesis?",
    "content_id": 1
  }'
```

---

### 5. Content Library

```bash
# List content
curl http://localhost:8000/api/v1/content/library?limit=10

# Get specific content
curl http://localhost:8000/api/v1/content/1
```

---

## 🎨 Frontend Testing

Open in browser: **http://localhost:5173**

The frontend provides:
- Content simplification interface
- Translation tools
- Q&A chat interface
- User authentication

---

## 📊 API Documentation

Visit **http://localhost:8000/docs** for:
- Interactive API explorer (Swagger UI)
- All endpoint documentation
- Request/response schemas
- Try-it-out functionality

---

## 🔍 Monitoring & Logs

### View Logs
```bash
# Backend logs
tail -f /tmp/backend.log

# Frontend logs
tail -f /tmp/frontend.log

# Celery worker logs
tail -f /tmp/celery.log
```

### Check Health
```bash
curl http://localhost:8000/health
```

### Database Connection
```bash
# Connect to PostgreSQL
docker exec -it shiksha-postgres psql -U shiksha_user -d shiksha_setu

# List tables
\dt

# Check users
SELECT id, email, role FROM users;
```

### Redis Connection
```bash
# Connect to Redis
docker exec -it shikshasetu_redis redis-cli

# Check keys
KEYS *

# Monitor commands
MONITOR
```

---

## 🛠️ Service Management

### Start All Services
```bash
./start_all.sh
```

### Stop All Services
```bash
./stop_all.sh
```

### Restart Individual Services
```bash
# Stop all first
./stop_all.sh

# Start backend only
source .venv/bin/activate
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload > /tmp/backend.log 2>&1 &

# Start frontend only
cd frontend && npm run dev > /tmp/frontend.log 2>&1 &

# Start Celery workers only
source .venv/bin/activate
celery -A backend.tasks.celery_app worker --loglevel=info > /tmp/celery.log 2>&1 &
```

---

## 🧩 Testing Workflows

### Complete Workflow 1: Content Simplification
1. Login to get token
2. Submit text for simplification
3. Check task status
4. View simplified result

### Complete Workflow 2: Translation
1. Login to get token
2. Submit text for translation
3. Get translated text in target language
4. Verify translation quality

### Complete Workflow 3: RAG Q&A
1. Login to get token
2. Upload/process document
3. Ask questions about the content
4. Get AI-generated answers
5. View chat history

---

## 🐛 Troubleshooting

### Backend Not Starting
```bash
# Check port 8000
lsof -i :8000

# Check logs
tail -50 /tmp/backend.log
```

### Frontend Not Starting
```bash
# Check port 5173
lsof -i :5173

# Rebuild frontend
cd frontend && npm install && npm run build
```

### Database Issues
```bash
# Check PostgreSQL is running
docker ps | grep postgres

# Restart PostgreSQL
docker restart shiksha-postgres
```

### Celery Not Processing Tasks
```bash
# Check Celery workers
ps aux | grep celery

# Restart Celery
pkill -f celery
celery -A backend.tasks.celery_app worker --loglevel=info > /tmp/celery.log 2>&1 &
```

---

## 📝 Notes

- **AI Models:** Currently running in fallback mode (no actual ML models loaded)
- **Authentication:** Required for Q&A and some admin endpoints
- **CORS:** Enabled for localhost:5173 frontend
- **Rate Limiting:** 60 requests/minute, 1000 requests/hour

---

## 🎯 Next Steps

1. Test all endpoints via Swagger UI (http://localhost:8000/docs)
2. Test frontend interface (http://localhost:5173)
3. Try authentication flow
4. Test content simplification
5. Test translation between languages
6. Test Q&A system with documents
7. Monitor Celery task processing
8. Check database records

Happy Testing! 🚀
