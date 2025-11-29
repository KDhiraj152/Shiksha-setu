# 🔐 Security Architecture

Comprehensive security guide for Shiksha Setu, covering authentication, authorization, rate limiting, and security best practices.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Authorization (RBAC)](#authorization-rbac)
4. [Rate Limiting](#rate-limiting)
5. [CORS & Security Headers](#cors--security-headers)
6. [Environment Security](#environment-security)
7. [API Security](#api-security)
8. [Database Security](#database-security)
9. [Best Practices](#best-practices)

---

## Overview

Shiksha Setu implements production-grade security following industry standards (OWASP Top 10, NIST, ISO 27001).

### Security Layers

```
┌─────────────────────────────────────────┐
│         TLS/HTTPS Transport             │ Layer 7: Encryption
├─────────────────────────────────────────┤
│     CORS & Security Headers             │ Layer 6: Browser Protection
├─────────────────────────────────────────┤
│     Rate Limiting (Redis)               │ Layer 5: DDoS Protection
├─────────────────────────────────────────┤
│     JWT Authentication                  │ Layer 4: Identity
├─────────────────────────────────────────┤
│     RBAC Authorization                  │ Layer 3: Access Control
├─────────────────────────────────────────┤
│     Input Validation                    │ Layer 2: Data Sanitization
├─────────────────────────────────────────┤
│     Database Encryption                 │ Layer 1: Data Protection
└─────────────────────────────────────────┘
```

---

## Authentication

### JWT (JSON Web Tokens)

**Implementation**: `backend/middleware/auth.py`

```python
from backend.middleware.auth import create_access_token, verify_token

# Create token
access_token = create_access_token(
    data={"sub": user.email, "role": user.role}
)

# Verify token
payload = verify_token(token)
# Returns: {"sub": "user@example.com", "role": "teacher"}
```

### Token Configuration

**File**: `.env`

```bash
# JWT Configuration
SECRET_KEY="your-secret-key-min-32-chars"  # Generate with: openssl rand -hex 32
ALGORITHM="HS256"                          # HMAC-SHA256
ACCESS_TOKEN_EXPIRE_MINUTES=30             # 30 minutes default
REFRESH_TOKEN_EXPIRE_DAYS=7                # 7 days for refresh tokens
```

### Token Lifecycle

```
┌──────────┐                  ┌──────────┐                 ┌──────────┐
│  Login   │ ─── Token ────>  │  Client  │ ─── Request ──> │  Backend │
│          │                  │          │ + Bearer Token  │          │
└──────────┘                  └──────────┘                 └──────────┘
                                    │                           │
                                    │                           ▼
                                    │                      Verify Token
                                    │                           │
                                    │ <───── Response ──────────┘
                                    │      (200 or 401)
```

### Protected Endpoints

```python
from fastapi import Depends
from backend.middleware.auth import get_current_user

@router.get("/profile")
async def get_profile(current_user: User = Depends(get_current_user)):
    """Protected endpoint - requires authentication"""
    return {"email": current_user.email, "role": current_user.role}

@router.post("/content/upload")
async def upload_content(
    file: UploadFile,
    current_user: User = Depends(get_current_user)
):
    """Only authenticated users can upload"""
    if current_user.role not in ["teacher", "admin"]:
        raise HTTPException(403, "Insufficient permissions")
    # ... process upload ...
```

### Login Flow

```bash
# 1. Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "secret123"}'

# Response:
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800  # 30 minutes
}

# 2. Use token for authenticated requests
curl http://localhost:8000/api/v1/content/library \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

# 3. Refresh token (before expiry)
curl -X POST http://localhost:8000/api/v1/auth/refresh \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

---

## Authorization (RBAC)

### Role Hierarchy

```
Admin
  │
  ├── Full system access
  ├── User management
  ├── System configuration
  └── View all content
      │
      └── Teacher
          │
          ├── Upload content
          ├── View own content
          ├── Manage own students
          └── Generate simplified content
              │
              └── Student
                  │
                  ├── View assigned content
                  ├── View simplified content
                  └── Submit responses
```

### Role Definitions

**File**: `backend/models/user.py`

```python
from enum import Enum

class UserRole(str, Enum):
    STUDENT = "student"      # Basic access
    TEACHER = "teacher"      # Content creation
    ADMIN = "admin"          # Full access

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True)
    email = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(Enum(UserRole), default=UserRole.STUDENT)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
```

### Permission Checks

```python
from backend.middleware.auth import require_role

# Role-based route protection
@router.post("/content/upload")
@require_role(["teacher", "admin"])
async def upload_content(
    file: UploadFile,
    current_user: User = Depends(get_current_user)
):
    """Only teachers and admins can upload"""
    # ... process upload ...

@router.get("/admin/users")
@require_role(["admin"])
async def list_users(current_user: User = Depends(get_current_user)):
    """Admin-only endpoint"""
    # ... list users ...

@router.get("/content/{content_id}")
async def get_content(
    content_id: int,
    current_user: User = Depends(get_current_user)
):
    """Check ownership or admin access"""
    content = await get_content_by_id(content_id)
    
    if current_user.role != "admin" and content.user_id != current_user.id:
        raise HTTPException(403, "Access denied")
    
    return content
```

### Permission Matrix

| Action | Student | Teacher | Admin |
|--------|---------|---------|-------|
| View own content | ✅ | ✅ | ✅ |
| View all content | ❌ | ❌ | ✅ |
| Upload content | ❌ | ✅ | ✅ |
| Delete own content | ❌ | ✅ | ✅ |
| Delete any content | ❌ | ❌ | ✅ |
| Manage users | ❌ | ❌ | ✅ |
| View system logs | ❌ | ❌ | ✅ |
| Configure system | ❌ | ❌ | ✅ |

---

## Rate Limiting

### Implementation

**File**: `backend/middleware/rate_limit.py`

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Apply to specific endpoints
@router.post("/content/upload")
@limiter.limit("10 per hour")
async def upload_content(request: Request, ...):
    """Limited to 10 uploads per hour per IP"""
    pass

@router.post("/auth/login")
@limiter.limit("5 per minute")
async def login(request: Request, ...):
    """Limited to 5 login attempts per minute"""
    pass
```

### Rate Limit Tiers

| Endpoint | Student | Teacher | Admin | Public |
|----------|---------|---------|-------|--------|
| `/api/v1/content/upload` | N/A | 20/hour | 100/hour | N/A |
| `/api/v1/content/process` | 10/hour | 50/hour | Unlimited | N/A |
| `/api/v1/auth/login` | 5/min | 5/min | 10/min | 5/min |
| `/api/v1/auth/register` | 2/hour | 2/hour | N/A | 2/hour |
| `/api/v1/health` | Unlimited | Unlimited | Unlimited | 100/min |

### Configuration

**File**: `.env`

```bash
# Redis Configuration (for rate limiting)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=                          # Optional

# Rate Limit Configuration
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_DAY=200
RATE_LIMIT_PER_HOUR=50
RATE_LIMIT_PER_MINUTE=10
```

### Testing Rate Limits

```bash
# Test upload rate limit (10/hour)
for i in {1..15}; do
  echo "Request $i:"
  curl -X POST http://localhost:8000/api/v1/content/upload \
    -H "Authorization: Bearer $TOKEN" \
    -F "file=@test.pdf" \
    -w "\n%{http_code}\n"
  sleep 1
done

# Expected: First 10 succeed (200), next 5 fail (429)
```

### Response Headers

```http
HTTP/1.1 429 Too Many Requests
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1640000000
Retry-After: 3600

{
  "detail": "Rate limit exceeded: 10 requests per hour"
}
```

---

## CORS & Security Headers

### CORS Configuration

**File**: `backend/main.py`

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Development frontend
        "https://shikshasetu.com"  # Production domain
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
    expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining"],
    max_age=3600  # Cache preflight for 1 hour
)
```

### Security Headers

**Implementation**: `backend/middleware/security.py`

```python
from fastapi import Request

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    
    # Prevent XSS attacks
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    
    # Enforce HTTPS
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    # Content Security Policy
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self' data:;"
    )
    
    # Permissions Policy
    response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
    
    return response
```

### Security Headers Explained

| Header | Purpose | Value |
|--------|---------|-------|
| `X-Content-Type-Options` | Prevent MIME sniffing | `nosniff` |
| `X-Frame-Options` | Prevent clickjacking | `DENY` |
| `X-XSS-Protection` | Enable browser XSS filter | `1; mode=block` |
| `Strict-Transport-Security` | Enforce HTTPS | `max-age=31536000` |
| `Content-Security-Policy` | Control resource loading | Custom policy |
| `Permissions-Policy` | Control browser features | Deny dangerous features |

---

## Environment Security

### Secret Management

**File**: `.env` (NEVER commit this)

```bash
# CRITICAL: Keep these secret!
SECRET_KEY=your-secret-key-min-32-chars-long
DATABASE_URL=postgresql://user:password@localhost:5432/shikshasetu
REDIS_PASSWORD=your-redis-password

# API Keys (encrypted at rest)
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=AIza...
AZURE_API_KEY=...
```

### Environment Template

**File**: `.env.example` (safe to commit)

```bash
# JWT Configuration
SECRET_KEY=generate-with-openssl-rand-hex-32
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/dbname

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# API Keys (get from providers)
OPENAI_API_KEY=
GOOGLE_API_KEY=
AZURE_API_KEY=
```

### Generate Secrets

```bash
# Generate SECRET_KEY
openssl rand -hex 32

# Generate random password
openssl rand -base64 32

# Generate UUID
python3 -c "import uuid; print(uuid.uuid4())"
```

### Protect Secrets

```bash
# Add to .gitignore
echo ".env" >> .gitignore
echo "*.key" >> .gitignore
echo "*.pem" >> .gitignore

# Set file permissions
chmod 600 .env
chmod 600 *.key

# Verify .env not tracked
git status --ignored | grep .env
```

---

## API Security

### Input Validation

**File**: `backend/schemas/content.py`

```python
from pydantic import BaseModel, Field, validator

class ContentUpload(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    grade_level: int = Field(..., ge=5, le=12)  # 5-12 only
    subject: str = Field(..., regex="^[A-Za-z ]+$")
    
    @validator('title')
    def sanitize_title(cls, v):
        # Remove dangerous characters
        return v.strip().replace('<', '').replace('>', '')
    
    class Config:
        # Reject extra fields
        extra = "forbid"
```

### SQL Injection Prevention

```python
from sqlalchemy import text

# ❌ BAD: String interpolation (vulnerable)
query = f"SELECT * FROM users WHERE email = '{email}'"
result = session.execute(query)

# ✅ GOOD: Parameterized queries
query = text("SELECT * FROM users WHERE email = :email")
result = session.execute(query, {"email": email})

# ✅ BETTER: ORM (automatic escaping)
user = session.query(User).filter(User.email == email).first()
```

### File Upload Security

```python
import magic
from pathlib import Path

ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.doc', '.docx'}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

async def validate_upload(file: UploadFile):
    # Check file extension
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Invalid file type: {ext}")
    
    # Check file size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large (max 100MB)")
    
    # Verify MIME type (prevent extension spoofing)
    mime_type = magic.from_buffer(content, mime=True)
    if mime_type not in ['application/pdf', 'text/plain']:
        raise HTTPException(400, f"Invalid file content: {mime_type}")
    
    await file.seek(0)  # Reset for further processing
    return content
```

### XSS Prevention

```python
from html import escape

# Always escape user input
def render_content(user_input: str):
    return escape(user_input)

# Example
user_input = "<script>alert('XSS')</script>"
safe_output = escape(user_input)
# Result: "&lt;script&gt;alert('XSS')&lt;/script&gt;"
```

---

## Database Security

### Password Hashing

**File**: `backend/utils/auth.py`

```python
from passlib.context import CryptContext

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    """Hash password with bcrypt"""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return pwd_context.verify(plain_password, hashed_password)
```

### Database Connection

```python
# Use SSL for production
DATABASE_URL = "postgresql://user:pass@host:5432/db?sslmode=require"

# Connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,  # Check connections before use
    pool_recycle=3600     # Recycle connections hourly
)
```

### Sensitive Data Encryption

```python
from cryptography.fernet import Fernet

# Generate key (store in environment)
key = Fernet.generate_key()
cipher = Fernet(key)

# Encrypt API keys before storing
encrypted_key = cipher.encrypt(api_key.encode())
user.encrypted_api_key = encrypted_key

# Decrypt when needed
decrypted_key = cipher.decrypt(user.encrypted_api_key).decode()
```

---

## Best Practices

### 1. Never Hardcode Secrets

```python
# ❌ BAD
SECRET_KEY = "my-secret-key-123"
OPENAI_API_KEY = "sk-abc123..."

# ✅ GOOD
import os
SECRET_KEY = os.getenv("SECRET_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
```

### 2. Validate All Input

```python
# ❌ BAD
@router.post("/content")
async def create_content(data: dict):
    # No validation!
    content = Content(**data)

# ✅ GOOD
@router.post("/content")
async def create_content(data: ContentCreate):
    # Pydantic validates automatically
    content = Content(**data.dict())
```

### 3. Use HTTPS in Production

```bash
# Nginx configuration
server {
    listen 443 ssl http2;
    server_name shikshasetu.com;
    
    ssl_certificate /path/to/fullchain.pem;
    ssl_certificate_key /path/to/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 4. Implement Logging & Monitoring

```python
import logging

logger = logging.getLogger(__name__)

@router.post("/auth/login")
async def login(credentials: LoginForm):
    try:
        user = authenticate(credentials)
        logger.info(f"Successful login: {user.email}")
        return {"access_token": create_token(user)}
    except AuthenticationError:
        logger.warning(f"Failed login attempt: {credentials.email}")
        raise HTTPException(401, "Invalid credentials")
```

### 5. Regular Security Audits

```bash
# Check for vulnerabilities
pip install safety
safety check --json

# Check for outdated packages
pip list --outdated

# Run security tests
pytest tests/security/ -v
```

---

## Security Checklist

- [ ] All secrets in `.env` (not committed)
- [ ] `.env` has `chmod 600` permissions
- [ ] JWT `SECRET_KEY` is 32+ characters
- [ ] HTTPS enabled in production
- [ ] CORS configured with specific origins
- [ ] Rate limiting enabled on all endpoints
- [ ] All passwords hashed with bcrypt
- [ ] SQL queries use parameterization
- [ ] File uploads validated (size, type, MIME)
- [ ] User input escaped/sanitized
- [ ] Security headers configured
- [ ] Database connections use SSL
- [ ] API keys encrypted at rest
- [ ] Logging configured (no sensitive data logged)
- [ ] Regular security audits scheduled

---

## Further Reading

- **[OWASP Top 10](https://owasp.org/www-project-top-ten/)** - Web application security risks
- **[JWT Best Practices](https://tools.ietf.org/html/rfc8725)** - JWT security considerations
- **[NIST Guidelines](https://www.nist.gov/cyberframework)** - Cybersecurity framework
- **[FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)** - Framework security features

---

## 👨‍💻 Author

**K Dhiraj** • [k.dhiraj.srihari@gmail.com](mailto:k.dhiraj.srihari@gmail.com) • [@KDhiraj152](https://github.com/KDhiraj152) • [LinkedIn](https://www.linkedin.com/in/k-dhiraj-83b025279/)

*Last updated: November 2025*
