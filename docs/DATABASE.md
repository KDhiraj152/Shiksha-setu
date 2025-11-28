# Database Architecture

**ShikshaSetu Database Documentation**

Complete reference for database schema, migrations, and vector search setup.

---

## Overview

- **Database**: PostgreSQL 17+
- **ORM**: SQLAlchemy 2.0
- **Migrations**: Alembic
- **Vector Extension**: pgvector (for RAG/Q&A)
- **Location**: `backend/models.py`, `database/migrations/`

---

## Schema Overview

### Core Tables (25+ tables)

#### 1. Content & Processing

**`processed_content`** - Main content storage
```sql
id                      UUID PRIMARY KEY
original_text           TEXT NOT NULL
simplified_text         TEXT
translated_text         TEXT
language                VARCHAR(50) NOT NULL
grade_level             INTEGER NOT NULL
subject                 VARCHAR(100) NOT NULL
audio_file_path         TEXT
ncert_alignment_score   FLOAT
audio_accuracy_score    FLOAT
created_at              TIMESTAMP
metadata                JSONB
user_id                 UUID
```

**Indexes**:
- `idx_user_content` - (user_id, created_at)
- `idx_grade_subject` - (grade_level, subject)
- `idx_language_grade` - (language, grade_level)
- `idx_subject_created` - (subject, created_at)

**`content_translations`** - Normalized translations
```sql
id                          UUID PRIMARY KEY
content_id                  UUID FOREIGN KEY → processed_content
language                    VARCHAR(50)
translated_text             TEXT
translation_model           VARCHAR(100)
translation_quality_score   FLOAT
created_at                  TIMESTAMP
```

**`content_audio`** - Audio files
```sql
id                  UUID PRIMARY KEY
content_id          UUID FOREIGN KEY → processed_content
language            VARCHAR(50)
audio_file_path     TEXT
audio_format        VARCHAR(20)
duration_seconds    FLOAT
tts_model           VARCHAR(100)
accuracy_score      FLOAT
created_at          TIMESTAMP
```

**`content_validation`** - Validation results
```sql
id                  UUID PRIMARY KEY
content_id          UUID FOREIGN KEY → processed_content
validation_type     VARCHAR(50)  -- 'ncert', 'script', 'factual'
alignment_score     FLOAT
passed              BOOLEAN
issues_found        JSONB
validated_at        TIMESTAMP
```

#### 2. RAG & Q&A System

**`document_chunks`** - Text chunks for vector search
```sql
id              UUID PRIMARY KEY
content_id      UUID FOREIGN KEY → processed_content
chunk_index     INTEGER NOT NULL
chunk_text      TEXT NOT NULL
chunk_size      INTEGER
chunk_metadata  JSONB  -- page number, section, etc.
created_at      TIMESTAMP
```

**`embeddings`** - Vector embeddings (pgvector)
```sql
id                  UUID PRIMARY KEY
chunk_id            UUID FOREIGN KEY → document_chunks
content_id          UUID FOREIGN KEY → processed_content
embedding           VECTOR(1024)  -- multilingual-e5-large dimension
embedding_model     VARCHAR(100) DEFAULT 'intfloat/multilingual-e5-large'
embedding_version   INTEGER DEFAULT 2
created_at          TIMESTAMP
```

**Indexes**:
- HNSW index on `embedding` for fast k-NN search
- IVFFlat index for larger datasets (alternative)

**`chat_history`** - Q&A conversation history
```sql
id                  UUID PRIMARY KEY
user_id             UUID FOREIGN KEY → users
content_id          UUID FOREIGN KEY → processed_content
question            TEXT NOT NULL
answer              TEXT NOT NULL
context_chunks      UUID[]  -- Array of chunk IDs used
confidence_score    FLOAT
created_at          TIMESTAMP
```

#### 3. User & Authentication

**`users`** - User accounts
```sql
id                  UUID PRIMARY KEY
email               VARCHAR(255) UNIQUE NOT NULL
hashed_password     VARCHAR(255) NOT NULL
full_name           VARCHAR(255)
role                VARCHAR(50) DEFAULT 'user'
is_active           BOOLEAN DEFAULT true
created_at          TIMESTAMP
updated_at          TIMESTAMP
```

**`refresh_tokens`** - JWT refresh tokens
```sql
id              UUID PRIMARY KEY
user_id         UUID FOREIGN KEY → users
token           VARCHAR(500) UNIQUE
expires_at      TIMESTAMP
created_at      TIMESTAMP
revoked         BOOLEAN DEFAULT false
```

#### 4. Feedback & Monitoring

**`feedback`** - User feedback
```sql
id              UUID PRIMARY KEY
content_id      UUID FOREIGN KEY → processed_content
user_id         UUID FOREIGN KEY → users
rating          INTEGER  -- 1-5 stars
feedback_text   TEXT
issue_type      VARCHAR(100)
created_at      TIMESTAMP
```

**`pipeline_logs`** - Processing logs
```sql
id                      UUID PRIMARY KEY
content_id              UUID FOREIGN KEY → processed_content
stage                   VARCHAR(50)  -- 'ocr', 'simplify', 'translate', etc.
status                  VARCHAR(20)  -- 'started', 'completed', 'failed'
processing_time_ms      INTEGER
error_message           TEXT
timestamp               TIMESTAMP
```

#### 5. Progress & Gamification

**`student_progress`** - Learning progress tracking
```sql
id                  INTEGER PRIMARY KEY
user_id             UUID FOREIGN KEY → users
content_id          UUID FOREIGN KEY → processed_content
progress_percent    FLOAT DEFAULT 0.0
time_spent_seconds  INTEGER
last_accessed       TIMESTAMP
completed           BOOLEAN DEFAULT false
```

**`quiz_scores`** - Quiz results
```sql
id              INTEGER PRIMARY KEY
user_id         UUID FOREIGN KEY → users
content_id      UUID FOREIGN KEY → processed_content
score           INTEGER
max_score       INTEGER
completed_at    TIMESTAMP
```

**`achievements`** - Gamification achievements
```sql
id              INTEGER PRIMARY KEY
user_id         UUID FOREIGN KEY → users
achievement_id  VARCHAR(100)
earned_at       TIMESTAMP
```

#### 6. Advanced Features

**`translation_reviews`** - Translation review workflow
```sql
id                  UUID PRIMARY KEY
content_id          UUID FOREIGN KEY → processed_content
original_text       TEXT
translated_text     TEXT
status              VARCHAR(50)  -- 'pending', 'approved', 'rejected'
reviewer_id         UUID FOREIGN KEY → users
created_at          TIMESTAMP
```

**`review_comments`** - Review feedback
```sql
id          UUID PRIMARY KEY
review_id   UUID FOREIGN KEY → translation_reviews
user_id     UUID FOREIGN KEY → users
comment     TEXT
position    INTEGER
resolved    BOOLEAN DEFAULT false
created_at  TIMESTAMP
```

**`ab_test_experiments`** - A/B testing
```sql
id                  UUID PRIMARY KEY
name                VARCHAR(255) UNIQUE
description         TEXT
status              VARCHAR(50)  -- 'draft', 'active', 'paused', 'completed'
traffic_allocation  FLOAT
start_date          TIMESTAMP
end_date            TIMESTAMP
```

**`learning_recommendations`** - Personalized recommendations
```sql
id                      INTEGER PRIMARY KEY
user_id                 UUID FOREIGN KEY → users
recommended_content_id  UUID FOREIGN KEY → processed_content
score                   FLOAT
reason                  VARCHAR(255)
created_at              TIMESTAMP
```

---

## Vector Database (pgvector)

### Overview

pgvector enables semantic search for RAG Q&A system.

### Setup

**Supabase (Recommended)**:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

**Local PostgreSQL**:
```bash
# macOS
brew install pgvector
psql -d shiksha_setu -c "CREATE EXTENSION vector;"

# Ubuntu
sudo apt install postgresql-server-dev-all
git clone https://github.com/pgvector/pgvector.git
cd pgvector && make && sudo make install
psql -d shiksha_setu -c "CREATE EXTENSION vector;"
```

**Docker**:
```yaml
services:
  postgres:
    image: pgvector/pgvector:pg17
    environment:
      POSTGRES_DB: shiksha_setu
```

### Vector Index Types

**HNSW (Hierarchical Navigable Small World)** - Recommended
```sql
CREATE INDEX embeddings_hnsw_idx 
ON embeddings 
USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

**IVFFlat** - Alternative for large datasets
```sql
CREATE INDEX embeddings_ivfflat_idx 
ON embeddings 
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

### RAG Query Flow

1. **Indexing**:
   - Document → Chunks (200-500 tokens, 50 token overlap)
   - Chunks → Embeddings (multilingual-e5-large, 1024-dim)
   - Store in `document_chunks` + `embeddings` tables

2. **Query**:
   - Question → Embedding
   - k-NN search in `embeddings` table
   - Retrieve top-N chunks
   - Generate answer with context

3. **API Endpoint**:
```python
# POST /api/v1/qa/ask
{
    "content_id": "uuid",
    "question": "What is photosynthesis?",
    "top_k": 5
}
```

### Vector Search Example

```sql
-- Find 5 most similar chunks
SELECT 
    dc.chunk_text,
    dc.chunk_metadata,
    1 - (e.embedding <=> query_embedding) AS similarity
FROM embeddings e
JOIN document_chunks dc ON e.chunk_id = dc.id
WHERE e.content_id = 'content-uuid'
ORDER BY e.embedding <=> query_embedding
LIMIT 5;
```

### Performance Optimization

**Index Configuration**:
- `m = 16` - Connections per layer (higher = better recall, slower build)
- `ef_construction = 64` - Search depth during construction
- `lists = 100` (IVFFlat) - Number of clusters

**Query Tuning**:
```sql
-- Adjust search quality
SET hnsw.ef_search = 40;  -- Higher = better accuracy, slower
```

**Scaling**:
- **< 1M vectors**: pgvector (HNSW) ✅
- **1M - 10M vectors**: pgvector (IVFFlat) or Qdrant
- **> 10M vectors**: Qdrant with sharding

---

## Migrations

### Alembic Configuration

**Location**: `database/migrations/`

**Key Migrations**:
- `001_initial_schema.py` - Base tables
- `002_add_feedback.py` - Feedback system
- `003_add_authentication.py` - Auth tables
- `007_enable_pgvector.py` - Vector extension
- `008_add_q_a_tables_for_rag_system.py` - RAG tables
- `009_add_hnsw_indexes.py` - Vector indexes
- `013_add_multi_tenancy.py` - Multi-tenant support

### Running Migrations

```bash
# Upgrade to latest
alembic upgrade head

# Downgrade one version
alembic downgrade -1

# Create new migration
alembic revision -m "description"

# Show current version
alembic current

# Show history
alembic history
```

### Migration Best Practices

1. **Concurrent Index Creation**:
```python
op.create_index(
    'idx_name',
    'table_name',
    ['column'],
    postgresql_concurrently=True
)
```

2. **Row Count Checks**:
```python
# Only create expensive indexes if enough data
connection = op.get_bind()
result = connection.execute("SELECT COUNT(*) FROM table")
if result.scalar() > 10000:
    op.create_index(...)
```

3. **Backfilling Data**:
```python
# Use batch updates for large tables
op.execute("""
    UPDATE table 
    SET new_column = default_value 
    WHERE new_column IS NULL
""")
```

---

## Backup & Recovery

### PostgreSQL Backup

**Full Backup**:
```bash
pg_dump -h localhost -U postgres -d shiksha_setu -F c -b -v -f backup.dump
```

**Restore**:
```bash
pg_restore -h localhost -U postgres -d shiksha_setu -v backup.dump
```

### Vector Data Backup

pgvector data is included in standard PostgreSQL backups.

**Point-in-Time Recovery**:
```bash
# Enable WAL archiving in postgresql.conf
archive_mode = on
archive_command = 'cp %p /archive/%f'
```

---

## Query Examples

### Content Search

```sql
-- Find content by grade and subject
SELECT * FROM processed_content
WHERE grade_level = 8 
  AND subject = 'Science'
  AND language = 'Hindi'
ORDER BY created_at DESC
LIMIT 10;
```

### User Progress

```sql
-- Get student progress summary
SELECT 
    u.full_name,
    COUNT(DISTINCT sp.content_id) AS items_started,
    SUM(CASE WHEN sp.completed THEN 1 ELSE 0 END) AS items_completed,
    AVG(sp.progress_percent) AS avg_progress
FROM users u
JOIN student_progress sp ON u.id = sp.user_id
WHERE u.id = 'user-uuid'
GROUP BY u.full_name;
```

### Content Quality

```sql
-- Find content needing improvement
SELECT 
    id,
    subject,
    grade_level,
    ncert_alignment_score,
    audio_accuracy_score
FROM processed_content
WHERE ncert_alignment_score < 0.7 
   OR audio_accuracy_score < 0.85
ORDER BY ncert_alignment_score ASC;
```

---

## Database Configuration

### Connection Pooling

**SQLAlchemy Settings**:
```python
engine = create_engine(
    DATABASE_URL,
    pool_size=10,          # Connections in pool
    max_overflow=20,       # Extra connections
    pool_timeout=30,       # Wait time for connection
    pool_recycle=3600,     # Recycle after 1 hour
    pool_pre_ping=True     # Test connections
)
```

### PostgreSQL Tuning

**For Development**:
```sql
-- postgresql.conf
shared_buffers = 256MB
work_mem = 16MB
maintenance_work_mem = 64MB
max_connections = 100
```

**For Production**:
```sql
-- postgresql.conf
shared_buffers = 4GB           # 25% of RAM
effective_cache_size = 12GB    # 75% of RAM
work_mem = 64MB
maintenance_work_mem = 1GB
max_connections = 200
```

---

## Monitoring

### Key Metrics

```sql
-- Table sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Index usage
SELECT 
    indexrelname AS index_name,
    idx_scan AS times_used,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- Slow queries
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    max_time
FROM pg_stat_statements
ORDER BY mean_time DESC
LIMIT 10;
```

---

## Related Documentation

- **[API.md](api.md)** - API endpoints using these tables
- **[BACKEND.md](backend.md)** - Backend services and ORM usage
- **[AI_PIPELINE.md](ai-ml-pipeline.md)** - Content processing pipeline
- **[DEPLOYMENT.md](deployment.md)** - Production database setup

---

## 👨‍💻 Made By

**K Dhiraj Srihari**

🔗 **Connect:**
- 📧 [k.dhiraj.srihari@gmail.com](mailto:k.dhiraj.srihari@gmail.com)
- 💼 [LinkedIn](https://linkedin.com/in/k-dhiraj)
- 🐙 [GitHub](https://github.com/KDhiraj152)
