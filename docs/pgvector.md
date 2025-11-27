# pgvector Setup

Vector database for RAG Q&A system.

## What is pgvector?

PostgreSQL extension for vector similarity search. Used for:
- Semantic search
- Question-Answer matching
- Content recommendations

## Quick Setup

### Supabase (Recommended)

**Already included!** Just enable it:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Or via Dashboard: Database → Extensions → Enable "vector"

### Local PostgreSQL

**macOS:**
```bash
brew install pgvector
psql -d education_content -c "CREATE EXTENSION vector;"
```

**Ubuntu/Debian:**
```bash
sudo apt install postgresql-server-dev-all build-essential
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make && sudo make install
sudo -u postgres psql -d education_content -c "CREATE EXTENSION vector;"
```

**Docker:**
```yaml
services:
  postgres:
    image: pgvector/pgvector:pg17
    environment:
      POSTGRES_DB: education_content
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
```

## Verify Installation

```sql
SELECT * FROM pg_extension WHERE extname = 'vector';
```

## Usage

The extension is used automatically by the RAG system for:
- Document embeddings storage
- Similarity search
- Q&A retrieval

No additional configuration needed!

## Troubleshooting

**Extension not found?**
```bash
# Restart PostgreSQL
brew services restart postgresql  # macOS
sudo systemctl restart postgresql # Linux
```

**Permission denied?**
```bash
sudo make install  # Run with sudo
```

## Resources

- [pgvector GitHub](https://github.com/pgvector/pgvector)
- [Supabase Vectors](https://supabase.com/docs/guides/database/extensions/pgvector)

### Option 3: AWS RDS

pgvector is available on AWS RDS PostgreSQL 13+:

```sql
-- Connect to your RDS instance
CREATE EXTENSION vector;
```

## Database Configuration

### 1. Enable the Extension

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify installation
SELECT * FROM pg_extension WHERE extname = 'vector';
```

### 2. Update ShikshaSetu Schema

The Q&A tables already support pgvector. Verify they have the correct structure:

```sql
-- Check questions table
\d questions

-- Should have:
-- question_embedding vector(384)  -- For multilingual embeddings
```

If you need to add vector columns to existing tables:

```sql
-- Add vector column (384 dimensions for sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2)
ALTER TABLE questions 
ADD COLUMN IF NOT EXISTS question_embedding vector(384);

-- Add index for faster similarity search
CREATE INDEX IF NOT EXISTS questions_embedding_idx 
ON questions USING ivfflat (question_embedding vector_cosine_ops)
WITH (lists = 100);
```

### 3. Optimize Vector Search

```sql
-- Create index for cosine similarity (recommended for normalized embeddings)
CREATE INDEX questions_embedding_cosine_idx 
ON questions USING ivfflat (question_embedding vector_cosine_ops)
WITH (lists = 100);

-- Or create index for L2 distance
CREATE INDEX questions_embedding_l2_idx 
ON questions USING ivfflat (question_embedding vector_l2_ops)
WITH (lists = 100);

-- Or create index for inner product
CREATE INDEX questions_embedding_ip_idx 
ON questions USING ivfflat (question_embedding vector_ip_ops)
WITH (lists = 100);
```

**Index Configuration:**
- `lists`: Number of inverted lists (typically sqrt of number of rows)
- For < 1M rows: `lists = 100`
- For 1M-10M rows: `lists = 1000`
- For > 10M rows: `lists = 10000`

## ShikshaSetu Integration

### 1. Update Environment Configuration

Verify your `.env` file has the correct database URL:

> **Security Note:** Do not use plaintext credentials in documentation — replace with placeholders or environment variables in examples. Never commit actual passwords to version control.

```bash
# For Supabase
DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@db.your-project.supabase.co:5432/postgres

# For local PostgreSQL
DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@localhost:5432/education_content
```

### 2. Update Python Code

The code in `src/services/qa_service.py` already supports pgvector. Verify it's configured correctly:

```python
from sentence_transformers import SentenceTransformer
from sqlalchemy import text

# Initialize embedding model
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Generate embedding
def generate_embedding(text: str) -> list:
    return model.encode(text).tolist()

# Similarity search
def search_similar_questions(query: str, limit: int = 5):
    query_embedding = generate_embedding(query)
    
    sql = text("""
        SELECT id, question_text, question_embedding <=> :query_embedding AS distance
        FROM questions
        WHERE question_embedding IS NOT NULL
        ORDER BY question_embedding <=> :query_embedding
        LIMIT :limit
    """)
    
    result = db.execute(sql, {
        "query_embedding": str(query_embedding),
        "limit": limit
    })
    return result.fetchall()
```

### 3. Run Migration

The pgvector extension is enabled via migration 007_enable_pgvector:

```bash
# Run migrations
cd /Users/kdhiraj_152/Downloads/shiksha_setu
source .venv/bin/activate
alembic upgrade head
```

**IVFFLAT Index Creation:**

Migration 007 includes smart IVFFLAT index creation that:
- Only creates indexes when table has >= 1000 rows (for proper clustering)
- Supports concurrent index creation: `alembic upgrade head -x concurrently=true`
- Automatically calculates optimal list count based on row count
- Provides clear feedback about index creation status

**For manual index creation after data ingestion:**

```sql
CREATE INDEX CONCURRENTLY questions_embedding_cosine_idx
ON questions USING ivfflat (question_embedding vector_cosine_ops) WITH (lists = 100);
```

Edit the new migration file:

```python
def upgrade():
    # Enable pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    
    # Add indexes if they don't exist
    op.execute("""
        CREATE INDEX IF NOT EXISTS questions_embedding_cosine_idx 
        ON questions USING ivfflat (question_embedding vector_cosine_ops)
        WITH (lists = 100)
    """)

def downgrade():
    op.execute('DROP INDEX IF EXISTS questions_embedding_cosine_idx')
    op.execute('DROP EXTENSION IF EXISTS vector CASCADE')
```

Run migration:

```bash
alembic upgrade head
```

## Testing pgvector

### 1. Test in SQL

```sql
-- Insert test data with embeddings
INSERT INTO questions (question_text, question_embedding, subject, grade_level)
VALUES 
    ('What is photosynthesis?', '[0.1, 0.2, 0.3, ...]', 'science', 8),
    ('Explain gravity', '[0.15, 0.25, 0.35, ...]', 'science', 8);

-- Test similarity search (cosine distance)
SELECT question_text, question_embedding <=> '[0.12, 0.22, 0.32, ...]' AS distance
FROM questions
ORDER BY question_embedding <=> '[0.12, 0.22, 0.32, ...]'
LIMIT 5;

-- Test using L2 distance
SELECT question_text, question_embedding <-> '[0.12, 0.22, 0.32, ...]' AS distance
FROM questions
ORDER BY question_embedding <-> '[0.12, 0.22, 0.32, ...]'
LIMIT 5;

-- Test using inner product
SELECT question_text, question_embedding <#> '[0.12, 0.22, 0.32, ...]' AS distance
FROM questions
ORDER BY question_embedding <#> '[0.12, 0.22, 0.32, ...]'
LIMIT 5;
```

### 2. Test via Python

```python
# test_pgvector.py
import os
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# Connect to database
engine = create_engine(os.getenv("DATABASE_URL"))
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Test query
query = "What is photosynthesis?"
query_embedding = model.encode(query).tolist()

with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT question_text, 
               question_embedding <=> :embedding AS distance
        FROM questions
        WHERE question_embedding IS NOT NULL
        ORDER BY question_embedding <=> :embedding
        LIMIT 5
    """), {"embedding": str(query_embedding)})
    
    for row in result:
        print(f"Question: {row[0]}, Distance: {row[1]}")
```

Run test:

```bash
python test_pgvector.py
```

### 3. Test via API

```bash
# Generate embeddings for existing questions
curl -X POST http://localhost:8000/api/v1/qa/generate-embeddings

# Search similar questions
curl -X POST http://localhost:8000/api/v1/qa/search \
  -H "Content-Type: application/json" \
  -d '{"query": "What is photosynthesis?", "limit": 5}'
```

## Performance Optimization

### 1. Tune IVFFLAT Index

```sql
-- Adjust lists parameter based on data size
-- Rule of thumb: lists = sqrt(total_rows)

-- For 10,000 rows
CREATE INDEX questions_embedding_idx 
ON questions USING ivfflat (question_embedding vector_cosine_ops)
WITH (lists = 100);

-- For 1,000,000 rows
CREATE INDEX questions_embedding_idx 
ON questions USING ivfflat (question_embedding vector_cosine_ops)
WITH (lists = 1000);
```

### 2. Optimize Query Parameters

```sql
-- Adjust probes for query-time accuracy vs speed tradeoff
SET ivfflat.probes = 10;  -- Higher = more accurate but slower

-- Test different probe values
SET ivfflat.probes = 1;   -- Fast, less accurate
SET ivfflat.probes = 10;  -- Balanced (default)
SET ivfflat.probes = 50;  -- Slow, more accurate
```

### 3. Batch Embedding Generation

```python
# Generate embeddings in batches for better performance
from sqlalchemy import text

def generate_embeddings_batch(batch_size=100):
    # Get questions without embeddings
    query = text("""
        SELECT id, question_text 
        FROM questions 
        WHERE question_embedding IS NULL
        LIMIT :batch_size
    """)
    
    with engine.connect() as conn:
        while True:
            rows = conn.execute(query, {"batch_size": batch_size}).fetchall()
            if not rows:
                break
            
            # Generate embeddings for batch
            texts = [row[1] for row in rows]
            embeddings = model.encode(texts)
            
            # Update database
            for (id, _), embedding in zip(rows, embeddings):
                update = text("""
                    UPDATE questions 
                    SET question_embedding = :embedding 
                    WHERE id = :id
                """)
                conn.execute(update, {
                    "id": id,
                    "embedding": embedding.tolist()
                })
            
            conn.commit()
```

## Troubleshooting

### Error: "extension vector does not exist"

**Solution:**
```sql
CREATE EXTENSION vector;
```

If that fails, pgvector is not installed. Follow installation steps above.

### Error: "vector type doesn't exist"

**Solution:** The vector extension is not enabled. Run:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### Slow Query Performance

**Solutions:**
1. Create/rebuild index:
   ```sql
   CREATE INDEX CONCURRENTLY questions_embedding_idx 
   ON questions USING ivfflat (question_embedding vector_cosine_ops)
   WITH (lists = 100);
   ```

2. Increase maintenance_work_mem:
   ```sql
   SET maintenance_work_mem = '1GB';
   ```

3. Adjust probes:
   ```sql
   SET ivfflat.probes = 10;
   ```

### Dimension Mismatch

**Error:** "expected 384 dimensions, got 768"

**Solution:** Verify embedding model dimensions match database:
```python
# Check model dimensions
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
dimensions = model.get_sentence_embedding_dimension()
print(f"Model dimensions: {dimensions}")  # Should be 384

# Update table if needed
ALTER TABLE questions ALTER COLUMN question_embedding TYPE vector(384);
```

## Production Checklist

- [ ] pgvector extension enabled in database
- [ ] Vector columns added to questions table
- [ ] IVFFLAT indexes created with appropriate `lists` parameter
- [ ] Embeddings generated for existing questions
- [ ] API endpoints tested for similarity search
- [ ] Query performance optimized (check with EXPLAIN ANALYZE)
- [ ] Monitoring configured for vector search queries
- [ ] Backup strategy includes vector columns
- [ ] Documentation updated for API users

## Resources

- pgvector GitHub: https://github.com/pgvector/pgvector
- Supabase pgvector Guide: https://supabase.com/docs/guides/database/extensions/pgvector
- Sentence Transformers: https://www.sbert.net/
- PostgreSQL IVFFLAT Index: https://github.com/pgvector/pgvector#ivfflat
