-- ShikshaSetu Database Initialization
-- PostgreSQL 15+ with pgvector extension

-- Enable pgvector extension for vector embeddings
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify installation
SELECT 'pgvector extension installed successfully' as status;
