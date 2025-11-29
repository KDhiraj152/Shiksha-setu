-- ShikshaSetu Database Schema
-- PostgreSQL 15+ with pgvector extension
-- This file creates ALL tables, indexes, and constraints
-- Safe to run multiple times (uses IF NOT EXISTS)

-- =============================================================================
-- EXTENSIONS
-- =============================================================================

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS vector;

-- =============================================================================
-- USERS & AUTHENTICATION
-- =============================================================================

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    hashed_password VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    organization VARCHAR(255),
    role VARCHAR(50) NOT NULL DEFAULT 'user',
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    is_verified BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_login TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);

-- API Keys for programmatic access
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    last_used TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_api_keys_user ON api_keys(user_id);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);

-- Token blacklist for logout/revocation
CREATE TABLE IF NOT EXISTS token_blacklist (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    token_jti VARCHAR(255) UNIQUE NOT NULL,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    reason VARCHAR(100) NOT NULL,
    blacklisted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_token_blacklist_jti ON token_blacklist(token_jti);
CREATE INDEX IF NOT EXISTS idx_token_blacklist_user ON token_blacklist(user_id);

-- Refresh tokens for rotation tracking
CREATE TABLE IF NOT EXISTS refresh_tokens (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    token_jti VARCHAR(255) UNIQUE NOT NULL,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    device_fingerprint VARCHAR(255),
    ip_address VARCHAR(50),
    user_agent TEXT,
    parent_jti VARCHAR(255),
    rotation_count INTEGER DEFAULT 0,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    last_used_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_refresh_tokens_jti ON refresh_tokens(token_jti);
CREATE INDEX IF NOT EXISTS idx_refresh_tokens_user ON refresh_tokens(user_id);

-- =============================================================================
-- CONTENT PROCESSING
-- =============================================================================

-- Main processed content table
CREATE TABLE IF NOT EXISTS processed_content (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    original_text TEXT NOT NULL,
    simplified_text TEXT,
    translated_text TEXT,
    language VARCHAR(50) NOT NULL,
    grade_level INTEGER NOT NULL CHECK (grade_level BETWEEN 5 AND 12),
    subject VARCHAR(100) NOT NULL,
    audio_file_path TEXT,
    ncert_alignment_score DECIMAL(4,3) CHECK (ncert_alignment_score BETWEEN 0 AND 1),
    audio_accuracy_score DECIMAL(4,3) CHECK (audio_accuracy_score BETWEEN 0 AND 1),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_content_user ON processed_content(user_id);
CREATE INDEX IF NOT EXISTS idx_content_language ON processed_content(language);
CREATE INDEX IF NOT EXISTS idx_content_grade ON processed_content(grade_level);
CREATE INDEX IF NOT EXISTS idx_content_subject ON processed_content(subject);
CREATE INDEX IF NOT EXISTS idx_content_created ON processed_content(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_content_user_created ON processed_content(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_content_grade_subject ON processed_content(grade_level, subject);

-- Content translations (normalized)
CREATE TABLE IF NOT EXISTS content_translations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_id UUID NOT NULL REFERENCES processed_content(id) ON DELETE CASCADE,
    language VARCHAR(50) NOT NULL,
    translated_text TEXT NOT NULL,
    translation_model VARCHAR(100),
    translation_quality_score DECIMAL(4,3),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(content_id, language)
);

CREATE INDEX IF NOT EXISTS idx_translations_content ON content_translations(content_id);
CREATE INDEX IF NOT EXISTS idx_translations_language ON content_translations(language);

-- Content audio files (normalized)
CREATE TABLE IF NOT EXISTS content_audio (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_id UUID NOT NULL REFERENCES processed_content(id) ON DELETE CASCADE,
    language VARCHAR(50) NOT NULL,
    audio_file_path TEXT NOT NULL,
    audio_format VARCHAR(20),
    duration_seconds DECIMAL(10,2),
    tts_model VARCHAR(100),
    accuracy_score DECIMAL(4,3),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(content_id, language)
);

CREATE INDEX IF NOT EXISTS idx_audio_content ON content_audio(content_id);

-- Content validation results
CREATE TABLE IF NOT EXISTS content_validation (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_id UUID NOT NULL REFERENCES processed_content(id) ON DELETE CASCADE,
    validation_type VARCHAR(50) NOT NULL,
    alignment_score DECIMAL(4,3) NOT NULL,
    passed BOOLEAN NOT NULL,
    issues_found JSONB DEFAULT '[]',
    validated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_validation_content ON content_validation(content_id);
CREATE INDEX IF NOT EXISTS idx_validation_type ON content_validation(validation_type);

-- =============================================================================
-- NCERT STANDARDS REFERENCE
-- =============================================================================

CREATE TABLE IF NOT EXISTS ncert_standards (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    grade_level INTEGER NOT NULL CHECK (grade_level BETWEEN 1 AND 12),
    subject VARCHAR(100) NOT NULL,
    topic VARCHAR(200) NOT NULL,
    description TEXT,
    learning_objectives TEXT[],
    keywords TEXT[],
    chapter_number INTEGER,
    unit_number INTEGER,
    embedding VECTOR(1024),  -- For semantic search
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ncert_grade ON ncert_standards(grade_level);
CREATE INDEX IF NOT EXISTS idx_ncert_subject ON ncert_standards(subject);
CREATE INDEX IF NOT EXISTS idx_ncert_grade_subject ON ncert_standards(grade_level, subject);

-- Vector index for semantic search (if pgvector available)
-- Using ivfflat for approximate nearest neighbor search
CREATE INDEX IF NOT EXISTS idx_ncert_embedding ON ncert_standards 
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- =============================================================================
-- FEEDBACK & REVIEWS
-- =============================================================================

CREATE TABLE IF NOT EXISTS feedback (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_id UUID NOT NULL REFERENCES processed_content(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
    feedback_text TEXT,
    issue_type VARCHAR(100),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_feedback_content ON feedback(content_id);
CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback(user_id);

-- Translation reviews (expert review workflow)
CREATE TABLE IF NOT EXISTS translation_reviews (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_id UUID NOT NULL REFERENCES processed_content(id) ON DELETE CASCADE,
    reviewer_id UUID REFERENCES users(id) ON DELETE SET NULL,
    language VARCHAR(50) NOT NULL,
    original_translation TEXT NOT NULL,
    corrected_translation TEXT,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    accuracy_score DECIMAL(4,3),
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    reviewed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_reviews_content ON translation_reviews(content_id);
CREATE INDEX IF NOT EXISTS idx_reviews_status ON translation_reviews(status);

-- Review comments
CREATE TABLE IF NOT EXISTS review_comments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    review_id UUID NOT NULL REFERENCES translation_reviews(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    comment_text TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Review versions (track changes)
CREATE TABLE IF NOT EXISTS review_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    review_id UUID NOT NULL REFERENCES translation_reviews(id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL,
    translated_text TEXT NOT NULL,
    changed_by UUID REFERENCES users(id) ON DELETE SET NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- STUDENT PROGRESS TRACKING
-- =============================================================================

CREATE TABLE IF NOT EXISTS student_progress (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content_id UUID REFERENCES processed_content(id) ON DELETE SET NULL,
    subject VARCHAR(100) NOT NULL,
    grade_level INTEGER NOT NULL,
    completion_percentage DECIMAL(5,2) DEFAULT 0,
    time_spent_seconds INTEGER DEFAULT 0,
    last_accessed TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_progress_user ON student_progress(user_id);
CREATE INDEX IF NOT EXISTS idx_progress_subject ON student_progress(subject);

CREATE TABLE IF NOT EXISTS quiz_scores (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    content_id UUID REFERENCES processed_content(id) ON DELETE SET NULL,
    subject VARCHAR(100) NOT NULL,
    grade_level INTEGER NOT NULL,
    score DECIMAL(5,2) NOT NULL,
    max_score DECIMAL(5,2) NOT NULL,
    time_taken_seconds INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_quiz_user ON quiz_scores(user_id);

CREATE TABLE IF NOT EXISTS learning_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    duration_seconds INTEGER,
    activities JSONB DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_sessions_user ON learning_sessions(user_id);

CREATE TABLE IF NOT EXISTS achievements (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    achievement_type VARCHAR(100) NOT NULL,
    achievement_name VARCHAR(200) NOT NULL,
    description TEXT,
    earned_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_achievements_user ON achievements(user_id);

-- Parent reports
CREATE TABLE IF NOT EXISTS parent_reports (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    student_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    parent_email VARCHAR(255),
    report_period_start DATE NOT NULL,
    report_period_end DATE NOT NULL,
    report_data JSONB NOT NULL,
    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- RAG / Q&A SYSTEM
-- =============================================================================

-- Document chunks for RAG
CREATE TABLE IF NOT EXISTS document_chunks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_id UUID REFERENCES processed_content(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(content_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_chunks_content ON document_chunks(content_id);

-- Embeddings for semantic search
CREATE TABLE IF NOT EXISTS embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    chunk_id UUID NOT NULL REFERENCES document_chunks(id) ON DELETE CASCADE,
    embedding VECTOR(1024),  -- E5-Large dimension
    model_id VARCHAR(100) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Vector index for semantic search
CREATE INDEX IF NOT EXISTS idx_embeddings_vector ON embeddings 
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Chat history for Q&A
CREATE TABLE IF NOT EXISTS chat_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    content_id UUID REFERENCES processed_content(id) ON DELETE SET NULL,
    session_id VARCHAR(100),
    role VARCHAR(20) NOT NULL,  -- 'user' or 'assistant'
    message TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_user ON chat_history(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_session ON chat_history(session_id);
CREATE INDEX IF NOT EXISTS idx_chat_content ON chat_history(content_id);

-- =============================================================================
-- PIPELINE LOGS & METRICS
-- =============================================================================

CREATE TABLE IF NOT EXISTS pipeline_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    content_id UUID REFERENCES processed_content(id) ON DELETE SET NULL,
    stage VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    processing_time_ms INTEGER,
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pipeline_content ON pipeline_logs(content_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_stage ON pipeline_logs(stage);
CREATE INDEX IF NOT EXISTS idx_pipeline_status ON pipeline_logs(status);
CREATE INDEX IF NOT EXISTS idx_pipeline_timestamp ON pipeline_logs(timestamp DESC);

-- =============================================================================
-- SEED DATA: Default Users
-- =============================================================================

-- Insert default users (only if they don't exist)
INSERT INTO users (email, hashed_password, full_name, role, is_active, is_verified)
SELECT 'test@shiksha.com', 
       '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/X4.O4bJwH3x5q5rLe',  -- Test@1234567
       'Test Student',
       'user',
       TRUE,
       TRUE
WHERE NOT EXISTS (SELECT 1 FROM users WHERE email = 'test@shiksha.com');

INSERT INTO users (email, hashed_password, full_name, role, is_active, is_verified)
SELECT 'teacher@shiksha.com',
       '$2b$12$92IXUNpkjO0rOQ5byMi.Ye4oKoEa3Ro9llC/.og/at2.uheWG/igi',  -- Teacher@123456
       'Test Teacher',
       'educator',
       TRUE,
       TRUE
WHERE NOT EXISTS (SELECT 1 FROM users WHERE email = 'teacher@shiksha.com');

INSERT INTO users (email, hashed_password, full_name, role, is_active, is_verified)
SELECT 'admin@shiksha.com',
       '$2b$12$8u8tK0EJ6VHx1QqH5VKzXOcUcJKxMVhHjY8dWmYKaGvJ8qDXPZxvi',  -- Admin@123456
       'Admin User',
       'admin',
       TRUE,
       TRUE
WHERE NOT EXISTS (SELECT 1 FROM users WHERE email = 'admin@shiksha.com');

-- =============================================================================
-- HELPER FUNCTIONS
-- =============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger for users table
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Verification
SELECT 'Database schema initialized successfully' as status;
SELECT COUNT(*) as user_count FROM users;
