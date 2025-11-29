# Comprehensive Test Suite

This directory contains **comprehensive tests** for the ShikshaSetu platform with the **new AI tech stack**.

## 🚀 Quick Start

### Run All Tests
```bash
./scripts/run_tests.sh
```

### Run Specific Tests
```bash
# Unit tests (fast)
pytest tests/unit/ -v -m "not slow"

# AI stack tests
pytest tests/unit/test_new_ai_stack.py -v
pytest tests/integration/test_new_ai_stack_integration.py -v

# Integration tests
pytest tests/integration/ -v -m "integration"

# E2E tests
pytest tests/e2e/ -v -m "e2e"

# Performance tests
pytest tests/performance/ -v -m "performance"

# With coverage
pytest tests/ --cov=backend --cov-report=html
```

### Prerequisites
- PostgreSQL running with `shiksha_setu_test` database
- Redis running on port 6379
- Ollama running on port 11434 (optional, for full AI tests)

---

## 🆕 New AI Tech Stack Tests

### Unit Tests
- **`unit/test_new_ai_stack.py`** - AI service unit tests (400+ lines)
  - NLLB-200 translator
  - Ollama simplifier
  - Edge TTS generator
  - BGE-M3 embeddings
  - Memory manager
  - AI orchestrator

### Integration Tests
- **`integration/test_new_ai_stack_integration.py`** - AI integration tests (500+ lines)
  - Translation, simplification, TTS workflows
  - RAG with BGE-M3 embeddings
  - Celery task execution
  - Redis caching
  - Memory management
  - Full pipelines

- **`integration/test_celery_redis.py`** - Celery/Redis tests (300+ lines)
  - Task queueing and states
  - Result caching
  - Cache invalidation
  - Task monitoring

- **`integration/test_rag_qa.py`** - RAG Q&A tests (400+ lines)
  - Document processing
  - BGE-M3 embeddings
  - Vector search with pgvector
  - Question answering
  - Conversation history

### E2E Tests
- **`e2e/test_ai_pipeline_e2e.py`** - End-to-end workflows (200+ lines)
  - Content upload → simplify → translate → audio
  - RAG document ingestion and querying
  - Health monitoring
  - Error handling

### Performance Tests
- **`performance/test_ai_performance.py`** - Performance benchmarks (250+ lines)
  - Memory management (10GB budget)
  - Latency benchmarks (<20s simplification, <5s translation)
  - Throughput (20 concurrent requests)
  - Cache speedup

---

## Original Test Coverage

### End-to-End Integration Tests (`test_end_to_end_integration.py`)

Covers all requirements from Task 11.1:

1. **Full Pipeline Processing**
   - Tests complete flow: Input → Simplification → Translation → Validation → Speech → Storage → Retrieval
   - Verifies all pipeline stages execute successfully
   - Tracks metrics and performance

2. **NCERT Alignment Verification** (Requirement 3.2)
   - Tests that NCERT alignment scores meet ≥80% threshold
   - Validates curriculum standards compliance
   - Ensures educational content quality

3. **Audio Accuracy Verification** (Requirement 4.3)
   - Tests that audio accuracy scores meet ≥90% threshold
   - Validates text-to-speech quality
   - Ensures pronunciation accuracy

4. **Multi-Language Support** (Requirement 1.4)
   - Tests all 5 MVP languages: Hindi, Tamil, Telugu, Bengali, Marathi
   - Verifies correct Unicode script rendering
   - Validates translation quality for each language

5. **Offline Functionality** (Requirement 7.4)
   - Tests content caching and offline access
   - Validates batch download package creation
   - Ensures package sizes are reasonable for low-bandwidth scenarios

6. **2G Network Performance** (Requirement 7.4)
   - Simulates 2G network conditions
   - Verifies content loads within 5 seconds
   - Tests text-only mode for faster loading

## Prerequisites

### 1. PostgreSQL Database

The integration tests require a PostgreSQL database with the following setup:

```bash
# Install PostgreSQL (if not already installed)
# On Ubuntu/Debian:
sudo apt-get install postgresql postgresql-contrib

# On macOS:
brew install postgresql

# On Windows:
# Download and install from https://www.postgresql.org/download/windows/

# Create test database
createdb test_education_content

# Or using psql:
psql -U postgres
CREATE DATABASE test_education_content;
```

### 2. Environment Variables

Set the following environment variables:

```bash
# Database connection
export DATABASE_URL="postgresql://user:password@localhost:5432/test_education_content"

# Hugging Face API key (for translation and speech generation)
export HUGGINGFACE_API_KEY="your_api_key_here"

# Optional: Enable SQL query logging
export SQL_ECHO="false"
```

### 3. Python Dependencies

Install all required dependencies:

```bash
pip install -r requirements/base.txt
```

## Running the Tests

### Run All Integration Tests

```bash
pytest tests/test_end_to_end_integration.py -v -s
```

### Run Specific Test

```bash
# Test NCERT alignment threshold
pytest tests/test_end_to_end_integration.py::TestEndToEndIntegration::test_ncert_alignment_threshold -v -s

# Test audio accuracy threshold
pytest tests/test_end_to_end_integration.py::TestEndToEndIntegration::test_audio_accuracy_threshold -v -s

# Test all MVP languages
pytest tests/test_end_to_end_integration.py::TestEndToEndIntegration::test_all_mvp_languages -v -s

# Test offline functionality
pytest tests/test_end_to_end_integration.py::TestEndToEndIntegration::test_offline_functionality -v -s

# Test 2G network performance
pytest tests/test_end_to_end_integration.py::TestEndToEndIntegration::test_2g_network_performance -v -s

# Comprehensive test with all requirements
pytest tests/test_end_to_end_integration.py::TestEndToEndIntegration::test_full_pipeline_with_all_requirements -v -s
```

### Run with Coverage

```bash
pytest tests/test_end_to_end_integration.py --cov=src --cov-report=html -v
```

## Test Output

The tests provide detailed output including:

- Processing time for each pipeline stage
- Quality scores (NCERT alignment, audio accuracy)
- Content IDs for verification
- Package sizes for offline downloads
- Load times for performance tests

Example output:

```
✓ Complete pipeline flow test passed: content_id=123e4567-e89b-12d3-a456-426614174000
✓ NCERT alignment threshold test passed: 85.50%
✓ Audio accuracy threshold test passed: 92.30%
✓ All MVP languages test passed: 5 languages
  - Hindi: NCERT=85.50%, Audio=92.30%
  - Tamil: NCERT=83.20%, Audio=91.50%
  - Telugu: NCERT=84.70%, Audio=93.10%
  - Bengali: NCERT=82.90%, Audio=90.80%
  - Marathi: NCERT=86.10%, Audio=94.20%
✓ Offline functionality test passed: 2.5MB package
✓ 2G network performance test passed: 3.45s load time
```

## Troubleshooting

### Database Connection Errors

If you see database connection errors:

1. Verify PostgreSQL is running:
   ```bash
   sudo service postgresql status  # Linux
   brew services list              # macOS
   ```

2. Check DATABASE_URL is correct:
   ```bash
   echo $DATABASE_URL
   ```

3. Test connection manually:
   ```bash
   psql $DATABASE_URL
   ```

### Missing Dependencies

If you see import errors:

```bash
pip install -r requirements.txt
```

### API Key Issues

If you see Hugging Face API errors:

1. Verify API key is set:
   ```bash
   echo $HUGGINGFACE_API_KEY
   ```

2. Get a free API key from: https://huggingface.co/settings/tokens

### Slow Tests

Integration tests may take several minutes to complete as they:
- Process real content through the full pipeline
- Generate audio files
- Test multiple languages
- Create offline packages

To run faster subset of tests:

```bash
# Run only parameter validation (fast)
pytest tests/test_end_to_end_integration.py::TestEndToEndIntegration::test_parameter_validation -v

# Run only text-only tests (no audio generation)
pytest tests/test_end_to_end_integration.py::TestEndToEndIntegration::test_text_only_output -v
```

## CI/CD Integration

For continuous integration, use the following configuration:

```yaml
# .github/workflows/integration-tests.yml
name: Integration Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_education_content
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -r requirements/base.txt
      
      - name: Run integration tests
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_education_content
          HUGGINGFACE_API_KEY: ${{ secrets.HUGGINGFACE_API_KEY }}
        run: pytest tests/test_end_to_end_integration.py -v
```

## Additional Tests

Other test files in this directory:

- `test_text_simplifier.py` - Unit tests for text simplification
- `test_speech_generator.py` - Unit tests for speech generation
- `test_orchestrator_basic.py` - Unit tests for pipeline orchestrator

Run all tests:

```bash
pytest tests/ -v
```

---

## 👨‍💻 Author

**K Dhiraj** • [k.dhiraj.srihari@gmail.com](mailto:k.dhiraj.srihari@gmail.com) • [@KDhiraj152](https://github.com/KDhiraj152) • [LinkedIn](https://www.linkedin.com/in/k-dhiraj-83b025279/)

*Last updated: November 2025*
