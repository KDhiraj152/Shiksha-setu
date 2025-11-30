# 🔧 Scripts Directory

Organized collection of utility scripts for Shiksha Setu setup, testing, and deployment.

---

## 📁 Directory Structure

```
scripts/
├── setup/              # Initial setup and configuration
├── validation/         # System validation and health checks  
├── testing/            # Test suites and verification
├── demo/               # Demo data and examples
├── deployment/         # Deployment automation
└── utils/              # General utilities
```

---

## 🚀 Setup Scripts

**Location**: `scripts/setup/`

### `init_db.py`
Initialize database schema and apply migrations.
```bash
python scripts/setup/init_db.py
```

### `setup_postgres.sh`
Automated PostgreSQL setup with pgvector extension.
```bash
./scripts/setup/setup_postgres.sh
```

### `download_models.py`
Download required AI/ML models (FLAN-T5, IndicTrans2, MMS-TTS).
```bash
python scripts/setup/download_models.py
```

### `install_optimal_models.sh`
Download optimal model sizes based on available resources.
```bash
./scripts/setup/install_optimal_models.sh
```

### `setup_huggingface_auth.sh`
Configure HuggingFace API authentication.
```bash
./scripts/setup/setup_huggingface_auth.sh
```

### `setup_complete.py`
Comprehensive setup verification and final configuration.
```bash
python scripts/setup/setup_complete.py
```

### `fix_migrations.py`
Fix database migration issues.
```bash
python scripts/setup/fix_migrations.py
```

---

## ✅ Validation Scripts

**Location**: `scripts/validation/`

### `validate_setup.py`
Comprehensive system validation (dependencies, services, configuration).
```bash
python scripts/validation/validate_setup.py
```

### `check_dependencies.py`
Check all Python dependencies and external services.
```bash
python scripts/validation/check_dependencies.py
```

**Checks**:
- Python version (3.11+)
- Required packages (FastAPI, Transformers, etc.)
- External services (PostgreSQL, Redis)
- Environment configuration
- Model files

---

## 🧪 Testing Scripts

**Location**: `scripts/testing/`

### `test_all_features.py`
End-to-end feature testing (14 features).
```bash
python scripts/testing/test_all_features.py
```

**Tests**:
- Configuration & Model Loading
- Document Processing & Embeddings
- Translation (IndicTrans2)
- Simplification (FLAN-T5)
- Text-to-Speech (MMS-TTS)
- RAG Q&A System
- Pipeline Orchestration
- API Endpoints (24 routes)

### `test_setup.py`
Quick setup verification test.
```bash
python scripts/testing/test_setup.py
```

### `run_comprehensive_tests.py`
Full test suite with coverage report.
```bash
python scripts/testing/run_comprehensive_tests.py
```

### `verify_issues_9_11.py`
Verify specific issue fixes (#9, #11).
```bash
python scripts/testing/verify_issues_9_11.py
```

---

## 🎭 Demo Scripts

**Location**: `scripts/demo/`

### `create_demo_user.py`
Create demo user accounts for testing.
```bash
python scripts/demo/create_demo_user.py
```

**Creates**:
- Student user: `student@demo.com` / `demo123`
- Teacher user: `teacher@demo.com` / `demo123`
- Admin user: `admin@demo.com` / `admin123`

### `demo_aiml_pipeline.py`
Interactive AI/ML pipeline demonstration.
```bash
python scripts/demo/demo_aiml_pipeline.py
```

**Demonstrates**:
- Text simplification
- Multi-language translation
- Speech synthesis
- NCERT validation
- RAG-based Q&A

### `ncert_indexer.py`
Index NCERT content for RAG system.
```bash
python scripts/demo/ncert_indexer.py
```

---

## 🚢 Deployment Scripts

**Location**: `scripts/deployment/`

### `backup-postgres.sh`
Automated PostgreSQL backup.
```bash
./scripts/deployment/backup-postgres.sh
```

**Creates**: `backups/postgres_YYYYMMDD_HHMMSS.sql`

### `deploy_vllm.sh`
Deploy vLLM inference server.
```bash
./scripts/deployment/deploy_vllm.sh
```

### `deploy_triton.sh`
Deploy NVIDIA Triton inference server.
```bash
./scripts/deployment/deploy_triton.sh
```

---

## 🛠️ Utility Scripts

**Location**: `scripts/utils/`

### `create_demo_accounts.py`
Create multiple demo accounts with different roles.
```bash
python scripts/utils/create_demo_accounts.py
```

### `reset_accounts.py`
Reset demo account passwords.
```bash
python scripts/utils/reset_accounts.py
```

### `check_status.py`
Check system health and status.
```bash
python scripts/utils/check_status.py
```

---

## 🔥 Common Workflows

### Initial Setup (First Time)

```bash
# 1. Install dependencies
./bin/setup

# 2. Initialize database
python scripts/setup/init_db.py

# 3. Download models
python scripts/setup/download_models.py

# 4. Setup HuggingFace (optional)
./scripts/setup/setup_huggingface_auth.sh

# 5. Validate setup
python scripts/validation/validate_setup.py

# 6. Create demo users
python scripts/demo/create_demo_user.py

# 7. Test all features
python scripts/testing/test_all_features.py
```

### Before Deployment

```bash
# Run comprehensive tests
python scripts/testing/run_comprehensive_tests.py

# Validate setup
python scripts/validation/validate_setup.py

# Backup database
./scripts/deployment/backup-postgres.sh
```

### Troubleshooting

```bash
# Check dependencies
python scripts/validation/check_dependencies.py

# Verify specific issues
python scripts/testing/verify_issues_9_11.py

# Check system status
python scripts/utils/check_status.py
```

---

## 📝 Script Conventions

All scripts follow these conventions:

1. **Exit Codes**:
   - `0` - Success
   - `1` - Error/Failure
   - `2` - Warnings present

2. **Output Format**:
   - ✅ Success messages
   - ❌ Error messages
   - ⚠️ Warning messages
   - ℹ️ Info messages

3. **Dependencies**:
   - Scripts assume virtual environment is activated
   - All Python scripts are Python 3.11+ compatible
   - Shell scripts are POSIX-compliant (bash/zsh)

4. **Environment**:
   - Scripts read from `.env` file
   - Can be overridden with environment variables
   - Safe defaults for development

---

## 🤝 Contributing

When adding new scripts:

1. Place in appropriate subdirectory
2. Add docstring explaining purpose
3. Include usage example in this README
4. Follow naming convention: `action_target.py/sh`
5. Add to appropriate workflow section

---

**Last Updated**: 2024-11-28  
**Maintained By**: Shiksha Setu Development Team
