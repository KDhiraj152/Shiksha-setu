"""
Pytest configuration and fixtures for integration tests.
"""
import pytest
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Set test environment BEFORE importing src modules
os.environ['TESTING'] = 'true'
os.environ['ENVIRONMENT'] = 'test'

# Use SQLite for testing (no database server required)
# SQLite is fast, in-memory capable, and perfect for integration tests
TEST_DATABASE_URL = 'sqlite:///test_shiksha_setu.db'
os.environ['DATABASE_URL'] = TEST_DATABASE_URL

# Redis configuration (test database 15 to avoid interfering with dev/prod)
TEST_REDIS_URL = 'redis://localhost:6379/15'
os.environ['REDIS_URL'] = TEST_REDIS_URL
os.environ['CELERY_BROKER_URL'] = TEST_REDIS_URL
os.environ['CELERY_RESULT_BACKEND'] = TEST_REDIS_URL

# Celery should run tasks synchronously during testing
os.environ['CELERY_TASK_ALWAYS_EAGER'] = 'true'
os.environ['CELERY_TASK_EAGER_PROPAGATES'] = 'true'

# Disable rate limiting for tests
os.environ['RATE_LIMIT_ENABLED'] = 'false'

# Set higher rate limits for tests
os.environ['RATE_LIMIT_PER_MINUTE'] = '10000'
os.environ['RATE_LIMIT_PER_HOUR'] = '100000'


@pytest.fixture(scope='session')
def test_data_dir():
    """Create test data directory."""
    data_dir = Path('data/test_samples')
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture(scope='session')
def celery_enable_logging():
    """Enable Celery logging in tests."""
    return True


@pytest.fixture(scope='session')
def celery_includes():
    """Celery task modules to include."""
    return ['src.tasks.pipeline_tasks']


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    # Use test database URL if not set
    if 'DATABASE_URL' not in os.environ:
        # Default to PostgreSQL test database
        os.environ['DATABASE_URL'] = 'postgresql://user:password@localhost:5432/test_education_content'
    
    # Disable SQL echo for cleaner test output
    if 'SQL_ECHO' not in os.environ:
        os.environ['SQL_ECHO'] = 'false'
    
    # Set test API keys if needed
    if 'HUGGINGFACE_API_KEY' not in os.environ:
        # Use test key - tests will be skipped if real API is needed
        os.environ['HUGGINGFACE_API_KEY'] = 'test_key_placeholder'
    
    yield


@pytest.fixture(scope="function")
def clean_database():
    """Clean database between tests."""
    from src.database import engine, Base
    
    # Drop all tables
    Base.metadata.drop_all(bind=engine)
    
    # Recreate all tables
    Base.metadata.create_all(bind=engine)
    
    yield
    
    # Cleanup after test
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def sample_content():
    """Sample educational content for testing."""
    return {
        "original_text": "Photosynthesis is the process by which plants make food.",
        "grade_level": 6,
        "subject": "Science",
        "language": "English"
    }
