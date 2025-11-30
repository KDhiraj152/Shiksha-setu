"""Unit tests for core configuration."""
import os
from backend.core.config import Settings, settings


def test_settings_initialization():
    """Test settings are initialized properly."""
    assert settings.APP_NAME == "ShikshaSetu AI Education API"
    assert settings.APP_VERSION == "2.0.0"
    assert settings.API_V1_PREFIX == "/api/v1"


def test_settings_directories_created():
    """Test that required directories are created on init."""
    assert settings.UPLOAD_DIR.exists()
    assert settings.MODEL_CACHE_DIR.exists()
    assert settings.LOG_DIR.exists()


def test_settings_environment_defaults():
    """Test default environment values."""
    assert settings.HOST == os.getenv("HOST", "0.0.0.0")
    assert settings.PORT == int(os.getenv("PORT", "8000"))
    

def test_settings_rate_limiting():
    """Test rate limiting configuration."""
    assert settings.RATE_LIMIT_PER_MINUTE >= 60  # At least 60/min
    assert settings.RATE_LIMIT_PER_HOUR >= 1000  # At least 1000/hour
    assert hasattr(settings, 'RATE_LIMIT_STORAGE')  # Storage backend configured


def test_settings_password_requirements():
    """Test password requirements are configured."""
    assert settings.MIN_PASSWORD_LENGTH >= 8  # Minimum 8 chars
    assert hasattr(settings, 'PASSWORD_REQUIRE_UPPERCASE')
    assert hasattr(settings, 'PASSWORD_REQUIRE_LOWERCASE')
    assert hasattr(settings, 'PASSWORD_REQUIRE_DIGIT')
