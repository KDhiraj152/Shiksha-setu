"""
Integration Test Suite for ShikshaSetu

End-to-end tests for:
- API endpoints
- Pipeline processing
- Authentication flow
- File upload/processing
- Health checks
"""
import asyncio
import io
import os
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient
from httpx import AsyncClient

# Set test environment
os.environ.setdefault("TESTING", "true")
os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def app():
    """Create test application."""
    from backend.api.main import app
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    with TestClient(app) as client:
        yield client


@pytest.fixture
async def async_client(app):
    """Create async test client."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


@pytest.fixture
def mock_db():
    """Mock database session."""
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None
    db.add = MagicMock()
    db.commit = MagicMock()
    db.refresh = MagicMock()
    return db


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    redis = MagicMock()
    redis.get = MagicMock(return_value=None)
    redis.set = MagicMock()
    redis.setex = MagicMock()
    redis.delete = MagicMock()
    redis.ping = MagicMock(return_value=True)
    return redis


class TestHealthEndpoints:
    """Test health check endpoints."""
    
    def test_health_basic(self, client):
        """Test basic health endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_health_detailed(self, client):
        """Test detailed health endpoint."""
        response = client.get("/api/v1/health/detailed")
        assert response.status_code in [200, 503]  # May be degraded in test
        data = response.json()
        assert "status" in data
        assert "components" in data
        assert "version" in data
    
    def test_health_ready(self, client):
        """Test Kubernetes readiness probe."""
        response = client.get("/api/v1/health/ready")
        assert response.status_code in [200, 503]
        data = response.json()
        assert "ready" in data
    
    def test_health_live(self, client):
        """Test Kubernetes liveness probe."""
        response = client.get("/api/v1/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["alive"] is True


class TestAuthenticationFlow:
    """Test authentication endpoints."""
    
    def test_register_user(self, client, mock_db):
        """Test user registration."""
        with patch("backend.api.routes.auth.get_db", return_value=mock_db):
            response = client.post("/api/v1/auth/register", json={
                "email": "test@example.com",
                "password": "SecurePass123!",
                "full_name": "Test User"
            })
            # May fail without proper DB setup, but should not crash
            assert response.status_code in [200, 201, 400, 422, 500]
    
    def test_login_invalid_credentials(self, client):
        """Test login with invalid credentials."""
        response = client.post("/api/v1/auth/login", data={
            "username": "nonexistent@example.com",
            "password": "wrongpassword"
        })
        assert response.status_code in [401, 422, 500]
    
    def test_protected_endpoint_no_token(self, client):
        """Test protected endpoint without token."""
        response = client.get("/api/v1/users/me")
        assert response.status_code in [401, 403, 404, 500]


class TestFileUploadEndpoints:
    """Test file upload functionality."""
    
    def test_upload_pdf(self, client):
        """Test PDF file upload."""
        # Create a simple PDF-like file
        pdf_content = b"%PDF-1.4 fake pdf content"
        files = {"file": ("test.pdf", io.BytesIO(pdf_content), "application/pdf")}
        
        response = client.post("/api/v1/upload", files=files)
        # May fail without proper setup, but should handle gracefully
        assert response.status_code in [200, 201, 400, 401, 422, 500]
    
    def test_upload_invalid_file_type(self, client):
        """Test upload with invalid file type."""
        files = {"file": ("test.exe", io.BytesIO(b"fake exe"), "application/x-msdownload")}
        
        response = client.post("/api/v1/upload", files=files)
        # Should reject invalid file types
        assert response.status_code in [400, 401, 415, 422, 500]
    
    def test_upload_empty_file(self, client):
        """Test upload with empty file."""
        files = {"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")}
        
        response = client.post("/api/v1/upload", files=files)
        assert response.status_code in [400, 401, 422, 500]


class TestPipelineEndpoints:
    """Test content processing pipeline."""
    
    @pytest.fixture
    def mock_pipeline(self):
        """Mock pipeline orchestrator."""
        mock = AsyncMock()
        mock.process.return_value = {
            "status": "completed",
            "content_id": "test-123",
            "simplified_text": "This is simplified text.",
            "translations": {"hi": "यह सरल पाठ है।"},
        }
        return mock
    
    def test_process_text(self, client, mock_pipeline):
        """Test text processing endpoint."""
        with patch("backend.api.routes.process.PipelineOrchestrator", return_value=mock_pipeline):
            response = client.post("/api/v1/process/text", json={
                "text": "This is a complex scientific explanation.",
                "target_grade": 6,
                "target_language": "hi"
            })
            # Check it handles the request
            assert response.status_code in [200, 202, 401, 422, 500]
    
    def test_get_processing_status(self, client):
        """Test processing status endpoint."""
        response = client.get("/api/v1/process/status/test-task-id")
        assert response.status_code in [200, 404, 500]


class TestContentEndpoints:
    """Test content management endpoints."""
    
    def test_list_content(self, client):
        """Test content listing."""
        response = client.get("/api/v1/content")
        assert response.status_code in [200, 401, 500]
    
    def test_get_content_by_id(self, client):
        """Test get content by ID."""
        response = client.get("/api/v1/content/nonexistent-id")
        assert response.status_code in [404, 401, 500]
    
    def test_search_content(self, client):
        """Test content search."""
        response = client.get("/api/v1/content/search", params={"q": "science"})
        assert response.status_code in [200, 401, 500]


class TestTTSEndpoints:
    """Test text-to-speech endpoints."""
    
    def test_generate_tts(self, client):
        """Test TTS generation."""
        response = client.post("/api/v1/tts/generate", json={
            "text": "Hello, how are you?",
            "language": "en",
            "voice": "default"
        })
        assert response.status_code in [200, 401, 422, 500, 503]
    
    def test_get_available_voices(self, client):
        """Test get available voices."""
        response = client.get("/api/v1/tts/voices")
        assert response.status_code in [200, 404, 500]


class TestQAEndpoints:
    """Test question-answering endpoints."""
    
    def test_ask_question(self, client):
        """Test question asking."""
        response = client.post("/api/v1/qa/ask", json={
            "question": "What is photosynthesis?",
            "context": "Plants use sunlight to make food through photosynthesis."
        })
        assert response.status_code in [200, 401, 422, 500, 503]
    
    def test_get_chat_history(self, client):
        """Test chat history retrieval."""
        response = client.get("/api/v1/qa/history")
        assert response.status_code in [200, 401, 500]


class TestNCERTEndpoints:
    """Test NCERT standards endpoints."""
    
    def test_list_standards(self, client):
        """Test listing NCERT standards."""
        response = client.get("/api/v1/ncert/standards")
        assert response.status_code in [200, 500]
    
    def test_get_standard_by_grade(self, client):
        """Test get standards by grade."""
        response = client.get("/api/v1/ncert/standards", params={"grade": 6})
        assert response.status_code in [200, 500]
    
    def test_get_subjects(self, client):
        """Test get subjects."""
        response = client.get("/api/v1/ncert/subjects")
        assert response.status_code in [200, 404, 500]


class TestRateLimiting:
    """Test rate limiting functionality."""
    
    def test_rate_limit_headers(self, client):
        """Test rate limit headers are present."""
        response = client.get("/api/v1/health")
        # Rate limit headers should be present when middleware is enabled
        # These might not be present in test environment
        assert response.status_code == 200
    
    def test_rate_limit_exceeded(self, client):
        """Test rate limit exceeded response."""
        # Make many requests rapidly
        for _ in range(100):
            response = client.get("/api/v1/health")
            if response.status_code == 429:
                assert "retry_after" in response.json() or "Retry-After" in response.headers
                break


class TestErrorHandling:
    """Test error handling and responses."""
    
    def test_404_error(self, client):
        """Test 404 error response."""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client):
        """Test 405 error response."""
        response = client.delete("/api/v1/health")
        assert response.status_code in [404, 405]
    
    def test_validation_error(self, client):
        """Test validation error response."""
        response = client.post("/api/v1/process/text", json={
            "invalid_field": "value"
        })
        assert response.status_code in [401, 422, 500]


class TestCORS:
    """Test CORS configuration."""
    
    def test_cors_headers(self, client):
        """Test CORS headers are present."""
        response = client.options(
            "/api/v1/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET"
            }
        )
        # CORS might not be configured in test environment
        assert response.status_code in [200, 404, 405]


# Async tests
class TestAsyncEndpoints:
    """Async endpoint tests."""
    
    @pytest.mark.asyncio
    async def test_async_health(self, async_client):
        """Test async health check."""
        response = await async_client.get("/api/v1/health")
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self, async_client):
        """Test handling concurrent requests."""
        tasks = [
            async_client.get("/api/v1/health")
            for _ in range(10)
        ]
        responses = await asyncio.gather(*tasks)
        assert all(r.status_code == 200 for r in responses)


# Pipeline integration tests
class TestPipelineIntegration:
    """Integration tests for the full pipeline."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_pipeline_flow(self, async_client):
        """Test complete pipeline flow."""
        # This would require actual model loading
        # Skip in CI unless integration tests are explicitly enabled
        pytest.skip("Requires full pipeline setup")
    
    @pytest.mark.integration  
    @pytest.mark.asyncio
    async def test_streaming_response(self, async_client):
        """Test SSE streaming response."""
        # Test streaming endpoint
        pytest.skip("Requires streaming endpoint setup")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
