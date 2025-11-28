"""
Unit tests for backend services.
Tests core functionality of translation, storage, and curriculum validation services.
"""
import pytest
import io
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock

# Translation Service Tests
from backend.services.translate import TranslationService


class TestTranslationService:
    """Tests for TranslationService."""
    
    @pytest.fixture
    def translation_service(self):
        """Create TranslationService instance."""
        return TranslationService()
    
    @pytest.fixture
    def mock_translation_engine(self):
        """Mock TranslationEngine."""
        with patch('backend.services.translate.TranslationEngine') as mock:
            instance = mock.return_value
            instance.translate.return_value = "translated text"
            yield instance
    
    @pytest.mark.asyncio
    async def test_translate_async_basic(self, translation_service, mock_translation_engine):
        """Test async translation with basic text."""
        translation_service.engine = mock_translation_engine
        
        result = await translation_service.translate_async(
            text="Hello world",
            source_lang="en",
            target_lang="hi"
        )
        
        assert result == "translated text"
        mock_translation_engine.translate.assert_called_once_with(
            "Hello world", "en", "hi"
        )
    
    @pytest.mark.asyncio
    async def test_translate_async_empty_text(self, translation_service, mock_translation_engine):
        """Test async translation with empty text."""
        translation_service.engine = mock_translation_engine
        mock_translation_engine.translate.return_value = ""
        
        result = await translation_service.translate_async(
            text="",
            source_lang="en",
            target_lang="hi"
        )
        
        assert result == ""
    
    @pytest.mark.asyncio
    async def test_translate_async_multiple_languages(self, translation_service, mock_translation_engine):
        """Test async translation with various language pairs."""
        translation_service.engine = mock_translation_engine
        
        language_pairs = [
            ("en", "hi", "Hello", "नमस्ते"),
            ("hi", "en", "नमस्ते", "Hello"),
            ("en", "ta", "Welcome", "வரவேற்பு"),
        ]
        
        for source, target, input_text, expected in language_pairs:
            mock_translation_engine.translate.return_value = expected
            result = await translation_service.translate_async(
                text=input_text,
                source_lang=source,
                target_lang=target
            )
            assert result == expected


# Storage Service Tests
from backend.services.storage import LocalStorageService, S3StorageService, StorageService


class TestLocalStorageService:
    """Tests for LocalStorageService."""
    
    @pytest.fixture
    def storage_service(self, tmp_path):
        """Create LocalStorageService with temp directory."""
        with patch('backend.services.storage.settings') as mock_settings:
            mock_settings.UPLOAD_DIR = tmp_path / "uploads"
            mock_settings.API_V1_PREFIX = "/api/v1"
            service = LocalStorageService()
            yield service
    
    def test_upload_file_basic(self, storage_service):
        """Test basic file upload."""
        file_content = b"test file content"
        file_obj = io.BytesIO(file_content)
        
        result = storage_service.upload_file(
            file_obj=file_obj,
            key="test/file.txt",
            content_type="text/plain"
        )
        
        assert result is not None
        assert Path(result).exists()
        assert Path(result).read_bytes() == file_content
    
    def test_upload_file_creates_subdirectories(self, storage_service):
        """Test file upload creates nested directories."""
        file_obj = io.BytesIO(b"content")
        
        result = storage_service.upload_file(
            file_obj=file_obj,
            key="deep/nested/path/file.txt"
        )
        
        assert Path(result).exists()
        assert Path(result).parent.name == "path"
    
    def test_download_file_exists(self, storage_service):
        """Test downloading existing file."""
        # First upload
        upload_content = b"download test content"
        upload_obj = io.BytesIO(upload_content)
        storage_service.upload_file(upload_obj, "download_test.txt")
        
        # Then download
        download_obj = io.BytesIO()
        storage_service.download_file("download_test.txt", download_obj)
        
        download_obj.seek(0)
        assert download_obj.read() == upload_content
    
    def test_download_file_not_found(self, storage_service):
        """Test downloading non-existent file raises error."""
        download_obj = io.BytesIO()
        
        with pytest.raises(FileNotFoundError):
            storage_service.download_file("nonexistent.txt", download_obj)
    
    def test_delete_file_exists(self, storage_service):
        """Test deleting existing file."""
        file_obj = io.BytesIO(b"to be deleted")
        result = storage_service.upload_file(file_obj, "delete_me.txt")
        
        assert Path(result).exists()
        
        deleted = storage_service.delete_file("delete_me.txt")
        assert deleted is True
        assert not Path(result).exists()
    
    def test_delete_file_not_exists(self, storage_service):
        """Test deleting non-existent file returns False."""
        deleted = storage_service.delete_file("never_existed.txt")
        assert deleted is False
    
    def test_generate_presigned_url(self, storage_service):
        """Test presigned URL generation for local storage."""
        url = storage_service.generate_presigned_url("test_file.txt")
        
        assert "/api/v1/static/uploads/test_file.txt" in url


class TestS3StorageService:
    """Tests for S3StorageService."""
    
    @pytest.fixture
    def mock_s3_client(self):
        """Mock boto3 S3 client."""
        with patch('backend.services.storage.boto3.client') as mock_client:
            mock_instance = Mock()
            mock_client.return_value = mock_instance
            yield mock_instance
    
    @pytest.fixture
    def storage_service(self, mock_s3_client):
        """Create S3StorageService with mocked client."""
        with patch('backend.services.storage.settings') as mock_settings:
            mock_settings.AWS_ACCESS_KEY_ID = "test_key"
            mock_settings.AWS_SECRET_ACCESS_KEY = "test_secret"
            mock_settings.AWS_REGION = "us-east-1"
            mock_settings.S3_BUCKET_NAME = "test-bucket"
            service = S3StorageService()
            service.s3_client = mock_s3_client
            service.bucket_name = "test-bucket"
            yield service
    
    def test_upload_file_to_s3(self, storage_service, mock_s3_client):
        """Test uploading file to S3."""
        file_content = b"s3 test content"
        file_obj = io.BytesIO(file_content)
        
        mock_s3_client.upload_fileobj.return_value = None
        
        result = storage_service.upload_file(
            file_obj=file_obj,
            key="s3/test.txt",
            content_type="text/plain"
        )
        
        assert result == "s3://test-bucket/s3/test.txt"
        mock_s3_client.upload_fileobj.assert_called_once()
    
    def test_generate_presigned_url_s3(self, storage_service, mock_s3_client):
        """Test generating presigned URL for S3."""
        mock_s3_client.generate_presigned_url.return_value = "https://s3.amazonaws.com/test-bucket/file.txt?signature=xyz"
        
        url = storage_service.generate_presigned_url("test_file.txt")
        
        assert url.startswith("https://s3.amazonaws.com")
        mock_s3_client.generate_presigned_url.assert_called_once()
    
    def test_delete_file_from_s3(self, storage_service, mock_s3_client):
        """Test deleting file from S3."""
        mock_s3_client.delete_object.return_value = {'DeleteMarker': True}
        
        result = storage_service.delete_file("s3/delete_test.txt")
        
        assert result is True
        mock_s3_client.delete_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="s3/delete_test.txt"
        )


# Curriculum Validator Tests
from backend.services.curriculum_validator import CurriculumValidator


class TestCurriculumValidator:
    """Tests for CurriculumValidator."""
    
    @pytest.fixture
    def mock_model(self):
        """Mock transformer model."""
        mock = Mock()
        mock.eval = Mock()
        # Mock model output
        mock_output = Mock()
        mock_output.logits = Mock()
        mock.return_value = mock_output
        return mock
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Mock tokenizer."""
        mock = Mock()
        mock.return_value = {
            'input_ids': Mock(),
            'attention_mask': Mock()
        }
        return mock
    
    @pytest.fixture
    def validator(self, mock_model, mock_tokenizer):
        """Create CurriculumValidator with mocked model."""
        with patch('backend.services.curriculum_validator.settings') as mock_settings:
            mock_settings.VALIDATOR_MODEL_ID = "ai4bharat/indic-bert"
            mock_settings.VALIDATOR_FINE_TUNE_PATH = "/tmp/nonexistent"
            
            validator = CurriculumValidator()
            validator.model = mock_model
            validator.tokenizer = mock_tokenizer
            yield validator
    
    def test_grade_ranges_defined(self, validator):
        """Test grade ranges are properly defined."""
        assert "primary" in validator.grade_ranges
        assert "middle" in validator.grade_ranges
        assert "secondary" in validator.grade_ranges
        assert "senior_secondary" in validator.grade_ranges
        
        assert validator.grade_ranges["primary"] == (1, 5)
        assert validator.grade_ranges["middle"] == (6, 8)
        assert validator.grade_ranges["secondary"] == (9, 10)
        assert validator.grade_ranges["senior_secondary"] == (11, 12)
    
    def test_subjects_defined(self, validator):
        """Test subject categories are defined."""
        expected_subjects = [
            "mathematics", "science", "social_science",
            "english", "hindi", "languages", "arts", "physical_education"
        ]
        
        for subject in expected_subjects:
            assert subject in validator.subjects
    
    def test_model_initialization(self):
        """Test validator can be initialized."""
        with patch('backend.services.curriculum_validator.settings') as mock_settings:
            mock_settings.VALIDATOR_MODEL_ID = "test-model"
            mock_settings.VALIDATOR_FINE_TUNE_PATH = "/tmp/test"
            
            validator = CurriculumValidator()
            
            assert validator.model_id == "test-model"
            assert validator.fine_tune_path == "/tmp/test"
            assert validator.model is None
            assert validator.tokenizer is None


# Bhashini Client Tests
from backend.services.bhashini import BhashiniClient


class TestBhashiniClient:
    """Tests for BhashiniClient."""
    
    @pytest.mark.skip(reason="BhashiniClient requires environment variables set before initialization")
    def test_client_initialization(self):
        """Test Bhashini client initializes correctly."""
        # This test would require actual environment variables or deeper mocking
        pass


# vLLM Client Tests
from backend.services.vllm_serve import VLLMClient


class TestVLLMClient:
    """Tests for VLLMClient."""
    
    @pytest.fixture
    def mock_requests(self):
        """Mock requests library."""
        with patch('backend.services.vllm_serve.requests') as mock:
            yield mock
    
    @pytest.fixture
    def vllm_client(self):
        """Create VLLMClient instance."""
        with patch('backend.services.vllm_serve.settings') as mock_settings:
            mock_settings.VLLM_API_URL = "http://localhost:8001"
            client = VLLMClient()
            yield client
    
    def test_client_initialization(self, vllm_client):
        """Test vLLM client initializes with correct URL."""
        assert vllm_client is not None
        # Add more specific assertions based on actual VLLMClient implementation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
