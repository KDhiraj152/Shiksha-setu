# Backend

The Shiksha Setu backend is built with FastAPI and provides the core API and ML pipeline functionality.

## Structure

- **api/** - REST API endpoints and routes
- **core/** - Core configuration, authentication, and utilities
- **models/** - Database models and schemas
- **services/** - Business logic and service layer
- **pipeline/** - AI/ML processing pipeline
- **tasks/** - Background jobs and async tasks
- **simplify/** - Text simplification features
- **speech/** - Speech-to-text and text-to-speech
- **translate/** - Translation services
- **validate/** - Input validation utilities
- **utils/** - Shared utility functions

## Key Features

- RESTful API with FastAPI
- RAG (Retrieval Augmented Generation) system
- Document processing and Q&A
- Multi-language support
- WebSocket streaming
- Background task processing
- OpenTelemetry observability
- Multi-tenancy support

## Development

```bash
# Install dependencies
pip install -r infrastructure/docker/requirements.txt

# Run backend
./bin/start-backend

# Run tests
./bin/test
```

## API Documentation

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
