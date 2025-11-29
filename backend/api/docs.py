"""
Enhanced API Documentation Configuration

Provides rich OpenAPI documentation with:
- Detailed descriptions
- Request/response examples
- Authentication documentation
- Error response schemas
"""
from typing import Dict, Any

# API metadata
API_TITLE = "ShikshaSetu API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
# ShikshaSetu - AI-Powered Education Platform

ShikshaSetu is an AI-powered education platform that simplifies complex educational 
content for students across India, supporting multiple Indian languages.

## Features

- **Document Processing**: Upload PDFs and extract text with OCR
- **Content Simplification**: Simplify complex text for different grade levels
- **Translation**: Translate content to 13+ Indian languages
- **Text-to-Speech**: Convert text to audio in regional languages
- **Q&A**: Ask questions about uploaded content
- **Progress Tracking**: Track learning progress and achievements

## Authentication

Most endpoints require authentication via JWT tokens. Include the token in the 
`Authorization` header:

```
Authorization: Bearer <your-token>
```

## Rate Limits

| Endpoint Type | Requests/Minute | Requests/Hour |
|--------------|-----------------|---------------|
| Default | 60 | 1000 |
| Processing | 10 | 100 |
| TTS | 20 | 200 |
| Upload | 5 | 50 |

## Supported Languages

Hindi (hi), Bengali (bn), Tamil (ta), Telugu (te), Marathi (mr), 
Gujarati (gu), Kannada (kn), Malayalam (ml), Odia (or), Punjabi (pa), 
Assamese (as), Urdu (ur), Sanskrit (sa)
"""

API_CONTACT = {
    "name": "ShikshaSetu Support",
    "email": "support@shiksha-setu.in",
    "url": "https://shiksha-setu.in",
}

API_LICENSE = {
    "name": "MIT",
    "url": "https://opensource.org/licenses/MIT",
}

# Tag descriptions
TAGS_METADATA = [
    {
        "name": "health",
        "description": "Health check endpoints for monitoring and orchestration",
    },
    {
        "name": "auth",
        "description": "Authentication and authorization endpoints",
    },
    {
        "name": "users",
        "description": "User management and profile operations",
    },
    {
        "name": "content",
        "description": "Content management and retrieval",
    },
    {
        "name": "process",
        "description": "Document processing pipeline endpoints",
    },
    {
        "name": "upload",
        "description": "File upload endpoints",
    },
    {
        "name": "tts",
        "description": "Text-to-speech generation",
    },
    {
        "name": "qa",
        "description": "Question-answering endpoints",
    },
    {
        "name": "ncert",
        "description": "NCERT curriculum standards",
    },
    {
        "name": "progress",
        "description": "Learning progress tracking",
    },
    {
        "name": "feedback",
        "description": "User feedback collection",
    },
]

# Common response examples
RESPONSE_EXAMPLES = {
    "success": {
        "summary": "Successful response",
        "value": {
            "status": "success",
            "data": {},
            "message": "Operation completed successfully"
        }
    },
    "error_400": {
        "summary": "Bad request",
        "value": {
            "detail": "Invalid request parameters",
            "errors": [
                {"field": "text", "message": "Text is required"}
            ]
        }
    },
    "error_401": {
        "summary": "Unauthorized",
        "value": {
            "detail": "Could not validate credentials",
            "error_code": "INVALID_TOKEN"
        }
    },
    "error_403": {
        "summary": "Forbidden",
        "value": {
            "detail": "You don't have permission to access this resource"
        }
    },
    "error_404": {
        "summary": "Not found",
        "value": {
            "detail": "Resource not found"
        }
    },
    "error_429": {
        "summary": "Rate limit exceeded",
        "value": {
            "error": "rate_limit_exceeded",
            "message": "Rate limit exceeded. Try again in 60 seconds.",
            "retry_after": 60
        }
    },
    "error_500": {
        "summary": "Internal server error",
        "value": {
            "detail": "An unexpected error occurred",
            "request_id": "abc-123-def"
        }
    }
}

# Endpoint-specific examples
ENDPOINT_EXAMPLES = {
    "process_text": {
        "request": {
            "summary": "Process text example",
            "description": "Simplify and translate educational content",
            "value": {
                "text": "Photosynthesis is the process by which plants convert light energy into chemical energy stored in glucose.",
                "target_grade": 6,
                "target_language": "hi",
                "options": {
                    "simplify": True,
                    "translate": True,
                    "generate_audio": False
                }
            }
        },
        "response": {
            "summary": "Successful processing",
            "value": {
                "content_id": "cnt_abc123",
                "status": "completed",
                "original_text": "Photosynthesis is the process...",
                "simplified_text": "Plants make their own food using sunlight. This process is called photosynthesis.",
                "translations": {
                    "hi": "पौधे सूर्य की रोशनी का उपयोग करके अपना भोजन बनाते हैं। इस प्रक्रिया को प्रकाश संश्लेषण कहते हैं।"
                },
                "grade_level": 6,
                "processing_time_ms": 1234
            }
        }
    },
    "upload_file": {
        "request": {
            "summary": "Upload PDF",
            "description": "Upload a PDF document for processing"
        },
        "response": {
            "summary": "Upload successful",
            "value": {
                "upload_id": "upl_xyz789",
                "filename": "science_chapter_1.pdf",
                "size_bytes": 1048576,
                "mime_type": "application/pdf",
                "pages": 10,
                "status": "queued",
                "task_id": "task_abc123"
            }
        }
    },
    "tts_generate": {
        "request": {
            "summary": "Generate speech",
            "value": {
                "text": "नमस्ते, आप कैसे हैं?",
                "language": "hi",
                "voice": "female",
                "speed": 1.0,
                "format": "mp3"
            }
        },
        "response": {
            "summary": "Audio generated",
            "value": {
                "audio_id": "aud_123abc",
                "url": "/api/v1/audio/aud_123abc.mp3",
                "duration_seconds": 3.5,
                "format": "mp3",
                "size_bytes": 56320
            }
        }
    },
    "qa_ask": {
        "request": {
            "summary": "Ask a question",
            "value": {
                "question": "What is photosynthesis?",
                "content_id": "cnt_abc123",
                "language": "en"
            }
        },
        "response": {
            "summary": "Answer generated",
            "value": {
                "answer": "Photosynthesis is the process by which plants make their food using sunlight, water, and carbon dioxide.",
                "confidence": 0.92,
                "sources": [
                    {
                        "content_id": "cnt_abc123",
                        "chunk": "Plants use sunlight to make food...",
                        "relevance": 0.95
                    }
                ],
                "related_questions": [
                    "What do plants need for photosynthesis?",
                    "Where does photosynthesis happen?"
                ]
            }
        }
    },
    "auth_login": {
        "request": {
            "summary": "Login request",
            "value": {
                "email": "student@example.com",
                "password": "securepassword123"
            }
        },
        "response": {
            "summary": "Login successful",
            "value": {
                "access_token": "eyJhbGciOiJIUzI1NiIs...",
                "refresh_token": "eyJhbGciOiJIUzI1NiIs...",
                "token_type": "bearer",
                "expires_in": 3600,
                "user": {
                    "id": "usr_123",
                    "email": "student@example.com",
                    "full_name": "Rahul Kumar",
                    "role": "student"
                }
            }
        }
    }
}


def get_openapi_config() -> Dict[str, Any]:
    """Get OpenAPI configuration for FastAPI."""
    return {
        "title": API_TITLE,
        "version": API_VERSION,
        "description": API_DESCRIPTION,
        "contact": API_CONTACT,
        "license_info": API_LICENSE,
        "openapi_tags": TAGS_METADATA,
    }


def get_example(endpoint: str, example_type: str = "response") -> Dict[str, Any]:
    """Get example for an endpoint."""
    if endpoint in ENDPOINT_EXAMPLES:
        return ENDPOINT_EXAMPLES[endpoint].get(example_type, {})
    return {}


def get_error_responses() -> Dict[int, Dict[str, Any]]:
    """Get common error response definitions."""
    return {
        400: {
            "description": "Bad Request - Invalid parameters",
            "content": {
                "application/json": {
                    "example": RESPONSE_EXAMPLES["error_400"]["value"]
                }
            }
        },
        401: {
            "description": "Unauthorized - Invalid or missing authentication",
            "content": {
                "application/json": {
                    "example": RESPONSE_EXAMPLES["error_401"]["value"]
                }
            }
        },
        403: {
            "description": "Forbidden - Insufficient permissions",
            "content": {
                "application/json": {
                    "example": RESPONSE_EXAMPLES["error_403"]["value"]
                }
            }
        },
        404: {
            "description": "Not Found - Resource does not exist",
            "content": {
                "application/json": {
                    "example": RESPONSE_EXAMPLES["error_404"]["value"]
                }
            }
        },
        429: {
            "description": "Too Many Requests - Rate limit exceeded",
            "content": {
                "application/json": {
                    "example": RESPONSE_EXAMPLES["error_429"]["value"]
                }
            }
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": RESPONSE_EXAMPLES["error_500"]["value"]
                }
            }
        }
    }
