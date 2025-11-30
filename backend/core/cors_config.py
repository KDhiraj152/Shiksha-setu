"""
Enhanced CORS Configuration

Issue: CODE-REVIEW-GPT #18 (MEDIUM)
Purpose: Environment-specific CORS with preflight caching
"""

from typing import List
from fastapi.middleware.cors import CORSMiddleware
import logging

logger = logging.getLogger(__name__)


def get_cors_origins(environment: str) -> List[str]:
    """
    Get allowed origins based on environment.
    
    Args:
        environment: Environment name (development, staging, production)
        
    Returns:
        List of allowed origins
    """
    if environment == "production":
        return [
            "https://shiksha-setu.com",
            "https://www.shiksha-setu.com",
            "https://app.shiksha-setu.com",
        ]
    elif environment == "staging":
        return [
            "https://staging.shiksha-setu.com",
            "https://staging-app.shiksha-setu.com",
            "http://localhost:3000",
            "http://localhost:5173",  # Vite dev server
        ]
    else:  # development
        return [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:8080",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
        ]


def configure_cors(app, environment: str = "development"):
    """
    Configure CORS middleware with environment-specific settings.
    
    Args:
        app: FastAPI application
        environment: Environment name
    """
    origins = get_cors_origins(environment)
    
    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-Request-ID",
            "X-API-Key",
            "Accept",
            "Origin",
            "User-Agent",
        ],
        expose_headers=[
            "X-Process-Time",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-Request-ID",
        ],
        max_age=3600,  # Cache preflight for 1 hour
    )
    
    logger.info(f"CORS configured for {environment} with {len(origins)} allowed origins")


# CORS configuration dict (alternative approach)
CORS_CONFIG = {
    "development": {
        "allow_origins": [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:8080",
        ],
        "allow_credentials": True,
        "allow_methods": ["*"],
        "allow_headers": ["*"],
        "max_age": 600,  # 10 minutes in development
    },
    "staging": {
        "allow_origins": [
            "https://staging.shiksha-setu.com",
            "https://staging-app.shiksha-setu.com",
            "http://localhost:3000",
        ],
        "allow_credentials": True,
        "allow_methods": ["GET", "POST", "PUT", "DELETE", "PATCH"],
        "allow_headers": [
            "Authorization",
            "Content-Type",
            "X-Request-ID",
        ],
        "max_age": 3600,
    },
    "production": {
        "allow_origins": [
            "https://shiksha-setu.com",
            "https://www.shiksha-setu.com",
            "https://app.shiksha-setu.com",
        ],
        "allow_credentials": True,
        "allow_methods": ["GET", "POST", "PUT", "DELETE", "PATCH"],
        "allow_headers": [
            "Authorization",
            "Content-Type",
            "X-Request-ID",
            "X-API-Key",
        ],
        "expose_headers": [
            "X-Process-Time",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
        ],
        "max_age": 7200,  # 2 hours in production
    }
}


def get_cors_config(environment: str = "development") -> dict:
    """
    Get CORS configuration for environment.
    
    Args:
        environment: Environment name
        
    Returns:
        CORS configuration dictionary
    """
    return CORS_CONFIG.get(environment, CORS_CONFIG["development"])
