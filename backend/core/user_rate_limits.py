"""
Enhanced Rate Limiting with Per-User Limits

Issue: CODE-REVIEW-GPT #15 (MEDIUM)
Enhancement: Add user-role-based rate limiting
"""

from typing import Dict, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class UserRole(str, Enum):
    """User roles with different rate limits."""
    ADMIN = "admin"
    TEACHER = "teacher"
    STUDENT = "student"
    GUEST = "guest"
    API_CLIENT = "api_client"


# Role-based rate limits (requests per minute, requests per hour)
ROLE_RATE_LIMITS = {
    UserRole.ADMIN: (1000, 10000),      # Admins: high limits
    UserRole.TEACHER: (200, 2000),      # Teachers: elevated limits
    UserRole.STUDENT: (60, 600),        # Students: standard limits
    UserRole.GUEST: (20, 100),          # Guests: restricted
    UserRole.API_CLIENT: (500, 5000),   # API clients: high limits
}

# Endpoint-specific multipliers by role
ENDPOINT_ROLE_MULTIPLIERS = {
    "/api/content/create": {
        UserRole.TEACHER: 2.0,   # Teachers can create more content
        UserRole.ADMIN: 5.0,     # Admins have highest limits
    },
    "/api/qa/search": {
        UserRole.STUDENT: 1.5,   # Students can search more
        UserRole.TEACHER: 2.0,
    },
}


def get_rate_limits_for_user(
    user_role: str,
    endpoint: str
) -> tuple[int, int]:
    """
    Get rate limits for user based on role and endpoint.
    
    Args:
        user_role: User's role
        endpoint: API endpoint path
        
    Returns:
        Tuple of (per_minute_limit, per_hour_limit)
    """
    try:
        role = UserRole(user_role)
    except ValueError:
        role = UserRole.GUEST
    
    base_minute, base_hour = ROLE_RATE_LIMITS.get(role, (20, 100))
    
    # Apply endpoint-specific multiplier
    if endpoint in ENDPOINT_ROLE_MULTIPLIERS:
        multiplier = ENDPOINT_ROLE_MULTIPLIERS[endpoint].get(role, 1.0)
        return int(base_minute * multiplier), int(base_hour * multiplier)
    
    return base_minute, base_hour


def get_user_quota_info(user_id: str, user_role: str) -> Dict[str, any]:
    """
    Get quota information for user.
    
    Args:
        user_id: User ID
        user_role: User role
        
    Returns:
        Dictionary with quota details
    """
    minute_limit, hour_limit = get_rate_limits_for_user(user_role, "")
    
    return {
        "user_id": user_id,
        "role": user_role,
        "limits": {
            "per_minute": minute_limit,
            "per_hour": hour_limit
        },
        "description": f"{user_role} tier rate limits"
    }


logger.info("Enhanced per-user rate limiting configuration loaded")
