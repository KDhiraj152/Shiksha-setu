"""Authentication and authorization schemas."""

from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class Token(BaseModel):
    """JWT token response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 1800  # 30 minutes in seconds


class TokenData(BaseModel):
    """JWT token payload data."""
    email: Optional[str] = None
    user_id: Optional[int] = None
    role: Optional[str] = None


class UserCreate(BaseModel):
    """User registration request."""
    email: EmailStr
    password: str = Field(min_length=8, max_length=100)
    full_name: str = Field(min_length=2, max_length=100)
    organization: Optional[str] = None
    role: Optional[str] = Field(default="user", pattern="^(user|educator|admin)$")


class UserLogin(BaseModel):
    """User login request."""
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """User information response."""
    id: int
    email: str
    full_name: str
    role: str
    is_active: bool
    created_at: str

    class Config:
        from_attributes = True


class RefreshTokenRequest(BaseModel):
    """Refresh token request."""
    refresh_token: str


__all__ = [
    "Token",
    "TokenData",
    "UserCreate",
    "UserLogin",
    "UserResponse",
    "RefreshTokenRequest",
]
