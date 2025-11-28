"""
Token Management Service

Issue: CODE-REVIEW-GPT #6 (CRITICAL) - Token Rotation Implementation
Handles token blacklisting, rotation, and session management.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional
from sqlalchemy.orm import Session
from jose import jwt, JWTError
import logging

from ..models import TokenBlacklist, RefreshToken, User
from ..core.config import Settings
from .auth import (
    SECRET_KEY,
    ALGORITHM,
    create_access_token,
    create_refresh_token,
    generate_device_fingerprint,
)

logger = logging.getLogger(__name__)
settings = Settings()


class TokenService:
    """Manage token lifecycle, rotation, and blacklisting."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def is_token_blacklisted(self, jti: str) -> bool:
        """Check if token JTI is blacklisted."""
        blacklisted = self.db.query(TokenBlacklist).filter(
            TokenBlacklist.token_jti == jti
        ).first()
        return blacklisted is not None
    
    def blacklist_token(
        self,
        jti: str,
        user_id: str,
        reason: str,
        expires_at: datetime
    ) -> TokenBlacklist:
        """Add token to blacklist."""
        blacklisted = TokenBlacklist(
            token_jti=jti,
            user_id=user_id,
            reason=reason,
            expires_at=expires_at
        )
        self.db.add(blacklisted)
        self.db.commit()
        logger.info(f"Token {jti} blacklisted for user {user_id}, reason: {reason}")
        return blacklisted
    
    def store_refresh_token(
        self,
        jti: str,
        user_id: str,
        expires_at: datetime,
        device_fingerprint: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        parent_jti: Optional[str] = None,
        rotation_count: int = 0
    ) -> RefreshToken:
        """Store refresh token in database for tracking."""
        refresh_token = RefreshToken(
            token_jti=jti,
            user_id=user_id,
            device_fingerprint=device_fingerprint,
            ip_address=ip_address,
            user_agent=user_agent,
            parent_jti=parent_jti,
            rotation_count=rotation_count,
            expires_at=expires_at
        )
        self.db.add(refresh_token)
        self.db.commit()
        return refresh_token
    
    def get_refresh_token(self, jti: str) -> Optional[RefreshToken]:
        """Get refresh token by JTI."""
        return self.db.query(RefreshToken).filter(
            RefreshToken.token_jti == jti,
            RefreshToken.is_active == True
        ).first()
    
    def invalidate_refresh_token(self, jti: str, reason: str = "rotation") -> bool:
        """Mark refresh token as inactive."""
        token = self.get_refresh_token(jti)
        if token:
            token.is_active = False
            self.db.commit()
            
            # Also blacklist it
            self.blacklist_token(jti, str(token.user_id), reason, token.expires_at)
            logger.info(f"Refresh token {jti} invalidated, reason: {reason}")
            return True
        return False
    
    def rotate_refresh_token(
        self,
        old_jti: str,
        user_id: str,
        email: str,
        role: str = "user",
        device_fingerprint: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Optional[tuple[str, str, str]]:
        """
        Rotate refresh token (invalidate old, create new).
        
        Returns:
            Tuple of (new_access_token, new_refresh_token, new_jti) or None if failed
        """
        # Get old token
        old_token = self.get_refresh_token(old_jti)
        if not old_token:
            logger.warning(f"Attempted to rotate non-existent token: {old_jti}")
            return None
        
        # Check rotation limit (prevent infinite rotation attacks)
        MAX_ROTATION_COUNT = 100
        if old_token.rotation_count >= MAX_ROTATION_COUNT:
            logger.warning(f"Token {old_jti} exceeded rotation limit")
            self.invalidate_refresh_token(old_jti, "rotation_limit_exceeded")
            return None
        
        # Invalidate old token
        self.invalidate_refresh_token(old_jti, "rotation")
        
        # Create new tokens
        token_data = {"sub": user_id, "email": email, "role": role}
        
        # Create new refresh token
        new_refresh_token, new_jti, expires_at = create_refresh_token(
            token_data,
            parent_jti=old_jti,
            rotation_count=old_token.rotation_count + 1
        )
        
        # Store new refresh token
        self.store_refresh_token(
            jti=new_jti,
            user_id=user_id,
            expires_at=expires_at,
            device_fingerprint=device_fingerprint,
            ip_address=ip_address,
            user_agent=user_agent,
            parent_jti=old_jti,
            rotation_count=old_token.rotation_count + 1
        )
        
        # Create new access token
        new_access_token = create_access_token(token_data)
        
        logger.info(f"Token rotated for user {user_id}, count: {old_token.rotation_count + 1}")
        return new_access_token, new_refresh_token, new_jti
    
    def revoke_all_user_tokens(self, user_id: str, reason: str = "logout_all") -> int:
        """Revoke all active tokens for a user."""
        # Get all active refresh tokens
        tokens = self.db.query(RefreshToken).filter(
            RefreshToken.user_id == user_id,
            RefreshToken.is_active == True
        ).all()
        
        count = 0
        for token in tokens:
            self.invalidate_refresh_token(token.token_jti, reason)
            count += 1
        
        logger.info(f"Revoked {count} tokens for user {user_id}, reason: {reason}")
        return count
    
    def cleanup_expired_tokens(self) -> tuple[int, int]:
        """Clean up expired tokens from database."""
        now = datetime.now(timezone.utc)
        
        # Delete expired blacklist entries
        blacklist_deleted = self.db.query(TokenBlacklist).filter(
            TokenBlacklist.expires_at < now
        ).delete()
        
        # Delete expired refresh tokens
        refresh_deleted = self.db.query(RefreshToken).filter(
            RefreshToken.expires_at < now
        ).delete()
        
        self.db.commit()
        logger.info(f"Cleaned up {blacklist_deleted} blacklist entries, {refresh_deleted} refresh tokens")
        return blacklist_deleted, refresh_deleted
    
    def verify_token_not_blacklisted(self, token: str, token_type: str = "access") -> Optional[dict]:
        """Verify token and check if blacklisted."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            
            # Check token type
            if payload.get("type") != token_type:
                logger.warning(f"Invalid token type: expected {token_type}, got {payload.get('type')}")
                return None
            
            # Check if blacklisted
            jti = payload.get("jti")
            if jti and self.is_token_blacklisted(jti):
                logger.warning(f"Token {jti} is blacklisted")
                return None
            
            return payload
            
        except JWTError as e:
            logger.error(f"JWT verification failed: {e}")
            return None


def get_token_service(db: Session) -> TokenService:
    """Get token service instance."""
    return TokenService(db)
