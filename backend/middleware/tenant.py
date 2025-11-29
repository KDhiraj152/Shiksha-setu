"""
Multi-tenancy Middleware

Tenant context injection for complete data isolation using
PostgreSQL Row Level Security (RLS).
"""
from fastapi import Request, HTTPException, status
from sqlalchemy import text
from typing import Optional
import uuid

from backend.core.database import get_db
from backend.models import User, Organization
from backend.utils.logging import get_logger

logger = get_logger(__name__)


class TenantContext:
    """Thread-local tenant context."""
    
    def __init__(self):
        self._organization_id: Optional[uuid.UUID] = None
        self._user_id: Optional[uuid.UUID] = None
    
    def set_context(self, organization_id: uuid.UUID, user_id: uuid.UUID):
        """Set tenant context."""
        self._organization_id = organization_id
        self._user_id = user_id
    
    def get_organization_id(self) -> Optional[uuid.UUID]:
        """Get current organization ID."""
        return self._organization_id
    
    def get_user_id(self) -> Optional[uuid.UUID]:
        """Get current user ID."""
        return self._user_id
    
    def clear(self):
        """Clear tenant context."""
        self._organization_id = None
        self._user_id = None


# Global tenant context
tenant_context = TenantContext()


async def tenant_middleware(request: Request, call_next):
    """
    Middleware to inject tenant context into database session.
    
    Sets PostgreSQL session variables for RLS policies:
    - app.current_organization_id
    - app.current_user_id
    """
    # Skip for health checks and public endpoints
    if request.url.path in ['/health', '/api/v1/auth/login', '/api/v1/auth/register']:
        response = await call_next(request)
        return response
    
    # Extract tenant from subdomain or header
    organization_id = None
    user_id = None
    
    # Method 1: Extract from subdomain (e.g., org1.example.com)
    host = request.headers.get('host', '')
    subdomain = host.split('.')[0] if '.' in host else None
    
    if subdomain and subdomain not in ['www', 'api', 'localhost']:
        # Lookup organization by slug
        with next(get_db()) as db:
            org = db.query(Organization).filter(
                Organization.slug == subdomain
            ).first()
            
            if org:
                organization_id = org.id
    
    # Method 2: Extract from X-Organization-ID header (for API clients)
    if not organization_id:
        org_header = request.headers.get('X-Organization-ID')
        if org_header:
            try:
                organization_id = uuid.UUID(org_header)
            except ValueError:
                logger.warning(f"Invalid organization ID in header: {org_header}")
    
    # Method 3: Extract from authenticated user
    if not organization_id and hasattr(request.state, 'user'):
        user = request.state.user
        if user and user.organization_id:
            organization_id = user.organization_id
            user_id = user.id
    
    # If no organization found, check if endpoint requires one
    if not organization_id:
        # Allow endpoints without organization (e.g., org creation)
        if not request.url.path.startswith('/api/v1/organizations/create'):
            logger.warning(f"No organization context for: {request.url.path}")
    
    # Set PostgreSQL session variables for RLS
    if organization_id or user_id:
        with next(get_db()) as db:
            if organization_id:
                db.execute(
                    text("SET LOCAL app.current_organization_id = :org_id"),
                    {"org_id": str(organization_id)}
                )
                tenant_context.set_context(organization_id, user_id)
                logger.debug(f"Set tenant context: org={organization_id}, user={user_id}")
            
            if user_id:
                db.execute(
                    text("SET LOCAL app.current_user_id = :user_id"),
                    {"user_id": str(user_id)}
                )
    
    try:
        response = await call_next(request)
        return response
    finally:
        # Clear context after request
        tenant_context.clear()


def require_organization():
    """
    Dependency to require organization context.
    
    Usage:
        @router.get("/data")
        async def get_data(org_id: uuid.UUID = Depends(require_organization)):
            ...
    """
    org_id = tenant_context.get_organization_id()
    
    if not org_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Organization context required"
        )
    
    return org_id


async def validate_organization_access(
    organization_id: uuid.UUID,
    user: User,
    db
) -> bool:
    """
    Validate that user has access to organization.
    
    Args:
        organization_id: Organization UUID
        user: Current user
        db: Database session
    
    Returns:
        True if user has access
    
    Raises:
        HTTPException if access denied
    """
    # Check user's organization
    if user.organization_id != organization_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this organization"
        )
    
    # Check organization is active
    org = db.query(Organization).filter(
        Organization.id == organization_id,
        Organization.is_active == True
    ).first()
    
    if not org:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found or inactive"
        )
    
    return True


def get_organization_from_request(request: Request) -> Optional[uuid.UUID]:
    """
    Extract organization ID from request.
    
    Checks in order:
    1. Subdomain
    2. X-Organization-ID header
    3. Authenticated user's organization
    """
    # Check subdomain
    host = request.headers.get('host', '')
    subdomain = host.split('.')[0] if '.' in host else None
    
    if subdomain and subdomain not in ['www', 'api', 'localhost']:
        with next(get_db()) as db:
            org = db.query(Organization).filter(
                Organization.slug == subdomain
            ).first()
            
            if org:
                return org.id
    
    # Check header
    org_header = request.headers.get('X-Organization-ID')
    if org_header:
        try:
            return uuid.UUID(org_header)
        except ValueError:
            logger.debug(f"Invalid organization ID format in header: {org_header}")
    
    # Check user
    if hasattr(request.state, 'user'):
        user = request.state.user
        if user and user.organization_id:
            return user.organization_id
    
    return None
