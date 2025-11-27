#!/usr/bin/env python3
"""Create demo user for testing."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.database import get_db_session
from src.models import User
from src.utils.auth import get_password_hash
from sqlalchemy.exc import IntegrityError

def create_demo_user():
    """Create demo user if it doesn't exist."""
    try:
        with get_db_session() as session:
            # Check if demo user exists
            existing_user = session.query(User).filter(User.email == 'demo@shiksha.com').first()
            
            if existing_user:
                print("✅ Demo user already exists")
                print("   Email: demo@shiksha.com")
                print("   Password: demo123")
                return True
            
            # Create demo user
            hashed_password = get_password_hash('demo123')
            demo_user = User(
                email='demo@shiksha.com',
                hashed_password=hashed_password,
                full_name='Demo User',
                organization='ShikshaSetu Demo',
                role='user',
                is_active=True,
                is_verified=True
            )
            
            session.add(demo_user)
            session.commit()
            
            print("✅ Demo user created successfully!")
            print("   Email: demo@shiksha.com")
            print("   Password: demo123")
            print("   Role: user")
            
            return True
            
    except IntegrityError:
        print("⚠️  Demo user already exists (constraint violation)")
        return True
    except Exception as e:
        print(f"❌ Failed to create demo user: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    success = create_demo_user()
    sys.exit(0 if success else 1)
