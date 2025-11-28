# Authentication Setup Guide

## ✅ Setup Complete

Authentication system is fully configured and operational.

## Test User Credentials

Three test users have been created with different roles:

### 1. Regular User
- **Email:** `test@shiksha.com`
- **Password:** `Test@1234567`
- **Role:** `user`
- **User ID:** `1c877a22-8ffa-4c9e-b4c8-009c58c5ed57`

### 2. Teacher
- **Email:** `teacher@shiksha.com`
- **Password:** `Teacher@123456`
- **Role:** `teacher`
- **User ID:** `f58e8cbf-da3f-4b08-8bff-094d602ca9f8`

### 3. Admin
- **Email:** `admin@shiksha.com`
- **Password:** `Admin@123456`
- **Role:** `admin`
- **User ID:** `04e141f1-6d80-48cd-9e3f-19e3e75558d2`

## JWT Configuration

- **JWT_SECRET_KEY:** Configured (86 characters)
- **Algorithm:** HS256
- **Access Token Expiry:** 30 minutes
- **Refresh Token Expiry:** 7 days

## Usage Examples

### 1. Login to Get Access Token

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "test@shiksha.com", "password": "Test@1234567"}'
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### 2. Access Protected Endpoints

Use the access token in the Authorization header:

```bash
# Example: Q&A Question
curl -X POST http://localhost:8000/api/v1/qa/ask \
  -H "Authorization: Bearer YOUR_ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is photosynthesis?",
    "content_id": 1
  }'
```

### 3. Refresh Access Token

```bash
curl -X POST http://localhost:8000/api/v1/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{"refresh_token": "YOUR_REFRESH_TOKEN"}'
```

### 4. Register New User

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "email": "newuser@example.com",
    "password": "SecurePass@123",
    "full_name": "New User",
    "organization": "Example School"
  }'
```

## Protected Endpoints

The following endpoints require authentication:

### Q&A System
- `POST /api/v1/qa/process` - Process document for Q&A
- `POST /api/v1/qa/ask` - Ask question about content
- `GET /api/v1/qa/history/{content_id}` - Get chat history

### User Management (Admin only)
- Various admin endpoints require `admin` role

## Password Requirements

- Minimum length: 12 characters
- Must contain: uppercase, lowercase, digit, special character
- Example valid password: `SecurePass@123`

## Security Notes

1. **JWT Secret Key:** Currently auto-generated. For production, set `JWT_SECRET_KEY` environment variable.
2. **HTTPS:** Always use HTTPS in production to protect tokens in transit.
3. **Token Storage:** Store access tokens securely (e.g., httpOnly cookies, secure storage).
4. **Token Rotation:** Refresh tokens support rotation for enhanced security.

## Troubleshooting

### "Not authenticated" Error
- Ensure you're including the `Authorization: Bearer TOKEN` header
- Check that the token hasn't expired (30 minutes for access tokens)
- Verify the token format is correct

### "Invalid credentials" Error
- Double-check email and password
- Password is case-sensitive
- Ensure user account is active

### Token Expired
- Use the refresh token endpoint to get a new access token
- Re-login if refresh token has also expired

## Re-running Setup

To recreate test users or reset authentication:

```bash
python setup_auth.py
```

This will:
- Verify/generate JWT secret key
- Create test users (if they don't exist)
- Generate sample access tokens

## Integration with Frontend

Example JavaScript code for authentication:

```javascript
// Login
const response = await fetch('http://localhost:8000/api/v1/auth/login', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    email: 'test@shiksha.com',
    password: 'Test@1234567'
  })
});

const { access_token } = await response.json();

// Use token for protected requests
const qaResponse = await fetch('http://localhost:8000/api/v1/qa/ask', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${access_token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    question: 'What is photosynthesis?',
    content_id: 1
  })
});
```

## Next Steps

1. ✅ Authentication system configured
2. ✅ Test users created
3. ✅ JWT tokens working
4. 🔄 Configure production JWT secret
5. 🔄 Implement token refresh logic in frontend
6. 🔄 Add role-based access control for admin features
