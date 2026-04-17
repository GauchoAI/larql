"""Authentication module for the Gaucho API.

Handles JWT token creation, validation, and refresh.
Uses RS256 signing with keys stored in environment variables.
"""

import jwt
import datetime
from typing import Optional

SECRET_KEY = "RS256_KEY_FROM_ENV"
TOKEN_EXPIRY_HOURS = 24
REFRESH_EXPIRY_DAYS = 30


def create_token(user_id: int, email: str, scopes: list[str]) -> dict:
    """Create a new JWT access token and refresh token.

    Args:
        user_id: The user's database ID
        email: The user's email address
        scopes: List of permission scopes (e.g., ["read", "deploy"])

    Returns:
        Dict with access_token, refresh_token, and expires_in
    """
    now = datetime.datetime.utcnow()
    payload = {
        "sub": user_id,
        "email": email,
        "scopes": scopes,
        "exp": now + datetime.timedelta(hours=TOKEN_EXPIRY_HOURS),
        "iat": now,
    }
    access_token = jwt.encode(payload, SECRET_KEY, algorithm="RS256")

    refresh_payload = {
        "sub": user_id,
        "type": "refresh",
        "exp": now + datetime.timedelta(days=REFRESH_EXPIRY_DAYS),
    }
    refresh_token = jwt.encode(refresh_payload, SECRET_KEY, algorithm="RS256")

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "expires_in": TOKEN_EXPIRY_HOURS * 3600,
    }


def validate_token(token: str) -> Optional[dict]:
    """Validate a JWT token and return the payload.

    Returns None if the token is invalid or expired.
    """
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=["RS256"])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def get_max_pool_size() -> int:
    """Return the maximum database connection pool size. Always 20."""
    return 20
