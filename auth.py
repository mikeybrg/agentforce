"""Authentication utilities for AgentForge."""
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional

# Simple token storage (in production, use Redis or database)
active_sessions = {}


def hash_password(password: str) -> str:
    """Hash a password using SHA-256 with salt."""
    salt = secrets.token_hex(16)
    hash_obj = hashlib.sha256((password + salt).encode())
    return f"{salt}:{hash_obj.hexdigest()}"


def verify_password(password: str, password_hash: str) -> bool:
    """Verify a password against its hash."""
    try:
        salt, stored_hash = password_hash.split(":")
        hash_obj = hashlib.sha256((password + salt).encode())
        return hash_obj.hexdigest() == stored_hash
    except ValueError:
        return False


def create_session_token(user_id: int) -> str:
    """Create a session token for a user."""
    token = secrets.token_urlsafe(32)
    active_sessions[token] = {
        "user_id": user_id,
        "created_at": datetime.utcnow(),
        "expires_at": datetime.utcnow() + timedelta(days=7)
    }
    return token


def get_user_from_token(token: str) -> Optional[int]:
    """Get user ID from session token."""
    if token not in active_sessions:
        return None

    session = active_sessions[token]
    if datetime.utcnow() > session["expires_at"]:
        del active_sessions[token]
        return None

    return session["user_id"]


def invalidate_session(token: str) -> bool:
    """Invalidate a session token."""
    if token in active_sessions:
        del active_sessions[token]
        return True
    return False
