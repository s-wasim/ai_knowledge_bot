"""
Simple authentication module.
Provides login, logout, and session management.
"""
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional

import jwt

from db import get_connection

SESSION_EXPIRY_HOURS = 24
JWT_SECRET = secrets.token_hex(32)


def _hash_password(password: str, salt: Optional[str] = None) -> tuple:
    if salt is None:
        salt = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000)
    return hashed.hex(), salt


def verify_password(stored_hash: str, salt: str, password: str) -> bool:
    computed, _ = _hash_password(password, salt)
    return computed == stored_hash


def authenticate_user(username: str, password: str) -> Optional[dict]:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT id, username, password_hash, salt FROM users WHERE username = ?",
        (username,),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    user_id, _, stored_hash, salt = row
    if not verify_password(stored_hash, salt, password):
        return None
    token = jwt.encode(
        {
            "user_id": user_id,
            "username": username,
            "exp": datetime.utcnow() + timedelta(hours=SESSION_EXPIRY_HOURS),
        },
        JWT_SECRET,
        algorithm="HS256",
    )
    return {"token": token, "user_id": user_id, "username": username}


def logout_user(token: str) -> bool:
    return True


def validate_session(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None
