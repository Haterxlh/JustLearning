# app/core/security.py —— 密码哈希与 JWT 的签发、解析

from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from app.core.settings import settings

ALGORITHM = "HS256"


def hash_password(password: str) -> str:
    """bcrypt 只取前 72 字节，超长密码在 schema 层已限制。"""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))


def create_access_token(user_id: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expires_minutes)
    payload = {"sub": user_id, "exp": expire}
    return jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)


def decode_access_token(token: str) -> str:
    """解析失败时抛 jwt.InvalidTokenError，由调用方转成 401。"""
    payload = jwt.decode(token, settings.jwt_secret, algorithms=[ALGORITHM])
    return payload["sub"]
