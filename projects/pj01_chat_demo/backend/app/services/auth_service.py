# app/services/auth_service.py —— 注册与登录的业务规则，路由层只负责收请求、转调这里

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import create_access_token, hash_password, verify_password
from app.models.user import User
from app.schemas.auth import LoginRequest, LoginResponse, RegisterRequest


class AuthService:
    @staticmethod
    async def register(db: AsyncSession, payload: RegisterRequest) -> User:
        email = payload.email.lower()
        if await db.scalar(select(User).where(User.email == email)) is not None:
            raise HTTPException(status.HTTP_409_CONFLICT, detail="邮箱已注册")
        user = User(email=email, password_hash=hash_password(payload.password))
        db.add(user)
        try:
            await db.commit()
        except IntegrityError:
            # 两个请求同时通过了上面的查重，唯一索引兜底
            await db.rollback()
            raise HTTPException(status.HTTP_409_CONFLICT, detail="邮箱已注册")
        return user

    @staticmethod
    async def login(db: AsyncSession, payload: LoginRequest) -> LoginResponse:
        user = await db.scalar(select(User).where(User.email == payload.email.lower()))
        if user is None or not verify_password(payload.password, user.password_hash):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="邮箱或密码错误")
        return LoginResponse(
            access_token=create_access_token(str(user.id)),
            user=user,
        )
