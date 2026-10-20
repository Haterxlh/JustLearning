# app/api/auth.py —— 用户模块的 HTTP 路由；只做收请求、转调 service

from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db
from app.core.deps import get_current_user
from app.models.user import User
from app.schemas.auth import LoginRequest, LoginResponse, RegisterRequest, UserResponse
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["用户"])


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    payload: RegisterRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> User:
    """注册。response_model 决定响应字段，密码不会出现在任何响应里。"""
    return await AuthService.register(db, payload)


@router.post("/login", response_model=LoginResponse)
async def login(
    payload: LoginRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
) -> LoginResponse:
    """登录成功返回 token 与用户信息。"""
    return await AuthService.login(db, payload)


@router.get("/me", response_model=UserResponse)
async def read_me(user: Annotated[User, Depends(get_current_user)]) -> User:
    """需要登录：token 校验通过后返回当前用户。"""
    return user
