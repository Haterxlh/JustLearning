# test/conftest.py —— 公共夹具：真实数据库上的应用实例、已登录用户

import uuid
from collections.abc import AsyncIterator

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete

from app.core.db import SessionLocal
from app.main import app
from app.models.user import User

PASSWORD = "12345678"


async def create_user(client: AsyncClient) -> dict[str, str]:
    """注册并登录一个新用户，返回邮箱、token 与可直接用的请求头。"""
    email = f"pytest-{uuid.uuid4().hex[:8]}@example.com"
    response = await client.post("/api/auth/register", json={"email": email, "password": PASSWORD})
    assert response.status_code == 201
    response = await client.post("/api/auth/login", json={"email": email, "password": PASSWORD})
    assert response.status_code == 200
    token = response.json()["access_token"]
    return {
        "email": email,
        "password": PASSWORD,
        "token": token,
        "headers": {"Authorization": f"Bearer {token}"},
    }


async def delete_user(email: str) -> None:
    """删用户，数据库的外键级联会带走他的会话与消息。"""
    async with SessionLocal() as session:
        await session.execute(delete(User).where(User.email == email))
        await session.commit()


@pytest_asyncio.fixture
async def client() -> AsyncIterator[AsyncClient]:
    """进 lifespan 建表，再给一个直接调用应用的异步客户端，不监听端口。"""
    async with app.router.lifespan_context(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as http_client:
            yield http_client


@pytest_asyncio.fixture
async def registered_user(client: AsyncClient) -> AsyncIterator[dict[str, str]]:
    """用例的主角用户，结束后删除。"""
    user = await create_user(client)
    yield user
    await delete_user(user["email"])


@pytest_asyncio.fixture
async def other_user(client: AsyncClient) -> AsyncIterator[dict[str, str]]:
    """第二个用户，用于验证数据隔离，结束后删除。"""
    user = await create_user(client)
    yield user
    await delete_user(user["email"])
