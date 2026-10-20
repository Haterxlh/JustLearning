# app/main.py —— 应用入口：创建 app、登记路由、管理进程级资源

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import models  # noqa: F401 导入模型，建表时 metadata 才完整
from app.api.auth import router as auth_router
from app.api.chat import router as chat_router, stream_router
from app.core.db import Base, engine
from app.core.settings import settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """yield 之前是启动阶段：连库建表；之后是关闭阶段：释放连接池。"""
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()


app = FastAPI(title="pj01 chat demo", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# → /api/auth/register、/api/auth/login、/api/auth/me
app.include_router(auth_router, prefix="/api")
# → /api/conversations、/api/conversations/{id}/messages
app.include_router(chat_router, prefix="/api")
# → /api/chat/stream
app.include_router(stream_router, prefix="/api")