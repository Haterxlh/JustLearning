# app/services/stream_service.py —— 转发模型的流式输出，并把完整回答写进库

import json
import uuid
from collections.abc import AsyncIterator

import anyio
import httpx
from sqlalchemy import func, update

from app.core.db import SessionLocal
from app.core.settings import settings
from app.models.conversation import Conversation, Message

# 关掉中间层的缓存与缓冲，否则前端要等整段生成完才收到内容
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def encode_frame(body: str) -> str:
    """按 SSE 规范封一帧：data: 行加一个空行"""
    return f"data: {body}\n\n"


async def save_reply(conversation_id: uuid.UUID, reply: str) -> None:
    """把这一段回答写进库，并刷新会话的更新时间。"""
    async with SessionLocal() as session:
        if reply:
            session.add(Message(conversation_id=conversation_id, role="assistant", content=reply))
        await session.execute(
            update(Conversation).where(Conversation.id == conversation_id).values(updated_at=func.now())
        )
        await session.commit()


async def stream_assistant_reply(
    conversation_id: uuid.UUID, history: list[dict[str, str]]
) -> AsyncIterator[str]:
    """把模型返回的增量逐段转给前端，结束时把整段回答写进库。"""
    collected: list[str] = []
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{settings.llm_base_url}/chat/completions",
                headers={"Authorization": f"Bearer {settings.llm_api_key}"},
                json={"model": settings.llm_model, "messages": history, "stream": True},
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        break
                    delta = json.loads(data)["choices"][0]["delta"].get("content")
                    if not delta:
                        continue
                    collected.append(delta)
                    yield encode_frame(json.dumps({"delta": delta}, ensure_ascii=False))
        yield encode_frame("[DONE]")
    except httpx.HTTPError as error:
        yield encode_frame(json.dumps({"error": f"模型接口调用失败：{error}"}, ensure_ascii=False))
    finally:
        # 正常说完会走到这里；浏览器断开导致生成器被取消时也会走到这里。
        # 取消状态下第一个 await 就会抛 CancelledError，所以要用屏蔽作用域保住这次写库。
        reply = "".join(collected)
        with anyio.CancelScope(shield=True):
            await save_reply(conversation_id, reply)