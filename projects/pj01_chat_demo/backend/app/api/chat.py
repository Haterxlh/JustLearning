# app/api/chat.py —— 会话模块的 HTTP 路由；只做收请求、转调 service

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.db import get_db
from app.core.deps import get_current_user
from app.models.conversation import Conversation, Message
from app.models.user import User
from app.schemas.chat import ChatRequest, ConversationCreate, ConversationResponse, MessageResponse
from app.services.chat_service import ChatService
from app.services.stream_service import SSE_HEADERS, stream_assistant_reply

router = APIRouter(prefix="/conversations", tags=["会话"])


@router.get("", response_model=list[ConversationResponse])
async def list_conversations(
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[Conversation]:
    """当前用户全部会话，按最近活动倒序。"""
    return await ChatService.list_conversations(db, user)


@router.post("", response_model=ConversationResponse, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    payload: ConversationCreate,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> Conversation:
    """新建会话，标题不传时用「新对话」。"""
    return await ChatService.create_conversation(db, user, payload.title)


@router.get("/{conversation_id}/messages", response_model=list[MessageResponse])
async def list_messages(
    conversation_id: uuid.UUID,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> list[Message]:
    """一个会话里的消息，按时间升序。"""
    return await ChatService.list_messages(db, user, conversation_id)


stream_router = APIRouter(prefix="/chat", tags=["对话"])


@stream_router.post("/stream")
async def chat_stream(
    payload: ChatRequest,
    user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> StreamingResponse:
    """先落用户消息，再把模型的增量逐帧转发。"""
    conversation = await ChatService.get_owned_conversation(db, user, payload.conversation_id)
    await ChatService.append_message(db, conversation.id, "user", payload.content)
    history = await ChatService.build_history(db, conversation.id)
    return StreamingResponse(
        stream_assistant_reply(conversation.id, history),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )