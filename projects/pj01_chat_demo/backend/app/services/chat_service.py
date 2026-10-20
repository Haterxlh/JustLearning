# app/services/chat_service.py —— 会话与消息的查询规则

import uuid

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.conversation import Conversation, Message
from app.models.user import User


class ChatService:
    @staticmethod
    async def list_conversations(db: AsyncSession, user: User) -> list[Conversation]:
        rows = await db.scalars(
            select(Conversation)
            .where(Conversation.user_id == user.id)
            .order_by(Conversation.updated_at.desc())
        )
        return list(rows)

    @staticmethod
    async def create_conversation(db: AsyncSession, user: User, title: str) -> Conversation:
        conversation = Conversation(user_id=user.id, title=title)
        db.add(conversation)
        await db.commit()
        return conversation

    @staticmethod
    async def list_messages(
        db: AsyncSession, user: User, conversation_id: uuid.UUID
    ) -> list[Message]:
        await ChatService.get_owned_conversation(db, user, conversation_id)
        rows = await db.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
        )
        return list(rows)
    

    @staticmethod
    async def get_owned_conversation(
        db: AsyncSession, user: User, conversation_id: uuid.UUID
    ) -> Conversation:
        conversation = await db.get(Conversation, conversation_id)
        # 别人的会话也按不存在处理，避免试探出会话 id 是否有效
        if conversation is None or conversation.user_id != user.id:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="会话不存在")
        return conversation

    @staticmethod
    async def append_message(
        db: AsyncSession, conversation_id: uuid.UUID, role: str, content: str
    ) -> Message:
        message = Message(conversation_id=conversation_id, role=role, content=content)
        db.add(message)
        await db.commit()
        return message

    @staticmethod
    async def build_history(
        db: AsyncSession, conversation_id: uuid.UUID
    ) -> list[dict[str, str]]:
        rows = await db.scalars(
            select(Message)
            .where(Message.conversation_id == conversation_id)
            .order_by(Message.created_at)
        )
        return [{"role": message.role, "content": message.content} for message in rows]