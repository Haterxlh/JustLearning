# app/schemas/chat.py —— 会话与消息的请求体、响应体

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ConversationCreate(BaseModel):
    """创建对话的请求体"""
    title: str = Field(default="新对话", max_length=200)


class ConversationResponse(BaseModel):
    """对话的响应体"""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    created_at: datetime
    updated_at: datetime


class MessageResponse(BaseModel):
    """消息的响应体"""
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: str
    content: str
    created_at: datetime
    
class ChatRequest(BaseModel):
    """发起一次流式回答的请求体"""
    conversation_id: uuid.UUID
    content: str = Field(min_length=1, max_length=8000)