# app/models/__init__.py —— 导出全部表模型，建表时 metadata 才完整

from app.models.conversation import Conversation, Message
from app.models.user import User

__all__ = ["User", "Conversation", "Message"]