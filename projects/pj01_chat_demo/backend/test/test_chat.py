# test/test_chat.py —— 会话与消息三个接口的行为

import uuid

from httpx import AsyncClient

from app.core.db import SessionLocal
from app.models.conversation import Message


async def create_conversation(
    client: AsyncClient, user: dict[str, str], title: str | None = None
) -> dict:
    payload = {} if title is None else {"title": title}
    response = await client.post("/api/conversations", json=payload, headers=user["headers"])
    assert response.status_code == 201
    return response.json()


async def test_create_conversation_returns_201(
    client: AsyncClient, registered_user: dict[str, str]
) -> None:
    body = await create_conversation(client, registered_user, "第一段对话")
    assert body["title"] == "第一段对话"
    assert uuid.UUID(body["id"])
    assert "user_id" not in body
    assert "conversation_id" not in body


async def test_create_conversation_without_title_uses_default(
    client: AsyncClient, registered_user: dict[str, str]
) -> None:
    body = await create_conversation(client, registered_user)
    assert body["title"] == "新对话"


async def test_create_conversation_title_too_long_returns_422(
    client: AsyncClient, registered_user: dict[str, str]
) -> None:
    response = await client.post(
        "/api/conversations", json={"title": "x" * 201}, headers=registered_user["headers"]
    )
    assert response.status_code == 422


async def test_create_conversation_without_token_returns_401(client: AsyncClient) -> None:
    response = await client.post("/api/conversations", json={"title": "无令牌"})
    assert response.status_code == 401


async def test_list_conversations_returns_own_only(
    client: AsyncClient, registered_user: dict[str, str], other_user: dict[str, str]
) -> None:
    mine = await create_conversation(client, registered_user, "我的会话")
    await create_conversation(client, other_user, "别人的会话")

    response = await client.get("/api/conversations", headers=registered_user["headers"])
    assert response.status_code == 200
    items = response.json()
    assert [item["id"] for item in items] == [mine["id"]]


async def test_list_conversations_ordered_by_updated_at(
    client: AsyncClient, registered_user: dict[str, str]
) -> None:
    first = await create_conversation(client, registered_user, "先建")
    second = await create_conversation(client, registered_user, "后建")

    response = await client.get("/api/conversations", headers=registered_user["headers"])
    ids = [item["id"] for item in response.json()]
    assert ids.index(second["id"]) < ids.index(first["id"])


async def test_messages_of_new_conversation_is_empty(
    client: AsyncClient, registered_user: dict[str, str]
) -> None:
    conversation = await create_conversation(client, registered_user, "空会话")
    response = await client.get(
        f"/api/conversations/{conversation['id']}/messages",
        headers=registered_user["headers"],
    )
    assert response.status_code == 200
    assert response.json() == []


async def test_messages_ordered_by_created_at(
    client: AsyncClient, registered_user: dict[str, str]
) -> None:
    conversation = await create_conversation(client, registered_user, "有消息")
    conversation_id = uuid.UUID(conversation["id"])

    async with SessionLocal() as session:
        session.add(Message(conversation_id=conversation_id, role="user", content="第一个问题"))
        await session.commit()
        session.add(Message(conversation_id=conversation_id, role="assistant", content="第一个回答"))
        await session.commit()

    response = await client.get(
        f"/api/conversations/{conversation_id}/messages", headers=registered_user["headers"]
    )
    assert response.status_code == 200
    body = response.json()
    assert [item["content"] for item in body] == ["第一个问题", "第一个回答"]
    assert [item["role"] for item in body] == ["user", "assistant"]
    assert "conversation_id" not in body[0]


async def test_messages_of_other_user_returns_404(
    client: AsyncClient, registered_user: dict[str, str], other_user: dict[str, str]
) -> None:
    conversation = await create_conversation(client, registered_user, "私有会话")
    response = await client.get(
        f"/api/conversations/{conversation['id']}/messages", headers=other_user["headers"]
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "会话不存在"


async def test_messages_of_unknown_conversation_returns_404(
    client: AsyncClient, registered_user: dict[str, str]
) -> None:
    response = await client.get(
        "/api/conversations/00000000-0000-0000-0000-000000000000/messages",
        headers=registered_user["headers"],
    )
    assert response.status_code == 404


async def test_messages_with_invalid_uuid_returns_422(
    client: AsyncClient, registered_user: dict[str, str]
) -> None:
    response = await client.get(
        "/api/conversations/not-a-uuid/messages", headers=registered_user["headers"]
    )
    assert response.status_code == 422

async def test_stream_rejects_other_user_conversation(
    client: AsyncClient, registered_user: dict[str, str], other_user: dict[str, str]
) -> None:
    body = await create_conversation(client, other_user, "别人的会话")
    response = await client.post(
        "/api/chat/stream",
        json={"conversation_id": body["id"], "content": "你好"},
        headers=registered_user["headers"],
    )
    assert response.status_code == 404


async def test_stream_rejects_empty_content(
    client: AsyncClient, registered_user: dict[str, str]
) -> None:
    body = await create_conversation(client, registered_user)
    response = await client.post(
        "/api/chat/stream",
        json={"conversation_id": body["id"], "content": ""},
        headers=registered_user["headers"],
    )
    assert response.status_code == 422


async def test_stream_requires_token(
    client: AsyncClient, registered_user: dict[str, str]
) -> None:
    body = await create_conversation(client, registered_user)
    response = await client.post(
        "/api/chat/stream", json={"conversation_id": body["id"], "content": "你好"}
    )
    assert response.status_code == 401