# test/test_auth.py —— 注册、登录、当前用户三个接口的行为

from httpx import AsyncClient

from app.core.security import create_access_token

PASSWORD = "12345678"


async def test_me_returns_registered_user(client: AsyncClient, registered_user: dict[str, str]) -> None:
    response = await client.get("/api/auth/me", headers=registered_user["headers"])
    assert response.status_code == 200
    body = response.json()
    assert body["email"] == registered_user["email"]
    assert "password" not in body
    assert "password_hash" not in body


async def test_register_duplicate_email_returns_409(
    client: AsyncClient, registered_user: dict[str, str]
) -> None:
    response = await client.post(
        "/api/auth/register",
        json={"email": registered_user["email"], "password": PASSWORD},
    )
    assert response.status_code == 409
    assert response.json()["detail"] == "邮箱已注册"


async def test_register_short_password_returns_422(client: AsyncClient) -> None:
    response = await client.post(
        "/api/auth/register",
        json={"email": "short-pass@example.com", "password": "123"},
    )
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "password"]


async def test_register_invalid_email_returns_422(client: AsyncClient) -> None:
    response = await client.post(
        "/api/auth/register",
        json={"email": "not-an-email", "password": PASSWORD},
    )
    assert response.status_code == 422


async def test_login_wrong_password_returns_401(
    client: AsyncClient, registered_user: dict[str, str]
) -> None:
    response = await client.post(
        "/api/auth/login",
        json={"email": registered_user["email"], "password": "wrong-pass"},
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "邮箱或密码错误"


async def test_login_returns_token_and_user(
    client: AsyncClient, registered_user: dict[str, str]
) -> None:
    response = await client.post(
        "/api/auth/login",
        json={"email": registered_user["email"], "password": registered_user["password"]},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"].startswith("eyJ")
    assert body["user"]["email"] == registered_user["email"]


async def test_me_without_token_returns_401(client: AsyncClient) -> None:
    response = await client.get("/api/auth/me")
    assert response.status_code == 401
    assert response.json()["detail"] == "缺少访问令牌"


async def test_me_with_broken_token_returns_401(client: AsyncClient) -> None:
    response = await client.get("/api/auth/me", headers={"Authorization": "Bearer not-a-jwt"})
    assert response.status_code == 401
    assert response.json()["detail"] == "令牌无效或已过期"


async def test_me_with_unknown_user_id_returns_401(client: AsyncClient) -> None:
    token = create_access_token("00000000-0000-0000-0000-000000000000")
    response = await client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401
    assert response.json()["detail"] == "令牌对应的用户不存在"
