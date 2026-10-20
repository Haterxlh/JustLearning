# 会话接口与 pytest 用例（第 3 步小步 B）

2026-10-20 补上会话与消息的三个查询接口，并给用户模块与会话模块写了 pytest 用例，20 个用例全部通过。

## 决定

- 三个接口：`GET /api/conversations`（本人会话，按 `updated_at` 倒序）、`POST /api/conversations`（标题默认「新对话」）、`GET /api/conversations/{id}/messages`（按 `created_at` 升序）。
- 归属校验写在 `ChatService` 里，别人的会话与不存在的会话都返回 404，不暴露 id 是否有效。
- 响应模型不含 `user_id` 与 `conversation_id`，前端不需要这两个字段。
- 测试用 `httpx.ASGITransport` 直连应用，不监听端口；`client` 夹具手动进入 `app.router.lifespan_context` 复现启动过程。
- 测试写真实数据库，不 mock：每个用例注册独立用户，结束后按邮箱删用户，外键级联清掉会话与消息。
- `pytest.ini` 设 `asyncio_mode = auto` 与 `asyncio_default_fixture_loop_scope = function`，异步用例与夹具不用逐个加标记。

## 考虑过的选项

- 用 `TestClient`（同步）：内部起一个线程跑事件循环，与 `asyncpg` 的连接池配合时容易踩事件循环不一致的问题；直接用异步客户端更贴近真实调用。
- 用 SQLite 内存库跑测试：`vector(1024)` 列在 SQLite 上不存在，等于换一套 DDL，测不到真实表结构。
- 用例里为每个用户建独立 schema 隔离：清理路径还要额外维护，删用户加级联已足够。

## 后果

- 用例分布：`test/test_auth.py` 9 个、`test/test_chat.py` 11 个，覆盖正常路径与 401/404/409/422 边界。
- `requirements.txt` 增加 `pytest`、`pytest-asyncio`、`httpx` 三项。
- 用例会向 `chatdemo` 库写入临时数据，跑之前要确保容器在运行。
- 下一步做小步 C：`POST /api/chat/stream` 的 SSE 流式接口。
