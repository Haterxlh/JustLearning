# pj01 对话 demo

前后端分离的对话应用，界面参照 DeepSeek 网页版：左侧会话列表，右侧消息流。后端保存会话与消息历史，回答通过 SSE 流式返回，数据库使用 pgvector。

## 目标

- 前端：会话列表（新建、切换）、消息滚动区、输入框，回答逐字出现。
- 后端：登录注册（JWT）、会话与消息的增查、SSE 流式转发模型输出。
- 数据：PostgreSQL 加 pgvector 扩展，`messages` 预留 `vector(1024)` 列，供后续接检索。
- 过程：本人执行命令，agent 给步骤与预期结果；一次只推进一个小步。

## 范围

本期只做上面对话链路，不做检索增强、不做多轮摘要、不做部署上线。

## 技术选型

| 部分 | 选择 |
|---|---|
| 前端 | Vite + React + TypeScript + ESLint |
| 后端 | FastAPI + SQLAlchemy 2.0（异步） + asyncpg + Pydantic v2 |
| 虚拟环境 | uv（`uv venv` 建环境，`uv pip install` 装依赖，环境内不带 pip） |
| 数据库 | `pgvector/pgvector:pg18` 容器 |
| 模型 | DeepSeek 兼容接口，地址、模型名、密钥放 `.env`；可换成宿舍 llama.cpp 的 OpenAI 兼容端点 |

## 端口与容器

| 服务 | 端口 |
|---|---|
| 后端 | 8000 |
| 前端 dev | 5173（被占用时 Vite 顺延到 5174，CORS 配置需跟着改） |
| pgvector | 宿主 5433 映射到容器 5432 |

```powershell
docker run -d --name justlearning-pj01-pgvector -e POSTGRES_USER=chat -e POSTGRES_PASSWORD=chat -e POSTGRES_DB=chatdemo -p 5433:5432 -v pj01_pgvector_data:/var/lib/postgresql pgvector/pgvector:pg18
```

PostgreSQL 18 的官方镜像把数据目录改成 `/var/lib/postgresql/18/docker`，卷声明的挂载点随之改成 `/var/lib/postgresql`；pg17 及更早的镜像用 `/var/lib/postgresql/data`。

## 目录结构

```
projects/pj01_chat_demo/
├─ README.md                 本文件：目标、契约、计划、协作约定
├─ docs/adr/                 架构决策记录，进度与决定都在这里
├─ docs/questions/           提问记录，一天一个文件
├─ backend/                  FastAPI 服务
│  ├─ requirements.txt
│  ├─ pytest.ini
│  ├─ .venv/                 uv 创建的虚拟环境（不入库）
│  ├─ app/                   api / core / models / schemas / services 分层
│  └─ test/                  pytest 用例
└─ frontend/                 Vite + React + TypeScript
```

## 接口契约

| 接口 | 说明 |
|---|---|
| `GET /api/health` | 健康检查，返回 `{"status": "ok"}` |
| `POST /api/auth/register` | 入参 `{email, password}`，返回 201 与用户信息（不含密码）；邮箱重复返回 409 |
| `POST /api/auth/login` | 入参同上，返回 `{access_token, token_type, user}`；密码错返回 401 |
| `GET /api/auth/me` | 带 `Authorization: Bearer <token>`，返回当前用户 |
| `GET /api/conversations` | 本人会话列表，按 `updated_at` 倒序 |
| `POST /api/conversations` | 入参 `{"title": "可选"}`，建会话 |
| `GET /api/conversations/{id}/messages` | 该会话全部消息，按 `created_at` 升序 |
| `POST /api/chat/stream` | 入参 `{conversation_id, content}`，返回 `text/event-stream` |

SSE 帧格式：每段增量一帧 `data: {"delta": "文字"}\n\n`，结束帧 `data: [DONE]\n\n`。会话不属于当前用户时返回 404。

数据表：`users`（uuid 主键、邮箱唯一、bcrypt 哈希）、`conversations`（属于用户、标题、创建与更新时间）、`messages`（属于会话、角色、内容、`embedding vector(1024)` 预留、创建时间）。

## 分步实施

第 1 步搭框架（本人执行命令，agent 只讲步骤）：

| 小步 | 内容 |
|---|---|
| 1 | 后端 uv 虚拟环境与依赖 |
| 2 | 前端 Vite 骨架（React + TypeScript + ESLint） |
| 3 | pgvector 容器 |
| 4 | `.env`（后端与前端各一份） |
| 5 | 配置生效验证（宿主端口连通、扩展版本、依赖可导入） |

第 2 步登录注册（agent 写代码）：`app/config.py`、`app/db.py`、`app/models.py`、`app/schemas.py`、`app/security.py`、`app/deps.py`、`app/routers/auth.py`、`app/main.py`。验收：注册 201、重复注册 409、密码错 401、无 token 访问 `/api/auth/me` 返回 401、`password_hash` 以 `$2b$12$` 开头。

第 3 步对话与 SSE（agent 讲步骤，本人写代码）：先加 `conversations` 与 `messages` 两张表，再做三个非流式接口用 curl 验证，然后写 SSE 生成器（先落用户消息再开流、用 `httpx` 读 `data:` 行、生成器内部新建 `SessionLocal` 写库、`finally` 里存已生成的部分），最后接前端（`fetch` 加 `ReadableStream`、按 `\n\n` 切帧、`AbortController` 停止）。验收：两个账号数据隔离、刷新后历史仍在、首字延迟有记录、停止后数据库有截断的助手消息、会话顺序随最后一条消息更新。

进度与决定记录在 `docs/adr/`：每完成一个小步或一个阶段，追加一条 ADR。

## 协作约定

1. 本人执行所有命令、安装与运行；agent 只给命令与预期结果。
2. 一次只给一个小步，工作量适中，不一次性输出整条流程。第 1 步与第 3 步 agent 不改文件，第 2 步由 agent 写代码。
3. 本人自行判断结果，不粘贴终端输出；出现报错时贴报错原文。
4. 讲解用通俗简洁的话，涉及"这个""它"这类代词时指明具体对象。
5. 静默记录提问：`docs/questions/YYYY-MM-DD.md`，一天一个文件，代词换写成具体对象，只记问题与简要回答，不在对话里声明已记录。
6. 密钥与模型地址从 `.env` 读取，代码里不写死。

