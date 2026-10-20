# pgvector 容器就位（第 1 步小步 3）

2026-10-20 起 PostgreSQL 18 加 pgvector 的容器，作为这个 demo 的数据库。

## 决定

- 镜像 `pgvector/pgvector:pg18`，容器名 `justlearning-pj01-pgvector`。
- 卷 `pj01_pgvector_data` 挂到 `/var/lib/postgresql`；PostgreSQL 18 的 `PGDATA` 是 `/var/lib/postgresql/18/docker`，挂载点必须上移一层。
- 宿主 5433 映射容器 5432；库名 `chatdemo`，用户与密码都是 `chat`。
- 启动命令：

  ```powershell
  docker run -d --name justlearning-pj01-pgvector -e POSTGRES_USER=chat -e POSTGRES_PASSWORD=chat -e POSTGRES_DB=chatdemo -p 5433:5432 -v pj01_pgvector_data:/var/lib/postgresql pgvector/pgvector:pg18
  ```

- 扩展在默认库 `chatdemo` 里启用：`CREATE EXTENSION IF NOT EXISTS vector;`。

## 考虑过的选项

- 沿用 pg16 与 `/var/lib/postgresql/data`：PostgreSQL 18 的镜像把卷挂载点改到 `/var/lib/postgresql`，旧路径下命名卷不会被写入，数据落在匿名卷里，删除容器即丢失。
- 用绑定挂载把宿主的目录挂进容器：Windows 上文件权限与 `fsync` 行为经过转换，PostgreSQL 容易在初始化时报权限错误；命名卷没有这一层。

## 后果

- 首次按 pg17 的路径挂卷，容器启动后 `exit=1`；删除容器与卷，改挂 `/var/lib/postgresql` 后重新启动，容器 `Up`，端口 `5433->5432` 正常。
- `CREATE EXTENSION` 执行成功，扩展版本待 `SELECT extversion` 确认。
- 下一步做小步 4：`backend/.env` 与 `frontend/.env.local`。
