# app/core/settings.py —— 从 backend/.env 读取配置

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """配置项字段名与 .env 里的变量名一一对应，大小写不敏感。"""

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str
    jwt_secret: str
    jwt_expires_minutes: int = 10080
    llm_base_url: str
    llm_api_key: str = ""
    llm_model: str
    cors_origins: str = "http://localhost:5173"

    @property
    def cors_origin_list(self) -> list[str]:
        """把逗号分隔的来源拆成列表，CORS 中间件要的是列表。"""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings()