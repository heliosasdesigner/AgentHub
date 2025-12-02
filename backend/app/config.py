from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_env: str = "dev"
    backend_host: str = "0.0.0.0"
    backend_port: int = 11111
    db_url: str = "sqlite:///./agent_lab.db"
    openai_api_key: str | None = None
    lmstudio_base_url: str | None = None
    lmstudio_api_key: str | None = None
    lmstudio_model: str | None = None
    vllm_base_url: str | None = "http://localhost:8000/v1"
    vllm_api_key: str | None = None
    vllm_model: str = "openai/gpt-oss-20b"
    embed_model_id: str = "BAAI/bge-m3"

    neo4j_uri: str | None = None
    neo4j_user: str | None = None
    neo4j_password: str | None = None
    cognee_data_dir: str | None = "./cognee_data"

    class Config:
        env_file = ".env"


settings = Settings()
