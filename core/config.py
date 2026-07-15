"""Central configuration for the Insurance Document Intelligence platform."""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    chroma_db_dir: str = "./data/chroma_db"
    chroma_collection: str = "insurance_docs"
    max_retrieval_k: int = 6
    app_host: str = "0.0.0.0"
    app_port: int = 8000

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
