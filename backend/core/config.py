"""
Core configuration for the RAG Document Q&A System.
Uses Pydantic BaseSettings for type-safe environment variable handling.
"""
from functools import lru_cache
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # ── Application ──────────────────────────────────────────
    APP_NAME: str = "RAG Document Q&A"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=False, env="DEBUG")
    SECRET_KEY: str = Field(..., env="SECRET_KEY")

    # ── API ───────────────────────────────────────────────────
    API_PREFIX: str = "/api/v1"
    ALLOWED_HOSTS: List[str] = ["*"]
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "https://your-frontend.vercel.app",
    ]

    # ── OpenAI ────────────────────────────────────────────────
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    OPENAI_CHAT_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_MAX_TOKENS: int = 2048
    OPENAI_TEMPERATURE: float = 0.1   # Low temp → factual, grounded answers

    # ── Vector Database ───────────────────────────────────────
    VECTOR_DB_TYPE: str = "chromadb"  # "chromadb" | "faiss"
    CHROMA_PERSIST_DIR: str = "./embeddings/chroma_db"
    CHROMA_COLLECTION_NAME: str = "documents"
    FAISS_INDEX_PATH: str = "./embeddings/faiss_index"

    # ── Document Processing ───────────────────────────────────
    UPLOAD_DIR: str = "./data/raw"
    PROCESSED_DIR: str = "./data/processed"
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: List[str] = ["pdf", "docx", "txt", "md"]

    # ── RAG Pipeline ──────────────────────────────────────────
    CHUNK_SIZE: int = 1000          # Characters per chunk
    CHUNK_OVERLAP: int = 200        # Overlap prevents context loss at boundaries
    TOP_K_RETRIEVAL: int = 5        # Number of chunks retrieved per query
    SIMILARITY_THRESHOLD: float = 0.3  # Minimum similarity score to include chunk
    RERANKING_ENABLED: bool = True

    # ── Redis Cache ───────────────────────────────────────────
    REDIS_URL: Optional[str] = Field(default=None, env="REDIS_URL")
    CACHE_TTL_SECONDS: int = 3600   # 1 hour cache for query results

    # ── Logging ───────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"        # "json" for production, "text" for dev

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings instance. lru_cache ensures Settings is only
    instantiated once per process — no repeated env var reads.
    """
    return Settings()


settings = get_settings()
