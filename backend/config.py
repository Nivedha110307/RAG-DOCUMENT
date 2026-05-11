"""
config.py — Central configuration using Pydantic Settings.

Why Pydantic Settings?
- Type-safe env vars with validation
- Automatic .env file loading
- IDE autocomplete support
- Fails loudly on missing required vars (better than silent KeyError at runtime)
"""

from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── App ───────────────────────────────────────────────────────────────
    APP_NAME: str = "RAG Document Q&A API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"

    # ── API Keys ──────────────────────────────────────────────────────────
    OPENAI_API_KEY: str = "your-key-here"  # Required in production

    # ── Embedding Model ───────────────────────────────────────────────────
    # text-embedding-3-small: 1536 dims, great cost/quality for most apps
    # text-embedding-3-large: 3072 dims, best quality, 2x cost
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    EMBEDDING_DIMENSIONS: int = 1536

    # ── LLM ───────────────────────────────────────────────────────────────
    LLM_MODEL: str = "gpt-4o-mini"   # Cheaper, fast; swap to gpt-4o for better reasoning
    LLM_TEMPERATURE: float = 0.1     # Low temp = more factual, less hallucination
    LLM_MAX_TOKENS: int = 1500

    # ── Chunking Strategy ─────────────────────────────────────────────────
    # Tradeoff: smaller chunks -> more precise retrieval, but lose context
    #           larger chunks -> more context, but may retrieve irrelevant parts
    CHUNK_SIZE: int = 1000       # Characters per chunk (empirically good for RAG)
    CHUNK_OVERLAP: int = 200     # Overlap prevents losing info at chunk boundaries
    CHUNK_STRATEGY: Literal["recursive", "semantic", "sentence"] = "recursive"

    # ── Retrieval ─────────────────────────────────────────────────────────
    RETRIEVAL_TOP_K: int = 5              # Return top 5 most similar chunks
    SIMILARITY_THRESHOLD: float = 0.7    # Minimum cosine similarity to include

    # ── Vector Store ──────────────────────────────────────────────────────
    # FAISS: In-process, blazing fast, no server needed - great for MVP/small scale
    # ChromaDB: Persistent, has metadata filtering, better for production
    VECTOR_STORE_TYPE: Literal["faiss", "chroma"] = "faiss"
    VECTOR_STORE_PATH: str = "./embeddings/vectorstore"
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8001

    # ── File Upload ───────────────────────────────────────────────────────
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: list = [".pdf", ".docx", ".txt", ".md"]
    UPLOAD_DIR: str = "./data/raw"

    # ── CORS ──────────────────────────────────────────────────────────────
    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:5173",
        "https://your-frontend-domain.vercel.app",
    ]

    # ── Redis (optional caching) ──────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379"
    CACHE_TTL_SECONDS: int = 3600  # 1 hour embedding cache

    # ── Rate Limiting ─────────────────────────────────────────────────────
    RATE_LIMIT_REQUESTS: int = 60    # per minute per IP
    RATE_LIMIT_WINDOW: int = 60

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024


@lru_cache()
def get_settings() -> Settings:
    """
    Cached singleton - reads .env once, not on every request.
    Use lru_cache so FastAPI's Depends() doesn't reload on each call.
    """
    return Settings()
