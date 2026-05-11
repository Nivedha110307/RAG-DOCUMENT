"""
embeddings.py — Embedding generation with optional Redis caching.

WHAT ARE EMBEDDINGS?
Text embeddings convert words/sentences into high-dimensional vectors (lists of
numbers) where semantic similarity = geometric proximity.

"King" - "Man" + "Woman" ~ "Queen" (famous word2vec example)

For RAG, we embed both:
1. All document chunks (once, at upload time) -> stored in vector DB
2. Each user query (at search time) -> compared against stored embeddings

DIMENSIONS & MODELS:
- text-embedding-3-small: 1536 dims, best cost/perf ratio
- text-embedding-3-large: 3072 dims, 20% better but 2x cost
- ada-002 (legacy): 1536 dims, OpenAI's previous model

CACHING:
Embedding the same text twice wastes API credits. We cache by:
hash(text + model_name) -> embedding vector in Redis
TTL = 1 hour (configurable). On cache miss -> call API -> store result.
"""

import hashlib
import json
import logging
import time
from typing import Optional

from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

logger = logging.getLogger(__name__)


class EmbeddingService:
    """
    Wraps OpenAI's embedding API with caching and batching.
    
    Batching: We embed 100 chunks at once instead of 1 at a time.
    This reduces API roundtrips from O(n) to O(n/100) — critical for
    large documents.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        redis_client=None,          # Optional Redis client for caching
        cache_ttl: int = 3600,      # Cache embeddings for 1 hour
        batch_size: int = 100,      # Embed N texts per API call
    ):
        self.model = model
        self.dimensions = dimensions
        self.redis = redis_client
        self.cache_ttl = cache_ttl
        self.batch_size = batch_size

        # LangChain wrapper — handles retries, rate limits, token counting
        self._embedder = OpenAIEmbeddings(
            model=model,
            dimensions=dimensions,
            # chunk_size controls the batch size within LangChain itself
            chunk_size=batch_size,
        )

    def embed_documents(self, chunks: list[Document]) -> list[Document]:
        """
        Embed all chunks and attach embeddings to metadata.
        
        We mutate chunks in-place (add 'embedding' key to metadata) because
        LangChain's vectorstore.add_documents() expects this format.
        """
        texts = [c.page_content for c in chunks]
        
        start = time.perf_counter()
        embeddings = self._embed_batch(texts)
        elapsed_ms = (time.perf_counter() - start) * 1000

        logger.info(
            "Embedded %d chunks in %.1fms (%.1f ms/chunk)",
            len(chunks),
            elapsed_ms,
            elapsed_ms / max(len(chunks), 1),
        )

        for chunk, embedding in zip(chunks, embeddings):
            chunk.metadata["embedding"] = embedding

        return chunks

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single query string for similarity search.
        
        Note: We use embed_query() not embed_documents() because some models
        (like E5) use different prefixes for queries vs passages:
        - Query: "query: What is X?"
        - Passage: "passage: X is a..."
        OpenAI models don't need this, but it's good practice.
        """
        cache_key = self._cache_key(query)
        
        # Try cache first
        if self.redis:
            cached = self._get_cached(cache_key)
            if cached:
                logger.debug("Cache hit for query embedding")
                return cached

        embedding = self._embedder.embed_query(query)
        
        # Store in cache
        if self.redis:
            self._set_cached(cache_key, embedding)

        return embedding

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed texts in batches to avoid API rate limits.
        
        OpenAI limits: 1M tokens/minute. A typical chunk is ~250 tokens,
        so 4000 chunks/minute. For larger scale, implement exponential backoff.
        """
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            
            # Check cache for each item in batch
            embeddings_for_batch = []
            texts_to_embed = []
            indices_to_embed = []
            
            for j, text in enumerate(batch):
                cache_key = self._cache_key(text)
                if self.redis:
                    cached = self._get_cached(cache_key)
                    if cached:
                        embeddings_for_batch.append((j, cached))
                        continue
                texts_to_embed.append(text)
                indices_to_embed.append(j)
            
            # Embed cache misses
            if texts_to_embed:
                fresh = self._embedder.embed_documents(texts_to_embed)
                for idx, (orig_idx, embedding) in enumerate(
                    zip(indices_to_embed, fresh)
                ):
                    embeddings_for_batch.append((orig_idx, embedding))
                    if self.redis:
                        self._set_cached(
                            self._cache_key(texts_to_embed[idx]), embedding
                        )
            
            # Sort by original index to maintain order
            embeddings_for_batch.sort(key=lambda x: x[0])
            all_embeddings.extend([e for _, e in embeddings_for_batch])

        return all_embeddings

    def _cache_key(self, text: str) -> str:
        """
        Deterministic cache key: hash of (model + text).
        Including the model name prevents stale cache hits when model changes.
        """
        content = f"{self.model}:{text}"
        return f"embed:{hashlib.sha256(content.encode()).hexdigest()[:32]}"

    def _get_cached(self, key: str) -> Optional[list[float]]:
        try:
            data = self.redis.get(key)
            return json.loads(data) if data else None
        except Exception:
            return None

    def _set_cached(self, key: str, embedding: list[float]) -> None:
        try:
            self.redis.setex(key, self.cache_ttl, json.dumps(embedding))
        except Exception:
            pass  # Caching is best-effort, don't break on Redis failure
