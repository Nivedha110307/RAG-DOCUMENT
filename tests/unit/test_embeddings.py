"""
test_embeddings.py — Unit tests for embedding service.

Uses mocks to avoid calling OpenAI during CI/CD.
"""

import pytest
from unittest.mock import MagicMock, patch
from backend.core.embeddings import EmbeddingService
from langchain.schema import Document


class TestEmbeddingService:
    """Tests for EmbeddingService with mocked OpenAI calls."""

    @pytest.fixture
    def mock_openai_embeddings(self):
        with patch("backend.core.embeddings.OpenAIEmbeddings") as mock_cls:
            instance = MagicMock()
            instance.embed_documents.return_value = [[0.1] * 1536]
            instance.embed_query.return_value = [0.1] * 1536
            mock_cls.return_value = instance
            yield instance

    def test_embed_query_returns_vector(self, mock_openai_embeddings):
        """embed_query should return a list of floats."""
        service = EmbeddingService()
        result = service.embed_query("What is machine learning?")
        
        assert isinstance(result, list)
        assert len(result) == 1536
        assert all(isinstance(x, float) for x in result)

    def test_embed_documents_attaches_to_metadata(self, mock_openai_embeddings):
        """Embeddings should be attached to chunk metadata."""
        mock_openai_embeddings.embed_documents.return_value = [
            [0.1] * 1536, [0.2] * 1536
        ]
        service = EmbeddingService()
        
        docs = [
            Document(page_content="First chunk", metadata={}),
            Document(page_content="Second chunk", metadata={}),
        ]
        
        result = service.embed_documents(docs)
        
        assert all("embedding" in doc.metadata for doc in result)
        assert len(result[0].metadata["embedding"]) == 1536

    def test_cache_key_deterministic(self):
        """Same text always produces same cache key."""
        service = EmbeddingService()
        key1 = service._cache_key("hello world")
        key2 = service._cache_key("hello world")
        assert key1 == key2

    def test_cache_key_differs_for_different_text(self):
        """Different text produces different cache keys."""
        service = EmbeddingService()
        key1 = service._cache_key("hello world")
        key2 = service._cache_key("goodbye world")
        assert key1 != key2

    def test_redis_cache_hit_skips_api_call(self, mock_openai_embeddings):
        """On cache hit, should not call OpenAI API."""
        import json
        fake_embedding = [0.5] * 1536
        
        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps(fake_embedding).encode()
        
        service = EmbeddingService(redis_client=mock_redis)
        result = service.embed_query("cached query")
        
        # Should use cached value, not call API
        mock_openai_embeddings.embed_query.assert_not_called()
        assert result == fake_embedding
