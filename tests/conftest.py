"""
conftest.py — pytest fixtures shared across all tests.

Fixtures:
- mock_settings: Override env vars for testing
- mock_vector_store: In-memory vector store (no OpenAI calls)
- mock_embeddings: Returns fake 1536-dim vectors
- test_client: FastAPI TestClient for API tests
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from langchain.schema import Document


@pytest.fixture(scope="session")
def mock_settings():
    """Override settings for test environment."""
    with patch("backend.config.get_settings") as mock:
        settings = MagicMock()
        settings.OPENAI_API_KEY = "test-key-not-real"
        settings.EMBEDDING_MODEL = "text-embedding-3-small"
        settings.EMBEDDING_DIMENSIONS = 1536
        settings.LLM_MODEL = "gpt-4o-mini"
        settings.LLM_TEMPERATURE = 0.1
        settings.LLM_MAX_TOKENS = 1500
        settings.CHUNK_SIZE = 500
        settings.CHUNK_OVERLAP = 50
        settings.CHUNK_STRATEGY = "recursive"
        settings.RETRIEVAL_TOP_K = 3
        settings.SIMILARITY_THRESHOLD = 0.5
        settings.VECTOR_STORE_TYPE = "faiss"
        settings.VECTOR_STORE_PATH = "/tmp/test_vectorstore"
        settings.ALLOWED_EXTENSIONS = [".pdf", ".docx", ".txt"]
        settings.MAX_FILE_SIZE_MB = 10
        settings.max_file_size_bytes = 10 * 1024 * 1024
        settings.UPLOAD_DIR = "/tmp/test_uploads"
        settings.CORS_ORIGINS = ["http://localhost:3000"]
        settings.APP_NAME = "Test RAG API"
        settings.APP_VERSION = "test"
        mock.return_value = settings
        yield settings


@pytest.fixture
def sample_documents():
    """Sample LangChain Documents for testing."""
    return [
        Document(
            page_content="Machine learning is a subset of artificial intelligence.",
            metadata={"document_id": "doc1", "filename": "ml.txt", "chunk_index": 0}
        ),
        Document(
            page_content="Neural networks are inspired by biological neurons.",
            metadata={"document_id": "doc1", "filename": "ml.txt", "chunk_index": 1}
        ),
        Document(
            page_content="RAG combines retrieval with language model generation.",
            metadata={"document_id": "doc2", "filename": "rag.txt", "chunk_index": 0}
        ),
    ]


@pytest.fixture
def mock_embeddings():
    """Return fake embeddings (1536 zeros) without calling OpenAI."""
    with patch("langchain_openai.OpenAIEmbeddings") as mock_cls:
        instance = mock_cls.return_value
        instance.embed_documents.return_value = [[0.0] * 1536, [0.0] * 1536, [0.0] * 1536]
        instance.embed_query.return_value = [0.0] * 1536
        yield instance


@pytest.fixture
def sample_text():
    return """
    Introduction to Machine Learning
    
    Machine learning is a method of data analysis that automates analytical model building.
    It is based on the idea that systems can learn from data, identify patterns and make
    decisions with minimal human intervention.
    
    Types of Machine Learning:
    
    Supervised Learning uses labeled training data to learn a mapping from inputs to outputs.
    Common algorithms include linear regression, decision trees, and neural networks.
    
    Unsupervised Learning finds hidden patterns or intrinsic structures in input data.
    Clustering and dimensionality reduction are common unsupervised tasks.
    
    Reinforcement Learning trains agents to make sequences of decisions by rewarding
    desired behaviors and punishing undesired ones.
    """
