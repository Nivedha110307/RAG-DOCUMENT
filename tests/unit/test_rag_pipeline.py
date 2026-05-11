"""
Unit tests for RAGPipeline — mocks LLM and vector store.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from langchain.schema import Document

from backend.models.schemas import QueryRequest
from backend.services.rag_pipeline import RAGPipeline


@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    store.similarity_search.return_value = [
        (
            Document(
                page_content="The capital of France is Paris.",
                metadata={
                    "document_id": "doc1",
                    "filename": "geography.pdf",
                    "chunk_id": "doc1_chunk_0",
                    "chunk_index": 0,
                },
            ),
            0.95,
        )
    ]
    return store


@pytest.fixture
def pipeline(mock_vector_store):
    with patch("backend.services.rag_pipeline.ChatOpenAI") as mock_llm_class:
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = MagicMock(
            content="Paris is the capital of France.",
            usage_metadata={"total_tokens": 42},
        )
        mock_llm_class.return_value = mock_llm
        p = RAGPipeline(vector_store=mock_vector_store)
        p.llm = mock_llm
        return p


class TestRAGPipeline:
    @pytest.mark.asyncio
    async def test_query_returns_answer_and_sources(self, pipeline, mock_vector_store):
        request = QueryRequest(query="What is the capital of France?", stream=False)
        response = await pipeline.query(request)

        assert "Paris" in response.answer
        assert len(response.sources) == 1
        assert response.sources[0].document_name == "geography.pdf"
        assert response.sources[0].similarity_score == 0.95

    @pytest.mark.asyncio
    async def test_query_with_no_results_handles_gracefully(self, pipeline, mock_vector_store):
        mock_vector_store.similarity_search.return_value = []
        request = QueryRequest(query="What is the meaning of life?", stream=False)
        response = await pipeline.query(request)
        # Should still return a response (LLM will say it doesn't have info)
        assert response.answer is not None
        assert response.sources == []

    def test_build_context_formats_sources(self, pipeline):
        results = [
            (
                Document(page_content="Content A", metadata={"filename": "docA.pdf", "page_number": 3}),
                0.9,
            )
        ]
        context = pipeline._build_context(results)
        assert "docA.pdf" in context
        assert "Content A" in context
        assert "Source 1" in context

    def test_sanitize_query_strips_whitespace(self):
        request = QueryRequest(query="  hello   world  ", stream=False)
        assert request.query == "hello world"
