"""
test_endpoints.py — API integration tests using FastAPI TestClient.

These tests hit the actual HTTP endpoints but mock external services
(OpenAI, vector store) to run without credentials.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
import io


@pytest.fixture(scope="module")
def test_client():
    """Create a TestClient with mocked dependencies."""
    with patch("backend.main.document_service") as mock_doc_svc, \
         patch("backend.main.chat_service") as mock_chat_svc:
        
        # Mock document service
        from backend.models.schemas import DocumentUploadResponse, DocumentListResponse
        mock_doc_svc.process_document = AsyncMock(return_value=DocumentUploadResponse(
            document_id="test-doc-123",
            filename="test.txt",
            status="completed",
            chunk_count=5,
            message="Processed 5 chunks",
            processing_time_ms=150.0,
        ))
        mock_doc_svc.list_documents.return_value = []
        mock_doc_svc.delete_document.return_value = True
        mock_doc_svc.vector_store.get_document_count.return_value = 0
        
        # Mock chat service
        from backend.models.schemas import ChatResponse
        mock_chat_svc.chat = AsyncMock(return_value=ChatResponse(
            answer="Machine learning is a subset of AI.",
            sources=[],
            model_used="gpt-4o-mini",
            tokens_used=150,
            latency_ms=800.0,
        ))
        
        from backend.main import app
        with TestClient(app) as client:
            yield client


class TestHealthEndpoint:
    def test_health_returns_200(self, test_client):
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_health_has_required_fields(self, test_client):
        data = test_client.get("/health").json()
        assert "status" in data
        assert "version" in data
        assert "services" in data


class TestDocumentEndpoints:
    def test_list_documents_empty(self, test_client):
        response = test_client.get("/api/documents/")
        assert response.status_code == 200
        data = response.json()
        assert "documents" in data
        assert "total" in data

    def test_upload_txt_document(self, test_client):
        fake_file = io.BytesIO(b"This is a test document about AI and machine learning.")
        response = test_client.post(
            "/api/documents/upload",
            files={"file": ("test.txt", fake_file, "text/plain")},
        )
        assert response.status_code == 201
        data = response.json()
        assert "document_id" in data
        assert data["status"] == "completed"

    def test_delete_document_not_found(self, test_client):
        with patch("backend.main.document_service") as mock:
            mock.delete_document.return_value = False
            response = test_client.delete("/api/documents/nonexistent-id")
            assert response.status_code == 404


class TestChatEndpoints:
    def test_query_returns_answer(self, test_client):
        response = test_client.post(
            "/api/chat/query",
            json={
                "question": "What is machine learning?",
                "top_k": 3,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert "latency_ms" in data

    def test_query_too_short_rejected(self, test_client):
        """Questions under 3 chars should be rejected by Pydantic."""
        response = test_client.post(
            "/api/chat/query",
            json={"question": "Hi"},
        )
        assert response.status_code == 422  # Validation error

    def test_query_with_document_filter(self, test_client):
        response = test_client.post(
            "/api/chat/query",
            json={
                "question": "What are the main topics?",
                "document_ids": ["doc-abc", "doc-xyz"],
            },
        )
        assert response.status_code == 200
