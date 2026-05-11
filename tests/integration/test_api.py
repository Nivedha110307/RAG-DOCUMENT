"""
Integration tests for FastAPI endpoints.
Tests the full request/response cycle with mocked services.
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import io

# Patch settings before importing app
with patch.dict("os.environ", {
    "OPENAI_API_KEY": "test-key",
    "SECRET_KEY": "test-secret",
}):
    from backend.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self):
        with patch("backend.api.routes.health.get_vector_store") as mock:
            mock.return_value.get_collection_stats.return_value = {"total_chunks": 0}
            res = client.get("/api/v1/health")
        assert res.status_code == 200
        data = res.json()
        assert "status" in data
        assert "version" in data


class TestDocumentEndpoints:
    def test_list_documents_empty(self):
        res = client.get("/api/v1/documents/")
        assert res.status_code == 200
        data = res.json()
        assert "documents" in data
        assert isinstance(data["documents"], list)

    def test_upload_invalid_extension_returns_400(self):
        file_content = b"malicious content"
        res = client.post(
            "/api/v1/documents/upload",
            files={"file": ("malware.exe", io.BytesIO(file_content), "application/octet-stream")},
        )
        assert res.status_code == 400

    def test_get_nonexistent_document_returns_404(self):
        res = client.get("/api/v1/documents/nonexistent-id")
        assert res.status_code == 404


class TestChatEndpoints:
    def test_query_with_stream_true_returns_400(self):
        res = client.post(
            "/api/v1/chat/query",
            json={"query": "test", "stream": True},
        )
        assert res.status_code == 400

    def test_query_empty_string_returns_422(self):
        res = client.post(
            "/api/v1/chat/query",
            json={"query": "", "stream": False},
        )
        assert res.status_code == 422
