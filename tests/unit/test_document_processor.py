"""
Unit tests for DocumentProcessor.
Uses pytest + mocking to avoid real file system / API calls.
"""
import os
import sys
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parents[2]))

from backend.services.document_processor import DocumentProcessor


class TestDocumentProcessor:
    def setup_method(self):
        self.processor = DocumentProcessor()

    def test_validate_file_valid_pdf(self):
        """Valid PDF under size limit should not raise."""
        self.processor.validate_file("test.pdf", "application/pdf", 1024 * 1024)

    def test_validate_file_invalid_extension(self):
        """Unsupported extension should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported file type"):
            self.processor.validate_file("malware.exe", "application/octet-stream", 100)

    def test_validate_file_too_large(self):
        """Files over 50MB should raise ValueError."""
        with pytest.raises(ValueError, match="too large"):
            self.processor.validate_file("big.pdf", "application/pdf", 60 * 1024 * 1024)

    def test_clean_text_removes_extra_whitespace(self):
        from langchain.schema import Document
        doc = Document(page_content="Hello    world\n\n\n\nNew paragraph")
        cleaned = self.processor._clean_text(doc)
        assert "    " not in cleaned.page_content
        assert "\n\n\n" not in cleaned.page_content

    @pytest.mark.asyncio
    @patch("backend.services.document_processor.PyPDFLoader")
    async def test_process_file_returns_chunks_and_metadata(self, mock_loader_class):
        """process_file should return non-empty chunks with correct metadata."""
        from langchain.schema import Document

        mock_doc = Document(
            page_content="This is a test document with enough content to form at least one chunk.",
            metadata={"page": 1},
        )
        mock_loader = MagicMock()
        mock_loader.load.return_value = [mock_doc]
        mock_loader_class.return_value = mock_loader

        with patch("pathlib.Path.stat") as mock_stat:
            mock_stat.return_value.st_size = 1024
            file_path = Path("/fake/test.pdf")
            file_path.suffix = ".pdf"

            chunks, meta = await self.processor.process_file(
                file_path=file_path,
                document_id="test-doc-123",
                filename="test.pdf",
            )

        assert len(chunks) >= 1
        assert all(c.metadata["document_id"] == "test-doc-123" for c in chunks)
        assert meta["filename"] == "test.pdf"
        assert meta["num_chunks"] == len(chunks)
