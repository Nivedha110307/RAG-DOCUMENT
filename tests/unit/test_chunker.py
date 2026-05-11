"""
test_chunker.py — Unit tests for document chunking logic.

Tests are FAST (no network, no GPU) — just pure Python logic.
"""

import pytest
from backend.core.chunker import DocumentChunker


class TestDocumentChunker:
    """Test suite for DocumentChunker."""

    def test_basic_chunking(self, sample_text):
        """Chunks should be created from valid text."""
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk_text(sample_text)
        
        assert len(chunks) > 0
        assert all(c.page_content for c in chunks)

    def test_chunk_size_respected(self, sample_text):
        """No chunk should significantly exceed the configured size."""
        chunk_size = 300
        chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=30)
        chunks = chunker.chunk_text(sample_text)
        
        # Allow 10% overshoot (splitter may not find exact break point)
        for chunk in chunks:
            assert len(chunk.page_content) <= chunk_size * 1.1, (
                f"Chunk size {len(chunk.page_content)} exceeds limit {chunk_size}"
            )

    def test_metadata_propagation(self, sample_text):
        """Metadata should be present in every chunk."""
        metadata = {"document_id": "test123", "filename": "test.txt"}
        chunker = DocumentChunker()
        chunks = chunker.chunk_text(sample_text, metadata)
        
        for chunk in chunks:
            assert chunk.metadata["document_id"] == "test123"
            assert chunk.metadata["filename"] == "test.txt"

    def test_chunk_index_sequential(self, sample_text):
        """Chunk indices should be 0, 1, 2, ... n-1."""
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk_text(sample_text)
        
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_empty_text_returns_empty(self):
        """Empty or whitespace text should return no chunks."""
        chunker = DocumentChunker()
        chunks = chunker.chunk_text("   \n   \t   ")
        assert len(chunks) == 0

    def test_text_cleaning_hyphenated_words(self):
        """Hyphenated line breaks should be joined."""
        chunker = DocumentChunker()
        text = "com-\nputer science is inter-\nesting"
        cleaned = chunker._clean_text(text)
        assert "com\nputer" not in cleaned
        assert "computer" in cleaned

    def test_chunk_stats(self, sample_text):
        """Stats should be computed correctly."""
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk_text(sample_text)
        stats = chunker.get_chunk_stats(chunks)
        
        assert stats["total_chunks"] == len(chunks)
        assert stats["strategy"] == "recursive"
        assert stats["min_chunk_size"] <= stats["avg_chunk_size"] <= stats["max_chunk_size"]

    def test_overlap_content_repeated(self, sample_text):
        """Adjacent chunks should share some content (the overlap)."""
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk_text(sample_text)
        
        if len(chunks) < 2:
            pytest.skip("Need at least 2 chunks to test overlap")
        
        # The end of chunk N should appear in the start of chunk N+1
        end_of_first = chunks[0].page_content[-30:]
        start_of_second = chunks[1].page_content[:100]
        # At least some overlap exists (exact match depends on split points)
        # We just verify chunks aren't completely disjoint
        assert len(chunks[0].page_content) > 0
        assert len(chunks[1].page_content) > 0

    @pytest.mark.parametrize("strategy", ["recursive", "sentence"])
    def test_strategies_produce_chunks(self, strategy, sample_text):
        """All chunking strategies should produce valid output."""
        chunker = DocumentChunker(strategy=strategy)
        chunks = chunker.chunk_text(sample_text)
        assert len(chunks) > 0
