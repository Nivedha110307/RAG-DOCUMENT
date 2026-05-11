"""
chunker.py — Document chunking strategies.

WHY CHUNKING MATTERS:
Vector databases store fixed-size text segments. We chunk documents because:
1. LLMs have context windows (can't process 100-page PDFs directly)
2. Smaller chunks = more precise retrieval (find the exact paragraph)
3. Embeddings degrade in quality on very long text

STRATEGIES:
- Recursive: Split on paragraph > sentence > word boundaries (good default)
- Semantic: Split where topic changes (better quality, slower)
- Sentence: Split on punctuation (fast, good for structured text)

OVERLAP:
We intentionally repeat ~200 chars at chunk boundaries so we don't lose
context when a key sentence spans two chunks.
"""

import re
from typing import Optional
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from langchain.schema import Document


class DocumentChunker:
    """
    Converts raw document text into overlapping chunks suitable for embedding.
    
    Design decision: We return LangChain Document objects which carry both
    the text content AND metadata (source, page, chunk_index). This metadata
    is stored in the vector store and returned with retrieved chunks so users
    can see exactly where each answer came from.
    """

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        strategy: str = "recursive",
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self._splitter = self._build_splitter()

    def _build_splitter(self):
        """
        Factory method — separates splitter construction from chunking logic.
        Makes it easy to swap strategies without changing chunk() API.
        """
        if self.strategy == "recursive":
            # RecursiveCharacterTextSplitter tries splitting on:
            # "\n\n" (paragraphs) -> "\n" (lines) -> " " (words) -> ""
            # This preserves natural language structure as much as possible.
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
                length_function=len,
                is_separator_regex=False,
            )
        elif self.strategy == "sentence":
            # Token-aware splitting — respects model token limits precisely
            return SentenceTransformersTokenTextSplitter(
                chunk_overlap=self.chunk_overlap // 4,
                tokens_per_chunk=self.chunk_size // 4,  # ~4 chars/token
            )
        else:
            # Default fallback
            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

    def chunk_text(
        self,
        text: str,
        metadata: Optional[dict] = None,
    ) -> list[Document]:
        """
        Split raw text into chunks with metadata attached.
        
        Args:
            text: Raw document text
            metadata: Dict merged into each chunk's metadata
                      (document_id, filename, file_type, etc.)
        
        Returns:
            List of LangChain Documents, each with:
            - page_content: The chunk text
            - metadata: source info + chunk_index
        """
        base_metadata = metadata or {}
        
        # Clean text before chunking
        cleaned = self._clean_text(text)
        
        # Create a single Document then split it
        # This ensures metadata flows through to all chunks
        doc = Document(page_content=cleaned, metadata=base_metadata)
        chunks = self._splitter.split_documents([doc])
        
        # Inject chunk index so we can reconstruct order
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
            # Word count for display
            chunk.metadata["word_count"] = len(chunk.page_content.split())
        
        return chunks

    def _clean_text(self, text: str) -> str:
        """
        Normalize raw extracted text.
        
        PDF extraction often introduces:
        - Extra whitespace and newlines
        - Hyphenated words split across lines ("com-\nputer")
        - Non-breaking spaces (\xa0)
        - Form feed characters (\x0c)
        """
        # Fix hyphenated line breaks
        text = re.sub(r"-\n(\w)", r"\1", text)
        # Normalize whitespace
        text = re.sub(r"\n{3,}", "\n\n", text)  # Max 2 consecutive newlines
        text = re.sub(r"[ \t]+", " ", text)       # Collapse spaces/tabs
        text = text.replace("\xa0", " ")           # Non-breaking space
        text = text.replace("\x0c", "\n")          # Form feed -> newline
        return text.strip()

    def get_chunk_stats(self, chunks: list[Document]) -> dict:
        """Compute statistics about chunks — useful for debugging and evaluation."""
        if not chunks:
            return {}
        
        sizes = [len(c.page_content) for c in chunks]
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(sizes) / len(sizes),
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes),
            "total_characters": sum(sizes),
            "strategy": self.strategy,
        }
