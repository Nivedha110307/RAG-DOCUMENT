"""
Document ingestion pipeline: load → clean → chunk → embed → store.
Supports PDF, DOCX, and TXT files with async processing.
"""
import asyncio
import hashlib
import os
import time
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain.schema import Document

from backend.core.config import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


class DocumentProcessor:
    """
    Handles the complete document ingestion pipeline.
    
    Why RecursiveCharacterTextSplitter?
    - Tries to split on paragraphs → sentences → words in that order
    - Preserves semantic meaning better than fixed-size splitting
    - The overlap parameter ensures context isn't lost at chunk boundaries
    """

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    async def process_file(
        self,
        file_path: Path,
        document_id: str,
        filename: str,
    ) -> Tuple[List[Document], dict]:
        """
        Full pipeline: load raw file → extract text → chunk → attach metadata.
        Returns (chunks, metadata_dict).
        """
        start_time = time.time()
        logger.info("processing_document", doc_id=document_id, filename=filename)

        # 1. Load raw text from file
        raw_docs = await self._load_document(file_path)
        num_pages = len(raw_docs)

        # 2. Clean text
        cleaned_docs = [self._clean_text(doc) for doc in raw_docs]

        # 3. Chunk into segments
        chunks = self.text_splitter.split_documents(cleaned_docs)

        # 4. Attach rich metadata to every chunk
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "document_id": document_id,
                "filename": filename,
                "chunk_index": i,
                "chunk_id": f"{document_id}_chunk_{i}",
                "file_type": file_path.suffix.lstrip("."),
                "content_hash": hashlib.md5(chunk.page_content.encode()).hexdigest(),
            })

        elapsed = (time.time() - start_time) * 1000
        logger.info(
            "document_processed",
            doc_id=document_id,
            num_pages=num_pages,
            num_chunks=len(chunks),
            elapsed_ms=round(elapsed, 2),
        )

        metadata = {
            "document_id": document_id,
            "filename": filename,
            "num_pages": num_pages,
            "num_chunks": len(chunks),
            "file_type": file_path.suffix.lstrip("."),
            "file_size_bytes": file_path.stat().st_size,
        }

        return chunks, metadata

    async def _load_document(self, file_path: Path) -> List[Document]:
        """
        Select the correct loader based on file extension.
        All loaders run in a thread pool to avoid blocking the event loop.
        """
        ext = file_path.suffix.lower()

        loader_map = {
            ".pdf": lambda: PyPDFLoader(str(file_path)),
            ".docx": lambda: Docx2txtLoader(str(file_path)),
            ".txt": lambda: TextLoader(str(file_path), encoding="utf-8"),
            ".md": lambda: UnstructuredMarkdownLoader(str(file_path)),
        }

        if ext not in loader_map:
            raise ValueError(f"Unsupported file type: {ext}")

        loader = loader_map[ext]()

        # Run blocking IO in thread pool — keeps FastAPI's event loop free
        loop = asyncio.get_event_loop()
        documents = await loop.run_in_executor(None, loader.load)
        return documents

    def _clean_text(self, doc: Document) -> Document:
        """
        Normalize whitespace, remove control characters.
        Clean text → better embeddings → better retrieval.
        """
        text = doc.page_content
        # Collapse multiple newlines / spaces
        import re
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"[^\x00-\x7F\u00C0-\u024F]", " ", text)  # keep latin chars
        text = text.strip()
        doc.page_content = text
        return doc

    def validate_file(self, filename: str, content_type: str, size_bytes: int) -> None:
        """
        Pre-upload validation. Raises ValueError with a user-friendly message.
        Never trust client-provided MIME types — validate extension too.
        """
        ext = Path(filename).suffix.lstrip(".").lower()
        if ext not in settings.ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type '.{ext}'. "
                f"Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )

        max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        if size_bytes > max_bytes:
            raise ValueError(
                f"File too large ({size_bytes / 1024 / 1024:.1f} MB). "
                f"Maximum: {settings.MAX_UPLOAD_SIZE_MB} MB"
            )
