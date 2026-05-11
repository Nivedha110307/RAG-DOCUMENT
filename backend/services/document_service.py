"""
document_service.py — Business logic for document ingestion pipeline.

This is the "upload -> process -> store" flow:
1. Validate file (type, size, content)
2. Extract text (PDF/DOCX/TXT each need different parsers)
3. Chunk text into overlapping segments
4. Generate embeddings for each chunk
5. Store in vector database with metadata
"""

import logging
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import UploadFile

from backend.config import get_settings
from backend.core.chunker import DocumentChunker
from backend.core.vectorstore import VectorStoreBase
from backend.models.schemas import DocumentMetadata, DocumentUploadResponse
from backend.utils.file_utils import (
    extract_text_from_pdf,
    extract_text_from_docx,
    save_upload,
    compute_file_hash,
)

logger = logging.getLogger(__name__)
settings = get_settings()


class DocumentService:
    """
    Orchestrates the document ingestion pipeline.
    
    Dependencies (injected):
    - vector_store: Where embeddings are stored
    - chunker: How to split documents
    
    We keep a simple in-memory registry of document metadata.
    In production, replace this with PostgreSQL or DynamoDB.
    """

    def __init__(
        self,
        vector_store: VectorStoreBase,
        chunker: Optional[DocumentChunker] = None,
    ):
        self.vector_store = vector_store
        self.chunker = chunker or DocumentChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            strategy=settings.CHUNK_STRATEGY,
        )
        # Simple in-memory store — replace with DB in production
        self._documents: dict[str, DocumentMetadata] = {}

    async def process_document(self, file: UploadFile) -> DocumentUploadResponse:
        """
        Main ingestion pipeline: validate -> extract -> chunk -> embed -> store.
        
        Returns immediately with document_id; processing happens synchronously
        in this implementation. For production, offload to a background task queue
        (Celery/RQ/FastAPI BackgroundTasks) to avoid blocking the API.
        """
        start_time = time.perf_counter()
        document_id = str(uuid.uuid4())

        # ── Step 1: Validate ──────────────────────────────────────────────
        self._validate_file(file)

        # ── Step 2: Save to disk ──────────────────────────────────────────
        upload_path = await save_upload(file, settings.UPLOAD_DIR)
        file_size = upload_path.stat().st_size
        
        logger.info("Processing document: %s (%d bytes)", file.filename, file_size)

        # ── Step 3: Extract text ──────────────────────────────────────────
        file_ext = Path(file.filename).suffix.lower()
        text, page_metadata = self._extract_text(upload_path, file_ext)
        
        if not text.strip():
            raise ValueError(f"No extractable text found in {file.filename}. "
                           "The file may be image-based or corrupted.")

        # ── Step 4: Chunk ─────────────────────────────────────────────────
        metadata = {
            "document_id": document_id,
            "filename": file.filename,
            "file_type": file_ext,
            "file_path": str(upload_path),
        }
        chunks = self.chunker.chunk_text(text, metadata)
        
        logger.info("Split into %d chunks", len(chunks))

        # ── Step 5: Add unique chunk IDs ──────────────────────────────────
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = f"{document_id}_{i}"
            # Merge page-level metadata if available
            if page_metadata and i < len(page_metadata):
                chunk.metadata.update(page_metadata[i])

        # ── Step 6: Store in vector database ──────────────────────────────
        # Embedding happens inside vector_store.add_documents()
        self.vector_store.add_documents(chunks)

        # ── Step 7: Record metadata ───────────────────────────────────────
        doc_meta = DocumentMetadata(
            document_id=document_id,
            filename=file.filename,
            file_type=file_ext,
            file_size_bytes=file_size,
            chunk_count=len(chunks),
            processed=True,
        )
        self._documents[document_id] = doc_meta

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        
        logger.info(
            "Document %s processed: %d chunks in %.1fms",
            document_id, len(chunks), elapsed_ms
        )

        return DocumentUploadResponse(
            document_id=document_id,
            filename=file.filename,
            status="completed",
            chunk_count=len(chunks),
            message=f"Successfully processed {len(chunks)} chunks from {file.filename}",
            processing_time_ms=round(elapsed_ms, 2),
        )

    def _validate_file(self, file: UploadFile) -> None:
        """Security validation before processing any file."""
        if not file.filename:
            raise ValueError("No filename provided")
        
        ext = Path(file.filename).suffix.lower()
        if ext not in settings.ALLOWED_EXTENSIONS:
            raise ValueError(
                f"File type '{ext}' not supported. "
                f"Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )
        
        # Note: We check size after saving for streaming uploads.
        # Could check Content-Length header early if needed.

    def _extract_text(
        self,
        file_path: Path,
        file_ext: str,
    ) -> tuple[str, Optional[list[dict]]]:
        """
        Route to appropriate text extractor based on file type.
        
        Returns:
            - text: Full document text
            - page_metadata: Optional per-chunk metadata (e.g. page numbers)
        """
        if file_ext == ".pdf":
            return extract_text_from_pdf(file_path)
        elif file_ext == ".docx":
            text = extract_text_from_docx(file_path)
            return text, None
        elif file_ext in (".txt", ".md"):
            text = file_path.read_text(encoding="utf-8", errors="replace")
            return text, None
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

    def get_document(self, document_id: str) -> Optional[DocumentMetadata]:
        return self._documents.get(document_id)

    def list_documents(self) -> list[DocumentMetadata]:
        return list(self._documents.values())

    def delete_document(self, document_id: str) -> bool:
        if document_id not in self._documents:
            return False
        self.vector_store.delete_documents(document_id)
        del self._documents[document_id]
        return True
