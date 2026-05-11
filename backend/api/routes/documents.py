"""
Document management API routes.
Upload, list, delete, and inspect documents.
"""
import os
import shutil
import uuid
from pathlib import Path
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.models.schemas import (
    DocumentListResponse,
    DocumentMetadata,
    DocumentUploadResponse,
    ErrorResponse,
)
from backend.services.document_processor import DocumentProcessor
from backend.services.vector_store import BaseVectorStore, get_vector_store

router = APIRouter(prefix="/documents", tags=["documents"])
logger = get_logger(__name__)

# In-memory document registry (replace with PostgreSQL in production)
_document_registry: dict = {}


def get_processor() -> DocumentProcessor:
    return DocumentProcessor()


def get_store() -> BaseVectorStore:
    return get_vector_store()


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and process a document",
)
async def upload_document(
    file: UploadFile = File(...),
    processor: DocumentProcessor = Depends(get_processor),
    store: BaseVectorStore = Depends(get_store),
):
    """
    Upload a PDF, DOCX, or TXT file for indexing.
    
    Pipeline:
    1. Validate file type and size
    2. Save to disk
    3. Extract and chunk text
    4. Generate embeddings
    5. Store in vector database
    """
    # Read file bytes for size validation
    content = await file.read()
    file_size = len(content)

    try:
        processor.validate_file(file.filename, file.content_type, file_size)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Generate unique document ID
    doc_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix
    safe_filename = f"{doc_id}{ext}"
    file_path = Path(settings.UPLOAD_DIR) / safe_filename

    # Save file to disk
    Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(content)

    # Process document (extract → chunk → embed)
    try:
        chunks, metadata = await processor.process_file(
            file_path=file_path,
            document_id=doc_id,
            filename=file.filename,
        )
        store.add_documents(chunks)
    except Exception as e:
        # Clean up file on failure
        file_path.unlink(missing_ok=True)
        logger.error("document_processing_failed", doc_id=doc_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")

    # Register document
    _document_registry[doc_id] = metadata

    logger.info("document_uploaded", doc_id=doc_id, filename=file.filename)

    return DocumentUploadResponse(
        document_id=doc_id,
        filename=file.filename,
        file_size_bytes=file_size,
        num_chunks=metadata["num_chunks"],
        num_pages=metadata.get("num_pages"),
        status="ready",
    )


@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="List all uploaded documents",
)
async def list_documents():
    """Returns all documents currently indexed in the vector store."""
    from datetime import datetime
    docs = [
        DocumentMetadata(
            document_id=v["document_id"],
            filename=v["filename"],
            file_type=v["file_type"],
            file_size_bytes=v["file_size_bytes"],
            num_chunks=v["num_chunks"],
            num_pages=v.get("num_pages"),
            created_at=datetime.utcnow(),
            status="ready",
        )
        for v in _document_registry.values()
    ]
    return DocumentListResponse(documents=docs, total=len(docs))


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document and its embeddings",
)
async def delete_document(
    document_id: str,
    store: BaseVectorStore = Depends(get_store),
):
    """Remove document chunks from the vector store and clean up storage."""
    if document_id not in _document_registry:
        raise HTTPException(status_code=404, detail="Document not found")

    store.delete_document(document_id)
    _document_registry.pop(document_id, None)

    logger.info("document_deleted", document_id=document_id)


@router.get(
    "/{document_id}",
    response_model=DocumentMetadata,
    summary="Get document metadata",
)
async def get_document(document_id: str):
    """Fetch metadata for a specific document."""
    from datetime import datetime
    if document_id not in _document_registry:
        raise HTTPException(status_code=404, detail="Document not found")
    v = _document_registry[document_id]
    return DocumentMetadata(
        document_id=v["document_id"],
        filename=v["filename"],
        file_type=v["file_type"],
        file_size_bytes=v["file_size_bytes"],
        num_chunks=v["num_chunks"],
        num_pages=v.get("num_pages"),
        created_at=datetime.utcnow(),
        status="ready",
    )
