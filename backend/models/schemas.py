"""
Pydantic models for request validation and response serialization.
Strong typing prevents runtime surprises and auto-generates OpenAPI docs.
"""
from datetime import datetime
from typing import List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator


class DocumentUploadResponse(BaseModel):
    document_id: UUID = Field(default_factory=uuid4)
    filename: str
    file_size_bytes: int
    num_chunks: int
    num_pages: Optional[int] = None
    status: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    message: str = "Document processed successfully"


class DocumentMetadata(BaseModel):
    document_id: UUID
    filename: str
    file_type: str
    file_size_bytes: int
    num_chunks: int
    num_pages: Optional[int]
    created_at: datetime
    status: str


class DocumentListResponse(BaseModel):
    documents: List[DocumentMetadata]
    total: int


class SourceChunk(BaseModel):
    chunk_id: str
    document_id: str
    document_name: str
    content: str
    page_number: Optional[int] = None
    similarity_score: float = Field(ge=0.0, le=1.0)
    chunk_index: int


class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    document_ids: Optional[List[str]] = None
    top_k: int = Field(default=5, ge=1, le=20)
    chat_history: Optional[List[ChatMessage]] = Field(default=[])
    stream: bool = Field(default=True)

    @validator("query")
    def sanitize_query(cls, v: str) -> str:
        return " ".join(v.strip().split())


class QueryResponse(BaseModel):
    query_id: UUID = Field(default_factory=uuid4)
    query: str
    answer: str
    sources: List[SourceChunk]
    model_used: str
    tokens_used: int
    latency_ms: float
    created_at: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    status: str
    version: str
    components: dict
    uptime_seconds: float


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    status_code: int
