"""
chat_service.py — Thin service layer wrapping the RAG pipeline.

Separates HTTP concerns (FastAPI routes) from business logic (RAG pipeline).
In a larger system, this layer would handle:
- Conversation persistence (save/load from DB)
- User-specific document filtering
- Rate limiting per user
- Usage metering / billing
"""

import logging
import uuid
from typing import AsyncGenerator, Optional

from backend.core.rag_pipeline import RAGPipeline
from backend.models.schemas import ChatMessage, ChatRequest, ChatResponse

logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, rag_pipeline: RAGPipeline):
        self.rag = rag_pipeline
        # In-memory conversation store — replace with Redis or PostgreSQL
        self._conversations: dict[str, list[ChatMessage]] = {}

    async def chat(self, request: ChatRequest) -> ChatResponse:
        """Process a chat request through the RAG pipeline."""
        
        # Load conversation history if continuing a session
        history = request.conversation_history
        
        response = await self.rag.query(
            question=request.question,
            document_ids=request.document_ids,
            conversation_history=history,
            top_k=request.top_k,
        )
        
        return response

    async def stream_chat(
        self, request: ChatRequest
    ) -> AsyncGenerator[str, None]:
        """Streaming version for real-time token-by-token responses."""
        async for token in self.rag.stream_query(
            question=request.question,
            document_ids=request.document_ids,
            conversation_history=request.conversation_history,
        ):
            yield token
