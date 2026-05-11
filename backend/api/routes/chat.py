"""
Chat / query API routes.
Supports both standard and streaming (SSE) responses.
"""
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from backend.core.logging import get_logger
from backend.models.schemas import QueryRequest, QueryResponse
from backend.services.rag_pipeline import RAGPipeline
from backend.services.vector_store import get_vector_store

router = APIRouter(prefix="/chat", tags=["chat"])
logger = get_logger(__name__)


def get_pipeline() -> RAGPipeline:
    store = get_vector_store()
    return RAGPipeline(vector_store=store)


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question about uploaded documents",
)
async def query_documents(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """
    Standard (non-streaming) query endpoint.
    Returns complete answer with source citations.
    
    For streaming responses, use POST /chat/stream instead.
    """
    if request.stream:
        raise HTTPException(
            status_code=400,
            detail="Set stream=false for this endpoint. Use /chat/stream for streaming.",
        )
    try:
        return await pipeline.query(request)
    except Exception as e:
        logger.error("query_failed", error=str(e), query=request.query[:80])
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")


@router.post(
    "/stream",
    summary="Stream an answer token by token (SSE)",
    response_class=StreamingResponse,
)
async def stream_query(
    request: QueryRequest,
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """
    Server-Sent Events (SSE) streaming endpoint.
    
    Response format (newline-delimited JSON):
      data: {"type": "sources", "data": [...]}   ← citations first
      data: {"type": "token", "data": "Hello"}   ← each token
      data: {"type": "done"}                      ← sentinel
    
    Frontend should use EventSource or fetch() with ReadableStream.
    """
    try:
        return StreamingResponse(
            pipeline.stream_query(request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )
    except Exception as e:
        logger.error("stream_query_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
