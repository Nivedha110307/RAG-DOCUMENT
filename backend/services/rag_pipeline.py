"""
The RAG (Retrieval-Augmented Generation) pipeline.

Flow:
  User query
    → Embed query
    → Retrieve top-K similar chunks
    → (Optional) Rerank chunks
    → Build augmented prompt
    → Call LLM
    → Return answer + citations

Why RAG beats plain LLM:
- LLM knowledge is frozen at training cutoff
- RAG retrieves real-time, user-specific context
- Source citations make answers verifiable
- Smaller, cheaper models perform better with retrieval
"""
import time
from typing import AsyncGenerator, List, Optional

from langchain_community.llms import Ollama
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import ChatPromptTemplate

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.models.schemas import QueryRequest, QueryResponse, SourceChunk
from backend.services.vector_store import BaseVectorStore

logger = get_logger(__name__)


SYSTEM_PROMPT = """You are a precise, helpful assistant that answers questions \
based ONLY on the provided document context.

Rules:
1. Answer using ONLY information from the context below.
2. If the context doesn't contain the answer, say "I don't have enough information \
   in the provided documents to answer this."
3. Always cite which document/section your answer comes from.
4. Be concise and factual — avoid speculation.
5. If multiple sources support an answer, synthesize them coherently.

Context from documents:
{context}

Conversation history is provided for continuity — reference it for follow-up questions.
"""


class RAGPipeline:
    """
    Orchestrates retrieval + generation.
    Stateless — safe to use as a singleton in FastAPI dependency injection.
    """

    def __init__(self, vector_store: BaseVectorStore):
        self.vector_store = vector_store
        self.llm = Ollama(
            model="phi3:mini",
            temperature=0.1,
        )

    async def query(self, request: QueryRequest) -> QueryResponse:
        """
        Non-streaming query. Returns complete response with sources.
        """
        start_time = time.time()

        # 1. Build metadata filter if specific docs requested
        filter_meta = None
        if request.document_ids:
            filter_meta = {"document_id": {"$in": request.document_ids}}

        # 2. Retrieve relevant chunks
        raw_results = self.vector_store.similarity_search(
            query=request.query,
            k=request.top_k,
            filter_metadata=filter_meta,
            score_threshold=settings.SIMILARITY_THRESHOLD,
        )

        if not raw_results:
            logger.warning("no_chunks_retrieved", query=request.query[:80])

        # 3. Optional reranking (cross-encoder improves precision)
        if settings.RERANKING_ENABLED and len(raw_results) > 1:
            raw_results = await self._rerank(request.query, raw_results)

        # 4. Build context string for prompt
        context = self._build_context(raw_results)

        # 5. Build source citations for response
        sources = self._build_sources(raw_results)

        # 6. Assemble messages (system + history + user query)
        messages = self._build_messages(request, raw_results)

        # 7. Call LLM
        response = await self.llm.ainvoke(messages)
        answer = str(answer)

        elapsed_ms = (time.time() - start_time) * 1000
        tokens_used = response.usage_metadata.get("total_tokens", 0) if hasattr(response, "usage_metadata") else 0

        logger.info(
            "query_completed",
            query=request.query[:60],
            num_sources=len(sources),
            elapsed_ms=round(elapsed_ms, 2),
            tokens=tokens_used,
        )

        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            model_used=settings.OPENAI_CHAT_MODEL,
            tokens_used=tokens_used,
            latency_ms=round(elapsed_ms, 2),
        )

    async def stream_query(
        self, request: QueryRequest
    ) -> AsyncGenerator[str, None]:
        """
        Streaming query. Yields tokens as they arrive from the LLM.
        Use with FastAPI StreamingResponse for real-time UX.
        """
        filter_meta = None
        if request.document_ids:
            filter_meta = {"document_id": {"$in": request.document_ids}}

        raw_results = self.vector_store.similarity_search(
            query=request.query,
            k=request.top_k,
            filter_metadata=filter_meta,
            score_threshold=settings.SIMILARITY_THRESHOLD,
        )

        if settings.RERANKING_ENABLED and len(raw_results) > 1:
            raw_results = await self._rerank(request.query, raw_results)

        context = self._build_context(raw_results)
        sources = self._build_sources(raw_results)
        messages = self._build_messages(request, raw_results)

        # Yield sources first (as JSON header), then stream tokens
        import json
        sources_json = json.dumps([s.dict() for s in sources])
        yield f"data: {{\"type\": \"sources\", \"data\": {sources_json}}}\n\n"

        # Stream tokens
        async for chunk in self.llm.astream(messages):
            token = str(chunk).replace("\n", "\\n")

            if token.strip():
                yield f"data: {{\"type\": \"token\", \"data\": \"{token}\"}}\n\n"
        yield "data: {\"type\": \"done\"}\n\n"

    def _build_context(self, results: list) -> str:
        """
        Format retrieved chunks into a readable context block.
        Numbers each source for citation in the answer.
        """
        if not results:
            return "No relevant documents found."

        parts = []
        for i, (doc, score) in enumerate(results, 1):
            meta = doc.metadata
            header = f"[Source {i}] {meta.get('filename', 'Unknown')} "
            if page := meta.get("page_number"):
                header += f"(Page {page})"
            parts.append(f"{header}\n{doc.page_content}")

        return "\n\n---\n\n".join(parts)

    def _build_sources(self, results: list) -> List[SourceChunk]:
        """Convert retrieval results into SourceChunk response objects."""
        sources = []
        for i, (doc, score) in enumerate(results):
            meta = doc.metadata
            sources.append(SourceChunk(
                chunk_id=meta.get("chunk_id", f"chunk_{i}"),
                document_id=meta.get("document_id", ""),
                document_name=meta.get("filename", "Unknown"),
                content=doc.page_content[:500] + ("..." if len(doc.page_content) > 500 else ""),
                page_number=meta.get("page_number"),
                similarity_score=round(abs(float(score)), 4),
                chunk_index=meta.get("chunk_index", i),
            ))
        return sources

    def _build_messages(self, request: QueryRequest, retrieved_docs) -> str:
        """Build prompt using retrieved document context."""

        # Combine retrieved document chunks
        context = "\n\n".join(
            [doc.page_content for doc, _ in retrieved_docs]
        )

        # Include recent chat history
        history = ""

        if request.chat_history:
            for msg in request.chat_history[-6:]:
                history += f"{msg.role}: {msg.content}\n"

        # Final prompt
        prompt = f"""
        You are a helpful AI assistant.

        Use ONLY the provided context to answer the user's question.

        If the answer is not present in the context, say:
        "I could not find the answer in the uploaded documents."

        Context:
        {context}

        Chat History:
        {history}

        Question:
        {request.query}

        Answer:
        """

        return prompt

    async def _rerank(self, query: str, results: list) -> list:
        """
        Lightweight reranking using semantic similarity scoring.
        In production, use a cross-encoder (ms-marco-MiniLM-L-6-v2).
        Here we re-sort by score as a fallback if cross-encoder unavailable.
        """
        try:
            from sentence_transformers import CrossEncoder
            model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            pairs = [(query, doc.page_content) for doc, _ in results]
            scores = model.predict(pairs)
            reranked = sorted(
                zip([doc for doc, _ in results], scores),
                key=lambda x: x[1],
                reverse=True,
            )
            return [(doc, float(score)) for doc, score in reranked]
        except ImportError:
            # Graceful degradation: keep original ranking
            logger.warning("cross_encoder_unavailable_skipping_rerank")
            return results
