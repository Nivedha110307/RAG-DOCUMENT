"""
rag_pipeline.py — The core RAG orchestration layer.

RAG PIPELINE EXPLAINED:
1. User asks a question
2. We RETRIEVE the most relevant document chunks using vector similarity
3. We AUGMENT the LLM prompt with those chunks as context
4. We GENERATE an answer grounded in the retrieved context

WHY RAG BEATS FINE-TUNING FOR DOCUMENT QA:
- Fine-tuning bakes knowledge into model weights (expensive, slow, stale)
- RAG retrieves fresh knowledge at query time (cheap, fast, always current)
- RAG can cite sources; fine-tuned models cannot
- RAG handles private/proprietary docs without exposing them to training

HALLUCINATION REDUCTION TECHNIQUES USED HERE:
1. Grounding check: if similarity scores are low, we say "I don't know"
2. System prompt: explicit instruction to only use provided context
3. Low temperature: reduces creative (but wrong) generation
4. Source citation: forces model to justify claims with evidence
"""

import logging
import time
from typing import AsyncGenerator, Optional

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from backend.core.vectorstore import VectorStoreBase
from backend.models.schemas import (
    ChatMessage,
    ChatResponse,
    SourceChunk,
)

logger = logging.getLogger(__name__)


# ── System Prompt Engineering ──────────────────────────────────────────────────
# This is the most impactful single change you can make to improve answer quality.
# Key principles:
# 1. Be explicit that the model should ONLY use provided context
# 2. Tell it to say "I don't know" rather than hallucinate
# 3. Ask for structured answers with citations
# 4. Set the tone/persona

RAG_SYSTEM_PROMPT = """You are a precise document analysis assistant. Your role is to answer questions based EXCLUSIVELY on the provided document context.

CRITICAL RULES:
1. Only use information from the [CONTEXT] sections below
2. If the context doesn't contain enough information to answer, say: "I don't have enough information in the provided documents to answer this question."
3. Never make up information not present in the context
4. Always cite which document/section your information comes from
5. Be concise and direct in your answers

FORMAT:
- Lead with a direct answer
- Support with relevant details from context
- End with the source citation: [Source: filename, chunk X]

If multiple chunks support your answer, cite all relevant sources."""

QUERY_CONTEXTUALIZATION_PROMPT = """Given the chat history and the latest user question, 
reformulate the question to be standalone (self-contained) so it can be understood 
without the chat history. Do NOT answer it, just reformulate if needed, otherwise return it as-is.

Chat history:
{chat_history}

Latest question: {question}

Standalone question:"""


class RAGPipeline:
    """
    Orchestrates the full Retrieval-Augmented Generation pipeline.
    
    This class is the heart of the system. It:
    1. Takes a user question + optional conversation history
    2. (Optional) Contextualizes the question using chat history
    3. Retrieves relevant chunks from the vector store
    4. Builds a grounded prompt with the retrieved context
    5. Generates an answer using the LLM
    6. Returns the answer with source citations
    """

    def __init__(
        self,
        vector_store: VectorStoreBase,
        llm_model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 1500,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
    ):
        self.vector_store = vector_store
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

        # Initialize LLM
        # Why gpt-4o-mini as default?
        # - 128K context window (can fit many chunks)
        # - ~10x cheaper than gpt-4o
        # - Excellent instruction-following for RAG tasks
        self.llm = ChatOpenAI(
            model=llm_model,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=True,  # Enable for streaming responses
        )
        
        # Smaller, cheaper model just for question contextualization
        self.contextualization_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=200,
        )

    async def query(
        self,
        question: str,
        document_ids: Optional[list[str]] = None,
        conversation_history: Optional[list[ChatMessage]] = None,
        top_k: Optional[int] = None,
    ) -> ChatResponse:
        """
        Full RAG pipeline — retrieve then generate.
        
        Args:
            question: User's question
            document_ids: Optional filter to specific documents
            conversation_history: Previous Q&A turns for multi-turn memory
            top_k: Override default retrieval count
        """
        start_time = time.perf_counter()
        k = top_k or self.top_k

        # Step 1: Contextualize question if there's conversation history
        # Example: "What does it say about that?" -> "What does the paper say about climate change?"
        standalone_question = question
        if conversation_history and len(conversation_history) > 0:
            standalone_question = await self._contextualize_question(
                question, conversation_history
            )
            logger.debug("Contextualized: '%s' -> '%s'", question, standalone_question)

        # Step 2: Retrieve relevant chunks
        filter_dict = None
        if document_ids:
            # Note: Chroma supports this natively; FAISS filters post-retrieval
            filter_dict = {"document_id": {"$in": document_ids}}

        retrieved = self.vector_store.similarity_search(
            query=standalone_question,
            k=k,
            filter=filter_dict,
            score_threshold=0.0,  # Get all scores, threshold below
        )

        # Step 3: Apply similarity threshold (grounding check)
        # If no chunks meet the threshold, return "I don't know" instead of hallucinating
        high_quality_chunks = [
            (doc, score) for doc, score in retrieved
            if score >= self.similarity_threshold
        ]
        
        grounded = len(high_quality_chunks) > 0
        chunks_to_use = high_quality_chunks if grounded else retrieved[:2]  # Use top 2 anyway

        # Step 4: Build context string for the prompt
        context_str = self._build_context_string(chunks_to_use)

        # Step 5: Build messages list (with conversation history for multi-turn)
        messages = self._build_messages(
            question=question,
            context=context_str,
            conversation_history=conversation_history or [],
            grounded=grounded,
        )

        # Step 6: Generate answer
        response = await self.llm.ainvoke(messages)
        answer = response.content

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Step 7: Build response with source citations
        sources = [
            SourceChunk(
                chunk_id=doc.metadata.get("chunk_id", f"chunk_{i}"),
                document_id=doc.metadata.get("document_id", ""),
                filename=doc.metadata.get("filename", "unknown"),
                content=doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                similarity_score=round(score, 4),
                page_number=doc.metadata.get("page_number"),
                chunk_index=doc.metadata.get("chunk_index", i),
            )
            for i, (doc, score) in enumerate(chunks_to_use)
        ]

        logger.info(
            "RAG query completed: %d chunks retrieved, grounded=%s, latency=%.1fms",
            len(chunks_to_use), grounded, elapsed_ms
        )

        return ChatResponse(
            answer=answer,
            sources=sources,
            model_used=self.llm.model_name,
            tokens_used=response.response_metadata.get("token_usage", {}).get("total_tokens", 0),
            latency_ms=round(elapsed_ms, 2),
            retrieval_scores=[round(score, 4) for _, score in chunks_to_use],
            grounded=grounded,
        )

    async def stream_query(
        self,
        question: str,
        document_ids: Optional[list[str]] = None,
        conversation_history: Optional[list[ChatMessage]] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Streaming variant — yields tokens as they're generated.
        
        Use this for the chat UI so the response appears word-by-word
        instead of waiting for the full response (reduces perceived latency).
        """
        retrieved = self.vector_store.similarity_search(
            query=question,
            k=self.top_k,
        )
        context_str = self._build_context_string(retrieved)
        messages = self._build_messages(
            question=question,
            context=context_str,
            conversation_history=conversation_history or [],
        )

        async for chunk in self.llm.astream(messages):
            if chunk.content:
                yield chunk.content

    def _build_context_string(self, chunks: list[tuple]) -> str:
        """
        Format retrieved chunks into a numbered context block.
        
        Each chunk is labeled with its source so the LLM can cite it.
        Example output:
            [Context 1 - report.pdf, chunk 3]
            The revenue grew by 23% in Q3...
            
            [Context 2 - report.pdf, chunk 7]
            Operating costs decreased due to...
        """
        if not chunks:
            return "No relevant context found in the documents."
        
        parts = []
        for i, (doc, score) in enumerate(chunks, 1):
            filename = doc.metadata.get("filename", "unknown")
            chunk_idx = doc.metadata.get("chunk_index", i)
            page = doc.metadata.get("page_number")
            
            header = f"[Context {i} - {filename}, chunk {chunk_idx}"
            if page:
                header += f", page {page}"
            header += f"] (relevance: {score:.2f})"
            
            parts.append(f"{header}\n{doc.page_content}")
        
        return "\n\n".join(parts)

    def _build_messages(
        self,
        question: str,
        context: str,
        conversation_history: list[ChatMessage],
        grounded: bool = True,
    ) -> list:
        """
        Construct the full message array for the LLM.
        
        Structure:
        [SystemMessage] <- RAG instructions + context
        [HumanMessage, AIMessage, ...] <- conversation history
        [HumanMessage] <- current question
        
        We inject the context into the system message so it's available
        for the entire conversation, not just the current turn.
        """
        system_content = f"{RAG_SYSTEM_PROMPT}\n\n--- DOCUMENT CONTEXT ---\n{context}\n--- END CONTEXT ---"
        
        if not grounded:
            system_content += "\n\nNOTE: The retrieved context has low relevance to the question. If you cannot answer from the context, clearly state you don't have sufficient information."

        messages = [SystemMessage(content=system_content)]
        
        # Add conversation history (limited to last 6 turns to avoid token bloat)
        for msg in conversation_history[-6:]:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))
        
        # Add current question
        messages.append(HumanMessage(content=question))
        
        return messages

    async def _contextualize_question(
        self,
        question: str,
        history: list[ChatMessage],
    ) -> str:
        """
        Rewrite a follow-up question to be standalone.
        
        Example:
        History: Q: "What is the revenue?" A: "Revenue was $5M in Q3."
        Follow-up: "And what about expenses?"
        -> Standalone: "What were the expenses in Q3?"
        
        This is crucial for multi-turn conversations where pronouns and
        references only make sense in context.
        """
        history_str = "\n".join(
            f"{msg.role}: {msg.content}" for msg in history[-4:]
        )
        
        prompt = QUERY_CONTEXTUALIZATION_PROMPT.format(
            chat_history=history_str,
            question=question,
        )
        
        response = await self.contextualization_llm.ainvoke(
            [HumanMessage(content=prompt)]
        )
        return response.content.strip()
