"""
vectorstore.py — Vector database abstraction layer.

FAISS vs ChromaDB — WHEN TO USE WHICH:

FAISS (Facebook AI Similarity Search)
+ Blazing fast (pure C++ under the hood, no network overhead)
+ No server needed (runs in-process)
+ Great for up to ~1M vectors on a single machine
- No persistence by default (must save/load manually)
- No metadata filtering (filter post-retrieval)
- Harder to scale horizontally

ChromaDB
+ Persistent out of the box
+ Rich metadata filtering (WHERE file_type='pdf' AND uploaded_after='2024')
+ REST API server mode for multi-process access
+ Growing ecosystem
- Slower than FAISS for pure ANN search
- Adds operational complexity (another service to run)

RECOMMENDATION: Start with FAISS. When you need metadata filtering or
multiple backend processes, migrate to ChromaDB. The abstraction layer
below makes this a 1-line config change.

HOW VECTOR SEARCH WORKS:
1. We store N embedding vectors of dimension D (e.g. 1536)
2. At query time: compute cosine similarity between query vector and all N stored vectors
3. Return top-K most similar vectors (and their associated text chunks)

Cosine similarity = dot(a, b) / (|a| * |b|)
Range: -1 (opposite) to 1 (identical). Good threshold: >0.7 for RAG.

FAISS uses HNSW (Hierarchical Navigable Small World) index for approximate
nearest neighbor search — finds ~99% of true nearest neighbors in O(log N)
instead of O(N), making million-vector search practical.
"""

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)


class VectorStoreBase(ABC):
    """Abstract interface — lets us swap FAISS/Chroma without changing RAG pipeline code."""
    
    @abstractmethod
    def add_documents(self, documents: list[Document]) -> list[str]:
        """Store chunks, return list of chunk IDs."""
        ...

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
        score_threshold: float = 0.0,
    ) -> list[tuple[Document, float]]:
        """Return (Document, score) pairs, ordered by relevance."""
        ...

    @abstractmethod
    def delete_documents(self, document_id: str) -> bool:
        """Remove all chunks belonging to a document."""
        ...

    @abstractmethod
    def get_document_count(self) -> int:
        """Total number of chunks stored."""
        ...


class FAISSVectorStore(VectorStoreBase):
    """
    FAISS-backed vector store with disk persistence.
    
    Persistence strategy:
    - On add: index saved to disk immediately (durability)
    - On startup: load existing index if present
    - Index format: FAISS index file + LangChain docstore pickle
    """

    def __init__(
        self,
        embeddings: OpenAIEmbeddings,
        persist_path: str = "./embeddings/vectorstore",
    ):
        self.embeddings = embeddings
        self.persist_path = Path(persist_path)
        self.persist_path.mkdir(parents=True, exist_ok=True)
        self._store: Optional[FAISS] = self._load_or_init()

    def _load_or_init(self) -> Optional[FAISS]:
        index_file = self.persist_path / "index.faiss"
        if index_file.exists():
            try:
                store = FAISS.load_local(
                    str(self.persist_path),
                    self.embeddings,
                    allow_dangerous_deserialization=True,  # Required by LangChain
                )
                count = store.index.ntotal
                logger.info("Loaded existing FAISS index with %d vectors", count)
                return store
            except Exception as e:
                logger.warning("Failed to load FAISS index: %s. Starting fresh.", e)
        return None

    def add_documents(self, documents: list[Document]) -> list[str]:
        """
        Add document chunks to FAISS index.
        
        Strategy: if index exists, merge. If not, create new.
        Saves to disk after each add for durability.
        """
        if not documents:
            return []

        if self._store is None:
            self._store = FAISS.from_documents(documents, self.embeddings)
        else:
            self._store.add_documents(documents)

        self._persist()
        
        # Return IDs (FAISS generates UUIDs internally)
        return [doc.metadata.get("chunk_id", f"chunk_{i}") 
                for i, doc in enumerate(documents)]

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
        score_threshold: float = 0.0,
    ) -> list[tuple[Document, float]]:
        """
        Retrieve top-K chunks with similarity scores.
        
        Uses cosine similarity (FAISS normalizes vectors internally).
        filter: e.g. {"document_id": "abc123"} — applied post-retrieval
        """
        if self._store is None:
            return []

        # Retrieve more than k, then filter — because metadata filtering
        # happens AFTER vector search in FAISS (no native support)
        fetch_k = k * 4 if filter else k  # Over-fetch to account for filter loss
        
        results = self._store.similarity_search_with_score(
            query,
            k=min(fetch_k, self._store.index.ntotal),
        )

        # Apply metadata filter
        if filter:
            results = [
                (doc, score)
                for doc, score in results
                if all(doc.metadata.get(key) == val for key, val in filter.items())
            ]

        # FAISS returns L2 distance (lower = better).
        # Convert to similarity score (higher = better) for consistency.
        # similarity = 1 / (1 + distance) is a common normalization.
        results_with_similarity = []
        for doc, distance in results:
            similarity = 1.0 / (1.0 + distance)
            if similarity >= score_threshold:
                results_with_similarity.append((doc, similarity))

        return results_with_similarity[:k]

    def delete_documents(self, document_id: str) -> bool:
        """
        Remove all chunks belonging to a document.
        
        Note: FAISS doesn't support deletion natively. We rebuild the index
        excluding the target document. For large indices, consider Chroma instead.
        """
        if self._store is None:
            return False
        
        # Get all documents
        all_ids = list(self._store.docstore._dict.keys())
        docs_to_keep = [
            self._store.docstore.search(doc_id)
            for doc_id in all_ids
            if self._store.docstore.search(doc_id).metadata.get("document_id") != document_id
        ]
        
        if not docs_to_keep:
            self._store = None
            # Remove persisted index
            for f in self.persist_path.glob("*"):
                f.unlink()
            return True
        
        # Rebuild index from remaining documents
        self._store = FAISS.from_documents(docs_to_keep, self.embeddings)
        self._persist()
        return True

    def get_document_count(self) -> int:
        if self._store is None:
            return 0
        return self._store.index.ntotal

    def _persist(self) -> None:
        """Save FAISS index to disk."""
        if self._store:
            self._store.save_local(str(self.persist_path))
            logger.debug("FAISS index persisted to %s", self.persist_path)


class ChromaVectorStore(VectorStoreBase):
    """
    ChromaDB-backed vector store with native persistence and metadata filtering.
    
    Use when you need:
    - Filter by document_id, file_type, date range
    - Shared access from multiple processes
    - A web UI (Chroma has a dashboard)
    """

    def __init__(
        self,
        embeddings: OpenAIEmbeddings,
        persist_path: str = "./embeddings/chroma",
        collection_name: str = "rag_documents",
    ):
        import chromadb
        from langchain_community.vectorstores import Chroma

        self._chroma_client = chromadb.PersistentClient(path=persist_path)
        self._store = Chroma(
            client=self._chroma_client,
            collection_name=collection_name,
            embedding_function=embeddings,
        )
        logger.info("ChromaDB initialized at %s", persist_path)

    def add_documents(self, documents: list[Document]) -> list[str]:
        ids = self._store.add_documents(documents)
        return ids

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[dict] = None,
        score_threshold: float = 0.0,
    ) -> list[tuple[Document, float]]:
        # Chroma uses where= for metadata filtering (native support!)
        results = self._store.similarity_search_with_relevance_scores(
            query,
            k=k,
            filter=filter,        # e.g. {"document_id": "abc"}
            score_threshold=score_threshold,
        )
        return results

    def delete_documents(self, document_id: str) -> bool:
        self._store.delete(where={"document_id": document_id})
        return True

    def get_document_count(self) -> int:
        return self._store._collection.count()


def create_vector_store(
    store_type: str,
    embeddings: OpenAIEmbeddings,
    **kwargs,
) -> VectorStoreBase:
    """
    Factory function — creates the appropriate vector store based on config.
    Called once at application startup.
    """
    if store_type == "faiss":
        return FAISSVectorStore(embeddings, **kwargs)
    elif store_type == "chroma":
        return ChromaVectorStore(embeddings, **kwargs)
    else:
        raise ValueError(f"Unknown vector store type: {store_type}. Use 'faiss' or 'chroma'.")
