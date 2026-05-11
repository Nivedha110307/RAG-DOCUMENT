"""
Vector store abstraction layer.
Supports ChromaDB (default) and FAISS.

Why an abstraction?
- Swap vector DBs without changing RAG pipeline code
- Easy to add Pinecone / Weaviate / Qdrant later
- Unit-testable with mock implementations

ChromaDB vs FAISS:
- ChromaDB: persistent, metadata filtering, REST API, good for dev/mid-scale
- FAISS: blazing fast similarity search, in-memory, ideal for large-scale read-heavy workloads
"""

import os
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

from backend.core.config import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


class BaseVectorStore(ABC):
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        ...

    @abstractmethod
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[dict] = None,
        score_threshold: float = 0.0,
    ) -> List[Tuple[Document, float]]:
        ...

    @abstractmethod
    def delete_document(self, document_id: str) -> None:
        ...

    @abstractmethod
    def get_collection_stats(self) -> dict:
        ...


class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB-backed vector store.
    Persists embeddings to disk — survives server restarts.
    """

    def __init__(self):
        from langchain_chroma import Chroma

        # FREE local embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)

        self.db = Chroma(
            collection_name=settings.CHROMA_COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=settings.CHROMA_PERSIST_DIR,
        )

        logger.info(
            "chroma_initialized",
            persist_dir=settings.CHROMA_PERSIST_DIR,
        )

    def add_documents(self, documents: List[Document]) -> None:
        """Embed and store documents."""
        start = time.time()

        self.db.add_documents(documents)

        elapsed = (time.time() - start) * 1000

        logger.info(
            "documents_embedded",
            count=len(documents),
            elapsed_ms=round(elapsed, 2),
        )

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[dict] = None,
        score_threshold: float = 0.0,
        ) -> List[Tuple[Document, float]]:

        results = self.db.similarity_search_with_relevance_scores(
            query=query,
            k=k,
            filter=filter_metadata,
        )

        logger.info(
            "similarity_search",
            query_preview=query[:60],
            k=k,
            results_returned=len(results),
        )

        return results
    def delete_document(self, document_id: str) -> None:
        """Remove all chunks belonging to a document from the index."""

        self.db.delete(where={"document_id": document_id})

        logger.info(
            "document_deleted",
            document_id=document_id,
        )

    def get_collection_stats(self) -> dict:
        collection = self.db._collection

        return {
            "total_chunks": collection.count(),
            "collection_name": settings.CHROMA_COLLECTION_NAME,
            "persist_dir": settings.CHROMA_PERSIST_DIR,
        }


class FAISSVectorStore(BaseVectorStore):
    """
    FAISS-backed vector store.
    Ultra-fast for large corpuses.
    """

    def __init__(self):
        from langchain_community.vectorstores import FAISS

        # FREE local embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        self._db: Optional[FAISS] = None
        self._index_path = settings.FAISS_INDEX_PATH

        # Load existing index if available
        if os.path.exists(f"{self._index_path}.faiss"):
            self._db = FAISS.load_local(
                self._index_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

            logger.info(
                "faiss_index_loaded",
                path=self._index_path,
            )

    def add_documents(self, documents: List[Document]) -> None:
        from langchain_community.vectorstores import FAISS

        if self._db is None:
            self._db = FAISS.from_documents(
                documents,
                self.embeddings,
            )
        else:
            self._db.add_documents(documents)

        self._db.save_local(self._index_path)

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_metadata: Optional[dict] = None,
        score_threshold: float = 0.0,
    ) -> List[Tuple[Document, float]]:
        if self._db is None:
            return []

        results = self._db.similarity_search_with_relevance_scores(
            query,
            k=k,
        )

        return [
            (doc, score)
            for doc, score in results
            if score >= score_threshold
        ]

    def delete_document(self, document_id: str) -> None:
        # FAISS deletion requires index rebuild
        logger.warning(
            "faiss_delete_requires_rebuild",
            document_id=document_id,
        )

    def get_collection_stats(self) -> dict:
        if self._db is None:
            return {"total_chunks": 0}

        return {
            "total_chunks": self._db.index.ntotal
        }


def get_vector_store() -> BaseVectorStore:
    """Factory function."""

    if settings.VECTOR_DB_TYPE == "faiss":
        return FAISSVectorStore()

    return ChromaVectorStore()