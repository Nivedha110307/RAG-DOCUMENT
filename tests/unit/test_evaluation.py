"""
RAG Evaluation Metrics — Precision@K, Recall@K, MRR, Latency.
Run with: pytest tests/unit/test_evaluation.py -v
"""
import time
import pytest


def precision_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """What fraction of the top-K retrieved chunks are relevant?"""
    top_k = retrieved_ids[:k]
    hits = sum(1 for r in top_k if r in relevant_ids)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    """What fraction of all relevant chunks did we retrieve in top-K?"""
    top_k = retrieved_ids[:k]
    hits = sum(1 for r in top_k if r in relevant_ids)
    return hits / len(relevant_ids) if relevant_ids else 0.0


def mean_reciprocal_rank(retrieved_ids: list, relevant_ids: set) -> float:
    """
    MRR: 1/rank of the first relevant chunk.
    Higher = first relevant result appears earlier.
    Perfect = 1.0 (first result is relevant).
    """
    for rank, chunk_id in enumerate(retrieved_ids, 1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


class TestRAGMetrics:
    """
    Benchmark with synthetic data.
    In production, run against a labeled dataset (query → expected_chunk_ids).
    """

    def test_perfect_retrieval(self):
        retrieved = ["chunk_1", "chunk_2", "chunk_3"]
        relevant = {"chunk_1", "chunk_2", "chunk_3"}
        assert precision_at_k(retrieved, relevant, k=3) == 1.0
        assert recall_at_k(retrieved, relevant, k=3) == 1.0
        assert mean_reciprocal_rank(retrieved, relevant) == 1.0

    def test_partial_retrieval(self):
        retrieved = ["chunk_x", "chunk_1", "chunk_y", "chunk_2"]
        relevant = {"chunk_1", "chunk_2"}
        assert precision_at_k(retrieved, relevant, k=4) == 0.5
        assert recall_at_k(retrieved, relevant, k=4) == 1.0
        # First relevant (chunk_1) is at rank 2
        assert mean_reciprocal_rank(retrieved, relevant) == pytest.approx(0.5)

    def test_no_relevant_retrieved(self):
        retrieved = ["chunk_x", "chunk_y"]
        relevant = {"chunk_1", "chunk_2"}
        assert precision_at_k(retrieved, relevant, k=2) == 0.0
        assert recall_at_k(retrieved, relevant, k=2) == 0.0
        assert mean_reciprocal_rank(retrieved, relevant) == 0.0

    def test_latency_under_threshold(self):
        """Retrieval should complete in < 500ms (simulated)."""
        def mock_retrieval():
            time.sleep(0.01)  # Simulated fast retrieval
            return ["chunk_1", "chunk_2"]

        start = time.time()
        results = mock_retrieval()
        elapsed_ms = (time.time() - start) * 1000

        assert elapsed_ms < 500, f"Retrieval too slow: {elapsed_ms:.1f}ms"
        assert len(results) > 0


# ── Evaluation dataset format ─────────────────────────────────────────────────
EVAL_DATASET = [
    {
        "query": "What is the return policy?",
        "expected_chunk_ids": {"doc_a_chunk_3", "doc_a_chunk_4"},
    },
    {
        "query": "How do I reset my password?",
        "expected_chunk_ids": {"doc_b_chunk_1"},
    },
    {
        "query": "What are the pricing tiers?",
        "expected_chunk_ids": {"doc_c_chunk_7", "doc_c_chunk_8", "doc_c_chunk_9"},
    },
]


def evaluate_retrieval_system(retrieval_fn, dataset: list, k: int = 5) -> dict:
    """
    Run evaluation over a labeled dataset.
    retrieval_fn: Callable[[str], List[str]] — takes query, returns chunk IDs
    """
    precisions, recalls, mrrs, latencies = [], [], [], []

    for item in dataset:
        start = time.time()
        retrieved_ids = retrieval_fn(item["query"])
        latency_ms = (time.time() - start) * 1000

        relevant = item["expected_chunk_ids"]
        precisions.append(precision_at_k(retrieved_ids, relevant, k))
        recalls.append(recall_at_k(retrieved_ids, relevant, k))
        mrrs.append(mean_reciprocal_rank(retrieved_ids, relevant))
        latencies.append(latency_ms)

    return {
        f"precision@{k}": round(sum(precisions) / len(precisions), 4),
        f"recall@{k}": round(sum(recalls) / len(recalls), 4),
        "mrr": round(sum(mrrs) / len(mrrs), 4),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
    }
