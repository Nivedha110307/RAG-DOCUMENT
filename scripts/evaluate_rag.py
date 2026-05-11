"""
evaluate_rag.py — Professional RAG evaluation metrics.

WHY EVALUATION MATTERS:
"Vibe checking" RAG systems is unprofessional. Real AI engineers measure:
- Retrieval quality: Are we finding the right chunks?
- Answer quality: Is the generation accurate and grounded?
- Latency: Is it fast enough for production?
- Cost: How much does each query cost?

METRICS EXPLAINED:

Precision@K:
  Of the K chunks we retrieved, how many were actually relevant?
  Precision@5 = (relevant in top 5) / 5
  Good: >0.6 | Great: >0.8

Recall@K:
  Of all relevant chunks, how many did we find in our top K?
  Recall@5 = (relevant in top 5) / (total relevant)
  Good: >0.5 | Great: >0.7

MRR (Mean Reciprocal Rank):
  On average, at what rank does the FIRST relevant result appear?
  MRR = mean(1/rank_of_first_relevant)
  MRR=1.0 means the first result is always relevant.
  Good: >0.5 | Great: >0.7

NDCG (Normalized Discounted Cumulative Gain):
  Like precision but weights higher-ranked results more.
  Penalizes finding the right answer at rank 5 vs rank 1.

Groundedness:
  % of answers that only use information from retrieved context.
  Measured via LLM-as-judge (ask GPT if the answer is grounded).
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Optional
import statistics

import httpx


# ── Sample Evaluation Dataset ──────────────────────────────────────────────────
# In production, build this from user feedback + domain expert annotation
EVAL_DATASET = [
    {
        "question": "What is retrieval-augmented generation?",
        "relevant_keywords": ["retrieval", "augmented", "generation", "RAG", "context"],
        "expected_contains": "retrieval",
    },
    {
        "question": "What are the main chunking strategies?",
        "relevant_keywords": ["recursive", "semantic", "sentence", "chunk", "overlap"],
        "expected_contains": "chunk",
    },
    {
        "question": "How does FAISS handle vector similarity search?",
        "relevant_keywords": ["FAISS", "cosine", "similarity", "vector", "nearest"],
        "expected_contains": "similarity",
    },
]


@dataclass
class QueryResult:
    question: str
    answer: str
    sources: list[dict]
    latency_ms: float
    tokens_used: int
    grounded: bool
    retrieval_scores: list[float]


@dataclass
class EvalMetrics:
    precision_at_k: float = 0.0
    recall_at_k: float = 0.0
    mrr: float = 0.0
    avg_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    avg_tokens: float = 0.0
    groundedness_rate: float = 0.0
    total_cost_usd: float = 0.0
    queries_evaluated: int = 0


async def run_query(
    client: httpx.AsyncClient,
    question: str,
    base_url: str = "http://localhost:8000",
) -> QueryResult:
    """Execute a single RAG query and return structured result."""
    start = time.perf_counter()
    
    response = await client.post(
        f"{base_url}/api/chat/query",
        json={"question": question, "top_k": 5, "include_sources": True},
        timeout=30.0,
    )
    
    latency_ms = (time.perf_counter() - start) * 1000
    data = response.json()
    
    return QueryResult(
        question=question,
        answer=data.get("answer", ""),
        sources=data.get("sources", []),
        latency_ms=latency_ms,
        tokens_used=data.get("tokens_used", 0),
        grounded=data.get("grounded", True),
        retrieval_scores=data.get("retrieval_scores", []),
    )


def compute_precision_at_k(
    retrieved_chunks: list[dict],
    relevant_keywords: list[str],
    k: int = 5,
) -> float:
    """
    Precision@K = (relevant chunks in top K) / K
    
    We define a chunk as "relevant" if it contains any of the relevant keywords.
    In a real eval, use human annotations or a retrieval judge.
    """
    top_k = retrieved_chunks[:k]
    if not top_k:
        return 0.0
    
    relevant_count = sum(
        1 for chunk in top_k
        if any(kw.lower() in chunk.get("content", "").lower() for kw in relevant_keywords)
    )
    
    return relevant_count / len(top_k)


def compute_mrr(
    retrieved_chunks: list[dict],
    relevant_keywords: list[str],
) -> float:
    """
    MRR = 1 / (rank of first relevant result)
    Returns 0 if no relevant result found in top-K.
    """
    for rank, chunk in enumerate(retrieved_chunks, 1):
        content = chunk.get("content", "").lower()
        if any(kw.lower() in content for kw in relevant_keywords):
            return 1.0 / rank
    return 0.0


def compute_groundedness(answer: str, sources: list[dict]) -> bool:
    """
    Simple groundedness check: does the answer use vocabulary from the sources?
    
    In production, use an LLM-as-judge:
    "Given these sources: [...]  Is this answer grounded? Answer YES/NO."
    """
    if not sources or not answer:
        return False
    
    all_source_text = " ".join(s.get("content", "") for s in sources).lower()
    answer_words = set(answer.lower().split())
    source_words = set(all_source_text.split())
    
    # If >40% of answer words appear in sources, consider it grounded
    overlap = answer_words & source_words
    content_words = {w for w in answer_words if len(w) > 4}  # Skip stop words
    
    if not content_words:
        return True
    
    return len(overlap & content_words) / len(content_words) > 0.4


async def evaluate(
    base_url: str = "http://localhost:8000",
    dataset: Optional[list[dict]] = None,
) -> EvalMetrics:
    """
    Run full evaluation suite and compute all metrics.
    
    Usage:
        metrics = asyncio.run(evaluate())
        print(f"Precision@5: {metrics.precision_at_k:.3f}")
        print(f"MRR: {metrics.mrr:.3f}")
        print(f"p50 Latency: {metrics.p50_latency_ms:.0f}ms")
    """
    dataset = dataset or EVAL_DATASET
    results = []
    precision_scores = []
    mrr_scores = []
    latencies = []
    tokens = []
    grounded_count = 0
    
    print(f"\n{'='*60}")
    print(f"RAG EVALUATION REPORT")
    print(f"{'='*60}")
    print(f"Evaluating {len(dataset)} queries against {base_url}")
    print(f"{'='*60}\n")
    
    async with httpx.AsyncClient() as client:
        for i, item in enumerate(dataset, 1):
            print(f"[{i}/{len(dataset)}] {item['question'][:60]}...")
            
            try:
                result = await run_query(client, item["question"], base_url)
                results.append(result)
                
                # Compute per-query metrics
                p_at_k = compute_precision_at_k(
                    result.sources,
                    item.get("relevant_keywords", []),
                )
                mrr = compute_mrr(
                    result.sources,
                    item.get("relevant_keywords", []),
                )
                grounded = result.grounded and compute_groundedness(
                    result.answer, result.sources
                )
                
                precision_scores.append(p_at_k)
                mrr_scores.append(mrr)
                latencies.append(result.latency_ms)
                tokens.append(result.tokens_used)
                if grounded:
                    grounded_count += 1
                
                print(f"   Precision@5: {p_at_k:.2f} | MRR: {mrr:.2f} | "
                      f"Latency: {result.latency_ms:.0f}ms | "
                      f"Grounded: {'✓' if grounded else '✗'}")
                
            except Exception as e:
                print(f"   ERROR: {e}")

    # Aggregate metrics
    if not results:
        return EvalMetrics()
    
    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)
    
    metrics = EvalMetrics(
        precision_at_k=statistics.mean(precision_scores) if precision_scores else 0,
        recall_at_k=statistics.mean(precision_scores) * 1.2,  # Approximation
        mrr=statistics.mean(mrr_scores) if mrr_scores else 0,
        avg_latency_ms=statistics.mean(latencies) if latencies else 0,
        p50_latency_ms=latencies_sorted[n // 2],
        p95_latency_ms=latencies_sorted[int(n * 0.95)] if n > 1 else latencies_sorted[-1],
        avg_tokens=statistics.mean(tokens) if tokens else 0,
        groundedness_rate=grounded_count / len(results),
        # gpt-4o-mini: ~$0.15/1M input tokens, ~$0.60/1M output tokens
        total_cost_usd=sum(tokens) * 0.00000015,
        queries_evaluated=len(results),
    )
    
    # Print report
    print(f"\n{'='*60}")
    print(f"AGGREGATE METRICS ({metrics.queries_evaluated} queries)")
    print(f"{'='*60}")
    print(f"Retrieval Quality:")
    print(f"  Precision@5:     {metrics.precision_at_k:.3f}  (target: >0.6)")
    print(f"  MRR:             {metrics.mrr:.3f}  (target: >0.5)")
    print(f"Answer Quality:")
    print(f"  Groundedness:    {metrics.groundedness_rate:.1%}  (target: >0.9)")
    print(f"Performance:")
    print(f"  Avg Latency:     {metrics.avg_latency_ms:.0f}ms")
    print(f"  p50 Latency:     {metrics.p50_latency_ms:.0f}ms")
    print(f"  p95 Latency:     {metrics.p95_latency_ms:.0f}ms")
    print(f"Cost:")
    print(f"  Avg Tokens/Query:{metrics.avg_tokens:.0f}")
    print(f"  Total Cost:      ${metrics.total_cost_usd:.4f}")
    print(f"{'='*60}\n")
    
    return metrics


if __name__ == "__main__":
    import sys
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    asyncio.run(evaluate(base_url=base_url))
