"""Deterministic graded retrieval metrics with explicit cutoff semantics."""

import math
from collections.abc import Sequence

from evidence_rag.schemas import QueryEvaluation, RelevanceJudgment


def evaluate_ranking(
    query_id: str,
    ranked_chunk_ids: Sequence[str],
    judgments: Sequence[RelevanceJudgment],
    *,
    k: int,
) -> QueryEvaluation:
    """Compute hit, recall, precision, RR, and nDCG at a fixed cutoff.

    Precision uses k as its denominator even when fewer than k hits are returned.
    Only positive relevance labels count as relevant for hit, recall, precision, and RR.
    """

    if not 1 <= k <= 100:
        raise ValueError("k must be between one and one hundred")
    if len(set(ranked_chunk_ids)) != len(ranked_chunk_ids):
        raise ValueError("ranked_chunk_ids must not contain duplicates")
    relevance = {item.chunk_id: item.relevance for item in judgments}
    relevant_ids = {chunk_id for chunk_id, value in relevance.items() if value > 0}
    if not relevant_ids:
        raise ValueError("Retrieval metrics require at least one positive judgment")

    top_ids = list(ranked_chunk_ids[:k])
    hits = [chunk_id for chunk_id in top_ids if relevance.get(chunk_id, 0) > 0]
    first_relevant = next(
        (rank for rank, chunk_id in enumerate(top_ids, start=1) if chunk_id in relevant_ids),
        None,
    )
    dcg = sum(
        (2 ** relevance.get(chunk_id, 0) - 1) / math.log2(rank + 1)
        for rank, chunk_id in enumerate(top_ids, start=1)
    )
    ideal_relevance = sorted((value for value in relevance.values() if value > 0), reverse=True)
    ideal_dcg = sum(
        (2**value - 1) / math.log2(rank + 1)
        for rank, value in enumerate(ideal_relevance[:k], start=1)
    )
    return QueryEvaluation(
        query_id=query_id,
        hit_at_k=bool(hits),
        recall_at_k=len(set(hits)) / len(relevant_ids),
        precision_at_k=len(hits) / k,
        reciprocal_rank=0.0 if first_relevant is None else 1 / first_relevant,
        ndcg_at_k=dcg / ideal_dcg,
    )
