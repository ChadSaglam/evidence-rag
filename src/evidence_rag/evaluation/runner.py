"""Run a retriever over answerable gold queries and macro-average metrics."""

from statistics import mean
from typing import Protocol

from evidence_rag.evaluation.metrics import evaluate_ranking
from evidence_rag.retrieval.base import resolve_query
from evidence_rag.schemas import GoldSet, Query, RerankedHit, RetrievalEvaluationReport


class EvaluableRetriever(Protocol):
    retriever_name: str

    def search(self, query: Query | str, top_k: int | None = None): ...


def evaluate_retriever(
    retriever: EvaluableRetriever,
    gold_set: GoldSet,
    *,
    k: int = 10,
    retriever_name: str | None = None,
) -> RetrievalEvaluationReport:
    if not 1 <= k <= 100:
        raise ValueError("k must be between one and one hundred")
    per_query = []
    skipped = 0
    for gold_query in gold_set.queries:
        if not gold_query.answerable:
            skipped += 1
            continue
        query = resolve_query(gold_query.query, k)
        hits = retriever.search(query)
        ranked_ids = [
            hit.base_hit.chunk.chunk_id if isinstance(hit, RerankedHit) else hit.chunk.chunk_id
            for hit in hits[:k]
        ]
        per_query.append(
            evaluate_ranking(query.query_id, ranked_ids, gold_query.judgments, k=k)
        )
    if not per_query:
        raise ValueError("The gold set contains no answerable queries")

    return RetrievalEvaluationReport(
        gold_set=gold_set.name,
        gold_label_source=gold_set.label_source,
        retriever=retriever_name or getattr(retriever, "retriever_name", type(retriever).__name__),
        k=k,
        evaluated_queries=len(per_query),
        skipped_unanswerable_queries=skipped,
        mean_hit_at_k=mean(item.hit_at_k for item in per_query),
        mean_recall_at_k=mean(item.recall_at_k for item in per_query),
        mean_precision_at_k=mean(item.precision_at_k for item in per_query),
        mean_reciprocal_rank=mean(item.reciprocal_rank for item in per_query),
        mean_ndcg_at_k=mean(item.ndcg_at_k for item in per_query),
        per_query=per_query,
    )
