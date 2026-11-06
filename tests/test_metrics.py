import math

import pytest
from conftest import make_chunk

from evidence_rag.evaluation import evaluate_ranking, evaluate_retriever
from evidence_rag.schemas import (
    GoldQuery,
    GoldSet,
    Query,
    RelevanceJudgment,
    RetrievalHit,
)


def judgment(query_id: str, chunk_id: str, relevance: int) -> RelevanceJudgment:
    return RelevanceJudgment(
        query_id=query_id,
        chunk_id=chunk_id,
        relevance=relevance,
        annotator="fixture",
    )


def test_graded_metrics_match_manual_calculation() -> None:
    judgments = [judgment("q", "a", 3), judgment("q", "b", 1), judgment("q", "c", 0)]
    result = evaluate_ranking("q", ["x", "a", "b"], judgments, k=3)
    dcg = 7 / math.log2(3) + 1 / math.log2(4)
    ideal = 7 + 1 / math.log2(3)
    assert result.hit_at_k is True
    assert result.recall_at_k == 1
    assert result.precision_at_k == pytest.approx(2 / 3)
    assert result.reciprocal_rank == 0.5
    assert result.ndcg_at_k == pytest.approx(dcg / ideal)


def test_metrics_handle_no_hits_and_short_rankings() -> None:
    judgments = [judgment("q", "a", 2)]
    missing = evaluate_ranking("q", ["x", "y"], judgments, k=3)
    assert missing.hit_at_k is False
    assert missing.recall_at_k == 0
    assert missing.reciprocal_rank == 0
    assert missing.ndcg_at_k == 0
    short = evaluate_ranking("q", ["a"], judgments, k=3)
    assert short.precision_at_k == pytest.approx(1 / 3)


def test_metrics_reject_duplicate_rankings_and_missing_positive_labels() -> None:
    judgments = [judgment("q", "a", 1)]
    with pytest.raises(ValueError, match="duplicates"):
        evaluate_ranking("q", ["a", "a"], judgments, k=2)
    with pytest.raises(ValueError, match="positive"):
        evaluate_ranking("q", ["a"], [judgment("q", "a", 0)], k=2)


class StaticRetriever:
    def search(self, query, top_k=None):
        return [
            RetrievalHit(
                rank=1,
                score=1.0,
                retriever="fixture",
                chunk=make_chunk("a", "alpha"),
            )
        ]


def test_runner_skips_unanswerable_queries_and_macro_averages() -> None:
    answerable = GoldQuery(
        query=Query(query_id="q", text="alpha"),
        judgments=[judgment("q", "a", 3)],
    )
    unanswerable = GoldQuery(
        query=Query(query_id="q-none", text="unknown"), answerable=False
    )
    gold_set = GoldSet(
        name="fixture",
        description="Synthetic fixture.",
        label_source="synthetic",
        queries=[answerable, unanswerable],
    )
    report = evaluate_retriever(StaticRetriever(), gold_set, k=2)
    assert report.evaluated_queries == 1
    assert report.skipped_unanswerable_queries == 1
    assert report.mean_hit_at_k == 1
    assert report.mean_precision_at_k == 0.5
    assert report.gold_label_source == "synthetic"
