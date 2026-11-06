"""Calibrate and evaluate the pre-generation evidence gate on labeled top-1 outcomes."""

import math
from typing import Protocol

from evidence_rag.retrieval.base import resolve_query
from evidence_rag.schemas import (
    FrozenSelectivePolicy,
    GoldSet,
    Query,
    RerankedHit,
    SelectiveCalibrationPoint,
    SelectiveCalibrationQuery,
    SelectiveCalibrationReport,
    SelectivePolicyEvaluationReport,
)


class CalibratableRetriever(Protocol):
    retriever_name: str

    def search(self, query: Query | str, top_k: int | None = None): ...


def _chunk_id(hit) -> str:
    return hit.base_hit.chunk.chunk_id if isinstance(hit, RerankedHit) else hit.chunk.chunk_id


def _candidate_point(
    rows: list[SelectiveCalibrationQuery], threshold: float
) -> SelectiveCalibrationPoint:
    answered = [
        row for row in rows if row.top_score is not None and row.top_score >= threshold
    ]
    abstained = [row for row in rows if row not in answered]
    correct_answers = sum(row.top_chunk_relevant for row in answered)
    wrong_answers = len(answered) - correct_answers
    correct_abstentions = sum(not row.top_chunk_relevant for row in abstained)
    return SelectiveCalibrationPoint(
        threshold=threshold,
        answered_queries=len(answered),
        correct_answers=correct_answers,
        wrong_answers=wrong_answers,
        correct_abstentions=correct_abstentions,
        coverage=len(answered) / len(rows),
        selective_accuracy=(correct_answers / len(answered)) if answered else 0.0,
        wrong_answer_rate=(wrong_answers / len(answered)) if answered else 0.0,
        decision_accuracy=(correct_answers + correct_abstentions) / len(rows),
    )


def _query_rows(
    retriever: CalibratableRetriever, gold_set: GoldSet
) -> list[SelectiveCalibrationQuery]:
    if not gold_set.queries:
        raise ValueError("Selective evaluation requires at least one gold query")
    rows = []
    for gold_query in gold_set.queries:
        query = resolve_query(gold_query.query, 1)
        hits = retriever.search(query)
        top = min(hits, key=lambda hit: (hit.rank, _chunk_id(hit))) if hits else None
        top_chunk_id = _chunk_id(top) if top is not None else None
        relevance_by_chunk = {
            judgment.chunk_id: judgment.relevance for judgment in gold_query.judgments
        }
        relevance = relevance_by_chunk.get(top_chunk_id, 0)
        rows.append(
            SelectiveCalibrationQuery(
                query_id=query.query_id,
                answerable=gold_query.answerable,
                top_score=top.score if top is not None else None,
                top_chunk_id=top_chunk_id,
                top_chunk_relevance=relevance,
                top_chunk_relevant=relevance > 0,
            )
        )
    return rows


def calibrate_selective_answerer(
    retriever: CalibratableRetriever,
    gold_set: GoldSet,
    *,
    max_risk: float = 0.0,
    retriever_name: str | None = None,
) -> SelectiveCalibrationReport:
    """Maximize coverage under an empirical wrong-answer risk constraint.

    An answer decision is correct only when the top-1 chunk has a positive gold
    judgment. Unjudged and missing chunks therefore count as incorrect answers.
    """
    if not 0 <= max_risk <= 1:
        raise ValueError("max_risk must be between zero and one")
    rows = _query_rows(retriever, gold_set)
    scores = sorted({row.top_score for row in rows if row.top_score is not None})
    thresholds = scores + ([math.nextafter(scores[-1], math.inf)] if scores else [0.0])
    candidates = [_candidate_point(rows, threshold) for threshold in thresholds]
    feasible = [point for point in candidates if point.wrong_answer_rate <= max_risk]
    selected = max(
        feasible,
        key=lambda point: (point.coverage, point.decision_accuracy, -point.threshold),
    )
    return SelectiveCalibrationReport(
        gold_set=gold_set.name,
        gold_label_source=gold_set.label_source,
        retriever=retriever_name or getattr(retriever, "retriever_name", type(retriever).__name__),
        max_risk=max_risk,
        selected_threshold=selected.threshold,
        query_count=len(rows),
        answerable_queries=sum(row.answerable for row in rows),
        unanswerable_queries=sum(not row.answerable for row in rows),
        selected=selected,
        candidates=candidates,
        per_query=rows,
    )


def evaluate_selective_policy(
    retriever: CalibratableRetriever,
    gold_set: GoldSet,
    policy: FrozenSelectivePolicy,
    *,
    retriever_name: str | None = None,
) -> SelectivePolicyEvaluationReport:
    """Evaluate one frozen threshold without generating or selecting candidates."""
    expected_batch = policy.evaluation["batch_id"]
    normalized_batch = expected_batch.replace("-", "").casefold()
    normalized_gold_name = gold_set.name.replace("-", "").casefold()
    if normalized_batch not in normalized_gold_name:
        raise ValueError("Gold set does not match the frozen policy evaluation batch")
    rows = _query_rows(retriever, gold_set)
    return SelectivePolicyEvaluationReport(
        policy_id=policy.policy_id,
        evaluation_split=str(policy.evaluation.get("split", "test")),
        gold_set=gold_set.name,
        gold_label_source=gold_set.label_source,
        retriever=retriever_name or getattr(retriever, "retriever_name", type(retriever).__name__),
        min_top_score=policy.min_top_score,
        require_citations=policy.require_citations,
        query_count=len(rows),
        answerable_queries=sum(row.answerable for row in rows),
        unanswerable_queries=sum(not row.answerable for row in rows),
        metrics=_candidate_point(rows, policy.min_top_score),
        per_query=rows,
    )
