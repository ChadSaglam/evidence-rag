import pytest
from conftest import make_chunk

from evidence_rag.evaluation import calibrate_selective_answerer, evaluate_selective_policy
from evidence_rag.schemas import (
    FrozenSelectivePolicy,
    GoldQuery,
    GoldSet,
    Query,
    RelevanceJudgment,
    RetrievalHit,
)


def hit(chunk_id: str, score: float) -> RetrievalHit:
    return RetrievalHit(
        rank=1,
        score=score,
        retriever="fixture",
        chunk=make_chunk(chunk_id, f"text for {chunk_id}"),
    )


def judgment(query_id: str, chunk_id: str, relevance: int) -> RelevanceJudgment:
    return RelevanceJudgment(
        query_id=query_id,
        chunk_id=chunk_id,
        relevance=relevance,
        annotator="fixture",
    )


class StaticRetriever:
    retriever_name = "fixture"

    def __init__(self, results: dict[str, list[RetrievalHit]]) -> None:
        self.results = results

    def search(self, query, top_k=None):
        assert query.top_k == 1
        return self.results.get(query.query_id, [])


def calibration_gold_set() -> GoldSet:
    return GoldSet(
        name="calibration-fixture",
        description="Synthetic calibration fixture.",
        label_source="synthetic",
        queries=[
            GoldQuery(
                query=Query(query_id="q-positive", text="positive"),
                judgments=[judgment("q-positive", "good:0", 3)],
            ),
            GoldQuery(
                query=Query(query_id="q-wrong", text="wrong"),
                judgments=[
                    judgment("q-wrong", "bad:0", 0),
                    judgment("q-wrong", "good:0", 3),
                ],
            ),
            GoldQuery(query=Query(query_id="q-none", text="none"), answerable=False),
        ],
    )


def calibration_retriever() -> StaticRetriever:
    return StaticRetriever(
        {
            "q-positive": [hit("good:0", 3.0)],
            "q-wrong": [hit("bad:0", 2.0)],
            "q-none": [hit("other:0", 1.0)],
        }
    )


def test_calibration_maximizes_coverage_within_zero_risk() -> None:
    report = calibrate_selective_answerer(calibration_retriever(), calibration_gold_set())
    assert report.selected_threshold == 3.0
    assert report.selected.answered_queries == 1
    assert report.selected.correct_answers == 1
    assert report.selected.wrong_answers == 0
    assert report.selected.correct_abstentions == 2
    assert report.selected.coverage == pytest.approx(1 / 3)
    assert report.selected.decision_accuracy == 1.0
    assert report.selected.wrong_answer_rate == 0.0
    assert len(report.candidates) == 4
    assert report.candidates[1].threshold == 2.0
    assert report.candidates[1].wrong_answer_rate == 0.5
    assert report.candidates[-1].answered_queries == 0


def test_calibration_is_deterministic_and_tracks_top_one_relevance() -> None:
    first = calibrate_selective_answerer(calibration_retriever(), calibration_gold_set())
    second = calibrate_selective_answerer(calibration_retriever(), calibration_gold_set())
    assert first.model_dump_json(indent=2) == second.model_dump_json(indent=2)
    rows = {row.query_id: row for row in first.per_query}
    assert rows["q-positive"].top_chunk_relevance == 3
    assert rows["q-positive"].top_chunk_relevant is True
    assert rows["q-wrong"].top_chunk_id == "bad:0"
    assert rows["q-wrong"].top_chunk_relevant is False
    assert rows["q-none"].top_chunk_relevance == 0


def test_calibration_handles_no_hits_and_rejects_invalid_inputs() -> None:
    gold_set = GoldSet(
        name="empty-retrieval",
        description="Synthetic no-hit fixture.",
        label_source="synthetic",
        queries=[GoldQuery(query=Query(query_id="q", text="q"), answerable=False)],
    )
    report = calibrate_selective_answerer(StaticRetriever({}), gold_set)
    assert report.selected_threshold == 0.0
    assert report.selected.coverage == 0.0
    assert report.selected.correct_abstentions == 1
    assert report.per_query[0].top_score is None

    with pytest.raises(ValueError, match="max_risk"):
        calibrate_selective_answerer(StaticRetriever({}), gold_set, max_risk=-0.1)
    empty_gold = GoldSet(
        name="empty",
        description="Empty fixture.",
        label_source="synthetic",
        queries=[],
    )
    with pytest.raises(ValueError, match="at least one"):
        calibrate_selective_answerer(StaticRetriever({}), empty_gold)


def test_frozen_policy_evaluation_uses_unchanged_threshold_and_matching_batch() -> None:
    policy = FrozenSelectivePolicy(
        policy_id="fixture-policy",
        status="frozen_for_independent_evaluation",
        frozen_at="2026-10-30",
        min_top_score=3.0,
        top_k=1,
        require_citations=True,
        decision_rule="Fixed threshold.",
        retriever={"method": "dense"},
        calibration={},
        corpus={},
        evaluation={
            "split": "test",
            "batch_id": "calibration-fixture",
            "retune_allowed": False,
        },
    )
    gold_set = calibration_gold_set()
    report = evaluate_selective_policy(calibration_retriever(), gold_set, policy)
    assert report.policy_id == "fixture-policy"
    assert report.min_top_score == 3.0
    assert report.metrics.threshold == 3.0
    assert report.metrics.answered_queries == 1
    assert report.metrics.correct_answers == 1
    assert report.metrics.wrong_answer_rate == 0.0
    assert report.metrics.coverage == pytest.approx(1 / 3)

    wrong_gold = gold_set.model_copy(update={"name": "another-batch"})
    with pytest.raises(ValueError, match="does not match"):
        evaluate_selective_policy(calibration_retriever(), wrong_gold, policy)
