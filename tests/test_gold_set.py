import pytest

from evidence_rag.evaluation import load_gold_set, summarize_gold_set
from evidence_rag.schemas import GoldQuery, GoldSet, Query, RelevanceJudgment


def make_judgment(query_id="q", chunk_id="a", relevance=3):
    return RelevanceJudgment(
        query_id=query_id,
        chunk_id=chunk_id,
        relevance=relevance,
        annotator="fixture-annotator",
    )


def make_gold_query(query_id="q", answerable=True):
    return GoldQuery(
        query=Query(query_id=query_id, text="query"),
        answerable=answerable,
        judgments=[make_judgment(query_id)] if answerable else [],
    )


def make_gold_set(queries):
    return GoldSet(
        name="fixture",
        description="Synthetic test fixture.",
        label_source="synthetic",
        queries=queries,
    )


def test_gold_query_rejects_mismatched_and_duplicate_judgments() -> None:
    with pytest.raises(ValueError, match="must match"):
        GoldQuery(
            query=Query(query_id="q", text="query"),
            judgments=[make_judgment(query_id="other")],
        )
    with pytest.raises(ValueError, match="Duplicate judgment"):
        GoldQuery(
            query=Query(query_id="q", text="query"),
            judgments=[make_judgment(), make_judgment()],
        )


def test_answerability_and_positive_judgments_are_consistent() -> None:
    with pytest.raises(ValueError, match="Answerable"):
        GoldQuery(query=Query(query_id="q", text="query"), judgments=[])
    with pytest.raises(ValueError, match="Unanswerable"):
        GoldQuery(
            query=Query(query_id="q", text="query"),
            answerable=False,
            judgments=[make_judgment()],
        )


def test_gold_set_rejects_duplicate_query_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        make_gold_set([make_gold_query(), make_gold_query()])


def test_sample_gold_set_has_explicit_synthetic_provenance() -> None:
    gold_set = load_gold_set("data/sample/gold_set.json")
    summary = summarize_gold_set(gold_set)
    assert summary == {
        "name": "synthetic-multilingual-smoke-set",
        "label_source": "synthetic",
        "queries": 5,
        "answerable_queries": 4,
        "unanswerable_queries": 1,
        "judgments": 4,
        "valid": True,
    }


def test_empty_human_annotation_template_is_valid() -> None:
    gold_set = load_gold_set("data/gold-template/gold_set.json")
    assert gold_set.label_source == "human"
    assert gold_set.queries == []
