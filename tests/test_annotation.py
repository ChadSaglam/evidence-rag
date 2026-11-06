import json

import pytest
from pydantic import ValidationError

from evidence_rag.annotation import (
    create_annotation_batch,
    export_human_gold_set,
    load_annotation_batch,
    load_annotation_queries,
    summarize_annotation_batch,
)
from evidence_rag.chunking import chunk_documents
from evidence_rag.cli import main
from evidence_rag.corpus import load_chunks, load_documents, write_jsonl
from evidence_rag.evaluation import load_gold_set, summarize_gold_set
from evidence_rag.schemas import AnnotationBatch

BATCH_PATH = "data/annotation-batch-01/annotations.json"


@pytest.fixture
def chunks_path(tmp_path):
    path = tmp_path / "chunks.jsonl"
    write_jsonl(path, chunk_documents(load_documents("data/sample/documents.jsonl")))
    return path


def make_pending_batch(chunks_path) -> AnnotationBatch:
    return create_annotation_batch(
        queries=load_annotation_queries("data/annotation-batch-01/queries.json"),
        chunks=load_chunks(chunks_path),
        chunks_path=chunks_path,
        batch_id="batch-01",
        name="human-evidence-batch-01-development",
        description="Rank-blind human annotation batch; no decisions have been collected.",
        split="development",
        corpus_name="synthetic-multilingual-smoke-corpus-v1",
    )


def completed_batch(chunks_path, annotator="human-adjudicator-01") -> AnnotationBatch:
    payload = make_pending_batch(chunks_path).model_dump(mode="json")
    for task_index, task in enumerate(payload["tasks"]):
        for candidate_index, candidate in enumerate(task["candidates"]):
            candidate["relevance"] = 3 if task_index < 4 and candidate_index == 0 else 0
            candidate["annotator"] = annotator
    return AnnotationBatch.model_validate(payload)


def test_first_batch_is_completed_rank_blind_and_corpus_bound(chunks_path) -> None:
    batch = load_annotation_batch(BATCH_PATH)
    summary = summarize_annotation_batch(batch)
    assert summary == {
        "batch_id": "batch-01",
        "split": "development",
        "label_source": "human",
        "queries": 5,
        "pairs": 20,
        "annotated_pairs": 20,
        "pending_pairs": 0,
        "annotators": [
            "human-annotator-01",
            "human-annotator-02",
            "human-annotator-03",
            "human-annotator-04",
        ],
        "complete": True,
    }
    assert batch.corpus_sha256 == make_pending_batch(chunks_path).corpus_sha256
    labels = [
        item.relevance for task in batch.tasks for item in task.candidates
    ]
    assert labels.count(0) == 14
    assert labels.count(3) == 6
    assert set(labels) == {0, 3}
    for task in batch.tasks:
        chunk_ids = [item.chunk.chunk_id for item in task.candidates]
        assert chunk_ids == sorted(chunk_ids)
        assert all(
            item.relevance is not None and item.annotator is not None
            for item in task.candidates
        )
        assert all("rank" not in item.model_dump() for item in task.candidates)


def test_incomplete_batch_cannot_become_human_gold(chunks_path) -> None:
    batch = make_pending_batch(chunks_path)
    with pytest.raises(ValueError, match="20 pending decisions"):
        export_human_gold_set(batch, chunks_path)


def test_complete_batch_exports_answerable_and_unanswerable_queries(chunks_path) -> None:
    gold_set = export_human_gold_set(completed_batch(chunks_path), chunks_path)
    assert gold_set.label_source == "human"
    assert len(gold_set.queries) == 5
    assert sum(item.answerable for item in gold_set.queries) == 4
    assert sum(len(item.judgments) for item in gold_set.queries) == 20
    assert gold_set.queries[-1].answerable is False
    assert all(
        judgment.annotator == "human-adjudicator-01"
        for query in gold_set.queries
        for judgment in query.judgments
    )


def test_human_gold_export_is_deterministic(chunks_path) -> None:
    batch = completed_batch(chunks_path)
    first = export_human_gold_set(batch, chunks_path)
    second = export_human_gold_set(batch, chunks_path)
    assert first.model_dump_json(indent=2) == second.model_dump_json(indent=2)


def test_partial_decision_is_rejected(chunks_path) -> None:
    payload = make_pending_batch(chunks_path).model_dump(mode="json")
    payload["tasks"][0]["candidates"][0]["relevance"] = 3
    with pytest.raises(ValidationError, match="requires an annotator"):
        AnnotationBatch.model_validate(payload)


def test_duplicate_candidates_are_rejected(chunks_path) -> None:
    payload = make_pending_batch(chunks_path).model_dump(mode="json")
    payload["tasks"][0]["candidates"][1] = payload["tasks"][0]["candidates"][0]
    with pytest.raises(ValidationError, match="must be unique"):
        AnnotationBatch.model_validate(payload)


def test_export_rejects_changed_corpus(chunks_path) -> None:
    batch = completed_batch(chunks_path)
    chunks_path.write_text(chunks_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="corpus hash"):
        export_human_gold_set(batch, chunks_path)


def test_export_rejects_obvious_nonhuman_annotator_placeholders(chunks_path) -> None:
    for annotator in ("synthetic-fixture", "template", "TODO"):
        with pytest.raises(ValueError, match="human identifier"):
            export_human_gold_set(completed_batch(chunks_path, annotator), chunks_path)


def test_annotation_create_cli_reproduces_pending_batch(tmp_path, chunks_path, capsys) -> None:
    output = tmp_path / "annotations.json"
    main(
        [
            "annotation-create",
            "--queries",
            "data/annotation-batch-01/queries.json",
            "--chunks",
            str(chunks_path),
            "--output",
            str(output),
            "--batch-id",
            "batch-01",
            "--name",
            "human-evidence-batch-01-development",
            "--corpus-name",
            "synthetic-multilingual-smoke-corpus-v1",
            "--split",
            "development",
        ]
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["pending_pairs"] == 20
    assert load_annotation_batch(output) == make_pending_batch(chunks_path)


def test_annotation_export_cli_waits_for_completion_then_writes_gold(
    tmp_path, chunks_path, capsys
) -> None:
    pending_batch_path = tmp_path / "pending.json"
    pending_batch_path.write_text(
        make_pending_batch(chunks_path).model_dump_json(indent=2), encoding="utf-8"
    )
    pending_output = tmp_path / "pending-gold.json"
    arguments = [
        "annotation-export",
        "--batch",
        str(pending_batch_path),
        "--chunks",
        str(chunks_path),
        "--output",
        str(pending_output),
    ]
    with pytest.raises(ValueError, match="pending decisions"):
        main(arguments)
    assert not pending_output.exists()

    batch_path = tmp_path / "completed.json"
    batch_path.write_text(
        completed_batch(chunks_path).model_dump_json(indent=2), encoding="utf-8"
    )
    output = tmp_path / "human-gold.json"
    main(
        [
            "annotation-export",
            "--batch",
            str(batch_path),
            "--chunks",
            str(chunks_path),
            "--output",
            str(output),
        ]
    )
    summary = summarize_gold_set(load_gold_set(output))
    assert summary["label_source"] == "human"
    assert summary["queries"] == 5
    assert summary["answerable_queries"] == 4
    assert json.loads(capsys.readouterr().out) == summary


def test_agreement_reads_completed_batch_json(tmp_path) -> None:
    import json

    from evidence_rag.annotation.agreement import (
        compare_annotations,
        load_independent_annotations,
    )

    source = json.loads(
        open("data/annotation-batch-03/annotations-annotator-a.json", encoding="utf-8").read()
    )
    task = source["tasks"][0]
    source["tasks"] = [task]
    copies = {}
    for role, offset in (("a", 0), ("b", 1)):
        data = json.loads(json.dumps(source))
        for index, candidate in enumerate(data["tasks"][0]["candidates"]):
            candidate["relevance"] = 3 if index == 0 else min(3, (index + offset) % 2)
            candidate["annotator"] = f"annotator-{role}"
        path = tmp_path / f"{role}.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        copies[role] = load_independent_annotations(path)
    report = compare_annotations(copies["a"], copies["b"], provenance="human")
    assert report.pair_count == len(task["candidates"])
    assert report.disagreements


def test_agreement_rejects_incomplete_batch_json() -> None:
    import pytest

    from evidence_rag.annotation.agreement import load_independent_annotations

    with pytest.raises(ValueError, match="not annotated yet"):
        load_independent_annotations("data/annotation-batch-03/annotations-annotator-a.json")
