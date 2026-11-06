import json

from evidence_rag.annotation import load_annotation_batch, summarize_annotation_batch
from evidence_rag.benchmark.kaggle import sha256_file
from evidence_rag.chunking import chunk_documents
from evidence_rag.corpus import load_documents, write_jsonl
from evidence_rag.evaluation import load_gold_set

BATCH02_PATH = "data/annotation-batch-02/annotations.json"
POLICY_PATH = "data/selective-policies/dense-e5-small-batch-01.json"


def test_batch02_is_completed_query_independent_and_corpus_bound(tmp_path) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    write_jsonl(
        chunks_path, chunk_documents(load_documents("data/sample/documents.jsonl"))
    )
    batch = load_annotation_batch(BATCH02_PATH)
    assert summarize_annotation_batch(batch) == {
        "batch_id": "batch-02",
        "split": "test",
        "label_source": "human",
        "queries": 5,
        "pairs": 20,
        "annotated_pairs": 20,
        "pending_pairs": 0,
        "annotators": ["human-annotator-01"],
        "complete": True,
    }
    assert batch.corpus_sha256 == sha256_file(chunks_path)

    previous_queries = {
        task.query.text
        for path in ("data/annotation-batch-01/annotations.json",)
        for task in load_annotation_batch(path).tasks
    }
    previous_queries.update(
        item.query.text for item in load_gold_set("data/sample/gold_set.json").queries
    )
    current_queries = {task.query.text for task in batch.tasks}
    assert current_queries.isdisjoint(previous_queries)
    labels = [
        item.relevance for task in batch.tasks for item in task.candidates
    ]
    assert labels.count(0) == 14
    assert labels.count(1) == 1
    assert labels.count(3) == 5
    for task in batch.tasks:
        chunk_ids = [item.chunk.chunk_id for item in task.candidates]
        assert len(chunk_ids) == 4
        assert chunk_ids == sorted(chunk_ids)
        assert all(
            item.relevance is not None and item.annotator is not None
            for item in task.candidates
        )
        assert all("rank" not in item.model_dump() for item in task.candidates)


def test_frozen_dense_policy_targets_batch02_without_retuning(tmp_path) -> None:
    policy = json.loads(open(POLICY_PATH, encoding="utf-8").read())
    batch = load_annotation_batch(BATCH02_PATH)
    chunks_path = tmp_path / "chunks.jsonl"
    write_jsonl(
        chunks_path, chunk_documents(load_documents("data/sample/documents.jsonl"))
    )
    assert policy["status"] == "frozen_for_independent_evaluation"
    assert policy["min_top_score"] == 0.822904370645214
    assert policy["require_citations"] is True
    assert policy["retriever"]["model"] == "intfloat/multilingual-e5-small"
    assert policy["retriever"]["model_revision"] == (
        "614241f622f53c4eeff9890bdc4f31cfecc418b3"
    )
    assert policy["corpus"]["chunks_sha256"] == sha256_file(chunks_path)
    assert policy["evaluation"] == {
        "split": "test",
        "batch_id": batch.batch_id,
        "status": "evaluation_completed",
        "completed_at": "2026-10-31",
        "gold_set": "artifacts/human-gold-batch-02.json",
        "gold_set_sha256": "1ba689cbb3038b860e506425b09f736e7cd699edb62647fa99a9e4b5e5b491cd",
        "report": "artifacts/selective-evaluation-batch-02-dense.json",
        "report_sha256": "8df42ecfe6baaec04e7d09c68025257d0add185c9e955bde988ea2028cfc7c42",
        "retune_allowed": False,
    }
