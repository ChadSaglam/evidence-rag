import json

from evidence_rag.benchmark.kaggle import sha256_file
from evidence_rag.chunking import chunk_documents
from evidence_rag.citation import ClaimLevelCitationVerifier
from evidence_rag.cli import main
from evidence_rag.corpus import load_documents, write_jsonl
from evidence_rag.retrieval.bm25 import BM25Retriever
from evidence_rag.schemas import Query, RetrievalResponse


def test_sample_evaluation_cli_is_deterministic(tmp_path, capsys) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    chunks = chunk_documents(load_documents("data/sample/documents.jsonl"))
    write_jsonl(chunks_path, chunks)
    arguments = [
        "evaluate",
        "--gold-set",
        "data/sample/gold_set.json",
        "--chunks",
        str(chunks_path),
        "--method",
        "bm25",
        "--k",
        "4",
    ]
    main(arguments)
    first = json.loads(capsys.readouterr().out)
    main(arguments)
    second = json.loads(capsys.readouterr().out)
    assert first == second
    assert first["evaluated_queries"] == 4
    assert first["skipped_unanswerable_queries"] == 1
    assert first["gold_label_source"] == "synthetic"


def test_selective_cli_combines_response_and_verification(tmp_path, capsys) -> None:
    chunks = chunk_documents(load_documents("data/sample/documents.jsonl"))
    hits = BM25Retriever(chunks).search("accounting records", top_k=1)
    response = RetrievalResponse(query=Query(query_id="q", text="accounting records"), hits=hits)
    response_path = tmp_path / "response.json"
    response_path.write_text(response.model_dump_json(), encoding="utf-8")
    report = ClaimLevelCitationVerifier().verify(
        "Accounting records must be retained for ten years [1].", hits
    )
    verification_path = tmp_path / "verification.json"
    verification_path.write_text(report.model_dump_json(), encoding="utf-8")
    main(
        [
            "selective",
            "--response",
            str(response_path),
            "--verification",
            str(verification_path),
            "--min-top-score",
            "1",
        ]
    )
    decision = json.loads(capsys.readouterr().out)
    assert decision["action"] == "answer"
    assert decision["reason"] == "evidence_and_citations_passed"


def test_selective_calibration_cli_writes_deterministic_report(tmp_path, capsys) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    chunks = chunk_documents(load_documents("data/sample/documents.jsonl"))
    write_jsonl(chunks_path, chunks)
    output = tmp_path / "calibration.json"
    arguments = [
        "calibrate-selective",
        "--gold-set",
        "data/sample/gold_set.json",
        "--chunks",
        str(chunks_path),
        "--method",
        "bm25",
        "--max-risk",
        "0",
        "--output",
        str(output),
    ]
    main(arguments)
    first = json.loads(capsys.readouterr().out)
    main(arguments)
    second = json.loads(capsys.readouterr().out)
    assert first == second == json.loads(output.read_text(encoding="utf-8"))
    assert first["retriever"] == "bm25"
    assert first["query_count"] == 5
    assert first["selected"]["wrong_answer_rate"] == 0.0
    thresholds = [point["threshold"] for point in first["candidates"]]
    assert thresholds == sorted(thresholds)


def test_frozen_policy_evaluation_cli_is_deterministic(tmp_path, capsys) -> None:
    chunks_path = tmp_path / "chunks.jsonl"
    chunks = chunk_documents(load_documents("data/sample/documents.jsonl"))
    write_jsonl(chunks_path, chunks)
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "policy_id": "fixture-bm25-policy",
                "status": "frozen_for_independent_evaluation",
                "frozen_at": "2026-10-30",
                "min_top_score": 0.0,
                "top_k": 1,
                "require_citations": True,
                "decision_rule": "Fixed threshold.",
                "retriever": {"method": "bm25"},
                "calibration": {},
                "corpus": {"chunks_sha256": sha256_file(chunks_path)},
                "evaluation": {
                    "split": "test",
                    "batch_id": "synthetic-multilingual-smoke-set",
                    "retune_allowed": False,
                },
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "evaluation.json"
    arguments = [
        "evaluate-selective",
        "--policy",
        str(policy_path),
        "--gold-set",
        "data/sample/gold_set.json",
        "--chunks",
        str(chunks_path),
        "--output",
        str(output),
    ]
    main(arguments)
    first = json.loads(capsys.readouterr().out)
    main(arguments)
    second = json.loads(capsys.readouterr().out)
    assert first == second == json.loads(output.read_text(encoding="utf-8"))
    assert first["policy_id"] == "fixture-bm25-policy"
    assert first["min_top_score"] == 0.0
    assert first["metrics"]["threshold"] == 0.0
    assert first["query_count"] == 5
    assert "candidates" not in first
