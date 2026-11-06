import json

import pytest
from pydantic import ValidationError

from evidence_rag.benchmark import (
    export_kaggle_dataset,
    run_benchmark,
    verify_manifest_files,
    write_benchmark_report,
)
from evidence_rag.benchmark.kaggle import load_benchmark_manifest
from evidence_rag.benchmark.runner import report_json
from evidence_rag.chunking import chunk_documents
from evidence_rag.cli import main
from evidence_rag.corpus import load_documents, write_jsonl
from evidence_rag.schemas import BenchmarkFile, BenchmarkManifest, BenchmarkReport


@pytest.fixture
def chunks_path(tmp_path):
    path = tmp_path / "input-chunks.jsonl"
    chunks = chunk_documents(load_documents("data/sample/documents.jsonl"))
    write_jsonl(path, chunks)
    return path


@pytest.fixture
def dataset_dir(tmp_path, chunks_path):
    destination = tmp_path / "dataset"
    export_kaggle_dataset(
        documents_path="data/sample/documents.jsonl",
        chunks_path=chunks_path,
        gold_set_path="data/sample/gold_set.json",
        output_dir=destination,
        dataset_id="fixture/evidence-rag-benchmark",
        name="Deterministic fixture benchmark",
        description="Synthetic labels for deterministic software testing.",
        k=4,
    )
    return destination


def test_kaggle_export_is_self_contained_and_hashed(dataset_dir) -> None:
    manifest = load_benchmark_manifest(dataset_dir / "benchmark-manifest.json")
    metadata = json.loads((dataset_dir / "dataset-metadata.json").read_text(encoding="utf-8"))
    paths = {item.path for item in manifest.files}
    assert {"documents.jsonl", "chunks.jsonl", "gold_set.json"}.issubset(paths)
    assert "evidence_rag/benchmark/runner.py" in paths
    assert metadata["id"] == "fixture/evidence-rag-benchmark"
    assert metadata["licenses"] == [{"name": "MIT"}]
    verify_manifest_files(manifest, dataset_dir)


def test_kaggle_export_is_byte_deterministic(tmp_path, chunks_path) -> None:
    for name in ("first", "second"):
        export_kaggle_dataset(
            documents_path="data/sample/documents.jsonl",
            chunks_path=chunks_path,
            gold_set_path="data/sample/gold_set.json",
            output_dir=tmp_path / name,
            dataset_id="fixture/evidence-rag-benchmark",
            name="Deterministic fixture benchmark",
            description="Synthetic labels for deterministic software testing.",
            k=4,
        )
    first = tmp_path / "first"
    second = tmp_path / "second"
    first_files = {path.relative_to(first) for path in first.rglob("*") if path.is_file()}
    second_files = {path.relative_to(second) for path in second.rglob("*") if path.is_file()}
    assert first_files == second_files
    assert all((first / path).read_bytes() == (second / path).read_bytes() for path in first_files)


def test_kaggle_notebook_is_valid_and_has_no_saved_outputs() -> None:
    notebook = json.loads(open("kaggle/benchmark.ipynb", encoding="utf-8").read())
    assert notebook["nbformat"] == 4
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert all(cell["execution_count"] is None and cell["outputs"] == [] for cell in code_cells)
    assert any("run_benchmark" in "".join(cell["source"]) for cell in code_cells)


def test_benchmark_report_and_markdown_are_byte_deterministic(dataset_dir, tmp_path) -> None:
    manifest_path = dataset_dir / "benchmark-manifest.json"
    first = run_benchmark(manifest_path)
    second = run_benchmark(manifest_path)
    assert report_json(first) == report_json(second)
    evaluation = first.evaluations[0]
    assert evaluation.evaluated_queries == 4
    assert evaluation.skipped_unanswerable_queries == 1
    assert evaluation.mean_hit_at_k == 1.0
    assert evaluation.gold_label_source == "synthetic"

    json_path, markdown_path = write_benchmark_report(first, tmp_path / "results")
    json_bytes = json_path.read_bytes()
    markdown_bytes = markdown_path.read_bytes()
    write_benchmark_report(second, tmp_path / "results")
    assert json_path.read_bytes() == json_bytes
    assert markdown_path.read_bytes() == markdown_bytes
    assert b"timestamp" not in markdown_bytes.lower() or b"no timestamp" in markdown_bytes.lower()


def test_benchmark_rejects_tampered_dataset_file(dataset_dir) -> None:
    chunks_path = dataset_dir / "chunks.jsonl"
    chunks_path.write_text(chunks_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch: chunks.jsonl"):
        run_benchmark(dataset_dir / "benchmark-manifest.json")


def test_benchmark_file_rejects_parent_paths() -> None:
    with pytest.raises(ValidationError, match="parent segments"):
        BenchmarkFile(path="../secret", sha256="0" * 64, size_bytes=1)


def test_manifest_requires_all_dataset_inputs(dataset_dir) -> None:
    manifest = load_benchmark_manifest(dataset_dir / "benchmark-manifest.json")
    payload = manifest.model_dump(mode="json")
    payload["files"] = [item for item in payload["files"] if item["path"] != "documents.jsonl"]
    with pytest.raises(ValidationError, match="required dataset files"):
        BenchmarkManifest.model_validate(payload)


def test_report_rejects_configuration_mismatch(dataset_dir) -> None:
    report = run_benchmark(dataset_dir / "benchmark-manifest.json")
    payload = report.model_dump(mode="json")
    payload["evaluations"][0]["k"] = 3
    with pytest.raises(ValidationError, match="cutoff"):
        BenchmarkReport.model_validate(payload)


def test_kaggle_and_benchmark_cli_round_trip(tmp_path, chunks_path, capsys) -> None:
    dataset_dir = tmp_path / "cli-dataset"
    main(
        [
            "kaggle-export",
            "--documents",
            "data/sample/documents.jsonl",
            "--chunks",
            str(chunks_path),
            "--gold-set",
            "data/sample/gold_set.json",
            "--output-dir",
            str(dataset_dir),
            "--dataset-id",
            "fixture/cli-benchmark",
            "--k",
            "4",
        ]
    )
    exported = json.loads(capsys.readouterr().out)
    assert exported["retriever"] == "bm25"

    arguments = ["benchmark-run", "--manifest", str(dataset_dir / "benchmark-manifest.json")]
    main(arguments)
    first = capsys.readouterr().out
    main(arguments)
    second = capsys.readouterr().out
    assert first == second
    assert json.loads(first)["evaluations"][0]["mean_hit_at_k"] == 1.0

    main(arguments + ["--output-dir", str(tmp_path / "cli-results")])
    outputs = json.loads(capsys.readouterr().out)
    assert (tmp_path / "cli-results" / "benchmark-report.json").is_file()
    assert (tmp_path / "cli-results" / "benchmark-report.md").is_file()
    assert outputs["json"].endswith("benchmark-report.json")
