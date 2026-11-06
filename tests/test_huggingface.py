import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from evidence_rag.chunking import chunk_documents
from evidence_rag.cli import main
from evidence_rag.corpus import load_documents, write_jsonl
from evidence_rag.huggingface import (
    DemoSearch,
    export_huggingface_dataset,
    render_dataset_card,
    verify_huggingface_dataset,
)
from evidence_rag.schemas import HuggingFaceDatasetManifest


@pytest.fixture
def chunks_path(tmp_path):
    path = tmp_path / "chunks.jsonl"
    write_jsonl(path, chunk_documents(load_documents("data/sample/documents.jsonl")))
    return path


def export_dataset(output_dir: Path, chunks_path: Path) -> HuggingFaceDatasetManifest:
    return export_huggingface_dataset(
        documents_path="data/sample/documents.jsonl",
        chunks_path=chunks_path,
        gold_set_path="data/sample/gold_set.json",
        output_dir=output_dir,
        dataset_id="fixture/evidence-rag",
        name="Evidence RAG fixture dataset",
        description="Synthetic multilingual retrieval fixtures.",
    )


def test_huggingface_export_has_configs_card_and_valid_hashes(tmp_path, chunks_path) -> None:
    destination = tmp_path / "dataset"
    manifest = export_dataset(destination, chunks_path)
    assert verify_huggingface_dataset(destination / "dataset-manifest.json") == manifest
    assert len(manifest.files) == 4
    card = (destination / "README.md").read_text(encoding="utf-8")
    assert card.startswith("---\nlicense: mit\n")
    assert '- "de"\n- "en"\n- "fr"' in card
    assert "config_name: documents" in card
    assert "config_name: chunks" in card
    assert "config_name: gold_set" in card
    assert "load_dataset(\"fixture/evidence-rag\", \"chunks\"" in card


def test_huggingface_export_is_byte_deterministic(tmp_path, chunks_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    export_dataset(first, chunks_path)
    export_dataset(second, chunks_path)
    first_files = {path.relative_to(first) for path in first.rglob("*") if path.is_file()}
    second_files = {path.relative_to(second) for path in second.rglob("*") if path.is_file()}
    assert first_files == second_files
    assert all((first / path).read_bytes() == (second / path).read_bytes() for path in first_files)


def test_huggingface_verification_rejects_tampering(tmp_path, chunks_path) -> None:
    destination = tmp_path / "dataset"
    export_dataset(destination, chunks_path)
    card = destination / "README.md"
    card.write_text(card.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch: README.md"):
        verify_huggingface_dataset(destination / "dataset-manifest.json")


def test_huggingface_manifest_requires_data_configs(tmp_path, chunks_path) -> None:
    destination = tmp_path / "dataset"
    manifest = export_dataset(destination, chunks_path)
    payload = manifest.model_dump(mode="json")
    payload["files"] = [
        item for item in payload["files"] if item["path"] != "data/gold_set.json"
    ]
    with pytest.raises(ValidationError, match="required data files"):
        HuggingFaceDatasetManifest.model_validate(payload)


def test_dataset_card_preserves_human_label_provenance() -> None:
    card = render_dataset_card(
        name="Human dataset",
        description="Human relevance labels.",
        dataset_id="fixture/human-evidence",
        languages=["en"],
        label_source="human",
    )
    assert "- human\n" in card
    assert "- synthetic\n" not in card
    assert "explicit `human` label provenance" in card


def test_demo_search_is_deterministic_and_multilingual() -> None:
    chunks = chunk_documents(load_documents("data/sample/documents.jsonl"))
    service = DemoSearch(chunks)
    english = service.search("How long must accounting records be retained?", 3)
    german = service.search("Wie lange müssen Geschäftsbücher aufbewahrt werden?", 3)
    assert english.hits[0].chunk.chunk_id == "en-retention:000000"
    assert german.hits[0].chunk.chunk_id == "de-retention:000000"
    assert english.hits[0].chunk.source.uri
    assert service.search_json(english.query.text, 3) == service.search_json(
        english.query.text, 3.0
    )


def test_demo_rejects_empty_queries_and_invalid_depth() -> None:
    chunks = chunk_documents(load_documents("data/sample/documents.jsonl"))
    service = DemoSearch(chunks)
    with pytest.raises(ValueError, match="non-empty"):
        service.search("  ")
    with pytest.raises(ValidationError):
        service.search("records", 0)


def test_huggingface_cli_export_and_verify(tmp_path, chunks_path, capsys) -> None:
    destination = tmp_path / "cli-dataset"
    main(
        [
            "hf-export",
            "--documents",
            "data/sample/documents.jsonl",
            "--chunks",
            str(chunks_path),
            "--gold-set",
            "data/sample/gold_set.json",
            "--output-dir",
            str(destination),
            "--dataset-id",
            "fixture/cli-evidence-rag",
        ]
    )
    exported = json.loads(capsys.readouterr().out)
    assert exported["label_source"] == "synthetic"
    main(["hf-verify", "--manifest", str(destination / "dataset-manifest.json")])
    assert json.loads(capsys.readouterr().out) == {"files": 4, "valid": True}


def test_demo_space_and_local_preview_configuration() -> None:
    package = json.loads(Path("package.json").read_text(encoding="utf-8"))
    requirements = Path("demo/requirements.txt").read_text(encoding="utf-8")
    space_card = Path("demo/README.md").read_text(encoding="utf-8")
    assert package["scripts"]["dev"] == ".venv/bin/python demo/app.py"
    assert "gradio>=5,<7" in requirements
    assert "sdk: gradio" in space_card
    assert "app_file: app.py" in space_card
