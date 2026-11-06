"""Export deterministic Hugging Face datasets with JSONL configs and a dataset card."""

import json
from collections.abc import Sequence
from pathlib import Path

from evidence_rag.benchmark.kaggle import copy_file, sha256_file, write_json
from evidence_rag.corpus import load_chunks, load_documents
from evidence_rag.evaluation.gold import load_gold_set
from evidence_rag.schemas import BenchmarkFile, HuggingFaceDatasetManifest

MANIFEST_NAME = "dataset-manifest.json"


def render_dataset_card(
    *,
    name: str,
    description: str,
    dataset_id: str,
    languages: Sequence[str],
    label_source: str,
    license_id: str = "mit",
    attribution: str | None = None,
) -> str:
    attribution_section = (
        f"\n## Source and license\n\n{attribution}\n" if attribution else ""
    )
    language_yaml = "\n".join(f"- {json.dumps(language)}" for language in languages)
    return f"""---
license: {license_id}
pretty_name: {json.dumps(name, ensure_ascii=False)}
task_categories:
- text-retrieval
language:
{language_yaml}
tags:
- retrieval-augmented-generation
- information-retrieval
- bm25
- {label_source}
configs:
- config_name: documents
  data_files:
  - split: train
    path: data/documents.jsonl
- config_name: chunks
  data_files:
  - split: train
    path: data/chunks.jsonl
- config_name: gold_set
  data_files:
  - split: train
    path: data/gold_set.json
---
# {name}

{description}

Dataset ID: `{dataset_id}`

This dataset preserves document attribution, exact chunk offsets, graded relevance
judgments, and explicit `{label_source}` label provenance. Synthetic fixtures are not
human-labeled benchmarks. The bundled sample is not legal advice.

## Load a config

```python
from datasets import load_dataset

chunks = load_dataset("{dataset_id}", "chunks", split="train")
documents = load_dataset("{dataset_id}", "documents", split="train")
gold_set = load_dataset("{dataset_id}", "gold_set", split="train")
```

Each config contains one JSONL record per row. `dataset-manifest.json` records file
sizes and SHA-256 hashes for reproducibility. No files are uploaded by local export.
{attribution_section}"""


def export_huggingface_dataset(
    *,
    documents_path: str | Path,
    chunks_path: str | Path,
    gold_set_path: str | Path,
    output_dir: str | Path,
    dataset_id: str,
    name: str,
    description: str,
    license_id: str = "mit",
    attribution: str | None = None,
) -> HuggingFaceDatasetManifest:
    documents = load_documents(documents_path)
    load_chunks(chunks_path)
    gold_set = load_gold_set(gold_set_path)
    if not documents:
        raise ValueError("At least one document is required for dataset export")
    destination = Path(output_dir)
    inputs = {
        "data/documents.jsonl": Path(documents_path),
        "data/chunks.jsonl": Path(chunks_path),
        "data/gold_set.json": Path(gold_set_path),
    }
    for relative_path, source in inputs.items():
        copy_file(source, destination / relative_path)

    languages = sorted({document.source.language for document in documents})
    card = render_dataset_card(
        name=name,
        description=description,
        dataset_id=dataset_id,
        languages=languages,
        label_source=gold_set.label_source,
        license_id=license_id,
        attribution=attribution,
    )
    (destination / "README.md").write_text(card, encoding="utf-8")
    files = [
        BenchmarkFile(
            path=path.relative_to(destination).as_posix(),
            sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
        )
        for path in sorted(destination.rglob("*"))
        if path.is_file() and path.name != MANIFEST_NAME
    ]
    manifest = HuggingFaceDatasetManifest(
        name=name,
        description=description,
        dataset_id=dataset_id,
        gold_set=gold_set.name,
        label_source=gold_set.label_source,
        files=files,
    )
    write_json(destination / MANIFEST_NAME, manifest.model_dump(mode="json"))
    return manifest


def load_huggingface_manifest(path: str | Path) -> HuggingFaceDatasetManifest:
    return HuggingFaceDatasetManifest.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )


def verify_huggingface_dataset(manifest_path: str | Path) -> HuggingFaceDatasetManifest:
    source = Path(manifest_path)
    manifest = load_huggingface_manifest(source)
    for item in manifest.files:
        path = source.parent.joinpath(*item.path.split("/"))
        if not path.is_file():
            raise ValueError(f"Dataset file is missing: {item.path}")
        if path.stat().st_size != item.size_bytes or sha256_file(path) != item.sha256:
            raise ValueError(f"Dataset file hash mismatch: {item.path}")
    return manifest
