"""Export a self-contained Kaggle dataset with deterministic content hashes."""

import hashlib
import json
import shutil
from pathlib import Path

from evidence_rag.evaluation.gold import load_gold_set
from evidence_rag.schemas import BenchmarkFile, BenchmarkManifest

MANIFEST_NAME = "benchmark-manifest.json"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def export_kaggle_dataset(
    *,
    documents_path: str | Path,
    chunks_path: str | Path,
    gold_set_path: str | Path,
    output_dir: str | Path,
    dataset_id: str,
    name: str,
    description: str,
    k: int = 4,
    license_name: str = "MIT",
) -> BenchmarkManifest:
    """Copy data and source code into an offline, Kaggle-ready dataset directory."""

    gold_set = load_gold_set(gold_set_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    inputs = {
        "documents.jsonl": Path(documents_path),
        "chunks.jsonl": Path(chunks_path),
        "gold_set.json": Path(gold_set_path),
    }
    for relative_path, source in inputs.items():
        copy_file(source, destination / relative_path)

    package_root = Path(__file__).resolve().parents[1]
    for source in sorted(package_root.rglob("*.py")):
        relative_path = source.relative_to(package_root)
        copy_file(source, destination / "evidence_rag" / relative_path)

    files = [
        BenchmarkFile(
            path=path.relative_to(destination).as_posix(),
            sha256=sha256_file(path),
            size_bytes=path.stat().st_size,
        )
        for path in sorted(destination.rglob("*"))
        if path.is_file() and path.name not in {MANIFEST_NAME, "dataset-metadata.json"}
    ]
    manifest = BenchmarkManifest(
        name=name,
        description=description,
        dataset_id=dataset_id,
        gold_set=gold_set.name,
        label_source=gold_set.label_source,
        k=k,
        files=files,
    )
    write_json(destination / MANIFEST_NAME, manifest.model_dump(mode="json"))
    write_json(
        destination / "dataset-metadata.json",
        {"title": name, "id": dataset_id, "licenses": [{"name": license_name}]},
    )
    return manifest


def load_benchmark_manifest(path: str | Path) -> BenchmarkManifest:
    return BenchmarkManifest.model_validate_json(Path(path).read_text(encoding="utf-8"))
