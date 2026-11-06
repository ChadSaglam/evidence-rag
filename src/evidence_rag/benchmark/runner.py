"""Verify a benchmark dataset and produce deterministic JSON and Markdown reports."""

import json
from pathlib import Path

from evidence_rag.benchmark.kaggle import (
    load_benchmark_manifest,
    sha256_file,
    write_json,
)
from evidence_rag.corpus import load_chunks
from evidence_rag.evaluation.gold import load_gold_set
from evidence_rag.evaluation.runner import evaluate_retriever
from evidence_rag.retrieval.bm25 import BM25Retriever
from evidence_rag.schemas import BenchmarkManifest, BenchmarkReport


def verify_manifest_files(manifest: BenchmarkManifest, root: str | Path) -> None:
    dataset_root = Path(root)
    for item in manifest.files:
        path = dataset_root.joinpath(*item.path.split("/"))
        if not path.is_file():
            raise ValueError(f"Benchmark file is missing: {item.path}")
        if path.stat().st_size != item.size_bytes or sha256_file(path) != item.sha256:
            raise ValueError(f"Benchmark file hash mismatch: {item.path}")


def run_benchmark(manifest_path: str | Path) -> BenchmarkReport:
    source = Path(manifest_path)
    root = source.parent
    manifest = load_benchmark_manifest(source)
    verify_manifest_files(manifest, root)
    gold_set = load_gold_set(root / "gold_set.json")
    if gold_set.name != manifest.gold_set or gold_set.label_source != manifest.label_source:
        raise ValueError("Gold-set identity does not match the benchmark manifest")
    retriever = BM25Retriever(load_chunks(root / "chunks.jsonl"))
    evaluation = evaluate_retriever(
        retriever,
        gold_set,
        k=manifest.k,
        retriever_name=manifest.retriever,
    )
    return BenchmarkReport(manifest=manifest, evaluations=[evaluation])


def render_benchmark_markdown(report: BenchmarkReport) -> str:
    manifest = report.manifest
    lines = [
        f"# {manifest.name}",
        "",
        manifest.description,
        "",
        f"- Dataset ID: `{manifest.dataset_id}`",
        f"- Gold set: `{manifest.gold_set}`",
        f"- Label source: `{manifest.label_source}`",
        f"- Retriever: `{manifest.retriever}`",
        f"- Cutoff: `{manifest.k}`",
        "",
        "## Macro-averaged metrics",
        "",
        "| Retriever | Queries | Skipped | Hit@k | Recall@k | Precision@k | MRR | nDCG@k |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report.evaluations:
        lines.append(
            f"| {item.retriever} | {item.evaluated_queries} | "
            f"{item.skipped_unanswerable_queries} | {item.mean_hit_at_k:.6f} | "
            f"{item.mean_recall_at_k:.6f} | {item.mean_precision_at_k:.6f} | "
            f"{item.mean_reciprocal_rank:.6f} | {item.mean_ndcg_at_k:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Content-addressed inputs",
            "",
            "| Path | Bytes | SHA-256 |",
            "|---|---:|---|",
        ]
    )
    for item in manifest.files:
        lines.append(f"| `{item.path}` | {item.size_bytes} | `{item.sha256}` |")
    lines.extend(
        [
            "",
            "Synthetic labels are software fixtures, not human relevance judgments. "
            "This report contains no timestamp or machine-specific path.",
            "",
        ]
    )
    return "\n".join(lines)


def write_benchmark_report(report: BenchmarkReport, output_dir: str | Path) -> tuple[Path, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "benchmark-report.json"
    markdown_path = destination / "benchmark-report.md"
    write_json(json_path, report.model_dump(mode="json"))
    markdown_path.write_text(render_benchmark_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def report_json(report: BenchmarkReport) -> str:
    return json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2)
