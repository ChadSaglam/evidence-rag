"""Benchmark first-stage retrievers and cross-encoder reranking on the real GDPR corpus.

Every model is pinned to the Hugging Face commit it was resolved to, latency is
measured per query on the machine that ran the script, and rerank deltas carry a
paired bootstrap confidence interval because 27 answerable queries is a small sample.

Usage (each first stage is cached, so long runs can be split and resumed):
    python scripts/run_gdpr_benchmark.py --only bm25
    python scripts/run_gdpr_benchmark.py --only dense:multilingual-e5-small
    python scripts/run_gdpr_benchmark.py --report-only
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from evidence_rag.corpus import load_jsonl
from evidence_rag.evaluation.gold import load_gold_set
from evidence_rag.evaluation.runner import evaluate_retriever
from evidence_rag.reranking.cross_encoder import (
    DEFAULT_RERANKER_MODEL,
    CrossEncoderReranker,
    SentenceTransformerCrossEncoder,
)
from evidence_rag.reranking.pipeline import RerankedRetriever
from evidence_rag.retrieval.bm25 import BM25Retriever
from evidence_rag.retrieval.dense import DEFAULT_DENSE_MODEL, DenseRetriever, E5Encoder
from evidence_rag.retrieval.hybrid import HybridRetriever
from evidence_rag.schemas import Chunk

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("mean_hit_at_k", "mean_recall_at_k", "mean_reciprocal_rank", "mean_ndcg_at_k")


def resolve_revision(model_name: str) -> str:
    from huggingface_hub import HfApi

    return HfApi().model_info(model_name).sha


class TimedRetriever:
    """Record wall-clock search latency without changing results."""

    def __init__(self, inner, name: str, *, candidate_k: int | None = None) -> None:
        self.inner = inner
        self.retriever_name = name
        self.candidate_k = candidate_k
        self.latencies_ms: list[float] = []

    def search(self, query, top_k=None):
        started = time.perf_counter()
        if self.candidate_k is None:
            hits = self.inner.search(query, top_k=top_k)
        else:
            hits = self.inner.search(query, candidate_k=self.candidate_k, top_k=top_k)
        self.latencies_ms.append((time.perf_counter() - started) * 1000)
        return hits


def paired_bootstrap(
    baseline: list[float], treatment: list[float], *, samples: int = 10_000, seed: int = 13
) -> dict[str, float]:
    rng = random.Random(seed)
    deltas = [b - a for a, b in zip(baseline, treatment, strict=True)]
    n = len(deltas)
    draws = sorted(mean(deltas[rng.randrange(n)] for _ in range(n)) for _ in range(samples))
    return {
        "mean_delta": mean(deltas),
        "ci95_low": draws[int(0.025 * samples)],
        "ci95_high": draws[int(0.975 * samples) - 1],
        "queries_improved": sum(d > 1e-12 for d in deltas),
        "queries_worse": sum(d < -1e-12 for d in deltas),
    }


def by_language(report) -> dict[str, float]:
    groups: dict[str, list[float]] = {}
    for item in report.per_query:
        groups.setdefault(item.query_id.split("-", 1)[0], []).append(item.ndcg_at_k)
    return {language: mean(values) for language, values in sorted(groups.items())}


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", type=Path, default=ROOT / "data/real-gdpr/chunks.jsonl")
    parser.add_argument("--gold-set", type=Path, default=ROOT / "data/real-gdpr/gold_set.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/gdpr-benchmark")
    parser.add_argument("--dense-model", action="append", default=None)
    parser.add_argument("--rerank-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--only", action="append", default=None, help="First-stage run name.")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    dense_models = args.dense_model or [
        DEFAULT_DENSE_MODEL,
        "intfloat/multilingual-e5-base",
    ]
    cache_dir = args.output_dir / "runs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        run_stages(args, dense_models, cache_dir)
    write_report(args, dense_models, cache_dir)


def stage_names(dense_models: list[str]) -> list[str]:
    names = ["bm25"]
    for model in dense_models:
        short = model.split("/")[-1]
        names += [f"dense:{short}", f"hybrid:{short}"]
    return names


def run_stages(args, dense_models: list[str], cache_dir: Path) -> None:
    wanted = args.only or stage_names(dense_models)

    chunks = load_jsonl(args.chunks, Chunk)
    gold = load_gold_set(args.gold_set)
    reranker = CrossEncoderReranker(
        SentenceTransformerCrossEncoder(args.rerank_model, device=args.device)
    )
    reranker.model_name = args.rerank_model
    revisions = {args.rerank_model: resolve_revision(args.rerank_model)}

    bm25 = BM25Retriever(chunks)
    dense_cache: dict[str, DenseRetriever] = {}
    for name in wanted:
        if name == "bm25":
            retriever, dense_model = bm25, None
        else:
            kind, short = name.split(":", 1)
            dense_model = next(m for m in dense_models if m.split("/")[-1] == short)
            revisions[dense_model] = resolve_revision(dense_model)
            if short not in dense_cache:
                revision = revisions[dense_model]
                encoder = E5Encoder(dense_model, device=args.device, revision=revision)
                dense_cache[short] = DenseRetriever(chunks, encoder, model_name=dense_model)
            dense = dense_cache[short]
            retriever = dense if kind == "dense" else HybridRetriever(bm25, dense)
        for rerank in (False, True):
            label = f"{name}+rerank" if rerank else name
            inner = RerankedRetriever(retriever, reranker) if rerank else retriever
            timed = TimedRetriever(inner, label, candidate_k=args.candidate_k if rerank else None)
            report = evaluate_retriever(timed, gold, k=args.k, retriever_name=label)
            record = {
                "run": label,
                "first_stage": name,
                "reranked": rerank,
                "models": {
                    key: revisions[key]
                    for key in (dense_model, args.rerank_model if rerank else None)
                    if key
                },
                **{metric: getattr(report, metric) for metric in METRICS},
                "ndcg_by_language": by_language(report),
                "latency_ms_p50": percentile(timed.latencies_ms, 0.5),
                "latency_ms_p95": percentile(timed.latencies_ms, 0.95),
                "per_query_ndcg": {item.query_id: item.ndcg_at_k for item in report.per_query},
            }
            (cache_dir / f"{label.replace(':', '__')}.json").write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8"
            )
            print(f"{label:40s} nDCG@{args.k}={report.mean_ndcg_at_k:.3f}", flush=True)


def write_report(args, dense_models: list[str], cache_dir: Path) -> None:
    gold = load_gold_set(args.gold_set)
    records = {}
    for name in stage_names(dense_models):
        for label in (name, f"{name}+rerank"):
            path = cache_dir / f"{label.replace(':', '__')}.json"
            if path.exists():
                records[label] = json.loads(path.read_text(encoding="utf-8"))
    if not records:
        raise SystemExit("No cached runs found; run without --report-only first.")
    models: dict[str, str] = {}
    for record in records.values():
        models.update(record["models"])
    deltas = {}
    for name in stage_names(dense_models):
        if name in records and f"{name}+rerank" in records:
            base = records[name]["per_query_ndcg"]
            treated = records[f"{name}+rerank"]["per_query_ndcg"]
            ids = sorted(base)
            deltas[name] = paired_bootstrap([base[i] for i in ids], [treated[i] for i in ids])
    result = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus": str(args.chunks.resolve().relative_to(ROOT)),
        "gold_set": gold.name,
        "gold_label_source": gold.label_source,
        "k": args.k,
        "candidate_k": args.candidate_k,
        "answerable_queries": len(next(iter(records.values()))["per_query_ndcg"]),
        "models": models,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": args.device,
        },
        "runs": [
            {key: value for key, value in record.items() if key != "per_query_ndcg"}
            for record in records.values()
        ],
        "rerank_ndcg_delta_paired_bootstrap": deltas,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(render_markdown(result), encoding="utf-8")


def render_markdown(result: dict) -> str:
    k = result["k"]
    lines = [
        "# GDPR retrieval and reranking benchmark",
        "",
        f"Corpus `{result['corpus']}`, gold set `{result['gold_set']}` "
        f"(label source: **{result['gold_label_source']}**, not human), "
        f"{result['answerable_queries']} answerable queries in EN/DE/FR, k={k}, "
        f"rerank candidate depth={result['candidate_k']}.",
        "",
        "| Run | Hit@k | Recall@k | MRR | nDCG@k | nDCG EN | nDCG DE | nDCG FR | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for run in result["runs"]:
        lang = run["ndcg_by_language"]
        lines.append(
            f"| {run['run']} | {run['mean_hit_at_k']:.3f} | {run['mean_recall_at_k']:.3f} | "
            f"{run['mean_reciprocal_rank']:.3f} | **{run['mean_ndcg_at_k']:.3f}** | "
            f"{lang.get('en', 0):.3f} | {lang.get('de', 0):.3f} | {lang.get('fr', 0):.3f} | "
            f"{run['latency_ms_p50']:.0f} | {run['latency_ms_p95']:.0f} |"
        )
    lines += [
        "",
        f"## Reranking effect on nDCG@{k} (paired bootstrap, 10,000 resamples)",
        "",
        "| First stage | Δ nDCG | 95% CI | improved / worse queries |",
        "|---|---|---|---|",
    ]
    for name, delta in result["rerank_ndcg_delta_paired_bootstrap"].items():
        lines.append(
            f"| {name} | {delta['mean_delta']:+.3f} | "
            f"[{delta['ci95_low']:+.3f}, {delta['ci95_high']:+.3f}] | "
            f"{delta['queries_improved']} / {delta['queries_worse']} |"
        )
    lines += ["", "## Pinned models", ""]
    lines += [f"- `{name}` @ `{sha}`" for name, sha in result["models"].items()]
    env = result["environment"]
    lines += [
        "",
        f"Environment: Python {env['python']}, {env['platform']}, device `{env['device']}`. "
        "Latency is wall-clock per query on this machine, including query encoding.",
        "",
        "Labels are deterministic article-level source alignment. A CI that crosses zero "
        "means the reranker effect is not distinguishable from noise at this sample size.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
