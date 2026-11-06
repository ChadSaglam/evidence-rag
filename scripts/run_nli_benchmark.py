"""Compare lexical and NLI citation verification on constructed GDPR claims.

The headline number is the false-accept rate: how often a verifier marks a claim as
supported when the cited evidence does not support it (negations, altered numbers,
unrelated claims). For an evidence-first system that is the failure that matters.

Usage:
    python scripts/run_nli_benchmark.py --output-dir results/nli-benchmark
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from evidence_rag.citation.nli import (
    DEFAULT_NLI_MODEL,
    NLICitationVerifier,
    TransformersNLIModel,
)
from evidence_rag.citation.verifier import ClaimLevelCitationVerifier
from evidence_rag.corpus import load_jsonl
from evidence_rag.schemas import Chunk, RetrievalHit

ROOT = Path(__file__).resolve().parents[1]
LEXICAL_MAP = {"lexically_supported": "supported"}
NLI_MAP = {"semantically_supported": "supported", "contradicted": "contradicted"}


def summarize(rows: list[dict], key: str) -> dict:
    accepted_unsupported = [r for r in rows if r["label"] != "supported" and r[key] == "supported"]
    rejected_supported = [r for r in rows if r["label"] == "supported" and r[key] != "supported"]
    negatives = sum(r["label"] != "supported" for r in rows)
    positives = len(rows) - negatives
    binary_correct = sum((r["label"] == "supported") == (r[key] == "supported") for r in rows)
    return {
        "binary_accuracy": binary_correct / len(rows),
        "false_accept_rate": len(accepted_unsupported) / negatives if negatives else 0.0,
        "false_reject_rate": len(rejected_supported) / positives if positives else 0.0,
        "three_class_accuracy": sum(r["label"] == r[key] for r in rows) / len(rows),
        "false_accepts": [r["case_id"] for r in accepted_unsupported],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, default=ROOT / "data/citation-nli/cases.json")
    parser.add_argument("--chunks", type=Path, default=ROOT / "data/real-gdpr/chunks.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/nli-benchmark")
    parser.add_argument("--model", default=DEFAULT_NLI_MODEL)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from huggingface_hub import HfApi

    revision = HfApi().model_info(args.model).sha
    dataset = json.loads(args.cases.read_text(encoding="utf-8"))
    chunks = {chunk.chunk_id: chunk for chunk in load_jsonl(args.chunks, Chunk)}
    lexical = ClaimLevelCitationVerifier()
    nli = NLICitationVerifier(
        TransformersNLIModel(args.model, device=args.device, revision=revision)
    )

    rows = []
    for case in dataset["cases"]:
        hit = RetrievalHit(rank=1, score=1.0, retriever="fixture", chunk=chunks[case["chunk_id"]])
        lexical_claim = lexical.verify(case["answer"], [hit]).claims[0]
        nli_claim = nli.verify(case["answer"], [hit]).claims[0]
        rows.append(
            {
                "case_id": case["case_id"],
                "language": case["language"],
                "perturbation": case["perturbation"],
                "label": case["label"],
                "lexical": LEXICAL_MAP.get(lexical_claim.status, "not_supported"),
                "lexical_score": lexical_claim.support_score,
                "nli": NLI_MAP.get(nli_claim.status, "not_supported"),
                "entailment": nli_claim.entailment_score,
                "contradiction": nli_claim.contradiction_score,
            }
        )

    by_perturbation = defaultdict(list)
    for row in rows:
        by_perturbation[row["perturbation"]].append(row)
    result = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset["name"],
        "label_source": dataset["label_source"],
        "cases": len(rows),
        "nli_model": {"name": args.model, "revision": revision},
        "lexical": summarize(rows, "lexical"),
        "nli": summarize(rows, "nli"),
        "by_perturbation": {
            name: {
                "cases": len(group),
                "lexical_binary_accuracy": summarize(group, "lexical")["binary_accuracy"],
                "nli_binary_accuracy": summarize(group, "nli")["binary_accuracy"],
            }
            for name, group in sorted(by_perturbation.items())
        },
        "rows": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (args.output_dir / "report.md").write_text(render(result), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("lexical", "nli")}, indent=2))


def render(result: dict) -> str:
    lex, nli = result["lexical"], result["nli"]
    lines = [
        "# Citation verification: lexical overlap vs NLI",
        "",
        f"{result['cases']} constructed claims over official GDPR chunks (EN/DE/FR and "
        f"cross-lingual). Label source: **{result['label_source']}**, single curator, "
        "not independently annotated.",
        "",
        "| Verifier | Binary accuracy | False-accept rate | False-reject rate | 3-class accuracy |",
        "|---|---|---|---|---|",
        f"| lexical-overlap-v1 | {lex['binary_accuracy']:.2f} | {lex['false_accept_rate']:.2f} "
        f"| {lex['false_reject_rate']:.2f} | n/a |",
        f"| nli-entailment-v1 | {nli['binary_accuracy']:.2f} | {nli['false_accept_rate']:.2f} "
        f"| {nli['false_reject_rate']:.2f} | {nli['three_class_accuracy']:.2f} |",
        "",
        "## By perturbation (binary accuracy)",
        "",
        "| Perturbation | Cases | Lexical | NLI |",
        "|---|---|---|---|",
    ]
    for name, group in result["by_perturbation"].items():
        lines.append(
            f"| {name} | {group['cases']} | {group['lexical_binary_accuracy']:.2f} | "
            f"{group['nli_binary_accuracy']:.2f} |"
        )
    lines += [
        "",
        f"False accepts, lexical: {', '.join(lex['false_accepts']) or 'none'}",
        "",
        f"False accepts, NLI: {', '.join(nli['false_accepts']) or 'none'}",
        "",
        f"NLI model: `{result['nli_model']['name']}` @ `{result['nli_model']['revision']}`.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
