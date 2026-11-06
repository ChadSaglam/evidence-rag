"""Load and summarize validated gold-set artifacts."""

import json
from pathlib import Path

from evidence_rag.schemas import GoldSet


def load_gold_set(path: str | Path) -> GoldSet:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    return GoldSet.model_validate(payload)


def summarize_gold_set(gold_set: GoldSet) -> dict[str, int | str | bool]:
    answerable = sum(item.answerable for item in gold_set.queries)
    return {
        "name": gold_set.name,
        "label_source": gold_set.label_source,
        "queries": len(gold_set.queries),
        "answerable_queries": answerable,
        "unanswerable_queries": len(gold_set.queries) - answerable,
        "judgments": sum(len(item.judgments) for item in gold_set.queries),
        "valid": True,
    }
