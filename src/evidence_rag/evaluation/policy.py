"""Load frozen selective-answering policies with strict runtime validation."""

from pathlib import Path

from evidence_rag.schemas import FrozenSelectivePolicy


def load_frozen_selective_policy(path: str | Path) -> FrozenSelectivePolicy:
    return FrozenSelectivePolicy.model_validate_json(Path(path).read_text(encoding="utf-8"))
