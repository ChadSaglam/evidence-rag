"""Load validated poisoned-document security fixtures."""

from pathlib import Path

from pydantic import TypeAdapter

from evidence_rag.schemas import PoisonedDocumentCase

CASE_LIST = TypeAdapter(list[PoisonedDocumentCase])


def load_poisoned_cases(path: str | Path) -> list[PoisonedDocumentCase]:
    return CASE_LIST.validate_json(Path(path).read_text(encoding="utf-8"))
