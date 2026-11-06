"""Strict JSONL input and output for corpus artifacts."""

import json
from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from evidence_rag.schemas import Chunk, Document

ModelT = TypeVar("ModelT", bound=BaseModel)


def load_jsonl(path: str | Path, model: type[ModelT]) -> list[ModelT]:
    """Load UTF-8 JSONL, skipping blank lines and reporting exact error locations."""

    records: list[ModelT] = []
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(model.model_validate_json(line))
            except ValueError as error:
                raise ValueError(f"Invalid record at {source}:{line_number}: {error}") from error
    return records


def write_jsonl(path: str | Path, records: Iterable[BaseModel]) -> None:
    """Write stable, UTF-8 JSONL without ASCII escaping."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")


def load_documents(path: str | Path) -> list[Document]:
    return load_jsonl(path, Document)


def load_chunks(path: str | Path) -> list[Chunk]:
    return load_jsonl(path, Chunk)
