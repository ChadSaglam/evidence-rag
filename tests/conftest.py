from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from evidence_rag.schemas import Chunk, Source


def make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=chunk_id.split(":", maxsplit=1)[0],
        text=text,
        start_char=0,
        end_char=len(text),
        source=Source(
            uri=f"https://example.test/{chunk_id}",
            title=f"Fixture {chunk_id}",
            language="en",
        ),
    )


class MappingEncoder:
    """Fixed vectors make dense tests independent of models, networks, and hardware."""

    def __init__(self, vectors: dict[str, Sequence[float]]) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def encode(self, texts: Sequence[str], *, normalize_embeddings: bool = True) -> Any:
        assert normalize_embeddings is True
        self.calls.append(list(texts))
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float64)


@pytest.fixture
def lexical_chunks() -> list[Chunk]:
    return [
        make_chunk("doc-b:0", "alpha alpha beta"),
        make_chunk("doc-a:0", "alpha beta gamma gamma"),
        make_chunk("doc-c:0", "unrelated words"),
    ]
