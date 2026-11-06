"""Dependency-light Okapi BM25 with non-negative IDF and stable ties."""

from collections import Counter
from collections.abc import Sequence

import numpy as np

from evidence_rag.retrieval.base import rank_hits, resolve_query, validate_chunks
from evidence_rag.schemas import Chunk, Query, RetrievalHit
from evidence_rag.tokenization import tokenize


class BM25Retriever:
    """An in-memory BM25 baseline suitable for small research corpora."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.chunks = validate_chunks(chunks)
        if k1 <= 0:
            raise ValueError("k1 must be greater than zero")
        if not 0 <= b <= 1:
            raise ValueError("b must be between zero and one")
        self.k1 = float(k1)
        self.b = float(b)

        self._term_frequencies = [Counter(tokenize(chunk.text)) for chunk in self.chunks]
        self._lengths = np.asarray(
            [sum(frequencies.values()) for frequencies in self._term_frequencies],
            dtype=np.float64,
        )
        self._average_length = float(np.mean(self._lengths))
        document_count = len(self.chunks)
        document_frequencies = Counter(
            term for frequencies in self._term_frequencies for term in frequencies
        )
        self._idf = {
            term: float(np.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5)))
            for term, frequency in document_frequencies.items()
        }

    def get_scores(self, query: Query | str) -> np.ndarray:
        resolved = resolve_query(query)
        scores = np.zeros(len(self.chunks), dtype=np.float64)
        if self._average_length == 0:
            return scores
        for term in tokenize(resolved.text):
            idf = self._idf.get(term)
            if idf is None:
                continue
            for index, frequencies in enumerate(self._term_frequencies):
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                length_norm = 1 - self.b + self.b * self._lengths[index] / self._average_length
                denominator = frequency + self.k1 * length_norm
                scores[index] += idf * frequency * (self.k1 + 1) / denominator
        return scores

    def search(
        self,
        query: Query | str,
        top_k: int | None = None,
        *,
        include_zero: bool = False,
    ) -> list[RetrievalHit]:
        """Rank chunks by BM25.

        Chunks sharing no query term score exactly zero. Their relative order would
        only reflect chunk IDs, so they are omitted unless ``include_zero`` is set.
        """

        resolved = resolve_query(query, top_k)
        hits = rank_hits(resolved, self.chunks, self.get_scores(resolved), "bm25")
        if include_zero:
            return hits
        return [hit for hit in hits if hit.score > 0.0]
