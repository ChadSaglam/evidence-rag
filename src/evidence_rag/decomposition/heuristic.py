"""A conservative, deterministic multilingual query decomposition baseline."""

import re

from evidence_rag.retrieval.base import resolve_query
from evidence_rag.schemas import Query, QueryDecomposition, SubQuery
from evidence_rag.tokenization import tokenize

EXPLICIT_SEPARATOR = re.compile(r"\?+(?=\s*\S)|[;\n]+")
CLAUSE_CONJUNCTION = re.compile(r"\s+(?:and|und|et|ve)\s+", re.IGNORECASE)


class HeuristicQueryDecomposer:
    """Split explicit requests and substantial conjunction-linked clauses.

    The baseline supports English, German, French, and Turkish conjunctions. It
    deliberately avoids semantic claims and leaves short noun phrases intact.
    """

    strategy = "heuristic-v1"

    def __init__(self, *, max_subqueries: int = 4, min_clause_words: int = 3) -> None:
        if max_subqueries < 1:
            raise ValueError("max_subqueries must be greater than zero")
        if min_clause_words < 1:
            raise ValueError("min_clause_words must be greater than zero")
        self.max_subqueries = max_subqueries
        self.min_clause_words = min_clause_words

    def decompose(self, query: Query | str) -> QueryDecomposition:
        resolved = resolve_query(query)
        text = re.sub(r"[ \t]+", " ", resolved.text.strip())
        parts = [part.strip() for part in EXPLICIT_SEPARATOR.split(text) if part.strip()]
        if len(parts) == 1:
            conjunction_parts = [
                part.strip() for part in CLAUSE_CONJUNCTION.split(text) if part.strip()
            ]
            if len(conjunction_parts) > 1 and all(
                len(tokenize(part)) >= self.min_clause_words for part in conjunction_parts
            ):
                parts = conjunction_parts
        if not parts:
            parts = [text]
        if len(parts) > self.max_subqueries:
            parts = parts[: self.max_subqueries - 1] + [" ".join(parts[self.max_subqueries - 1 :])]

        is_question = "?" in text
        normalized_parts = []
        for part in parts:
            normalized = re.sub(r"\s+", " ", part).strip().rstrip("?").strip()
            normalized_parts.append(f"{normalized}?" if is_question else normalized)
        subqueries = [
            SubQuery(subquery_id=f"{resolved.query_id}:sub-{index:03d}", text=part)
            for index, part in enumerate(normalized_parts, start=1)
        ]
        return QueryDecomposition(query=resolved, strategy=self.strategy, subqueries=subqueries)
