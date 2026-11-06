"""Dependency-light, read-only search service used by the Hugging Face demo."""

import html
import json
from collections.abc import Sequence

from evidence_rag.citation import ClaimLevelCitationVerifier
from evidence_rag.retrieval.bm25 import BM25Retriever
from evidence_rag.schemas import Chunk, Query, RetrievalResponse


class DemoSearch:
    """Expose deterministic BM25 retrieval without generation or corpus mutation."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.retriever = BM25Retriever(chunks)

    def search(self, text: str, top_k: int = 5) -> RetrievalResponse:
        if not text.strip():
            raise ValueError("Enter a non-empty search query")
        query = Query(query_id="demo", text=text, top_k=int(top_k))
        return RetrievalResponse(query=query, hits=self.retriever.search(query))

    def search_json(self, text: str, top_k: int = 5) -> str:
        response = self.search(text, top_k)
        return json.dumps(response.model_dump(mode="json"), ensure_ascii=False, indent=2)

    def search_markdown(self, text: str, top_k: int = 5) -> str:
        """Readable ranked evidence; document text is escaped because it is untrusted."""

        response = self.search(text, top_k)
        if not response.hits:
            return "_No chunk shares a term with this query. BM25 abstains instead of guessing._"
        blocks = []
        for hit in response.hits:
            chunk = hit.chunk
            excerpt = html.escape(chunk.text[:700]) + ("…" if len(chunk.text) > 700 else "")
            blocks.append(
                f"**[{hit.rank}]** {html.escape(chunk.source.title)} "
                f"({chunk.source.language}) · BM25 {hit.score:.2f} · "
                f"`{chunk.chunk_id}` chars {chunk.start_char}–{chunk.end_char}\n\n"
                f"> {excerpt}\n\n[source]({chunk.source.uri})"
            )
        return "\n\n---\n\n".join(blocks)

    def verify_markdown(self, text: str, answer: str, top_k: int = 5) -> str:
        """Check an answer's [n] citations against the ranks returned for the query."""

        if not answer.strip():
            raise ValueError("Enter an answer with [n] citations")
        response = self.search(text, top_k)
        report = ClaimLevelCitationVerifier().verify(answer, response.hits)
        rows = ["| Claim | Status | Score | Cited chunks |", "|---|---|---|---|"]
        for item in report.claims:
            score = "–" if item.support_score is None else f"{item.support_score:.2f}"
            claim = html.escape(item.claim.text).replace("|", "\\|")
            cited = ", ".join(item.cited_chunk_ids) or "–"
            rows.append(f"| {claim} | `{item.status}` | {score} | {cited} |")
        verdict = "Fully supported" if report.fully_supported else "Not fully supported"
        return (
            f"**{verdict}** (lexical overlap + numeric guard; not semantic entailment).\n\n"
            + "\n".join(rows)
        )
