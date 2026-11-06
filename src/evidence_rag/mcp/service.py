"""Allowlisted read-only operations shared by MCP and deterministic contract tests."""

from collections.abc import Sequence
from typing import Any

from evidence_rag.citation import ClaimLevelCitationVerifier
from evidence_rag.retrieval.bm25 import BM25Retriever
from evidence_rag.schemas import (
    Chunk,
    DecomposedResponse,
    McpChunkInput,
    McpCitationVerificationInput,
    McpSearchInput,
    Query,
    RerankedResponse,
    RetrievalResponse,
)

READ_ONLY_ANNOTATIONS = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}


class ReadOnlyEvidenceService:
    """Expose retrieval and verification without mutation, generation, or I/O tools."""

    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = tuple(chunks)
        self.retriever = BM25Retriever(self.chunks)
        self.chunks_by_id = {chunk.chunk_id: chunk for chunk in self.chunks}

    @staticmethod
    def tool_contracts() -> list[dict[str, Any]]:
        specifications = [
            (
                "search_evidence",
                "Search frozen evidence with deterministic BM25 and return attributed hits.",
                McpSearchInput,
            ),
            (
                "get_evidence_chunk",
                "Return one exact source-attributed chunk by its stable chunk ID.",
                McpChunkInput,
            ),
            (
                "verify_citations",
                "Verify answer claims against an explicit retrieval response using lexical checks.",
                McpCitationVerificationInput,
            ),
        ]
        return [
            {
                "name": name,
                "description": description,
                "inputSchema": model.model_json_schema(),
                "annotations": dict(READ_ONLY_ANNOTATIONS),
            }
            for name, description, model in specifications
        ]

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "search_evidence":
            request = McpSearchInput.model_validate(arguments)
            query = Query(query_id="mcp", text=request.text, top_k=request.top_k)
            response = RetrievalResponse(query=query, hits=self.retriever.search(query))
            return response.model_dump(mode="json")
        if name == "get_evidence_chunk":
            request = McpChunkInput.model_validate(arguments)
            try:
                chunk = self.chunks_by_id[request.chunk_id]
            except KeyError as error:
                raise ValueError(f"Unknown chunk_id: {request.chunk_id}") from error
            return chunk.model_dump(mode="json")
        if name == "verify_citations":
            request = McpCitationVerificationInput.model_validate(arguments)
            response = self._parse_response(request.response)
            report = ClaimLevelCitationVerifier(
                support_threshold=request.support_threshold
            ).verify(request.answer, response.hits)
            return report.model_dump(mode="json")
        raise ValueError(f"Unknown read-only tool: {name}")

    @staticmethod
    def _parse_response(payload: dict[str, Any]):
        if "reranker_model" in payload:
            return RerankedResponse.model_validate(payload)
        if "decomposition" in payload:
            return DecomposedResponse.model_validate(payload)
        return RetrievalResponse.model_validate(payload)
