"""Load any public retrieval response for citation verification."""

import json
from pathlib import Path

from evidence_rag.schemas import DecomposedResponse, RerankedResponse, RetrievalResponse

Response = RetrievalResponse | DecomposedResponse | RerankedResponse


def load_retrieval_response(path: str | Path) -> Response:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if "reranker_model" in payload:
        return RerankedResponse.model_validate(payload)
    if "decomposition" in payload:
        return DecomposedResponse.model_validate(payload)
    return RetrievalResponse.model_validate(payload)
