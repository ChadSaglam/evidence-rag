"""Deterministic query decomposition and subquery result fusion."""

from evidence_rag.decomposition.heuristic import HeuristicQueryDecomposer
from evidence_rag.decomposition.pipeline import DecomposedRetriever

__all__ = ["DecomposedRetriever", "HeuristicQueryDecomposer"]
