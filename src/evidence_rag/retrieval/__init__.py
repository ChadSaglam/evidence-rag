"""Lexical and multilingual dense retrieval baselines."""

from evidence_rag.retrieval.bm25 import BM25Retriever
from evidence_rag.retrieval.dense import DenseRetriever, E5Encoder

__all__ = ["BM25Retriever", "DenseRetriever", "E5Encoder"]
