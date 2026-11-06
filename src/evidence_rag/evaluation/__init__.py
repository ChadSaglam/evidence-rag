"""Gold-set validation, deterministic retrieval metrics, and selective answering."""

from evidence_rag.evaluation.calibration import (
    calibrate_selective_answerer,
    evaluate_selective_policy,
)
from evidence_rag.evaluation.gold import load_gold_set, summarize_gold_set
from evidence_rag.evaluation.metrics import evaluate_ranking
from evidence_rag.evaluation.policy import load_frozen_selective_policy
from evidence_rag.evaluation.runner import evaluate_retriever
from evidence_rag.evaluation.selective import SelectiveAnswerer

__all__ = [
    "SelectiveAnswerer",
    "calibrate_selective_answerer",
    "evaluate_ranking",
    "evaluate_retriever",
    "evaluate_selective_policy",
    "load_frozen_selective_policy",
    "load_gold_set",
    "summarize_gold_set",
]
