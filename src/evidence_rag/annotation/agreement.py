"""Independent annotation comparison and explicit adjudication reporting."""

import json
from pathlib import Path

from pydantic import TypeAdapter

from evidence_rag.corpus import load_jsonl
from evidence_rag.schemas import (
    AdjudicationDecision,
    AnnotationAgreementReport,
    AnnotationDisagreement,
    RelevanceJudgment,
)

PairKey = tuple[str, str]


def load_independent_annotations(path: str | Path) -> list[RelevanceJudgment]:
    """Load judgments from JSONL, or from a completed annotation batch JSON file."""

    path = Path(path)
    if path.suffix == ".json":
        from evidence_rag.annotation.batch import load_annotation_batch

        batch = load_annotation_batch(path)
        judgments = []
        for task in batch.tasks:
            for candidate in task.candidates:
                if candidate.relevance is None or not candidate.annotator:
                    raise ValueError(
                        f"{path.name}: {task.query.query_id} / {candidate.chunk.chunk_id} "
                        "is not annotated yet"
                    )
                judgments.append(
                    RelevanceJudgment(
                        query_id=task.query.query_id,
                        chunk_id=candidate.chunk.chunk_id,
                        relevance=candidate.relevance,
                        annotator=candidate.annotator,
                        notes=candidate.notes,
                    )
                )
        return judgments
    return load_jsonl(path, RelevanceJudgment)


def load_adjudications(path: str | Path | None) -> dict[PairKey, AdjudicationDecision]:
    if path is None:
        return {}
    decisions = TypeAdapter(list[AdjudicationDecision]).validate_json(
        Path(path).read_text(encoding="utf-8")
    )
    keyed = {(item.query_id, item.chunk_id): item for item in decisions}
    if len(keyed) != len(decisions):
        raise ValueError("Adjudication decisions must have unique query-chunk pairs")
    return keyed


def _index(judgments: list[RelevanceJudgment], role: str) -> tuple[str, dict[PairKey, int]]:
    annotators = {item.annotator for item in judgments}
    if len(annotators) != 1:
        raise ValueError(f"{role} annotations must contain exactly one annotator")
    keyed = {(item.query_id, item.chunk_id): item.relevance for item in judgments}
    if len(keyed) != len(judgments):
        raise ValueError(f"{role} annotations contain duplicate query-chunk pairs")
    return annotators.pop(), keyed


def _kappa(
    primary: dict[PairKey, int],
    secondary: dict[PairKey, int],
    keys: list[PairKey],
    *,
    weighted: bool,
) -> float | None:
    labels = range(4)
    primary_counts = {label: 0 for label in labels}
    secondary_counts = {label: 0 for label in labels}
    observed = 0.0
    for key in keys:
        first = primary[key]
        second = secondary[key]
        primary_counts[first] += 1
        secondary_counts[second] += 1
        agreement = 1.0 - ((first - second) ** 2 / 9.0) if weighted else float(first == second)
        observed += agreement
    observed /= len(keys)
    expected = 0.0
    for first in labels:
        for second in labels:
            probability = primary_counts[first] * secondary_counts[second] / len(keys) ** 2
            agreement = 1.0 - ((first - second) ** 2 / 9.0) if weighted else float(first == second)
            expected += probability * agreement
    if expected == 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def compare_annotations(
    primary_judgments: list[RelevanceJudgment],
    secondary_judgments: list[RelevanceJudgment],
    *,
    adjudications: dict[PairKey, AdjudicationDecision] | None = None,
    provenance: str = "benchmark",
) -> AnnotationAgreementReport:
    """Compare identical pair sets without presenting benchmark labels as human work."""

    if provenance not in {"human", "benchmark", "synthetic"}:
        raise ValueError("Unsupported annotation provenance")
    primary_name, primary = _index(primary_judgments, "Primary")
    secondary_name, secondary = _index(secondary_judgments, "Secondary")
    if primary_name == secondary_name:
        raise ValueError("Agreement requires two distinct annotator identifiers")
    if not primary or primary.keys() != secondary.keys():
        raise ValueError("Both annotators must label the same non-empty query-chunk pairs")
    adjudications = adjudications or {}
    if not set(adjudications).issubset(primary.keys()):
        raise ValueError("Adjudication references an unknown query-chunk pair")

    keys = sorted(primary)
    disagreements = []
    for key in keys:
        if primary[key] == secondary[key]:
            continue
        decision = adjudications.get(key)
        disagreements.append(
            AnnotationDisagreement(
                query_id=key[0],
                chunk_id=key[1],
                primary_relevance=primary[key],
                secondary_relevance=secondary[key],
                adjudicated_relevance=decision.relevance if decision else None,
                adjudicator=decision.adjudicator if decision else None,
                notes=decision.notes if decision else None,
            )
        )
    return AnnotationAgreementReport(
        provenance=provenance,
        primary_annotator=primary_name,
        secondary_annotator=secondary_name,
        pair_count=len(keys),
        exact_agreement_rate=sum(primary[key] == secondary[key] for key in keys) / len(keys),
        cohens_kappa=_kappa(primary, secondary, keys, weighted=False),
        quadratic_weighted_kappa=_kappa(primary, secondary, keys, weighted=True),
        disagreement_count=len(disagreements),
        adjudicated_count=sum(item.adjudicated_relevance is not None for item in disagreements),
        complete=all(item.adjudicated_relevance is not None for item in disagreements),
        disagreements=disagreements,
    )


def agreement_summary(report: AnnotationAgreementReport) -> dict[str, object]:
    return json.loads(report.model_dump_json())
