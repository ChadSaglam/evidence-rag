"""Semantic claim verification with a multilingual natural-language-inference model.

The lexical verifier answers "do the cited chunks share the claim's words?". That
accepts negated or altered claims that reuse the evidence vocabulary, and rejects
faithful paraphrases and cross-lingual citations. This verifier keeps the same
citation-resolution contract and numeric guard, then asks an NLI model whether the
cited evidence entails or contradicts each claim.

NLI entailment is still a model judgment, not a proof of legal or factual truth.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from evidence_rag.citation.extractor import CitationClaimExtractor
from evidence_rag.citation.verifier import NUMBER_PATTERN, ClaimLevelCitationVerifier
from evidence_rag.schemas import (
    Chunk,
    CitationVerificationReport,
    ClaimVerification,
    RerankedHit,
    RetrievalHit,
)

DEFAULT_NLI_MODEL = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"
SUPPORTED_STATUSES = frozenset({"lexically_supported", "semantically_supported"})
SEGMENT_PATTERN = re.compile(r"(?<=[.;:!?])\s+")
_LABEL_ORDER = ("entailment", "neutral", "contradiction")


class NLIModel(Protocol):
    """Return one ``[entailment, neutral, contradiction]`` probability row per pair."""

    def predict(self, pairs: Sequence[tuple[str, str]]) -> Any: ...


class TransformersNLIModel:
    """Lazy Hugging Face adapter that normalizes label order from the model config."""

    def __init__(
        self,
        model_name: str = DEFAULT_NLI_MODEL,
        *,
        device: str | None = None,
        revision: str | None = None,
        batch_size: int = 16,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "NLI verification requires the optional dependency: pip install -e '.[nli]'"
            ) from error
        self._torch = torch
        self.model_name = model_name
        self.model_revision = revision
        self.batch_size = batch_size
        self.device = device or "cpu"
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name, revision=revision
        ).to(self.device)
        self._model.eval()
        labels = {
            str(label).lower(): int(index)
            for index, label in self._model.config.id2label.items()
        }
        missing = [label for label in _LABEL_ORDER if label not in labels]
        if missing:
            raise ValueError(f"NLI model {model_name} lacks labels: {missing}")
        self._columns = [labels[label] for label in _LABEL_ORDER]

    def predict(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        rows = []
        with self._torch.no_grad():
            for start in range(0, len(pairs), self.batch_size):
                batch = pairs[start : start + self.batch_size]
                encoded = self._tokenizer(
                    [premise for premise, _ in batch],
                    [hypothesis for _, hypothesis in batch],
                    truncation="only_first",
                    max_length=512,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                probabilities = self._model(**encoded).logits.softmax(dim=-1)
                rows.append(probabilities[:, self._columns].cpu().numpy())
        return np.concatenate(rows, axis=0)


def evidence_windows(text: str, size: int) -> list[str]:
    """Overlapping sentence/clause windows, so long legal chunks fit the model."""

    segments = [segment for segment in SEGMENT_PATTERN.split(text.strip()) if segment]
    if len(segments) <= size:
        return [text.strip()]
    return [" ".join(segments[i : i + size]) for i in range(len(segments) - size + 1)]


class NLICitationVerifier:
    """Claim-level citation verification backed by NLI entailment and contradiction."""

    verifier_name = "nli-entailment-v1"

    def __init__(
        self,
        model: NLIModel | None = None,
        *,
        model_name: str = DEFAULT_NLI_MODEL,
        device: str | None = None,
        entailment_threshold: float = 0.5,
        contradiction_threshold: float = 0.5,
        window_size: int = 3,
        extractor: CitationClaimExtractor | None = None,
    ) -> None:
        for name, value in (
            ("entailment_threshold", entailment_threshold),
            ("contradiction_threshold", contradiction_threshold),
        ):
            if not 0 < value <= 1:
                raise ValueError(f"{name} must be greater than zero and at most one")
        if window_size < 1:
            raise ValueError("window_size must be at least one")
        self.model = model or TransformersNLIModel(model_name, device=device)
        self.model_name = getattr(self.model, "model_name", model_name)
        self.entailment_threshold = float(entailment_threshold)
        self.contradiction_threshold = float(contradiction_threshold)
        self.window_size = window_size
        self.extractor = extractor or CitationClaimExtractor()
        # Reuse the lexical verifier for citation resolution and the uncited/invalid paths.
        self._lexical = ClaimLevelCitationVerifier(extractor=self.extractor)

    def verify(
        self, answer: str, hits: Sequence[RetrievalHit | RerankedHit]
    ) -> CitationVerificationReport:
        evidence = ClaimLevelCitationVerifier._evidence_by_rank(hits)
        claims = self.extractor.extract(answer)
        verifications = [self._verify_claim(claim, evidence) for claim in claims]
        return CitationVerificationReport(
            answer=answer,
            verifier=f"{self.verifier_name}:{self.model_name}",
            support_threshold=self.entailment_threshold,
            fully_supported=bool(verifications)
            and all(item.status in SUPPORTED_STATUSES for item in verifications),
            claims=verifications,
        )

    def _verify_claim(self, claim, evidence: dict[int, Chunk]) -> ClaimVerification:
        lexical = self._lexical._verify_claim(claim, evidence)
        if lexical.status in {"uncited", "invalid_citation"}:
            return lexical

        chunks = [evidence[rank] for rank in claim.citation_ranks]
        windows = [
            window for chunk in chunks for window in evidence_windows(chunk.text, self.window_size)
        ]
        scores = np.asarray(
            self.model.predict([(window, claim.text) for window in windows]), dtype=np.float64
        )
        if scores.shape != (len(windows), 3) or not np.all(np.isfinite(scores)):
            raise ValueError("NLI model must return finite [entail, neutral, contradict] rows")
        entailment = float(scores[:, 0].max())
        contradiction = float(scores[:, 2].max())

        claim_numbers = set(NUMBER_PATTERN.findall(claim.text))
        evidence_numbers = set().union(
            *(set(NUMBER_PATTERN.findall(chunk.text)) for chunk in chunks)
        )
        missing_numbers = sorted(claim_numbers - evidence_numbers)

        if contradiction >= self.contradiction_threshold and contradiction > entailment:
            status = "contradicted"
            reason = f"Cited evidence contradicts the claim (p={contradiction:.2f})."
        elif missing_numbers:
            status = "insufficient_support"
            reason = f"Claim numbers absent from cited evidence: {missing_numbers}."
        elif entailment >= self.entailment_threshold:
            status = "semantically_supported"
            reason = f"Cited evidence entails the claim (p={entailment:.2f})."
        else:
            status = "insufficient_support"
            reason = (
                f"Entailment {entailment:.2f} is below threshold "
                f"{self.entailment_threshold:.2f}."
            )
        return ClaimVerification(
            claim=claim,
            status=status,
            support_score=entailment,
            entailment_score=entailment,
            contradiction_score=contradiction,
            cited_chunk_ids=[chunk.chunk_id for chunk in chunks],
            matched_tokens=lexical.matched_tokens,
            missing_tokens=lexical.missing_tokens,
            reason=reason,
        )
