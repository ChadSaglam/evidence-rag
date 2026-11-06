# Claim-level citation verification

The citation verifier checks whether answer sentences resolve to retrieved evidence and whether the cited text provides sufficient lexical coverage. It does not establish semantic entailment or real-world truth.

## Workflow

1. Split the answer into sentence-level or line-level claims.
2. Extract numeric citations such as `[1]`, `[1][2]`, and `[1, 2]`.
3. Resolve each rank against the displayed retrieval or reranking response.
4. Compute unique content-token coverage over the union of cited chunks.
5. Require every Arabic numeral in the claim to occur in the cited evidence.
6. Emit a status, score, matched tokens, missing tokens, chunk IDs, and reason for every claim.

## Statuses

| Status | Meaning |
|---|---|
| `lexically_supported` | Citations resolve, numeric checks pass, and coverage meets the threshold |
| `insufficient_support` | Citations resolve, but coverage or numeric checks fail |
| `uncited` | The claim has no numeric citation |
| `invalid_citation` | At least one citation rank is absent from the response |

`fully_supported` is true only when the answer contains at least one claim and every claim is `lexically_supported`. The report always names the verifier as `lexical-overlap-v1` and records its threshold.

## Limits

Lexical overlap does not prove entailment. Paraphrases and translations can be rejected even when a human would accept them, while overlapping but contradictory text can receive a high score. The numeric guard uses digits and does not normalize number words. Sentence splitting is deterministic and is not a linguistic sentence tokenizer. A future semantic verifier must use a separate model name and evaluation report rather than silently changing these semantics.

## CLI

Save any retrieval response and verify an answer against it:

```bash
evidence-rag search \
  --chunks artifacts/chunks.jsonl \
  --query "accounting records" \
  --top-k 2 > artifacts/retrieval.json

evidence-rag verify \
  --response artifacts/retrieval.json \
  --answer "Accounting records must be retained for ten years [1]."
```

Use `--support-threshold` to select a value greater than zero and at most one. The default is `0.6`.


## NLI verification

`NLICitationVerifier` (`evidence_rag.citation.nli`) keeps the lexical contract for `uncited` and `invalid_citation` claims and keeps the numeric guard. For a resolved claim, it splits each cited chunk into overlapping windows of three sentences or clauses, so long legal chunks fit the model's 512-token limit. It then scores every window against the claim with a multilingual NLI model (default `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli`) and applies these rules in order:

1. `contradicted`: the maximum contradiction probability is at least 0.5 and exceeds the maximum entailment probability.
2. `insufficient_support`: a number in the claim is absent from the cited evidence.
3. `semantically_supported`: the maximum entailment probability is at least 0.5.
4. `insufficient_support`: otherwise.

Each `ClaimVerification` records `entailment_score` and `contradiction_score`. Thresholds are defaults and have not been calibrated; calibrate them on a labeled development set before relying on them.

`data/citation-nli/cases.json` holds 38 constructed claims over official GDPR chunks: restatements, paraphrases, negations, altered numbers, unrelated claims, and cross-lingual citations. A single curator assigned the labels while constructing the claims. They have not been independently annotated. Results are in `results/nli-benchmark/report.md`.

Known limitation: a small NLI model rarely bridges paraphrases that need inference, such as "three days" for "72 hours". Entailment is a model judgment, not proof of legal or factual truth.
