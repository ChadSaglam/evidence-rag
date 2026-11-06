# RAG security tests

The security module treats retrieved document text as untrusted data. It provides two separate evaluations: heuristic prompt-injection screening and poisoned-document retrieval exposure. Neither proves that a corpus or answering system is safe.

## Prompt-injection screening

`InjectionDetector` matches documented English, German, French, and Turkish instruction-override patterns, system-prompt exfiltration, secret exfiltration, role overrides, and fake system tags. Assessments contain a SHA-256 text digest, rule IDs, and character offsets rather than reproducing attack text.

`screen_chunks` returns the original unflagged chunks plus a quarantine report. It never edits or deletes the source corpus. Safe chunk IDs mean only that no configured heuristic rule matched.

```bash
evidence-rag security-scan --chunks artifacts/chunks.jsonl
```

## Poisoned-document retrieval experiments

Each `PoisonedDocumentCase` contains a query, known target chunk, known poisoned chunk, and a bounded synthetic corpus. The runner measures:

- **Top-1 attack success:** the poisoned chunk is ranked first.
- **Top-k exposure:** the poisoned chunk appears within the query's original top-k cutoff.
- **Outranking:** the poisoned chunk ranks above the target.
- **Quarantine:** which chunks screening removed before retrieval.

```bash
evidence-rag security-test --cases data/security/poisoned_cases.json
evidence-rag security-test --cases data/security/poisoned_cases.json --screen
```

The bundled case uses keyword stuffing and explicit injection. The unscreened BM25 run demonstrates exposure; the screened run demonstrates removal of that specific attack.

## Limits

Pattern matching cannot detect all injections, obfuscation, misinformation, or subtle poisoning. A dedicated test includes a false numeric claim without injection phrases and remains after screening. Retrieval exposure is not equivalent to answer-generation compromise; an end-to-end model test must separately measure whether the answering model follows the poisoned instruction. Never run document-supplied commands or tools during these tests.
