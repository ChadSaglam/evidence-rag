# Kaggle benchmark

The Kaggle workflow exports a self-contained dataset, verifies every input with SHA-256, and produces deterministic JSON and Markdown reports. It runs the dependency-light BM25 baseline offline; no retrieval model, GPU, timestamp, or machine-specific path is required.

The bundled gold set is synthetic. Its metrics are deterministic software fixtures, not human benchmark results or evidence of general retrieval quality.

## Export a dataset

Create the chunk artifact first, then export the corpus, labels, package source, and Kaggle metadata:

```bash
evidence-rag chunk \
  --documents data/sample/documents.jsonl \
  --output artifacts/chunks.jsonl

evidence-rag kaggle-export \
  --documents data/sample/documents.jsonl \
  --chunks artifacts/chunks.jsonl \
  --gold-set data/sample/gold_set.json \
  --output-dir artifacts/kaggle-dataset \
  --dataset-id YOUR_KAGGLE_USERNAME/evidence-rag-benchmark \
  --name "Evidence RAG deterministic benchmark" \
  --k 4
```

Replace the placeholder dataset ID before publishing. Export only writes local files; it does not use Kaggle credentials or upload anything.

The output contains:

- `documents.jsonl`, `chunks.jsonl`, and `gold_set.json` under canonical names.
- `evidence_rag/` package source for offline notebook execution.
- `benchmark-manifest.json` with the evaluation configuration, label provenance, relative paths, file sizes, and SHA-256 hashes.
- `dataset-metadata.json` with the supplied dataset ID and MIT license declaration.

Use an empty output directory for a clean publication artifact. Export overwrites known files but does not delete unrelated stale files.

## Run the deterministic report

```bash
evidence-rag benchmark-run \
  --manifest artifacts/kaggle-dataset/benchmark-manifest.json \
  --output-dir artifacts/kaggle-results
```

The runner verifies all manifest hashes before retrieval. A missing, modified, or truncated input stops the run. It writes `benchmark-report.json` and `benchmark-report.md`; omit `--output-dir` to print the JSON report to stdout.

The report includes Hit@k, Recall@k, Precision@k, reciprocal rank, nDCG@k, per-query results, skipped unanswerable queries, label provenance, and content hashes. The same manifest and file bytes produce byte-identical reports.

## Run on Kaggle

Upload the exported directory as a Kaggle dataset. Add `kaggle/benchmark.ipynb` to a notebook and set `DATASET_DIR` to the mounted dataset path. The notebook adds the exported package source to `sys.path`, runs the manifest, and writes reports under `/kaggle/working/benchmark-results`.

The notebook expects NumPy and Pydantic in the Kaggle Python environment. Internet access and model downloads are not needed for the BM25 run. Publishing and Kaggle API credentials are intentionally outside the deterministic test path.
