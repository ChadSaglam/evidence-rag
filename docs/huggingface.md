# Hugging Face dataset and demo

The Hugging Face workflow provides a deterministic dataset export and a read-only Gradio retrieval demo. Neither command publishes data, creates a repository, nor uses account credentials.

## Dataset export

```bash
evidence-rag hf-export \
  --documents data/sample/documents.jsonl \
  --chunks artifacts/chunks.jsonl \
  --gold-set data/sample/gold_set.json \
  --output-dir artifacts/huggingface-dataset \
  --dataset-id YOUR_HF_USERNAME/evidence-rag

evidence-rag hf-verify \
  --manifest artifacts/huggingface-dataset/dataset-manifest.json
```

Replace the placeholder dataset ID before publishing. Export validates the input records and writes:

- `data/documents.jsonl`, `data/chunks.jsonl`, and `data/gold_set.json`.
- `README.md` with Hugging Face dataset metadata and `documents`, `chunks`, and `gold_set` configs.
- `dataset-manifest.json` with relative paths, file sizes, SHA-256 hashes, and label provenance.

The same inputs and export arguments produce byte-identical files. Verification stops on missing or modified content. Use an empty output directory for publication because export does not remove unrelated stale files.

The bundled labels are synthetic software fixtures, not human relevance judgments. Dataset cards preserve `human`, `synthetic`, or `benchmark` provenance instead of upgrading it.

## Read-only demo

Install the optional dependency and start the local preview:

```bash
pip install -e ".[demo]"
npm run dev -- --host 127.0.0.1 --port 7100
```

The demo runs BM25 over the bundled synthetic documents and returns a `RetrievalResponse` JSON object with ranks, scores, source attribution, chunk text, and exact offsets. It does not generate answers, call external models, or mutate the corpus. Set `EVIDENCE_RAG_CHUNKS` to search another validated chunk JSONL file.

`demo/app.py` is the Gradio entry point. `demo/README.md` contains Space metadata, and `demo/requirements.txt` lists runtime dependencies. The dependency-light `DemoSearch` service is tested without starting a server or downloading a model.

## Space layout

For a Hugging Face Space, place the contents of `demo/` at the Space root and include the repository's `src/` and `data/sample/` directories. Keep `app_file: app.py`. The application inserts `src/` into `sys.path` and builds the sample index at startup.

Publishing datasets or Spaces remains a separate, explicit account operation.
