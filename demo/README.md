---
title: Evidence RAG
emoji: 🔎
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Multilingual GDPR evidence retrieval with citation checks
---

# Evidence RAG

Read-only multilingual BM25 retrieval over 12 official GDPR articles in English, German and French. The demo returns ranked evidence with source attribution and exact character offsets, and checks `[n]` citations in an answer. It does not generate answers and is not legal advice. GDPR text © European Union, https://eur-lex.europa.eu/, reused with attribution.

For a Space deployment, put this `README.md`, `app.py` and `requirements.txt` at the Space root, next to the repository's `src/` directory and the `data/real-gdpr/chunks.jsonl` and `data/sample/` files. The app adds `src/` to `sys.path`. It needs no model download and no API key.

Run locally from the repository root after installing the demo extra:

```bash
python demo/app.py --host 127.0.0.1 --port 7100
```

Set `EVIDENCE_RAG_CHUNKS` to a validated chunk JSONL file to search a different local corpus. The demo never writes to that corpus.
