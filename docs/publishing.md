# Publishing to Hugging Face and Kaggle

Nothing in this repository uploads automatically. These steps publish the GDPR dataset, the demo Space and the Kaggle benchmark under your own accounts. Run each command from the repository root with the virtual environment active. Replace `HF_USERNAME` and `KAGGLE_USERNAME` with your usernames.

## Before you publish

- GDPR text: © European Union, https://eur-lex.europa.eu/. Reuse is authorised with attribution under Commission Decision 2011/833/EU, and the Commission's standard reuse licence is CC BY 4.0. The dataset card already carries that attribution. Only the Official Journal editions are authentic.
- Labels are article-level source alignment, not human relevance judgments. The card and reports say so; keep that wording.
- Pick private or public deliberately. The commands below create **private** repos. Make them public on the website once you have checked them.

## Hugging Face

1. Install the tools: `pip install -U huggingface_hub`
2. Log in with a write token from https://huggingface.co/settings/tokens: `hf auth login`
3. Re-export the dataset with your username in the card:

   ```bash
   evidence-rag hf-export --documents data/real-gdpr/documents.jsonl --chunks data/real-gdpr/chunks.jsonl --gold-set data/real-gdpr/gold_set.json --output-dir artifacts/huggingface-dataset-gdpr --dataset-id HF_USERNAME/evidence-rag-gdpr --name "Evidence RAG: multilingual GDPR retrieval benchmark" --description "112 source-attributed chunks from 12 GDPR articles in English, German and French, with 30 queries (3 deliberately unanswerable) and graded article-level relevance labels." --license cc-by-4.0 --attribution "Legal text: Regulation (EU) 2016/679 (GDPR), official EN/DE/FR versions from EUR-Lex. © European Union, https://eur-lex.europa.eu/, 1998-2026. Reuse authorised under Commission Decision 2011/833/EU with attribution. Selected articles extracted, whitespace normalized, text chunked. Only the Official Journal editions are authentic. Labels are article-level source alignment, not human judgments. Not legal advice."
   ```

4. Check the hashes: `evidence-rag hf-verify --manifest artifacts/huggingface-dataset-gdpr/dataset-manifest.json`
5. Create the dataset repo: `hf repos create HF_USERNAME/evidence-rag-gdpr --repo-type dataset --private`
6. Upload it: `hf upload HF_USERNAME/evidence-rag-gdpr artifacts/huggingface-dataset-gdpr . --repo-type dataset`
7. Build the Space folder: `python scripts/build_space.py`
8. Create the Space: `hf repos create HF_USERNAME/evidence-rag --repo-type space --space-sdk gradio --private`
9. Upload it: `hf upload HF_USERNAME/evidence-rag artifacts/hf-space . --repo-type space`

The Space builds within a few minutes. It needs no API key and downloads no model.

## Kaggle

1. Install the tool: `pip install -U kaggle`
2. On https://www.kaggle.com/settings, under API, click **Create New Token**. Move the downloaded `kaggle.json` to `~/.kaggle/kaggle.json`.
3. Re-export with your username:

   ```bash
   evidence-rag kaggle-export --documents data/real-gdpr/documents.jsonl --chunks data/real-gdpr/chunks.jsonl --gold-set data/real-gdpr/gold_set.json --output-dir artifacts/kaggle-dataset-gdpr --dataset-id KAGGLE_USERNAME/evidence-rag-gdpr --name "Evidence RAG multilingual GDPR benchmark" --k 10 --license CC-BY-4.0
   ```

4. Create the dataset (private by default): `kaggle datasets create -p artifacts/kaggle-dataset-gdpr`
5. On Kaggle, create a notebook, attach the dataset, and upload `kaggle/benchmark.ipynb`. Set `DATASET_DIR` in its first cell to `/kaggle/input/evidence-rag-gdpr`.

## MCP Inspector (optional, external client)

The official MCP Inspector is a TypeScript client and does not share code with the Python SDK:

```bash
npx @modelcontextprotocol/inspector --cli .venv/bin/evidence-rag mcp-serve --chunks data/real-gdpr/chunks.jsonl --method tools/list
```
