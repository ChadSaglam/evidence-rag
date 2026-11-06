"""Run the read-only Evidence RAG demo locally or as a Hugging Face Space."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evidence_rag.chunking import chunk_documents  # noqa: E402
from evidence_rag.corpus import load_chunks, load_documents  # noqa: E402
from evidence_rag.huggingface import DemoSearch  # noqa: E402

GDPR_CHUNKS = ROOT / "data" / "real-gdpr" / "chunks.jsonl"
INTRO = """# Evidence RAG
Read-only multilingual BM25 retrieval over 12 official GDPR articles in English, German and
French (112 source-attributed chunks). It returns evidence with exact character offsets and
checks citations. It never generates answers and is not legal advice.
"""


def build_interface(service: DemoSearch):
    import gradio as gr

    with gr.Blocks(title="Evidence RAG") as interface:
        gr.Markdown(INTRO)
        with gr.Tab("Search evidence"):
            query = gr.Textbox(
                label="Query",
                placeholder="How quickly must a controller notify a personal data breach?",
                lines=2,
            )
            top_k = gr.Slider(1, 10, value=5, step=1, label="Results")
            search = gr.Button("Search", variant="primary")
            results = gr.Markdown()
            with gr.Accordion("Raw retrieval response (JSON)", open=False):
                raw = gr.Code(language="json", interactive=False)
            for trigger in (search.click, query.submit):
                trigger(service.search_markdown, inputs=[query, top_k], outputs=results)
                trigger(service.search_json, inputs=[query, top_k], outputs=raw)
            gr.Examples(
                examples=[
                    ["How quickly must a controller notify a personal data breach?", 5],
                    ["Wann hat eine betroffene Person das Recht auf Löschung?", 5],
                    ["Qu'est-ce que le droit à la portabilité des données ?", 5],
                    ["What fixed accounting-record retention period does the GDPR prescribe?", 5],
                ],
                inputs=[query, top_k],
            )
        with gr.Tab("Check citations"):
            gr.Markdown(
                "Write an answer with `[n]` citations. Each `n` refers to rank *n* in the "
                "results for the query."
            )
            verify_query = gr.Textbox(
                label="Query",
                value="How quickly must a controller notify a personal data breach?",
            )
            answer = gr.Textbox(
                label="Answer",
                lines=3,
                value=(
                    "The controller must notify the supervisory authority not later than "
                    "72 hours after having become aware of it [1]. "
                    "The controller must notify within 24 hours [1]."
                ),
            )
            verify_k = gr.Slider(1, 10, value=5, step=1, label="Evidence depth")
            verify = gr.Button("Verify", variant="primary")
            report = gr.Markdown()
            verify.click(
                service.verify_markdown, inputs=[verify_query, answer, verify_k], outputs=report
            )
    return interface


def load_demo_chunks():
    configured = os.environ.get("EVIDENCE_RAG_CHUNKS")
    if configured:
        return load_chunks(configured)
    if GDPR_CHUNKS.exists():
        return load_chunks(GDPR_CHUNKS)
    documents = load_documents(ROOT / "data" / "sample" / "documents.jsonl")
    return chunk_documents(documents)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evidence RAG read-only demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7100)
    args = parser.parse_args()
    interface = build_interface(DemoSearch(load_demo_chunks()))
    interface.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        show_error=False,
        quiet=True,
    )


if __name__ == "__main__":
    main()
