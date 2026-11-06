"""Create the two-annotator GDPR batch-03 package with pooled, rank-blind candidates.

Candidates are the union of the top five chunks from two different systems (BM25 and
multilingual-e5-base dense retrieval), the standard pooling approach for building
relevance judgments without labelling the full corpus. Pool order is discarded:
candidates are written in ascending chunk_id order so annotators see no rank signal.

Usage:
    python scripts/build_annotation_batch_03.py
"""

from __future__ import annotations

import json
from pathlib import Path

from evidence_rag.annotation.batch import create_annotation_batch, write_annotation_batch
from evidence_rag.corpus import load_jsonl
from evidence_rag.retrieval.bm25 import BM25Retriever
from evidence_rag.retrieval.dense import DenseRetriever, E5Encoder
from evidence_rag.schemas import AnnotationBatch, Chunk, Query

ROOT = Path(__file__).resolve().parents[1]
CHUNKS = ROOT / "data/real-gdpr/chunks.jsonl"
OUTPUT = ROOT / "data/annotation-batch-03"
QUERY_IDS = [
    "en-breach-deadline",
    "en-dpia",
    "de-erasure",
    "de-access",
    "fr-portability",
    "fr-security",
    "de-accounting-retention",
]
POOL_DEPTH = 5
DENSE_MODEL = "intfloat/multilingual-e5-base"
DENSE_REVISION = "d128750597153bb5987e10b1c3493a34e5a4502a"


def main() -> None:
    chunks = load_jsonl(CHUNKS, Chunk)
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    queries = {
        item["query_id"]: Query.model_validate(item)
        for item in json.loads((ROOT / "data/real-gdpr/queries.json").read_text("utf-8"))
    }
    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks, E5Encoder(DENSE_MODEL, revision=DENSE_REVISION))

    tasks = []
    pool_log = {}
    for query_id in QUERY_IDS:
        query = queries[query_id]
        pooled = {hit.chunk.chunk_id for hit in bm25.search(query, top_k=POOL_DEPTH)}
        pooled |= {hit.chunk.chunk_id for hit in dense.search(query, top_k=POOL_DEPTH)}
        pool_log[query_id] = sorted(pooled)
        single = create_annotation_batch(
            queries=[query],
            chunks=[by_id[chunk_id] for chunk_id in pooled],
            chunks_path=CHUNKS,
            batch_id="batch-03",
            name="gdpr-two-annotator-batch-03",
            description="pending",
            split="test",
            corpus_name="gdpr-selected-articles-multilingual-v1",
        )
        tasks.extend(single.tasks)

    pairs = sum(len(task.candidates) for task in tasks)
    template = create_annotation_batch(
        queries=[queries[QUERY_IDS[0]]],
        chunks=chunks[:1],
        chunks_path=CHUNKS,
        batch_id="batch-03",
        name="gdpr-two-annotator-batch-03",
        description="pending",
        split="test",
        corpus_name="gdpr-selected-articles-multilingual-v1",
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for role in ("a", "b"):
        batch = AnnotationBatch.model_validate(
            {
                **template.model_dump(),
                "description": (
                    f"Independent annotator {role.upper()} copy. {len(tasks)} queries, "
                    f"{pairs} pooled query-chunk pairs (BM25 top-{POOL_DEPTH} union "
                    f"{DENSE_MODEL} top-{POOL_DEPTH}). Do not look at the other copy."
                ),
                "tasks": [task.model_dump() for task in tasks],
            }
        )
        write_annotation_batch(batch, OUTPUT / f"annotations-annotator-{role}.json")
    (OUTPUT / "pool.json").write_text(
        json.dumps(
            {
                "pool_depth": POOL_DEPTH,
                "systems": ["bm25", f"{DENSE_MODEL}@{DENSE_REVISION}"],
                "pairs": pairs,
                "candidates": pool_log,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"queries": len(tasks), "pairs": pairs}))


if __name__ == "__main__":
    main()
