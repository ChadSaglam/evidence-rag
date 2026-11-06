"""Command-line entry points for retrieval, verification, and evaluation."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from evidence_rag.annotation import (
    compare_annotations,
    create_annotation_batch,
    export_human_gold_set,
    load_adjudications,
    load_annotation_batch,
    load_annotation_queries,
    load_independent_annotations,
    summarize_annotation_batch,
    write_annotation_batch,
)
from evidence_rag.benchmark import (
    export_kaggle_dataset,
    run_benchmark,
    write_benchmark_report,
)
from evidence_rag.benchmark.kaggle import sha256_file, write_json
from evidence_rag.benchmark.runner import report_json
from evidence_rag.chunking import chunk_documents
from evidence_rag.citation import ClaimLevelCitationVerifier, load_retrieval_response
from evidence_rag.corpus import load_chunks, load_documents, write_jsonl
from evidence_rag.decomposition import DecomposedRetriever, HeuristicQueryDecomposer
from evidence_rag.evaluation import (
    SelectiveAnswerer,
    calibrate_selective_answerer,
    evaluate_retriever,
    evaluate_selective_policy,
    load_frozen_selective_policy,
    load_gold_set,
    summarize_gold_set,
)
from evidence_rag.huggingface import export_huggingface_dataset, verify_huggingface_dataset
from evidence_rag.mcp import ReadOnlyEvidenceService
from evidence_rag.mcp.server import run_stdio
from evidence_rag.reranking import (
    DEFAULT_RERANKER_MODEL,
    CrossEncoderReranker,
    RerankedRetriever,
)
from evidence_rag.retrieval.bm25 import BM25Retriever
from evidence_rag.retrieval.dense import DEFAULT_DENSE_MODEL, DenseRetriever, E5Encoder
from evidence_rag.retrieval.hybrid import HybridRetriever
from evidence_rag.schema_export import export_schemas
from evidence_rag.schemas import (
    CitationVerificationReport,
    DecomposedResponse,
    Query,
    RerankedResponse,
    RetrievalResponse,
)
from evidence_rag.security import (
    evaluate_poisoned_retrieval,
    load_poisoned_cases,
    screen_chunks,
)


def add_retrieval_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--method", choices=("bm25", "dense", "hybrid"), default="bm25")
    parser.add_argument("--model", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--device")
    parser.add_argument("--decompose", action="store_true", help="Decompose the query first.")
    parser.add_argument("--max-subqueries", type=int, default=4)
    parser.add_argument("--rerank", action="store_true", help="Apply cross-encoder reranking.")
    parser.add_argument("--rerank-model", default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--candidate-k", type=int)


def build_base_retriever(args, chunks):
    bm25 = BM25Retriever(chunks)
    if args.method == "bm25":
        return bm25
    dense = DenseRetriever(chunks, E5Encoder(args.model, device=args.device))
    return dense if args.method == "dense" else HybridRetriever(bm25, dense)


def build_evaluation_retriever(args, chunks):
    retriever = build_base_retriever(args, chunks)
    retriever_name = args.method
    if args.decompose:
        decomposer = HeuristicQueryDecomposer(max_subqueries=args.max_subqueries)
        retriever = DecomposedRetriever(retriever, decomposer)
        retriever_name += "+decompose"
    if args.rerank:
        reranker = CrossEncoderReranker(model_name=args.rerank_model, device=args.device)
        retriever = RerankedRetriever(retriever, reranker)
        retriever_name += "+rerank"
    return retriever, retriever_name


def build_frozen_policy_retriever(policy, chunks, *, device=None):
    method = policy.retriever.get("method")
    if method not in {"bm25", "dense", "hybrid"}:
        raise ValueError("Frozen policy method must be bm25, dense, or hybrid")
    bm25 = BM25Retriever(chunks)
    if method == "bm25":
        return bm25
    dense = DenseRetriever(
        chunks,
        E5Encoder(
            policy.retriever.get("model", DEFAULT_DENSE_MODEL),
            device=device,
            revision=policy.retriever.get("model_revision"),
        ),
        query_prefix=policy.retriever.get("query_prefix", "query: "),
        passage_prefix=policy.retriever.get("passage_prefix", "passage: "),
    )
    return dense if method == "dense" else HybridRetriever(bm25, dense)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evidence-rag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate a document JSONL corpus.")
    validate.add_argument("--documents", required=True, type=Path)

    chunk = subparsers.add_parser("chunk", help="Chunk documents into a JSONL artifact.")
    chunk.add_argument("--documents", required=True, type=Path)
    chunk.add_argument("--output", required=True, type=Path)
    chunk.add_argument("--size", type=int, default=900)
    chunk.add_argument("--overlap", type=int, default=150)

    search = subparsers.add_parser("search", help="Search a chunk JSONL artifact.")
    search.add_argument("--chunks", required=True, type=Path)
    search.add_argument("--query", required=True)
    search.add_argument("--query-id", default="adhoc")
    search.add_argument("--language")
    search.add_argument("--top-k", type=int, default=5)
    add_retrieval_options(search)

    verify = subparsers.add_parser("verify", help="Verify answer citations against a response.")
    verify.add_argument("--response", required=True, type=Path)
    verify.add_argument("--answer", required=True)
    verify.add_argument("--support-threshold", type=float, default=0.6)
    verify.add_argument(
        "--verifier",
        choices=["lexical", "nli"],
        default="lexical",
        help="lexical: token overlap (offline). nli: multilingual entailment model.",
    )
    verify.add_argument("--nli-model", default=None)
    verify.add_argument("--device", default=None)

    gold = subparsers.add_parser("validate-gold", help="Validate and summarize a gold set.")
    gold.add_argument("--gold-set", required=True, type=Path)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate retrieval against a gold set.")
    evaluate.add_argument("--gold-set", required=True, type=Path)
    evaluate.add_argument("--chunks", required=True, type=Path)
    evaluate.add_argument("--k", type=int, default=10)
    add_retrieval_options(evaluate)

    selective = subparsers.add_parser("selective", help="Make an answer-or-abstain decision.")
    selective.add_argument("--response", required=True, type=Path)
    selective.add_argument("--verification", type=Path)
    selective.add_argument("--min-top-score", required=True, type=float)
    selective.add_argument("--allow-unverified", action="store_true")

    calibration = subparsers.add_parser(
        "calibrate-selective", help="Calibrate a top-score threshold on a labeled gold set."
    )
    calibration.add_argument("--gold-set", required=True, type=Path)
    calibration.add_argument("--chunks", required=True, type=Path)
    calibration.add_argument("--max-risk", type=float, default=0.0)
    calibration.add_argument("--output", type=Path)
    add_retrieval_options(calibration)

    policy_evaluation = subparsers.add_parser(
        "evaluate-selective", help="Evaluate a frozen policy without threshold retuning."
    )
    policy_evaluation.add_argument("--policy", required=True, type=Path)
    policy_evaluation.add_argument("--gold-set", required=True, type=Path)
    policy_evaluation.add_argument("--chunks", required=True, type=Path)
    policy_evaluation.add_argument("--device")
    policy_evaluation.add_argument("--output", type=Path)

    scan = subparsers.add_parser("security-scan", help="Screen chunks without modifying them.")
    scan.add_argument("--chunks", required=True, type=Path)

    security = subparsers.add_parser("security-test", help="Run poisoned retrieval fixtures.")
    security.add_argument("--cases", required=True, type=Path)
    security.add_argument("--screen", action="store_true")

    kaggle = subparsers.add_parser("kaggle-export", help="Export a hashed Kaggle dataset.")
    kaggle.add_argument("--documents", required=True, type=Path)
    kaggle.add_argument("--chunks", required=True, type=Path)
    kaggle.add_argument("--gold-set", required=True, type=Path)
    kaggle.add_argument("--output-dir", required=True, type=Path)
    kaggle.add_argument("--dataset-id", required=True)
    kaggle.add_argument("--name", default="Evidence RAG deterministic benchmark")
    kaggle.add_argument(
        "--description",
        default="Offline BM25 evaluation over a frozen corpus and explicit gold labels.",
    )
    kaggle.add_argument("--k", type=int, default=4)
    kaggle.add_argument("--license", default="MIT", help="Kaggle license name, e.g. CC-BY-4.0.")

    benchmark = subparsers.add_parser("benchmark-run", help="Run a hashed benchmark manifest.")
    benchmark.add_argument("--manifest", required=True, type=Path)
    benchmark.add_argument("--output-dir", type=Path)

    huggingface = subparsers.add_parser("hf-export", help="Export a Hugging Face dataset.")
    huggingface.add_argument("--documents", required=True, type=Path)
    huggingface.add_argument("--chunks", required=True, type=Path)
    huggingface.add_argument("--gold-set", required=True, type=Path)
    huggingface.add_argument("--output-dir", required=True, type=Path)
    huggingface.add_argument("--dataset-id", required=True)
    huggingface.add_argument("--name", default="Evidence RAG multilingual evidence dataset")
    huggingface.add_argument("--license", default="mit", help="Hub license id, e.g. cc-by-4.0.")
    huggingface.add_argument("--attribution", default=None, help="Source and license note.")
    huggingface.add_argument(
        "--description",
        default="Source-attributed multilingual retrieval fixtures with explicit label provenance.",
    )

    verify_dataset = subparsers.add_parser("hf-verify", help="Verify dataset content hashes.")
    verify_dataset.add_argument("--manifest", required=True, type=Path)

    mcp = subparsers.add_parser("mcp-serve", help="Serve read-only MCP tools over stdio.")
    mcp.add_argument("--chunks", required=True, type=Path)

    create = subparsers.add_parser("annotation-create", help="Create a rank-blind batch.")
    create.add_argument("--queries", required=True, type=Path)
    create.add_argument("--chunks", required=True, type=Path)
    create.add_argument("--output", required=True, type=Path)
    create.add_argument("--batch-id", required=True)
    create.add_argument("--name", required=True)
    create.add_argument("--corpus-name", required=True)
    create.add_argument("--split", choices=("development", "test"), default="development")
    create.add_argument(
        "--description",
        default="Rank-blind human annotation batch; no decisions have been collected.",
    )
    create.add_argument("--guidelines-version", default="1.0")

    annotation = subparsers.add_parser("annotation-summary", help="Summarize batch completion.")
    annotation.add_argument("--batch", required=True, type=Path)

    export = subparsers.add_parser("annotation-export", help="Export a complete human gold set.")
    export.add_argument("--batch", required=True, type=Path)
    export.add_argument("--chunks", required=True, type=Path)
    export.add_argument("--output", required= True, type=Path)

    agreement = subparsers.add_parser(
        "annotation-agreement", help="Compare two independent annotation files."
    )
    agreement.add_argument("--primary", required=True, type=Path)
    agreement.add_argument("--secondary", required=True, type=Path)
    agreement.add_argument("--adjudications", type=Path)
    agreement.add_argument(
        "--provenance", choices=("human", "benchmark", "synthetic"), required=True
    )
    agreement.add_argument("--output", type=Path)

    schemas = subparsers.add_parser("export-schemas", help="Write public JSON Schemas.")
    schemas.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    if args.command == "validate":
        documents = load_documents(args.documents)
        chunks = chunk_documents(documents)
        print(json.dumps({"documents": len(documents), "chunks": len(chunks), "valid": True}))
        return

    if args.command == "chunk":
        chunks = chunk_documents(
            load_documents(args.documents), size=args.size, overlap=args.overlap
        )
        write_jsonl(args.output, chunks)
        print(json.dumps({"chunks": len(chunks), "output": str(args.output)}))
        return

    if args.command == "verify":
        response = load_retrieval_response(args.response)
        if args.verifier == "nli":
            from evidence_rag.citation.nli import DEFAULT_NLI_MODEL, NLICitationVerifier

            verifier = NLICitationVerifier(
                model_name=args.nli_model or DEFAULT_NLI_MODEL, device=args.device
            )
        else:
            verifier = ClaimLevelCitationVerifier(support_threshold=args.support_threshold)
        report = verifier.verify(args.answer, response.hits)
        print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return

    if args.command == "validate-gold":
        summary = summarize_gold_set(load_gold_set(args.gold_set))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.command == "evaluate":
        retriever, retriever_name = build_evaluation_retriever(args, load_chunks(args.chunks))
        report = evaluate_retriever(
            retriever, load_gold_set(args.gold_set), k=args.k, retriever_name=retriever_name
        )
        print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return

    if args.command == "selective":
        response = load_retrieval_response(args.response)
        verification = (
            CitationVerificationReport.model_validate_json(
                args.verification.read_text(encoding="utf-8")
            )
            if args.verification is not None
            else None
        )
        decision = SelectiveAnswerer(
            min_top_score=args.min_top_score,
            require_citations=not args.allow_unverified,
        ).decide(response.hits, verification)
        print(json.dumps(decision.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return

    if args.command == "calibrate-selective":
        retriever, retriever_name = build_evaluation_retriever(args, load_chunks(args.chunks))
        report = calibrate_selective_answerer(
            retriever,
            load_gold_set(args.gold_set),
            max_risk=args.max_risk,
            retriever_name=retriever_name,
        )
        if args.output is not None:
            write_json(args.output, report.model_dump(mode="json"))
        print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return

    if args.command == "evaluate-selective":
        policy = load_frozen_selective_policy(args.policy)
        if policy.corpus.get("chunks_sha256") != sha256_file(args.chunks):
            raise ValueError("Frozen policy corpus hash does not match the chunks")
        retriever = build_frozen_policy_retriever(
            policy, load_chunks(args.chunks), device=args.device
        )
        report = evaluate_selective_policy(
            retriever,
            load_gold_set(args.gold_set),
            policy,
            retriever_name=policy.retriever["method"],
        )
        if args.output is not None:
            write_json(args.output, report.model_dump(mode="json"))
        print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return

    if args.command == "security-scan":
        _, report = screen_chunks(load_chunks(args.chunks))
        print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return

    if args.command == "security-test":
        report = evaluate_poisoned_retrieval(
            BM25Retriever,
            load_poisoned_cases(args.cases),
            screened=args.screen,
        )
        print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return

    if args.command == "kaggle-export":
        manifest = export_kaggle_dataset(
            documents_path=args.documents,
            chunks_path=args.chunks,
            gold_set_path=args.gold_set,
            output_dir=args.output_dir,
            dataset_id=args.dataset_id,
            name=args.name,
            description=args.description,
            k=args.k,
            license_name=args.license,
        )
        print(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return

    if args.command == "benchmark-run":
        report = run_benchmark(args.manifest)
        if args.output_dir is None:
            print(report_json(report))
        else:
            json_path, markdown_path = write_benchmark_report(report, args.output_dir)
            print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}))
        return

    if args.command == "hf-export":
        manifest = export_huggingface_dataset(
            documents_path=args.documents,
            chunks_path=args.chunks,
            gold_set_path=args.gold_set,
            output_dir=args.output_dir,
            dataset_id=args.dataset_id,
            name=args.name,
            description=args.description,
            license_id=args.license,
            attribution=args.attribution,
        )
        print(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return

    if args.command == "hf-verify":
        manifest = verify_huggingface_dataset(args.manifest)
        print(json.dumps({"files": len(manifest.files), "valid": True}))
        return

    if args.command == "mcp-serve":
        run_stdio(ReadOnlyEvidenceService(load_chunks(args.chunks)))
        return

    if args.command == "annotation-create":
        batch = create_annotation_batch(
            queries=load_annotation_queries(args.queries),
            chunks=load_chunks(args.chunks),
            chunks_path=args.chunks,
            batch_id=args.batch_id,
            name=args.name,
            description=args.description,
            split=args.split,
            corpus_name=args.corpus_name,
            guidelines_version=args.guidelines_version,
        )
        write_annotation_batch(batch, args.output)
        print(json.dumps(summarize_annotation_batch(batch), ensure_ascii=False, indent=2))
        return

    if args.command == "annotation-summary":
        summary = summarize_annotation_batch(load_annotation_batch(args.batch))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    if args.command == "annotation-export":
        gold_set = export_human_gold_set(load_annotation_batch(args.batch), args.chunks)
        write_json(args.output, gold_set.model_dump(mode="json"))
        print(json.dumps(summarize_gold_set(gold_set), ensure_ascii=False, indent=2))
        return

    if args.command == "annotation-agreement":
        report = compare_annotations(
            load_independent_annotations(args.primary),
            load_independent_annotations(args.secondary),
            adjudications=load_adjudications(args.adjudications),
            provenance=args.provenance,
        )
        if args.output is not None:
            write_json(args.output, report.model_dump(mode="json"))
        print(json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return

    if args.command == "export-schemas":
        paths = export_schemas(args.output_dir)
        print(json.dumps({"schemas": len(paths), "output_dir": str(args.output_dir)}))
        return

    chunks = load_chunks(args.chunks)
    query = Query(
        query_id=args.query_id,
        text=args.query,
        language=args.language,
        top_k=args.top_k,
    )
    retriever = build_base_retriever(args, chunks)
    decomposition = None
    if args.decompose:
        decomposer = HeuristicQueryDecomposer(max_subqueries=args.max_subqueries)
        decomposition = decomposer.decompose(query)
        retriever = DecomposedRetriever(retriever, decomposer)

    if args.rerank:
        reranker = CrossEncoderReranker(model_name=args.rerank_model, device=args.device)
        reranked_hits = RerankedRetriever(retriever, reranker).search(
            query, candidate_k=args.candidate_k
        )
        response = RerankedResponse(
            query=query,
            decomposition=decomposition,
            reranker_model=reranker.model_name,
            hits=reranked_hits,
        )
    elif decomposition is not None:
        response = DecomposedResponse(
            query=query,
            decomposition=decomposition,
            hits=retriever.search_decomposition(decomposition),
        )
    else:
        response = RetrievalResponse(query=query, hits=retriever.search(query))
    print(json.dumps(response.model_dump(mode="json"), ensure_ascii=False, indent=2))
