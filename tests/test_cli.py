import json

from conftest import make_chunk

from evidence_rag.cli import main
from evidence_rag.corpus import write_jsonl


def test_bm25_cli_emits_auditable_json(tmp_path, capsys) -> None:
    path = tmp_path / "chunks.jsonl"
    write_jsonl(path, [make_chunk("doc:0", "alpha"), make_chunk("doc:1", "beta")])
    main(
        [
            "search",
            "--chunks",
            str(path),
            "--query",
            "alpha",
            "--query-id",
            "q-test",
            "--top-k",
            "1",
        ]
    )
    output = json.loads(capsys.readouterr().out)
    assert output["query"]["query_id"] == "q-test"
    assert output["hits"][0]["chunk"]["chunk_id"] == "doc:0"
    assert output["hits"][0]["retriever"] == "bm25"


def test_sample_corpus_validates(capsys) -> None:
    main(["validate", "--documents", "data/sample/documents.jsonl"])
    assert json.loads(capsys.readouterr().out) == {"documents": 4, "chunks": 4, "valid": True}
