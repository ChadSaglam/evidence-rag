"""Build a frozen, source-attributed multilingual GDPR retrieval corpus.

The source is the official Publications Office Cellar XHTML representation of
Regulation (EU) 2016/679. Network access is only needed with --fetch. All later
steps use the saved source snapshots and standard-library XML parsing.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evidence_rag.benchmark.kaggle import sha256_file, write_json  # noqa: E402
from evidence_rag.chunking import chunk_documents  # noqa: E402
from evidence_rag.corpus import write_jsonl  # noqa: E402
from evidence_rag.schemas import (  # noqa: E402
    Document,
    GoldQuery,
    GoldSet,
    Query,
    RelevanceJudgment,
    Source,
)

CELEX = "32016R0679"
SOURCE_URL = "https://publications.europa.eu/resource/celex/32016R0679"
ARTICLES = (5, 6, 12, 13, 15, 17, 20, 25, 30, 32, 33, 35)
LANGUAGES = {"en": "eng", "de": "deu", "fr": "fra"}
RETRIEVED_AT = "2026-10-31"
XHTML = "{http://www.w3.org/1999/xhtml}"

# Query-to-article alignment is benchmark curation, not human annotation.
QUERY_SPECS = [
    ("en-principles", "en", "What principles apply to the processing of personal data?", [5]),
    ("en-lawfulness", "en", "When is the processing of personal data lawful?", [6]),
    ("en-collection-information", "en", "What information must a controller provide when personal data are collected from the data subject?", [13]),
    ("en-access", "en", "What information can a data subject obtain under the right of access?", [15]),
    ("en-erasure", "en", "When does a data subject have the right to erasure?", [17]),
    ("en-portability", "en", "What is the right to data portability?", [20]),
    ("en-security", "en", "What security measures must controllers and processors implement?", [32]),
    ("en-breach-deadline", "en", "How quickly must a controller notify a personal data breach to the supervisory authority?", [33]),
    ("en-dpia", "en", "When is a data protection impact assessment required?", [35]),
    ("en-accounting-retention", "en", "What fixed accounting-record retention period does the GDPR prescribe?", []),
    ("de-principles", "de", "Welche Grundsätze gelten für die Verarbeitung personenbezogener Daten?", [5]),
    ("de-lawfulness", "de", "Wann ist die Verarbeitung personenbezogener Daten rechtmäßig?", [6]),
    ("de-collection-information", "de", "Welche Informationen muss der Verantwortliche bei der Erhebung personenbezogener Daten bereitstellen?", [13]),
    ("de-access", "de", "Welche Informationen erhält die betroffene Person durch das Auskunftsrecht?", [15]),
    ("de-erasure", "de", "Wann besteht ein Recht der betroffenen Person auf Löschung?", [17]),
    ("de-portability", "de", "Was umfasst das Recht auf Datenübertragbarkeit?", [20]),
    ("de-security", "de", "Welche Sicherheitsmaßnahmen müssen Verantwortliche und Auftragsverarbeiter treffen?", [32]),
    ("de-breach-deadline", "de", "Wie schnell muss der Verantwortliche eine Verletzung des Schutzes personenbezogener Daten der Aufsichtsbehörde melden?", [33]),
    ("de-dpia", "de", "Wann ist eine Datenschutz-Folgenabschätzung erforderlich?", [35]),
    ("de-accounting-retention", "de", "Welche feste Aufbewahrungsfrist für Buchhaltungsunterlagen schreibt die DSGVO vor?", []),
    ("fr-principles", "fr", "Quels principes s'appliquent au traitement des données à caractère personnel?", [5]),
    ("fr-lawfulness", "fr", "Quand le traitement des données à caractère personnel est-il licite?", [6]),
    ("fr-collection-information", "fr", "Quelles informations le responsable du traitement doit-il fournir lors de la collecte des données auprès de la personne concernée?", [13]),
    ("fr-access", "fr", "Quelles informations la personne concernée peut-elle obtenir grâce à son droit d'accès?", [15]),
    ("fr-erasure", "fr", "Quand la personne concernée a-t-elle le droit à l'effacement?", [17]),
    ("fr-portability", "fr", "Que comprend le droit à la portabilité des données?", [20]),
    ("fr-security", "fr", "Quelles mesures de sécurité les responsables du traitement et les sous-traitants doivent-ils mettre en œuvre?", [32]),
    ("fr-breach-deadline", "fr", "Dans quel délai le responsable du traitement doit-il notifier une violation de données à l'autorité de contrôle?", [33]),
    ("fr-dpia", "fr", "Quand une analyse d'impact relative à la protection des données est-elle obligatoire?", [35]),
    ("fr-accounting-retention", "fr", "Quel délai fixe de conservation des documents comptables le RGPD prescrit-il?", []),
]


def normalize_text(element: ET.Element) -> str:
    """Normalize layout whitespace while preserving the article's textual content."""

    return " ".join("".join(element.itertext()).split())


def fetch_sources(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for language, accept_language in LANGUAGES.items():
        destination = raw_dir / f"gdpr-{language}.xhtml"
        subprocess.run(
            [
                "curl",
                "-L",
                "--fail",
                "--max-time",
                "90",
                "-H",
                "Accept: application/xhtml+xml",
                "-H",
                f"Accept-Language: {accept_language}",
                "-o",
                str(destination),
                SOURCE_URL,
            ],
            check=True,
        )
        if destination.stat().st_size < 100_000:
            raise RuntimeError(f"Source snapshot is unexpectedly small: {destination}")


def extract_documents(raw_dir: Path) -> list[Document]:
    documents = []
    for language in LANGUAGES:
        snapshot = raw_dir / f"gdpr-{language}.xhtml"
        root = ET.parse(snapshot).getroot()
        for article in ARTICLES:
            elements = root.findall(f".//{XHTML}div[@id='art_{article}']")
            if len(elements) != 1:
                raise ValueError(f"Expected one Article {article} in {snapshot}")
            article_element = elements[0]
            title_elements = article_element.findall(
                f"./{XHTML}div[@id='art_{article}.tit_1']"
            )
            title = normalize_text(title_elements[0]) if title_elements else f"Article {article}"
            text = normalize_text(article_element)
            documents.append(
                Document(
                    document_id=f"gdpr-{language}-article-{article:02d}",
                    text=text,
                    source=Source(
                        uri=f"https://eur-lex.europa.eu/eli/reg/2016/679/oj/{LANGUAGES[language]}",
                        title=f"Regulation (EU) 2016/679 — Article {article}: {title}",
                        language=language,
                        retrieved_at=RETRIEVED_AT,
                    ),
                    metadata={
                        "authority": "European Union Publications Office",
                        "celex": CELEX,
                        "article": article,
                        "article_title": title,
                        "reuse_notice": "EUR-Lex legal document reuse; source acknowledged; whitespace normalized and selected articles extracted.",
                    },
                )
            )
    return documents


def build_gold_set(chunks) -> tuple[list[Query], GoldSet]:
    queries = []
    gold_queries = []
    for query_id, language, text, articles in QUERY_SPECS:
        query = Query(query_id=query_id, text=text, language=language, top_k=10)
        queries.append(query)
        judgments = [
            RelevanceJudgment(
                query_id=query_id,
                chunk_id=chunk.chunk_id,
                relevance=3,
                annotator="benchmark-source-alignment-v1",
                notes=f"Aligned to GDPR Article {chunk.metadata['article']}.",
            )
            for chunk in chunks
            if chunk.metadata.get("article") in articles
            and chunk.source.language == language
        ]
        gold_queries.append(
            GoldQuery(query=query, answerable=bool(judgments), judgments=judgments)
        )
    gold_set = GoldSet(
        name="gdpr-multilingual-source-aligned-v1",
        description=(
            "Thirty English, German, and French queries over selected official GDPR "
            "articles. Relevance labels are deterministic query-to-article benchmark "
            "curation, not human annotations or production legal judgments."
        ),
        label_source="benchmark",
        queries=gold_queries,
    )
    return queries, gold_set


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fetch", action="store_true", help="Refresh official XHTML snapshots.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "real-gdpr")
    args = parser.parse_args()
    raw_dir = args.output_dir / "raw"
    if args.fetch:
        fetch_sources(raw_dir)

    documents = extract_documents(raw_dir)
    chunks = chunk_documents(documents, size=1200, overlap=150)
    queries, gold_set = build_gold_set(chunks)
    write_jsonl(args.output_dir / "documents.jsonl", documents)
    write_jsonl(args.output_dir / "chunks.jsonl", chunks)
    write_json(args.output_dir / "queries.json", [q.model_dump(mode="json") for q in queries])
    write_json(args.output_dir / "gold_set.json", gold_set.model_dump(mode="json"))

    snapshots = {
        language: {
            "path": f"raw/gdpr-{language}.xhtml",
            "sha256": sha256_file(raw_dir / f"gdpr-{language}.xhtml"),
        }
        for language in LANGUAGES
    }
    manifest = {
        "schema_version": "1.0",
        "corpus_id": "gdpr-selected-articles-multilingual-v1",
        "source_url": SOURCE_URL,
        "retrieved_at": RETRIEVED_AT,
        "articles": list(ARTICLES),
        "languages": list(LANGUAGES),
        "documents": len(documents),
        "chunks": len(chunks),
        "queries": len(queries),
        "answerable_queries": sum(q.answerable for q in gold_set.queries),
        "unanswerable_queries": sum(not q.answerable for q in gold_set.queries),
        "label_source": gold_set.label_source,
        "source_snapshots": snapshots,
        "chunks_sha256": sha256_file(args.output_dir / "chunks.jsonl"),
        "license_notice": "https://eur-lex.europa.eu/content/legal-notice/legal-notice.html",
        "changes": "Selected articles extracted and layout whitespace normalized; no legal text paraphrased.",
    }
    write_json(args.output_dir / "corpus-manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
