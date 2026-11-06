"""Assemble a self-contained Hugging Face Space folder for the read-only demo.

Usage:
    python scripts/build_space.py --output-dir artifacts/hf-space
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "demo/app.py": "app.py",
    "demo/README.md": "README.md",
    "demo/requirements.txt": "requirements.txt",
    "data/real-gdpr/chunks.jsonl": "data/real-gdpr/chunks.jsonl",
    "data/sample/documents.jsonl": "data/sample/documents.jsonl",
    "LICENSE": "LICENSE",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/hf-space")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    for source, target in FILES.items():
        (out / target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / source, out / target)
    shutil.copytree(
        ROOT / "src/evidence_rag",
        out / "src/evidence_rag",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    # app.py resolves the repository root as its parent's parent; in a Space the root
    # is the app's own folder.
    app = out / "app.py"
    app.write_text(
        app.read_text(encoding="utf-8").replace(
            "Path(__file__).resolve().parents[1]", "Path(__file__).resolve().parent"
        ),
        encoding="utf-8",
    )
    print(f"Space folder ready: {out}")


if __name__ == "__main__":
    main()
