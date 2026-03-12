"""Ingest corpus.json into GraphRAG input directory as .txt files."""
from __future__ import annotations

import json
from pathlib import Path


def ingest_corpus(
    corpus_path: Path | str = "data/corpus.json",
    input_dir: Path | str = "input",
) -> list[Path]:
    """Convert corpus.json documents to individual .txt files in input_dir.

    GraphRAG expects plain-text files in the input directory.
    Each document becomes one .txt file named by its id.

    Returns list of written file paths.
    """
    corpus_path = Path(corpus_path)
    input_dir = Path(input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    with corpus_path.open(encoding="utf-8") as f:
        documents = json.load(f)

    written: list[Path] = []
    for doc in documents:
        doc_id = doc.get("id", f"doc_{len(written):04d}")
        title = doc.get("title", "")
        content = doc.get("content", "")

        text = f"{title}\n\n{content}\n" if title else f"{content}\n"
        out_path = input_dir / f"{doc_id}.txt"
        out_path.write_text(text, encoding="utf-8")
        written.append(out_path)

    print(f"Ingested {len(written)} documents to {input_dir}/")
    return written
