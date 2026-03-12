#!/usr/bin/env python3
"""Entry point: ingest corpus, build GraphRAG index, print stats."""
import os
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from graph_rag.ingest import ingest_corpus
from graph_rag.pipeline import GraphRAGPipeline


def main(corpus_file: str = "corpus.json") -> None:
    root = Path(__file__).parent.parent
    corpus_name = Path(corpus_file).stem  # e.g. "corpus_tree" from "corpus_tree.json"

    corpus_path = root / "data" / corpus_file
    if not corpus_path.exists():
        print(f"Error: corpus file not found: {corpus_path}")
        sys.exit(1)

    print(f"Step 1: Ingesting corpus from {corpus_file}...")
    ingest_corpus(
        corpus_path=corpus_path,
        input_dir=root / "input" / corpus_name,
    )

    print("\nStep 2: Building knowledge graph index (this may take 5-10 minutes)...")
    pipeline = GraphRAGPipeline(root_dir=root, corpus_name=corpus_name)
    results = pipeline.build_index()

    errors = [r for r in results if r.error]
    if errors:
        print("\nIndexing completed with errors:")
        for r in errors:
            print(f"  Workflow '{r.workflow}': {r.error}")
        sys.exit(1)
    else:
        print(f"\nIndexing completed successfully ({len(results)} workflows).")

    print("\nStep 3: Knowledge graph statistics:")
    pipeline.print_stats()

    print("\nStep 4: Running sample queries...")
    queries = [
        "How do I fix BSOD MEMORY_MANAGEMENT?",
        "How do I troubleshoot DHCP issues?",
        "How do I debug a Kubernetes CrashLoopBackOff?",
    ]
    for query in queries:
        print(f"\nQ: {query}")
        answer = pipeline.search(query)
        print(f"A: {answer}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env")

    api_key = os.environ.get("OPENROUTER_API_KEY", "NOT SET")
    print(f"OPENROUTER_API_KEY: {api_key[:8]}...{api_key[-4:] if len(api_key) > 12 else api_key}")

    main(corpus_file="corpus.json")