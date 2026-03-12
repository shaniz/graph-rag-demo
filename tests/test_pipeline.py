"""Integration tests for the GraphRAG IT troubleshooting pipeline.

These tests use a session-scoped fixture so the index is built only once.
Requires OPENROUTER_API_KEY to be set in the environment.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
CORPUS_FILE = "corpus.json"
CORPUS_NAME = Path(CORPUS_FILE).stem  # "corpus"


@pytest.fixture(scope="session")
def pipeline():
    """Build the index once for all tests in the session."""
    from graph_rag.ingest import ingest_corpus
    from graph_rag.pipeline import GraphRAGPipeline

    # Ingest corpus if input files don't exist
    input_dir = ROOT / "input" / CORPUS_NAME
    if not any(input_dir.glob("*.txt")):
        ingest_corpus(
            corpus_path=ROOT / "data" / CORPUS_FILE,
            input_dir=input_dir,
        )

    # Build index if output parquet not present
    p = GraphRAGPipeline(root_dir=ROOT, corpus_name=CORPUS_NAME)
    output = ROOT / "output" / CORPUS_NAME / "create_final_entities.parquet"
    if not output.exists():
        results = p.build_index()
        errors = [r for r in results if r.error]
        assert not errors, f"Indexing errors: {errors}"

    return p


@pytest.mark.skipif(
    not os.getenv("OPENROUTER_API_KEY"),
    reason="OPENROUTER_API_KEY not set",
)
class TestPipeline:
    def test_entities_parquet_exists(self, pipeline):
        """Entities parquet file should exist after indexing."""
        path = ROOT / "output" / CORPUS_NAME / "entities.parquet"
        assert path.exists(), f"Missing: {path}"

    def test_relationships_parquet_exists(self, pipeline):
        """Relationships parquet file should exist after indexing."""
        path = ROOT / "output" / CORPUS_NAME / "relationships.parquet"
        assert path.exists(), f"Missing: {path}"

    def test_community_reports_exist(self, pipeline):
        """Community reports parquet should exist after indexing."""
        path = ROOT / "output" / CORPUS_NAME / "community_reports.parquet"
        assert path.exists(), f"Missing: {path}"

    def test_entities_not_empty(self, pipeline):
        """The knowledge graph should contain entities."""
        import pandas as pd
        df = pd.read_parquet(ROOT / "output" / CORPUS_NAME / "entities.parquet")
        assert len(df) > 0, "No entities found in the knowledge graph"

    def test_search_bsod(self, pipeline):
        """Search should return a relevant answer about BSOD."""
        answer = pipeline.search("How do I troubleshoot a Windows Blue Screen of Death?")
        assert isinstance(answer, str)
        assert len(answer) > 50
        lower = answer.lower()
        assert any(kw in lower for kw in ["bsod", "blue screen", "memory", "driver", "crash", "minidump"])

    def test_search_network(self, pipeline):
        """Search should return relevant network troubleshooting steps."""
        answer = pipeline.search("Why can't my workstation get a DHCP IP address?")
        assert isinstance(answer, str)
        assert len(answer) > 50
        lower = answer.lower()
        assert any(kw in lower for kw in ["dhcp", "ip", "network", "ipconfig", "gateway"])

    def test_search_kubernetes(self, pipeline):
        """Search should answer Kubernetes-related questions."""
        answer = pipeline.search("How do I fix CrashLoopBackOff in Kubernetes?")
        assert isinstance(answer, str)
        assert len(answer) > 50
        lower = answer.lower()
        assert any(kw in lower for kw in ["crash", "kubernetes", "pod", "container", "oom", "logs"])
