"""GraphRAG pipeline: build index and local search."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any


class GraphRAGPipeline:
    """Wraps graphrag.api.build_index and local_search for IT troubleshooting queries."""

    def __init__(self, root_dir: Path | str = ".", corpus_name: str = "corpus") -> None:
        self.root_dir = Path(root_dir).resolve()
        self.corpus_name = corpus_name

    def _load_config(self) -> Any:
        from graphrag.config.load_config import load_config

        config = load_config(self.root_dir)
        self._patch_config(config)
        return config

    def _patch_config(self, config: Any) -> None:
        """Override input/output paths to use corpus-specific subdirectory."""
        sub = self.corpus_name
        if hasattr(config, "input_storage") and config.input_storage:
            config.input_storage.base_dir = f"input/{sub}"
        if hasattr(config, "output_storage") and config.output_storage:
            config.output_storage.base_dir = f"output/{sub}"
        if hasattr(config, "vector_store") and config.vector_store:
            vs = config.vector_store
            if hasattr(vs, "db_uri"):
                vs.db_uri = f"output/{sub}/lancedb"
            elif isinstance(vs, dict):
                for v in vs.values():
                    if hasattr(v, "db_uri"):
                        v.db_uri = f"output/{sub}/lancedb"
        if hasattr(config, "reporting") and config.reporting:
            config.reporting.base_dir = f"logs/{sub}"

    def build_index(self) -> list[Any]:
        """Run the full GraphRAG indexing pipeline synchronously.

        Returns list of PipelineRunResult objects.
        """
        return asyncio.run(self._build_index_async())

    async def _build_index_async(self) -> list[Any]:
        import graphrag.api as api

        config = self._load_config()
        results = await api.build_index(config=config)
        return results

    def search(self, query: str, community_level: int = 2) -> str:
        """Run a local search query against the indexed knowledge graph.

        Args:
            query: Natural language question about IT troubleshooting.
            community_level: Community level to search (higher = broader context).

        Returns:
            Answer string from the LLM.
        """
        return asyncio.run(self._search_async(query, community_level))

    async def _search_async(self, query: str, community_level: int) -> str:
        import graphrag.api as api

        config = self._load_config()

        response, context = await api.local_search(
            config=config,
            entities=self._load_parquet("entities"),
            communities=self._load_parquet("communities"),
            community_reports=self._load_parquet("community_reports"),
            text_units=self._load_parquet("text_units"),
            relationships=self._load_parquet("relationships"),
            covariates=None,
            community_level=community_level,
            response_type="multiple paragraphs",
            query=query,
        )
        return response

    def _load_parquet(self, table_name: str):
        import pandas as pd

        path = self.root_dir / "output" / self.corpus_name / f"{table_name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Parquet file not found: {path}\n"
                "Run build_index() first to generate the knowledge graph."
            )
        return pd.read_parquet(path)

    def print_stats(self) -> None:
        """Print statistics about the indexed knowledge graph."""
        tables = {
            "Entities": "entities",
            "Relationships": "relationships",
            "Communities": "communities",
            "Community Reports": "community_reports",
            "Text Units": "text_units",
        }

        print("\n=== Knowledge Graph Statistics ===")
        for label, table in tables.items():
            try:
                df = self._load_parquet(table)
                print(f"  {label}: {len(df)}")
            except FileNotFoundError:
                print(f"  {label}: (not found)")
        print("==================================\n")
