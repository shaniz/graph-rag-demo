# GraphRAG Demo

A Graph Retrieval-Augmented Generation (GraphRAG) demo built on **Microsoft GraphRAG v3** that indexes an IT troubleshooting corpus into a knowledge graph and answers natural-language queries using local search over entity communities.

## Architecture

```
data/corpus.json
      │
      ▼
 ingest.py ──────────► input/*.txt
                              │
                              ▼
                    graphrag build_index
                    (LiteLLM → OpenRouter)
                              │
                    ┌─────────┴──────────┐
                    ▼                    ▼
             output/*.parquet     output/lancedb
             (entities,           (vector embeddings)
              relationships,
              community_reports)
                    │
                    ▼
            pipeline.local_search()
                    │
                    ▼
                 answer
```

## Setup

```bash
# Install dependencies
uv sync --extra dev

# Set your OpenRouter API key
export OPENROUTER_API_KEY="sk-or-..."
```

## Usage

```bash
# Build the knowledge graph index from corpus (~5-10 min first run)
uv run python scripts/build_graph.py

# Visualize the graph as PNG
uv run python scripts/visualize_graph.py

# Run integration tests
uv run pytest tests/ -v
```

## Models

| Role | Model |
|---|---|
| Completion | `anthropic/claude-haiku-4-5` via OpenRouter |
| Embeddings | `openai/text-embedding-3-small` via OpenRouter |

To change models, edit `settings.yaml`.

## Project Structure

| File | Role |
|---|---|
| `settings.yaml` | GraphRAG configuration — models, storage, workflows |
| `src/graph_rag/ingest.py` | Converts `corpus.json` → `input/*.txt` |
| `src/graph_rag/pipeline.py` | Wraps `graphrag.api.build_index` and `local_search` |
| `scripts/build_graph.py` | Entry point: ingest + index + print stats |
| `scripts/visualize_graph.py` | Renders knowledge graph as `output/graph.png` |
| `tests/test_pipeline.py` | 7 integration tests |
