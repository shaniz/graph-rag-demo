# GraphRAG Demo

A Graph Retrieval-Augmented Generation (GraphRAG) demo built on **Microsoft GraphRAG v3** that indexes an IT troubleshooting corpus into a knowledge graph and answers natural-language queries using local search over entity communities.

## AI Assistant Rules

- **Always use Context7 MCP** when looking up library/API documentation, configuration options, or code generation steps — without waiting to be asked. Load the tool via `resolve-library-id` then `get-library-docs`.

## Commands

```bash
# Install dependencies
uv sync --extra dev

# Build the knowledge graph index from corpus (first run ~5-10 min)
uv run python scripts/build_graph.py

# Visualize the graph as PNG (requires output parquet files)
uv run python scripts/visualize_graph.py

# Run integration tests (requires ANTHROPIC_API_KEY or OPENROUTER_API_KEY)
uv run pytest tests/ -v
```

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
            (graphrag local search)
                    │
                    ▼
                 answer
```

### Components

| File | Role |
|---|---|
| `settings.yaml` | GraphRAG configuration — models, storage, workflows |
| `src/graph_rag/ingest.py` | Converts `corpus.json` → `input/*.txt` + writes default prompts |
| `src/graph_rag/pipeline.py` | `GraphRAGPipeline` — wraps `graphrag.api.build_index` and `local_search` |
| `scripts/build_graph.py` | Entry point: ingest + index + print stats |
| `scripts/visualize_graph.py` | Loads parquet output into NetworkX, renders `output/graph.png` |
| `tests/test_pipeline.py` | 7 integration tests (session-scoped fixture, real API calls) |

### GraphRAG Internals

- **Indexing** (`graphrag.api.build_index`): runs the full pipeline — chunk text, extract entities/relationships via LLM, cluster into communities, generate community reports, embed text units into LanceDB.
- **Local search** (`graphrag.api.local_search`): finds relevant entities via vector similarity, expands to community context, generates answer with the completion model.
- **Output tables** (Parquet in `output/`):
  - `create_final_entities.parquet`
  - `create_final_relationships.parquet`
  - `create_final_communities.parquet`
  - `create_final_community_reports.parquet`
  - `create_final_text_units.parquet`

## Configuration

### Environment Variables

| Variable | Purpose |
|---|---|
| `OPENROUTER_API_KEY` | API key for OpenRouter (set in `~/.zshrc`) |

Set in `~/.zshrc`:
```bash
export OPENROUTER_API_KEY="sk-or-..."
```

### Model Selection (`settings.yaml`)

| Role | Model |
|---|---|
| Completion (extraction, summarization, answering) | `anthropic/claude-haiku-4-5` via OpenRouter |
| Embeddings (vector store) | `openai/text-embedding-3-small` via OpenRouter |

To change models, edit `settings.yaml`:
```yaml
completion_models:
  default_completion_model:
    model: anthropic/claude-sonnet-4-5   # upgrade for better quality

embedding_models:
  default_embedding_model:
    model: openai/text-embedding-3-large  # larger embeddings
```

## Design Decisions

- **Microsoft GraphRAG over custom NetworkX**: GraphRAG provides battle-tested entity extraction prompts, hierarchical community clustering (Leiden algorithm), and both local and global search modes out of the box. NetworkX is still used only for visualization.
- **OpenRouter as backend**: OpenRouter provides a unified OpenAI-compatible API for Anthropic models, enabling Claude to be used without a direct Anthropic API key.
- **LiteLLM (internal)**: GraphRAG uses LiteLLM internally for all model calls, which handles retry, rate limiting, and provider routing.
- **Local search for queries**: Local search retrieves entity-level context + community summaries, well-suited for specific IT troubleshooting questions. Global search (community-level synthesis) is available for broader questions.
- **LanceDB for vector store**: Embedded vector database, no external service required.
- **Session-scoped test fixture**: Building the index is expensive (~5-10 min), so all 7 tests share a single indexed graph.

## Success Criteria

The project is considered complete when all of the following are true:

### Setup
- [ ] `uv sync --extra dev` completes without errors
- [ ] `OPENROUTER_API_KEY` is set in the environment

### Indexing
- [ ] `uv run python scripts/build_graph.py` runs to completion
- [ ] All 5 output parquet files exist in `output/`:
  - `entities.parquet` (non-empty)
  - `relationships.parquet` (non-empty)
  - `communities.parquet` (non-empty)
  - `community_reports.parquet` (non-empty)
  - `text_units.parquet` (non-empty)
- [ ] `output/lancedb/` exists and contains vector embeddings

### Search
- [ ] `GraphRAGPipeline.search("...")` returns a non-empty string answer
- [ ] Answers to IT questions (BSOD, DHCP, Kubernetes) contain domain-relevant keywords

### Visualization
- [ ] `uv run python scripts/visualize_graph.py` produces `output/graph.png`
- [ ] The PNG shows labeled nodes colored by entity type

### Tests
- [ ] `uv run pytest tests/ -v` passes all 7 tests with 0 failures
