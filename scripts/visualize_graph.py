#!/usr/bin/env python3
"""Visualize the GraphRAG knowledge graph as a PNG image using NetworkX."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "output" / "corpus"


def load_graph() -> nx.Graph:
    """Load entities and relationships from parquet files into a NetworkX graph."""
    entities_path = OUTPUT_DIR / "entities.parquet"
    relationships_path = OUTPUT_DIR / "relationships.parquet"

    if not entities_path.exists() or not relationships_path.exists():
        print("Error: Output parquet files not found.")
        print("Run 'uv run python scripts/build_graph.py' first.")
        sys.exit(1)

    entities_df = pd.read_parquet(entities_path)
    relationships_df = pd.read_parquet(relationships_path)

    G = nx.Graph()

    # Add entity nodes
    for _, row in entities_df.iterrows():
        node_id = str(row.get("title", row.get("name", row.get("id", ""))))
        entity_type = str(row.get("type", "unknown"))
        G.add_node(node_id, entity_type=entity_type)

    # Add relationship edges
    source_col = "source" if "source" in relationships_df.columns else "source_id"
    target_col = "target" if "target" in relationships_df.columns else "target_id"

    for _, row in relationships_df.iterrows():
        src = str(row.get(source_col, ""))
        tgt = str(row.get(target_col, ""))
        weight = float(row.get("weight", 1.0))
        description = str(row.get("description", ""))
        if src and tgt:
            G.add_edge(src, tgt, weight=weight, description=description)

    return G


def visualize(G: nx.Graph, output_path: Path) -> None:
    """Render the graph as a PNG."""
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Use spring layout; limit to largest connected component for clarity
    if G.number_of_nodes() > 100:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        print(f"Showing largest connected component: {G.number_of_nodes()} nodes")

    # Color nodes by entity type
    type_colors = {
        "problem": "#e74c3c",
        "solution": "#2ecc71",
        "component": "#3498db",
        "error": "#e67e22",
        "system": "#9b59b6",
        "technology": "#1abc9c",
        "procedure": "#f39c12",
        "unknown": "#95a5a6",
    }

    node_colors = [
        type_colors.get(G.nodes[n].get("entity_type", "unknown").lower(), "#95a5a6")
        for n in G.nodes()
    ]

    fig, ax = plt.subplots(figsize=(20, 16))
    pos = nx.spring_layout(G, seed=42, k=2.0 / (G.number_of_nodes() ** 0.5))

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=300, alpha=0.85, ax=ax)
    nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color="#7f8c8d", ax=ax)

    # Only show labels for high-degree nodes to reduce clutter
    degree_threshold = max(2, G.number_of_nodes() // 20)
    labels = {n: n for n in G.nodes() if G.degree(n) >= degree_threshold}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=7, ax=ax)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color, label=etype.capitalize())
        for etype, color in type_colors.items()
        if any(G.nodes[n].get("entity_type", "unknown").lower() == etype for n in G.nodes())
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)

    ax.set_title("IT Troubleshooting Knowledge Graph", fontsize=16, fontweight="bold")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Graph saved to: {output_path}")


def main() -> None:
    G = load_graph()
    output_path = OUTPUT_DIR / "graph.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    visualize(G, output_path)


if __name__ == "__main__":
    main()
