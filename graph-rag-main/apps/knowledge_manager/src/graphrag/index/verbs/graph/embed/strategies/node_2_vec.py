"""A module containing run method definition."""

from typing import Any

import networkx as nx

from apps.knowledge_manager.src.graphrag.index.graph.embedding import embed_nod2vec
from apps.knowledge_manager.src.graphrag.index.graph.utils import (
    stable_largest_connected_component,
)
from apps.knowledge_manager.src.graphrag.index.verbs.graph.embed.typing import (
    NodeEmbeddings,
)


def run(graph: nx.Graph, args: dict[str, Any]) -> NodeEmbeddings:
    """Run method definition."""
    if args.get("use_lcc", True):
        graph = stable_largest_connected_component(graph)

    # create graph embedding using node2vec
    embeddings = embed_nod2vec(
        graph=graph,
        dimensions=args.get("dimensions", 1536),
        num_walks=args.get("num_walks", 10),
        walk_length=args.get("walk_length", 40),
        window_size=args.get("window_size", 2),
        iterations=args.get("iterations", 3),
        random_seed=args.get("random_seed", 86),
    )

    pairs = zip(embeddings.nodes, embeddings.embeddings.tolist(), strict=True)
    sorted_pairs = sorted(pairs, key=lambda x: x[0])

    return dict(sorted_pairs)
