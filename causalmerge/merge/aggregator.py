"""Edge aggregator — collect edges from all source graphs.

The aggregator is the first stage of the merge pipeline.  It iterates every
source graph, converts each raw edge dict into a ``SourceEdge``, and groups
the results by their directed ``(cause, effect)`` key.

The output of this stage is consumed by the conflict detector (which checks
for opposing directions) and the consensus engine (which computes weighted
confidence scores).
"""

from __future__ import annotations

import logging

from causalmerge.data.schema import SourceEdge, SourceGraph

logger = logging.getLogger("causalmerge.aggregator")


def aggregate_edges(
    source_graphs: list[SourceGraph],
) -> dict[tuple[str, str], list[SourceEdge]]:
    """Collect all edges from all sources, keyed by ``(cause, effect)`` pair.

    Node names are normalised to lowercase so that ``"Smoking"`` and
    ``"smoking"`` are treated as the same node across different source graphs.

    Duplicate edges from the *same* source (same cause, effect, and source
    name) are kept so that confidence averaging reflects actual occurrences.
    Edges from *different* sources that share a direction are stored together
    under the same key; the resolver later decides which direction wins when
    the reverse direction also exists.

    Parameters
    ----------
    source_graphs:
        List of source graphs returned by the loader.

    Returns
    -------
    dict[tuple[str, str], list[SourceEdge]]
        Mapping of each directed edge ``(cause, effect)`` to the list of
        ``SourceEdge`` objects from every source that mentions this edge.
    """
    aggregated: dict[tuple[str, str], list[SourceEdge]] = {}

    for graph in source_graphs:
        logger.debug(
            "Aggregating %d edge(s) from source %r (weight=%.2f)",
            len(graph.edges),
            graph.source_name,
            graph.source_weight,
        )
        for raw in graph.edges:
            cause = str(raw["cause"]).strip().lower()
            effect = str(raw["effect"]).strip().lower()
            confidence = float(raw["confidence"])
            evidence = str(raw.get("evidence", ""))
            edge_type = str(raw.get("edge_type", "direct"))

            edge = SourceEdge(
                cause=cause,
                effect=effect,
                confidence=confidence,
                source_name=graph.source_name,
                source_weight=graph.source_weight,
                evidence=evidence,
                edge_type=edge_type,
            )

            key = (cause, effect)
            if key not in aggregated:
                aggregated[key] = []
            aggregated[key].append(edge)

    total_unique = len(aggregated)
    total_instances = sum(len(v) for v in aggregated.values())
    logger.info(
        "Aggregated %d edge instance(s) across %d unique directed edge(s) from %d source(s)",
        total_instances,
        total_unique,
        len(source_graphs),
    )
    return aggregated
