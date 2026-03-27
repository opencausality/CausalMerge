"""Graph builder — construct, save, and reload merged causal graphs.

The builder converts the flat list of ``MergedEdge`` objects produced by the
consensus engine into a ``networkx.DiGraph``, applying the confidence threshold
to filter out weak edges.  It also handles serialisation to and deserialisation
from the WhyNet-compatible JSON format so that merged graphs can be fed back
into WhyNet or stored for later analysis.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx

from causalmerge import __version__
from causalmerge.data.schema import MergedEdge
from causalmerge.exceptions import GraphLoadError

logger = logging.getLogger("causalmerge.builder")


def build_graph(
    merged_edges: list[MergedEdge],
    confidence_threshold: float = 0.4,
) -> nx.DiGraph:
    """Build a ``networkx.DiGraph`` from merged edges, filtered by confidence.

    Edges with ``merged_confidence`` strictly below ``confidence_threshold``
    are excluded.  Node names are taken directly from the ``MergedEdge``
    objects (already normalised to lowercase by the aggregator).

    Parameters
    ----------
    merged_edges:
        List of ``MergedEdge`` objects produced by the consensus engine.
    confidence_threshold:
        Minimum ``merged_confidence`` required to include an edge.

    Returns
    -------
    nx.DiGraph
        A directed graph with edge attributes:
        - ``merged_confidence`` (float)
        - ``source_agreement`` (float)
        - ``contributing_sources`` (list[str])
        - ``is_disputed`` (bool)
        - ``edge_type`` (str)
    """
    graph = nx.DiGraph()

    included = 0
    skipped = 0
    for edge in merged_edges:
        if edge.merged_confidence < confidence_threshold:
            logger.debug(
                "Skipping (%s → %s): confidence %.3f < threshold %.3f",
                edge.cause,
                edge.effect,
                edge.merged_confidence,
                confidence_threshold,
            )
            skipped += 1
            continue

        graph.add_edge(
            edge.cause,
            edge.effect,
            merged_confidence=edge.merged_confidence,
            source_agreement=edge.source_agreement,
            contributing_sources=edge.contributing_sources,
            is_disputed=edge.is_disputed,
            edge_type=edge.edge_type,
        )
        included += 1

    logger.info(
        "Built graph: %d node(s), %d edge(s) included, %d skipped by threshold %.2f",
        graph.number_of_nodes(),
        included,
        skipped,
        confidence_threshold,
    )
    return graph


def save_graph(graph: nx.DiGraph, merged_edges: list[MergedEdge], path: Path) -> None:
    """Save the merged graph to JSON in WhyNet-compatible format.

    The saved file can be loaded by WhyNet's graph tools or fed back into
    CausalMerge as one source in a future merge operation.

    Parameters
    ----------
    graph:
        The merged ``DiGraph`` to serialise.
    merged_edges:
        Full list of ``MergedEdge`` objects (including those filtered out by
        the threshold) so the file contains complete provenance information.
    path:
        Destination file path (will be created or overwritten).
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    nodes = list(graph.nodes())
    edges: list[dict[str, Any]] = []

    # Build a quick lookup by (cause, effect)
    edge_meta: dict[tuple[str, str], MergedEdge] = {
        (e.cause, e.effect): e for e in merged_edges
    }

    for u, v, data in graph.edges(data=True):
        meta = edge_meta.get((u, v))
        edges.append(
            {
                "cause": u,
                "effect": v,
                "confidence": data.get("merged_confidence", 0.0),
                "source_agreement": data.get("source_agreement", 0.0),
                "contributing_sources": data.get("contributing_sources", []),
                "is_disputed": data.get("is_disputed", False),
                "evidence": "",
                "edge_type": data.get("edge_type", "direct"),
            }
        )
        _ = meta  # meta available for future extension

    payload: dict[str, Any] = {
        "nodes": nodes,
        "edges": edges,
        "source_text": "merged causal graph",
        "model_used": f"causalmerge/{__version__}",
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "merged_edge_count": len(edges),
        "node_count": len(nodes),
    }

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved merged graph to %s (%d nodes, %d edges)", path, len(nodes), len(edges))


def load_merged_graph(path: Path) -> tuple[nx.DiGraph, list[MergedEdge]]:
    """Load a previously saved merged graph from JSON.

    Parameters
    ----------
    path:
        Path to a JSON file previously written by ``save_graph``.

    Returns
    -------
    tuple[nx.DiGraph, list[MergedEdge]]
        - Reconstructed ``DiGraph``.
        - List of ``MergedEdge`` objects reconstructed from the JSON.

    Raises
    ------
    GraphLoadError
        If the file is missing, invalid JSON, or has an incompatible schema.
    """
    if not path.exists():
        raise GraphLoadError(f"Merged graph file not found: {path}")

    try:
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GraphLoadError(f"Invalid JSON in {path}: {exc}") from exc

    if "edges" not in raw:
        raise GraphLoadError(f"Merged graph file {path} is missing 'edges' key")

    graph = nx.DiGraph()
    merged_edges: list[MergedEdge] = []

    for edge_dict in raw["edges"]:
        try:
            cause = str(edge_dict["cause"])
            effect = str(edge_dict["effect"])
            merged_confidence = float(edge_dict.get("confidence", 0.0))
            source_agreement = float(edge_dict.get("source_agreement", 0.0))
            contributing_sources = list(edge_dict.get("contributing_sources", []))
            is_disputed = bool(edge_dict.get("is_disputed", False))
            edge_type = str(edge_dict.get("edge_type", "direct"))
        except (KeyError, TypeError, ValueError) as exc:
            raise GraphLoadError(f"Malformed edge in {path}: {exc}") from exc

        graph.add_edge(
            cause,
            effect,
            merged_confidence=merged_confidence,
            source_agreement=source_agreement,
            contributing_sources=contributing_sources,
            is_disputed=is_disputed,
            edge_type=edge_type,
        )
        merged_edges.append(
            MergedEdge(
                cause=cause,
                effect=effect,
                merged_confidence=merged_confidence,
                source_agreement=source_agreement,
                contributing_sources=contributing_sources,
                is_disputed=is_disputed,
                edge_type=edge_type,
            )
        )

    logger.info(
        "Loaded merged graph from %s (%d nodes, %d edges)",
        path,
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    return graph, merged_edges
