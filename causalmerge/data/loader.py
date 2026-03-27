"""Graph loader — read causal graphs from JSON files.

Supports WhyNet's output format:
    {
        "nodes": ["A", "B", ...],
        "edges": [
            {"cause": "A", "effect": "B", "confidence": 0.8, "evidence": "...", "edge_type": "direct"},
            ...
        ],
        ...
    }

The ``source_name`` parameter is used to label edges during aggregation so
that conflicts can be attributed back to specific input files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from causalmerge.data.schema import SourceGraph
from causalmerge.exceptions import GraphLoadError

logger = logging.getLogger("causalmerge.loader")

# Required keys in every edge dict
_REQUIRED_EDGE_KEYS: frozenset[str] = frozenset({"cause", "effect", "confidence"})


def load_graph(path: Path, source_name: str, weight: float = 1.0) -> SourceGraph:
    """Load a single causal graph from a JSON file.

    Parameters
    ----------
    path:
        Filesystem path to the JSON graph file.
    source_name:
        Human-readable label for this source (e.g. ``"incident_report_A"``).
        Used to attribute edges during conflict detection.
    weight:
        Relative importance of this source (0.0–1.0).  Defaults to 1.0.
        Lower weights reduce this source's influence on merged confidence.

    Returns
    -------
    SourceGraph
        Validated source graph ready for aggregation.

    Raises
    ------
    GraphLoadError
        If the file does not exist, is not valid JSON, is missing required
        keys, or contains edges that fail schema validation.
    """
    if not path.exists():
        raise GraphLoadError(f"Graph file not found: {path}")
    if not path.is_file():
        raise GraphLoadError(f"Path is not a file: {path}")

    logger.debug("Loading graph from %s (source=%r, weight=%.2f)", path, source_name, weight)

    try:
        raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise GraphLoadError(f"Invalid JSON in {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise GraphLoadError(f"Expected a JSON object at the top level in {path}, got {type(raw).__name__}")

    if "edges" not in raw:
        raise GraphLoadError(f"Graph file {path} is missing required key 'edges'")

    edges: list[Any] = raw["edges"]
    if not isinstance(edges, list):
        raise GraphLoadError(f"'edges' in {path} must be a JSON array, got {type(edges).__name__}")

    _validate_edges(edges, path)

    nodes: list[str] = raw.get("nodes", [])
    if not isinstance(nodes, list):
        raise GraphLoadError(f"'nodes' in {path} must be a JSON array")

    # Collect extra metadata (everything except nodes/edges)
    metadata: dict[str, Any] = {k: v for k, v in raw.items() if k not in ("nodes", "edges")}

    graph = SourceGraph(
        source_name=source_name,
        source_weight=weight,
        nodes=[str(n) for n in nodes],
        edges=edges,
        metadata=metadata,
    )

    logger.info(
        "Loaded %d edge(s) from %s [source=%r]",
        len(graph.edges),
        path.name,
        source_name,
    )
    return graph


def load_graphs(paths: list[Path], weights: list[float] | None = None) -> list[SourceGraph]:
    """Load multiple causal graphs from a list of JSON files.

    Parameters
    ----------
    paths:
        Ordered list of paths to graph JSON files.
    weights:
        Optional list of source weights, one per path.  If omitted, all
        sources receive weight 1.0.  If provided, must be the same length
        as ``paths``.

    Returns
    -------
    list[SourceGraph]
        Loaded and validated source graphs in the same order as ``paths``.

    Raises
    ------
    GraphLoadError
        If ``weights`` is provided but has a different length than ``paths``,
        or if any individual graph fails to load.
    ValueError
        If ``paths`` is empty.
    """
    if not paths:
        raise ValueError("At least one graph path must be provided")

    if weights is not None and len(weights) != len(paths):
        raise GraphLoadError(
            f"Number of weights ({len(weights)}) must match number of paths ({len(paths)})"
        )

    resolved_weights: list[float] = weights if weights is not None else [1.0] * len(paths)

    graphs: list[SourceGraph] = []
    for path, weight in zip(paths, resolved_weights, strict=True):
        source_name = path.stem  # use filename without extension as default name
        graphs.append(load_graph(path, source_name=source_name, weight=weight))

    logger.info("Loaded %d source graph(s) total", len(graphs))
    return graphs


# ── Internal helpers ──────────────────────────────────────────────────────────


def _validate_edges(edges: list[Any], path: Path) -> None:
    """Validate that each edge dict contains the required keys.

    Raises
    ------
    GraphLoadError
        On the first invalid edge encountered.
    """
    for idx, edge in enumerate(edges):
        if not isinstance(edge, dict):
            raise GraphLoadError(
                f"Edge #{idx} in {path} must be a JSON object, got {type(edge).__name__}"
            )
        missing = _REQUIRED_EDGE_KEYS - edge.keys()
        if missing:
            raise GraphLoadError(
                f"Edge #{idx} in {path} is missing required keys: {sorted(missing)}"
            )
        confidence = edge["confidence"]
        if not isinstance(confidence, (int, float)):
            raise GraphLoadError(
                f"Edge #{idx} in {path}: 'confidence' must be a number, got {type(confidence).__name__}"
            )
        if not (0.0 <= float(confidence) <= 1.0):
            raise GraphLoadError(
                f"Edge #{idx} in {path}: 'confidence' must be between 0.0 and 1.0, got {confidence}"
            )
