"""Consensus engine — compute merged confidence and agreement scores.

After conflict resolution has decided which direction survives for each
disputed edge, this module computes the final ``MergedEdge`` for every
edge that passes the confidence threshold.

Merged confidence is the weighted average of all source confidences:
    merged = Σ(confidence_i × weight_i) / Σ(weight_i)

Source agreement is the fraction of sources that mention this edge at all:
    agreement = |sources_with_edge| / |all_sources|
"""

from __future__ import annotations

import logging

from causalmerge.data.schema import MergedEdge, SourceEdge

logger = logging.getLogger("causalmerge.consensus")


def compute_consensus(
    edge_key: tuple[str, str],
    source_edges: list[SourceEdge],
    all_source_names: list[str],
) -> MergedEdge:
    """Compute consensus for a single directed edge.

    Parameters
    ----------
    edge_key:
        The ``(cause, effect)`` tuple identifying this directed edge.
    source_edges:
        All ``SourceEdge`` instances that express this directed edge.
        Must not be empty.
    all_source_names:
        Names of *all* source graphs in the merge, used to compute the
        agreement fraction.

    Returns
    -------
    MergedEdge
        Consensus edge with weighted confidence and agreement score.

    Raises
    ------
    ValueError
        If ``source_edges`` is empty.
    """
    if not source_edges:
        raise ValueError(f"No source edges provided for edge key {edge_key!r}")

    cause, effect = edge_key

    # Weighted average confidence: Σ(conf * weight) / Σ(weight)
    weighted_sum = sum(e.confidence * e.source_weight for e in source_edges)
    weight_total = sum(e.source_weight for e in source_edges)
    merged_confidence = weighted_sum / weight_total if weight_total > 0 else 0.0

    # Agreement: fraction of all sources that include this edge
    contributing = list({e.source_name for e in source_edges})
    source_agreement = len(contributing) / len(all_source_names) if all_source_names else 0.0

    # Determine edge type — use the most common type among source edges
    type_counts: dict[str, int] = {}
    for e in source_edges:
        type_counts[e.edge_type] = type_counts.get(e.edge_type, 0) + 1
    dominant_type = max(type_counts, key=lambda t: type_counts[t])

    # Flag as disputed if agreement is below 100% (not all sources agree)
    is_disputed = source_agreement < 1.0

    logger.debug(
        "Consensus for (%s → %s): merged_conf=%.3f, agreement=%.2f, sources=%s",
        cause,
        effect,
        merged_confidence,
        source_agreement,
        contributing,
    )

    return MergedEdge(
        cause=cause,
        effect=effect,
        merged_confidence=round(merged_confidence, 4),
        source_agreement=round(source_agreement, 4),
        contributing_sources=sorted(contributing),
        is_disputed=is_disputed,
        edge_type=dominant_type,
    )


def compute_all_consensus(
    aggregated: dict[tuple[str, str], list[SourceEdge]],
    all_source_names: list[str],
) -> list[MergedEdge]:
    """Compute consensus for every edge in the aggregated mapping.

    This is a convenience wrapper around ``compute_consensus`` that
    processes the full aggregated dict returned by the aggregator.

    Parameters
    ----------
    aggregated:
        Output of ``aggregate_edges``.
    all_source_names:
        Names of all source graphs (for agreement fraction calculation).

    Returns
    -------
    list[MergedEdge]
        One ``MergedEdge`` per unique directed edge, sorted by
        ``merged_confidence`` descending.
    """
    merged: list[MergedEdge] = []
    for edge_key, source_edges in aggregated.items():
        merged.append(compute_consensus(edge_key, source_edges, all_source_names))

    # Sort highest confidence first for deterministic output
    merged.sort(key=lambda e: e.merged_confidence, reverse=True)
    return merged
