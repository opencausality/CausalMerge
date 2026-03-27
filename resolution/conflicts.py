"""Conflict detector — find directional disagreements between source graphs.

A conflict exists when:
- Source graph A contains the edge  X → Y
- Source graph B contains the edge  Y → X  (reversed)

Both directions existing in the same merge is structurally ambiguous and
must be resolved before building the final DAG.  This module identifies all
such conflicts and packages them as ``EdgeConflict`` objects for the resolver.
"""

from __future__ import annotations

import logging

from causalmerge.data.schema import EdgeConflict, SourceEdge

logger = logging.getLogger("causalmerge.conflicts")


def detect_conflicts(
    aggregated: dict[tuple[str, str], list[SourceEdge]],
) -> list[EdgeConflict]:
    """Find all pairs where A→B and B→A both exist in the aggregated map.

    Each conflict is reported once (not twice) — the canonical key is
    always the lexicographically smaller ``(cause, effect)`` pair so that
    ``("a", "b")`` and ``("b", "a")`` are not double-counted.

    Parameters
    ----------
    aggregated:
        Output of ``aggregate_edges`` — maps each directed edge to the list
        of ``SourceEdge`` objects that express it.

    Returns
    -------
    list[EdgeConflict]
        All detected conflicts, each with both directions populated.
        Resolution fields are left as empty strings; they are filled in by
        ``resolve_conflict``.
    """
    conflicts: list[EdgeConflict] = []
    seen: set[tuple[str, str]] = set()

    for (cause, effect), edges_a in aggregated.items():
        reverse_key = (effect, cause)
        if reverse_key not in aggregated:
            continue

        # Deduplicate: process the conflict only once
        canonical = min((cause, effect), reverse_key)
        if canonical in seen:
            continue
        seen.add(canonical)

        edges_b = aggregated[reverse_key]

        # Gather source names for each direction
        sources_a = sorted({e.source_name for e in edges_a})
        sources_b = sorted({e.source_name for e in edges_b})

        # Sum confidence weighted by source weight for each direction
        conf_a = _weighted_confidence_sum(edges_a)
        conf_b = _weighted_confidence_sum(edges_b)

        conflict = EdgeConflict(
            cause=cause,
            effect=effect,
            sources_for_direction_a=sources_a,
            confidence_direction_a=round(conf_a, 4),
            sources_for_direction_b=sources_b,
            confidence_direction_b=round(conf_b, 4),
            resolution="",           # filled in by resolver
            resolution_confidence=0.0,
        )
        conflicts.append(conflict)
        logger.debug(
            "Conflict: (%s → %s) [conf=%.3f, sources=%s] vs (%s → %s) [conf=%.3f, sources=%s]",
            cause, effect, conf_a, sources_a,
            effect, cause, conf_b, sources_b,
        )

    if conflicts:
        logger.info("Found %d directional conflict(s)", len(conflicts))
    else:
        logger.debug("No directional conflicts found")

    return conflicts


# ── Internal helpers ──────────────────────────────────────────────────────────


def _weighted_confidence_sum(edges: list[SourceEdge]) -> float:
    """Return the sum of source-weight-adjusted confidences for a list of edges.

    This gives higher-weight sources a proportionally larger say in which
    direction "wins" a conflict vote.
    """
    return sum(e.confidence * e.source_weight for e in edges)
