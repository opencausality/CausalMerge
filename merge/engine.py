"""MergeEngine — orchestrates the full graph merge pipeline.

Pipeline stages:
    1. aggregate_edges       — collect all directed edges from all sources
    2. detect_conflicts      — find A→B vs B→A disagreements
    3. resolve_conflicts     — pick a winner by confidence-weighted voting
    4. rewrite aggregated    — drop the losing direction from the aggregated map
    5. compute_all_consensus — compute weighted confidence + agreement per edge
    6. build_graph           — construct NetworkX DiGraph, filter by threshold
    7. enforce_dag           — break any remaining cycles
    8. assemble report       — build MergeReport with full provenance
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import networkx as nx

from causalmerge.config import Settings
from causalmerge.data.schema import MergedEdge, MergeReport, SourceEdge, SourceGraph
from causalmerge.exceptions import EmptyGraphError
from causalmerge.graph.builder import build_graph
from causalmerge.merge.aggregator import aggregate_edges
from causalmerge.merge.consensus import compute_all_consensus
from causalmerge.resolution.conflicts import detect_conflicts
from causalmerge.resolution.dag_enforcer import enforce_dag
from causalmerge.resolution.resolver import resolve_conflict

logger = logging.getLogger("causalmerge.engine")


class MergeEngine:
    """Orchestrates the complete graph merge pipeline.

    Parameters
    ----------
    settings:
        Application settings controlling confidence threshold and cycle
        resolution strategy.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def merge(
        self,
        source_graphs: list[SourceGraph],
    ) -> tuple[nx.DiGraph, MergeReport]:
        """Execute the full merge pipeline over the provided source graphs.

        Parameters
        ----------
        source_graphs:
            List of validated source graphs (from the loader).

        Returns
        -------
        tuple[nx.DiGraph, MergeReport]
            - The merged, acyclic directed graph.
            - A ``MergeReport`` containing full provenance information.

        Raises
        ------
        EmptyGraphError
            If no source graphs are provided, or all provided graphs have
            zero edges.
        """
        if not source_graphs:
            raise EmptyGraphError("No source graphs provided to merge")

        all_source_names = [g.source_name for g in source_graphs]
        source_weights = {g.source_name: g.source_weight for g in source_graphs}

        total_edges_before = sum(len(g.edges) for g in source_graphs)
        if total_edges_before == 0:
            raise EmptyGraphError("All provided source graphs have zero edges")

        logger.info(
            "Starting merge of %d source graph(s) with %d total edge(s)",
            len(source_graphs),
            total_edges_before,
        )

        # ── Stage 1: Aggregate ────────────────────────────────────────────
        aggregated: dict[tuple[str, str], list[SourceEdge]] = aggregate_edges(source_graphs)

        # ── Stage 2 & 3: Detect and resolve conflicts ─────────────────────
        conflicts = detect_conflicts(aggregated)
        logger.info("Detected %d directional conflict(s)", len(conflicts))

        resolved_conflicts = []
        for conflict in conflicts:
            resolved = resolve_conflict(conflict)
            resolved_conflicts.append(resolved)
            self._apply_conflict_resolution(aggregated, resolved)

        conflicts_resolved = sum(1 for c in resolved_conflicts if c.resolution != "DROPPED")
        dropped_from_conflicts: list[tuple[str, str]] = [
            (c.cause, c.effect)
            for c in resolved_conflicts
            if c.resolution == "DROPPED"
        ]

        # ── Stage 4: Compute consensus per edge ───────────────────────────
        merged_edges: list[MergedEdge] = compute_all_consensus(aggregated, all_source_names)

        # Mark edges that were involved in a conflict as disputed
        conflict_pairs: set[tuple[str, str]] = set()
        for c in resolved_conflicts:
            conflict_pairs.add((c.cause, c.effect))
            conflict_pairs.add((c.effect, c.cause))
        for edge in merged_edges:
            if (edge.cause, edge.effect) in conflict_pairs:
                edge.is_disputed = True

        # ── Stage 5: Filter by confidence threshold ───────────────────────
        threshold = self._settings.confidence_threshold
        passing = [e for e in merged_edges if e.merged_confidence >= threshold]
        dropped_by_threshold: list[tuple[str, str]] = [
            (e.cause, e.effect)
            for e in merged_edges
            if e.merged_confidence < threshold
        ]
        logger.info(
            "%d edge(s) passed confidence threshold %.2f (%d dropped)",
            len(passing),
            threshold,
            len(dropped_by_threshold),
        )

        # ── Stage 6: Build graph ──────────────────────────────────────────
        graph = build_graph(passing, confidence_threshold=threshold)

        # ── Stage 7: Enforce DAG ──────────────────────────────────────────
        graph, broken_cycle_edges = enforce_dag(graph, passing)
        logger.info("Broke %d cycle(s) during DAG enforcement", len(broken_cycle_edges))

        # Remove cycle-broken edges from passing list
        broken_set = set(broken_cycle_edges)
        final_edges = [e for e in passing if (e.cause, e.effect) not in broken_set]

        # Separate consensus and disputed edges for the report
        consensus_edges = [e for e in final_edges if not e.is_disputed]
        disputed_edges = [e for e in final_edges if e.is_disputed]

        all_dropped = (
            dropped_from_conflicts
            + dropped_by_threshold
            + list(broken_cycle_edges)
        )

        # ── Stage 8: Assemble report ──────────────────────────────────────
        report = MergeReport(
            sources_merged=all_source_names,
            source_weights=source_weights,
            total_edges_before=total_edges_before,
            total_edges_after=len(final_edges),
            conflicts_found=len(conflicts),
            conflicts_resolved=conflicts_resolved,
            cycles_broken=len(broken_cycle_edges),
            consensus_edges=consensus_edges,
            disputed_edges=disputed_edges,
            dropped_edges=all_dropped,
            created_at=datetime.now(tz=timezone.utc),
        )

        logger.info(
            "Merge complete: %d → %d edges | %d conflicts | %d cycles broken",
            total_edges_before,
            report.total_edges_after,
            report.conflicts_found,
            report.cycles_broken,
        )
        return graph, report

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _apply_conflict_resolution(
        self,
        aggregated: dict[tuple[str, str], list[SourceEdge]],
        resolved: "object",
    ) -> None:
        """Remove the losing direction from the aggregated map.

        After resolution, the direction that lost the vote is deleted from
        the aggregated dict so that downstream consensus computation only
        sees the winning direction.
        """
        from causalmerge.data.schema import EdgeConflict

        assert isinstance(resolved, EdgeConflict)

        if resolved.resolution == "DIRECTION_A":
            # Keep A→B, remove B→A
            losing_key = (resolved.effect, resolved.cause)
        elif resolved.resolution == "DIRECTION_B":
            # Keep B→A, remove A→B
            losing_key = (resolved.cause, resolved.effect)
        elif resolved.resolution == "DROPPED":
            # Remove both directions
            aggregated.pop((resolved.cause, resolved.effect), None)
            aggregated.pop((resolved.effect, resolved.cause), None)
            return
        else:
            logger.warning("Unknown conflict resolution %r — skipping", resolved.resolution)
            return

        aggregated.pop(losing_key, None)
        logger.debug(
            "Removed losing direction %s from aggregated map (resolution=%s)",
            losing_key,
            resolved.resolution,
        )
