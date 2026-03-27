"""Example: Merging three incident report graphs with CausalMerge.

This example demonstrates the full CausalMerge pipeline:
1. Loading three causal graphs extracted from different incident reports
2. Detecting directional conflicts between the sources
3. Resolving conflicts via confidence-weighted voting
4. Producing a unified, acyclic causal graph
5. Printing the merge report to the console
6. Saving the merged graph to JSON

Run from the project root:
    python examples/merge_incident_graphs.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from causalmerge.config import Settings, configure_logging
from causalmerge.data.loader import load_graphs
from causalmerge.graph.builder import save_graph
from causalmerge.merge.engine import MergeEngine
from causalmerge.reporting.report import print_merge_report, save_merge_report
from causalmerge.resolution.conflicts import detect_conflicts
from causalmerge.merge.aggregator import aggregate_edges

# ── Configuration ─────────────────────────────────────────────────────────────

FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures"
GRAPH_PATHS = [
    FIXTURES / "graph_a.json",
    FIXTURES / "graph_b.json",
    FIXTURES / "graph_c.json",
]

# Assign different weights to each source to reflect credibility
# graph_a (database team) is most authoritative; graph_b is the lowest
SOURCE_WEIGHTS = [1.0, 0.7, 0.9]

OUTPUT_GRAPH = Path("merged_incident_graph.json")
OUTPUT_REPORT = Path("merge_report.json")

logger = logging.getLogger("examples.merge_incident_graphs")


def main() -> None:
    """Run the full merge example."""
    # Set up logging
    settings = Settings(confidence_threshold=0.4, log_level="INFO")
    configure_logging(settings)

    logger.info("CausalMerge — Incident Graph Fusion Example")
    logger.info("=" * 60)

    # ── Step 1: Load source graphs ────────────────────────────────────────────
    logger.info("Step 1: Loading %d source graphs…", len(GRAPH_PATHS))
    source_graphs = load_graphs(GRAPH_PATHS, weights=SOURCE_WEIGHTS)
    for graph in source_graphs:
        logger.info(
            "  Loaded '%s' — %d edge(s), weight=%.1f",
            graph.source_name,
            len(graph.edges),
            graph.source_weight,
        )

    # ── Step 2: Preview conflicts before merging ──────────────────────────────
    logger.info("\nStep 2: Detecting conflicts before merge…")
    aggregated = aggregate_edges(source_graphs)
    pre_conflicts = detect_conflicts(aggregated)

    if pre_conflicts:
        logger.info("  Found %d directional conflict(s):", len(pre_conflicts))
        for conflict in pre_conflicts:
            logger.info(
                "    (%s → %s) [%.3f from %s]  vs  (%s → %s) [%.3f from %s]",
                conflict.cause, conflict.effect, conflict.confidence_direction_a,
                conflict.sources_for_direction_a,
                conflict.effect, conflict.cause, conflict.confidence_direction_b,
                conflict.sources_for_direction_b,
            )
    else:
        logger.info("  No conflicts detected.")

    # ── Step 3: Run merge pipeline ────────────────────────────────────────────
    logger.info("\nStep 3: Running merge pipeline…")
    engine = MergeEngine(settings=settings)
    merged_graph, report = engine.merge(source_graphs)

    # ── Step 4: Print report ──────────────────────────────────────────────────
    logger.info("\nStep 4: Merge report")
    print_merge_report(report)

    # ── Step 5: Analyse the merged graph ──────────────────────────────────────
    logger.info("\nStep 5: Analysing merged graph…")
    logger.info("  Nodes in merged graph: %d", merged_graph.number_of_nodes())
    logger.info("  Edges in merged graph: %d", merged_graph.number_of_edges())

    # Find the most central node by in-degree (most things cause this node)
    if merged_graph.number_of_nodes() > 0:
        import networkx as nx
        in_degrees = dict(merged_graph.in_degree())
        most_caused = max(in_degrees, key=lambda n: in_degrees[n])
        logger.info(
            "  Most-caused node: '%s' (in-degree %d)",
            most_caused,
            in_degrees[most_caused],
        )

        # Find root causes (nodes with no incoming edges)
        roots = [n for n in merged_graph.nodes() if merged_graph.in_degree(n) == 0]
        logger.info("  Root causes (no incoming edges): %s", roots)

        # Find leaf effects (nodes with no outgoing edges)
        leaves = [n for n in merged_graph.nodes() if merged_graph.out_degree(n) == 0]
        logger.info("  Terminal effects (no outgoing edges): %s", leaves)

        # Topological sort — causal ordering
        topo_order = list(nx.topological_sort(merged_graph))
        logger.info("  Causal ordering (topological): %s", " → ".join(topo_order))

    # ── Step 6: Save outputs ──────────────────────────────────────────────────
    logger.info("\nStep 6: Saving outputs…")
    all_edges = report.consensus_edges + report.disputed_edges
    save_graph(merged_graph, all_edges, OUTPUT_GRAPH)
    logger.info("  Merged graph saved to: %s", OUTPUT_GRAPH)

    save_merge_report(report, OUTPUT_REPORT)
    logger.info("  Merge report saved to: %s", OUTPUT_REPORT)

    logger.info("\nDone. Load merged_incident_graph.json into WhyNet or another")
    logger.info("CausalMerge run as a source graph for further fusion.")


if __name__ == "__main__":
    main()
