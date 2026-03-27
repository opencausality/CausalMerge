"""DAG enforcer — break cycles in the merged graph.

After conflict resolution, the merged graph should usually be acyclic.
However, indirect cycles can still arise when edges from different sources
form a loop that no single conflict pair captures.  For example:

    A → B (source 1), B → C (source 2), C → A (source 3)

None of these pairs individually form a conflict, but together they create
a cycle.  The enforcer detects and breaks all such cycles by iteratively
removing the lowest-confidence edge from each cycle until the graph is a DAG.
"""

from __future__ import annotations

import logging

import networkx as nx

from causalmerge.data.schema import MergedEdge
from causalmerge.exceptions import CycleBreakingError

logger = logging.getLogger("causalmerge.dag_enforcer")

# Maximum number of cycle-breaking iterations to prevent infinite loops
# in the unlikely event of a bug in the cycle detection logic.
_MAX_ITERATIONS = 1000


def enforce_dag(
    graph: nx.DiGraph,
    merged_edges: list[MergedEdge],
) -> tuple[nx.DiGraph, list[tuple[str, str]]]:
    """Break all cycles by removing the edge with the lowest merged confidence.

    The algorithm:
    1. Call ``networkx.find_cycle`` to locate any cycle.
    2. Among the edges in the cycle, find the one with the lowest
       ``merged_confidence`` in ``merged_edges``.
    3. Remove that edge from the graph.
    4. Repeat until ``networkx.find_cycle`` raises ``NetworkXNoCycle``.

    Parameters
    ----------
    graph:
        A ``networkx.DiGraph`` that may contain cycles.
    merged_edges:
        List of ``MergedEdge`` objects — used to look up confidence scores
        for edges in detected cycles.

    Returns
    -------
    tuple[nx.DiGraph, list[tuple[str, str]]]
        - The acyclic graph (same object mutated in-place).
        - A list of ``(cause, effect)`` tuples for every edge removed.

    Raises
    ------
    CycleBreakingError
        If the enforcer cannot produce an acyclic graph within the maximum
        number of iterations (indicates a logic error, not user data).
    """
    # Build a quick-lookup dict: (cause, effect) → merged_confidence
    confidence_map: dict[tuple[str, str], float] = {
        (e.cause, e.effect): e.merged_confidence for e in merged_edges
    }

    removed: list[tuple[str, str]] = []
    iterations = 0

    while True:
        try:
            cycle = nx.find_cycle(graph, orientation="original")
        except nx.NetworkXNoCycle:
            break

        iterations += 1
        if iterations > _MAX_ITERATIONS:
            raise CycleBreakingError(
                f"DAG enforcer exceeded {_MAX_ITERATIONS} iterations. "
                "This indicates a bug in the cycle-breaking logic."
            )

        # Extract edge tuples from the cycle; each element is (u, v, direction)
        cycle_edges: list[tuple[str, str]] = [(u, v) for u, v, *_ in cycle]

        # Pick the edge with the lowest confidence
        def _conf(edge: tuple[str, str]) -> float:
            return confidence_map.get(edge, 0.0)

        weakest = min(cycle_edges, key=_conf)
        graph.remove_edge(*weakest)
        removed.append(weakest)

        logger.info(
            "Cycle broken: removed edge (%s → %s) [conf=%.3f]",
            weakest[0],
            weakest[1],
            _conf(weakest),
        )

    if removed:
        logger.info("DAG enforcement complete: %d edge(s) removed to break cycles", len(removed))
    else:
        logger.debug("Graph is already a DAG — no cycle-breaking needed")

    # Final sanity check
    if not nx.is_directed_acyclic_graph(graph):
        raise CycleBreakingError(
            "DAG enforcement finished but graph is still not acyclic. "
            "This should never happen."
        )

    return graph, removed
