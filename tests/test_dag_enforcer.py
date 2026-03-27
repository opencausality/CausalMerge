"""Tests for causalmerge.resolution.dag_enforcer."""

from __future__ import annotations

import networkx as nx
import pytest

from causalmerge.data.schema import MergedEdge
from causalmerge.exceptions import CycleBreakingError
from causalmerge.resolution.dag_enforcer import enforce_dag


def _make_merged_edge(cause: str, effect: str, conf: float) -> MergedEdge:
    """Helper to build a MergedEdge for testing."""
    return MergedEdge(
        cause=cause,
        effect=effect,
        merged_confidence=conf,
        source_agreement=1.0,
        contributing_sources=["test"],
    )


def _build_graph(edges: list[tuple[str, str]]) -> nx.DiGraph:
    """Build a DiGraph from a list of (cause, effect) tuples."""
    g = nx.DiGraph()
    g.add_edges_from(edges)
    return g


# ── Already acyclic ───────────────────────────────────────────────────────────


def test_acyclic_graph_unchanged() -> None:
    """A DAG should pass through with zero edges removed."""
    g = _build_graph([("a", "b"), ("b", "c")])
    edges = [
        _make_merged_edge("a", "b", 0.9),
        _make_merged_edge("b", "c", 0.7),
    ]
    result, removed = enforce_dag(g, edges)
    assert removed == []
    assert nx.is_directed_acyclic_graph(result)


def test_acyclic_graph_node_edge_counts() -> None:
    """Node and edge counts should be unchanged for an already-DAG."""
    g = _build_graph([("a", "b"), ("b", "c"), ("a", "c")])
    edges = [
        _make_merged_edge("a", "b", 0.9),
        _make_merged_edge("b", "c", 0.7),
        _make_merged_edge("a", "c", 0.5),
    ]
    result, removed = enforce_dag(g, edges)
    assert result.number_of_edges() == 3
    assert removed == []


# ── Simple cycle ──────────────────────────────────────────────────────────────


def test_simple_cycle_broken() -> None:
    """A two-node cycle (a→b→a) should be broken by removing one edge."""
    g = _build_graph([("a", "b"), ("b", "a")])
    edges = [
        _make_merged_edge("a", "b", 0.8),
        _make_merged_edge("b", "a", 0.3),
    ]
    result, removed = enforce_dag(g, edges)
    assert nx.is_directed_acyclic_graph(result)
    assert len(removed) == 1


def test_simple_cycle_removes_lowest_confidence_edge() -> None:
    """The weakest edge in the cycle should be the one removed."""
    g = _build_graph([("a", "b"), ("b", "a")])
    edges = [
        _make_merged_edge("a", "b", 0.9),
        _make_merged_edge("b", "a", 0.2),  # weaker
    ]
    _, removed = enforce_dag(g, edges)
    assert removed == [("b", "a")]


def test_three_node_cycle_broken() -> None:
    """A three-node cycle a→b→c→a should be broken."""
    g = _build_graph([("a", "b"), ("b", "c"), ("c", "a")])
    edges = [
        _make_merged_edge("a", "b", 0.9),
        _make_merged_edge("b", "c", 0.7),
        _make_merged_edge("c", "a", 0.4),  # weakest — should be removed
    ]
    result, removed = enforce_dag(g, edges)
    assert nx.is_directed_acyclic_graph(result)
    assert len(removed) == 1
    assert removed[0] == ("c", "a")


def test_three_node_cycle_removes_weakest() -> None:
    """Among three cycle edges, the one with lowest confidence is removed."""
    g = _build_graph([("a", "b"), ("b", "c"), ("c", "a")])
    edges = [
        _make_merged_edge("a", "b", 0.9),
        _make_merged_edge("b", "c", 0.5),  # weakest
        _make_merged_edge("c", "a", 0.7),
    ]
    _, removed = enforce_dag(g, edges)
    assert removed == [("b", "c")]


# ── Multiple cycles ───────────────────────────────────────────────────────────


def test_two_independent_cycles_both_broken() -> None:
    """Two independent cycles should each have their weakest edge removed."""
    # Cycle 1: a→b→a  |  Cycle 2: c→d→c
    g = _build_graph([("a", "b"), ("b", "a"), ("c", "d"), ("d", "c")])
    edges = [
        _make_merged_edge("a", "b", 0.9),
        _make_merged_edge("b", "a", 0.2),
        _make_merged_edge("c", "d", 0.8),
        _make_merged_edge("d", "c", 0.1),
    ]
    result, removed = enforce_dag(g, edges)
    assert nx.is_directed_acyclic_graph(result)
    assert len(removed) == 2


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_empty_graph_no_error() -> None:
    """An empty graph should return immediately with no edges removed."""
    g = nx.DiGraph()
    result, removed = enforce_dag(g, [])
    assert removed == []
    assert nx.is_directed_acyclic_graph(result)


def test_single_node_no_edges() -> None:
    """A single isolated node is trivially a DAG."""
    g = nx.DiGraph()
    g.add_node("solo")
    result, removed = enforce_dag(g, [])
    assert removed == []
    assert nx.is_directed_acyclic_graph(result)


def test_result_is_same_graph_object() -> None:
    """The returned graph should be the same object (modified in-place)."""
    g = _build_graph([("a", "b")])
    edges = [_make_merged_edge("a", "b", 0.9)]
    result, _ = enforce_dag(g, edges)
    assert result is g


def test_missing_edge_in_confidence_map_treated_as_zero() -> None:
    """An edge in the graph with no MergedEdge entry is treated as confidence 0."""
    g = _build_graph([("a", "b"), ("b", "a")])
    # Only provide confidence for one direction; the other defaults to 0
    edges = [_make_merged_edge("a", "b", 0.8)]
    result, removed = enforce_dag(g, edges)
    assert nx.is_directed_acyclic_graph(result)
    # The edge with no confidence entry (b→a, conf=0) should be the one removed
    assert ("b", "a") in removed
