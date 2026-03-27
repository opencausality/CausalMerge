"""Tests for causalmerge.merge.aggregator."""

from __future__ import annotations

from causalmerge.data.schema import SourceEdge, SourceGraph
from causalmerge.merge.aggregator import aggregate_edges


def _make_graph(name: str, edges: list[dict], weight: float = 1.0) -> SourceGraph:
    """Helper to create an in-memory SourceGraph."""
    nodes = list({e["cause"] for e in edges} | {e["effect"] for e in edges})
    return SourceGraph(
        source_name=name,
        source_weight=weight,
        nodes=nodes,
        edges=edges,
    )


# ── Basic aggregation ─────────────────────────────────────────────────────────


def test_aggregate_single_source_single_edge() -> None:
    """A single edge from a single source should produce one entry."""
    g = _make_graph("s1", [{"cause": "a", "effect": "b", "confidence": 0.8}])
    result = aggregate_edges([g])
    assert ("a", "b") in result
    assert len(result[("a", "b")]) == 1


def test_aggregate_node_names_normalised_to_lowercase() -> None:
    """Cause and effect names should be lowercased during aggregation."""
    g = _make_graph("s1", [{"cause": "Smoking", "effect": "Lung Cancer", "confidence": 0.9}])
    result = aggregate_edges([g])
    assert ("smoking", "lung cancer") in result


def test_aggregate_two_sources_same_edge_grouped() -> None:
    """The same directed edge from two different sources should be grouped together."""
    g1 = _make_graph("s1", [{"cause": "a", "effect": "b", "confidence": 0.8}])
    g2 = _make_graph("s2", [{"cause": "a", "effect": "b", "confidence": 0.6}])
    result = aggregate_edges([g1, g2])
    assert len(result[("a", "b")]) == 2


def test_aggregate_opposite_directions_separate_keys() -> None:
    """A→B and B→A should be stored under separate keys."""
    g1 = _make_graph("s1", [{"cause": "a", "effect": "b", "confidence": 0.8}])
    g2 = _make_graph("s2", [{"cause": "b", "effect": "a", "confidence": 0.5}])
    result = aggregate_edges([g1, g2])
    assert ("a", "b") in result
    assert ("b", "a") in result


def test_aggregate_source_name_preserved() -> None:
    """Each SourceEdge should carry the correct source name."""
    g = _make_graph("my_source", [{"cause": "x", "effect": "y", "confidence": 0.7}])
    result = aggregate_edges([g])
    edges = result[("x", "y")]
    assert edges[0].source_name == "my_source"


def test_aggregate_source_weight_preserved() -> None:
    """Source weight from the SourceGraph should appear on each SourceEdge."""
    g = _make_graph("s1", [{"cause": "x", "effect": "y", "confidence": 0.7}], weight=0.5)
    result = aggregate_edges([g])
    assert result[("x", "y")][0].source_weight == 0.5


def test_aggregate_multiple_edges_from_one_source() -> None:
    """Multiple edges from the same source should all be present."""
    g = _make_graph("s1", [
        {"cause": "a", "effect": "b", "confidence": 0.8},
        {"cause": "b", "effect": "c", "confidence": 0.6},
        {"cause": "a", "effect": "c", "confidence": 0.4},
    ])
    result = aggregate_edges([g])
    assert len(result) == 3


def test_aggregate_empty_sources_returns_empty() -> None:
    """An empty list of source graphs should return an empty dict."""
    result = aggregate_edges([])
    assert result == {}


def test_aggregate_graph_with_no_edges() -> None:
    """A graph with no edges should contribute nothing to the aggregated map."""
    g = SourceGraph(source_name="empty", source_weight=1.0, nodes=["a", "b"], edges=[])
    result = aggregate_edges([g])
    assert result == {}


def test_aggregate_evidence_preserved() -> None:
    """The evidence string from the edge dict should be stored on SourceEdge."""
    g = _make_graph("s1", [{
        "cause": "a", "effect": "b",
        "confidence": 0.8, "evidence": "study proves this", "edge_type": "direct"
    }])
    result = aggregate_edges([g])
    edge = result[("a", "b")][0]
    assert edge.evidence == "study proves this"
    assert edge.edge_type == "direct"


def test_aggregate_three_sources() -> None:
    """Aggregating three sources with overlapping edges should collect all instances."""
    g1 = _make_graph("s1", [{"cause": "a", "effect": "b", "confidence": 0.9}])
    g2 = _make_graph("s2", [{"cause": "a", "effect": "b", "confidence": 0.7}])
    g3 = _make_graph("s3", [{"cause": "a", "effect": "b", "confidence": 0.5}])
    result = aggregate_edges([g1, g2, g3])
    assert len(result[("a", "b")]) == 3
