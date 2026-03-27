"""Shared pytest fixtures for CausalMerge tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from causalmerge.data.loader import load_graph, load_graphs
from causalmerge.data.schema import SourceEdge, SourceGraph

# ── Path constants ─────────────────────────────────────────────────────────────

FIXTURES_DIR = Path(__file__).parent / "fixtures"
GRAPH_A_PATH = FIXTURES_DIR / "graph_a.json"
GRAPH_B_PATH = FIXTURES_DIR / "graph_b.json"
GRAPH_C_PATH = FIXTURES_DIR / "graph_c.json"


# ── Source graph fixtures ──────────────────────────────────────────────────────


@pytest.fixture()
def graph_a() -> SourceGraph:
    """Load fixture graph A (database team report)."""
    return load_graph(GRAPH_A_PATH, source_name="graph_a", weight=1.0)


@pytest.fixture()
def graph_b() -> SourceGraph:
    """Load fixture graph B (platform team report — contains a conflict with A)."""
    return load_graph(GRAPH_B_PATH, source_name="graph_b", weight=1.0)


@pytest.fixture()
def graph_c() -> SourceGraph:
    """Load fixture graph C (SRE post-mortem — contains a conflict with A's CPU chain)."""
    return load_graph(GRAPH_C_PATH, source_name="graph_c", weight=1.0)


@pytest.fixture()
def all_three_graphs(graph_a: SourceGraph, graph_b: SourceGraph, graph_c: SourceGraph) -> list[SourceGraph]:
    """All three fixture graphs as a list."""
    return [graph_a, graph_b, graph_c]


# ── Minimal in-memory graphs ───────────────────────────────────────────────────


@pytest.fixture()
def simple_conflicting_graphs() -> list[SourceGraph]:
    """Two minimal graphs with a single directional conflict A→B vs B→A."""
    g1 = SourceGraph(
        source_name="source_1",
        source_weight=1.0,
        nodes=["a", "b"],
        edges=[{"cause": "a", "effect": "b", "confidence": 0.8, "evidence": "a causes b", "edge_type": "direct"}],
    )
    g2 = SourceGraph(
        source_name="source_2",
        source_weight=1.0,
        nodes=["a", "b"],
        edges=[{"cause": "b", "effect": "a", "confidence": 0.4, "evidence": "b causes a", "edge_type": "direct"}],
    )
    return [g1, g2]


@pytest.fixture()
def cyclic_source_edges() -> list[SourceEdge]:
    """A set of source edges that form a simple three-node cycle: a→b→c→a."""
    return [
        SourceEdge(cause="a", effect="b", confidence=0.9, source_name="s1"),
        SourceEdge(cause="b", effect="c", confidence=0.7, source_name="s1"),
        SourceEdge(cause="c", effect="a", confidence=0.5, source_name="s1"),
    ]
