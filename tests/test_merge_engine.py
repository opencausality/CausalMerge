"""End-to-end tests for causalmerge.merge.engine.MergeEngine."""

from __future__ import annotations

import networkx as nx
import pytest

from causalmerge.config import Settings
from causalmerge.data.schema import MergeReport, SourceGraph
from causalmerge.exceptions import EmptyGraphError
from causalmerge.merge.engine import MergeEngine


def _settings(**kwargs) -> Settings:  # type: ignore[return]
    """Create a Settings instance with optional overrides."""
    return Settings(**kwargs)


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def two_agreeing_graphs() -> list[SourceGraph]:
    """Two graphs that agree on all edges (no conflicts)."""
    g1 = SourceGraph(
        source_name="s1",
        source_weight=1.0,
        nodes=["a", "b", "c"],
        edges=[
            {"cause": "a", "effect": "b", "confidence": 0.8},
            {"cause": "b", "effect": "c", "confidence": 0.6},
        ],
    )
    g2 = SourceGraph(
        source_name="s2",
        source_weight=1.0,
        nodes=["a", "b", "c"],
        edges=[
            {"cause": "a", "effect": "b", "confidence": 0.9},
            {"cause": "b", "effect": "c", "confidence": 0.7},
        ],
    )
    return [g1, g2]


@pytest.fixture()
def two_conflicting_graphs() -> list[SourceGraph]:
    """Two graphs with a single directional conflict."""
    g1 = SourceGraph(
        source_name="s1",
        source_weight=1.0,
        nodes=["a", "b"],
        edges=[{"cause": "a", "effect": "b", "confidence": 0.9}],
    )
    g2 = SourceGraph(
        source_name="s2",
        source_weight=1.0,
        nodes=["a", "b"],
        edges=[{"cause": "b", "effect": "a", "confidence": 0.3}],
    )
    return [g1, g2]


@pytest.fixture()
def cyclic_graphs() -> list[SourceGraph]:
    """Three graphs that together form a cycle: a→b, b→c, c→a."""
    return [
        SourceGraph(source_name="s1", source_weight=1.0, nodes=["a", "b"],
                    edges=[{"cause": "a", "effect": "b", "confidence": 0.9}]),
        SourceGraph(source_name="s2", source_weight=1.0, nodes=["b", "c"],
                    edges=[{"cause": "b", "effect": "c", "confidence": 0.7}]),
        SourceGraph(source_name="s3", source_weight=1.0, nodes=["c", "a"],
                    edges=[{"cause": "c", "effect": "a", "confidence": 0.4}]),
    ]


# ── Basic merge ───────────────────────────────────────────────────────────────


def test_merge_returns_tuple(two_agreeing_graphs: list[SourceGraph]) -> None:
    """merge() should return a (DiGraph, MergeReport) tuple."""
    engine = MergeEngine(settings=_settings())
    result = engine.merge(two_agreeing_graphs)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_merge_returns_digraph(two_agreeing_graphs: list[SourceGraph]) -> None:
    """The first element should be a networkx DiGraph."""
    engine = MergeEngine(settings=_settings())
    graph, _ = engine.merge(two_agreeing_graphs)
    assert isinstance(graph, nx.DiGraph)


def test_merge_returns_report(two_agreeing_graphs: list[SourceGraph]) -> None:
    """The second element should be a MergeReport."""
    engine = MergeEngine(settings=_settings())
    _, report = engine.merge(two_agreeing_graphs)
    assert isinstance(report, MergeReport)


def test_merge_result_is_dag(two_agreeing_graphs: list[SourceGraph]) -> None:
    """The merged graph must always be a directed acyclic graph."""
    engine = MergeEngine(settings=_settings())
    graph, _ = engine.merge(two_agreeing_graphs)
    assert nx.is_directed_acyclic_graph(graph)


# ── Agreement merge ───────────────────────────────────────────────────────────


def test_merge_agreeing_graphs_no_conflicts(two_agreeing_graphs: list[SourceGraph]) -> None:
    """Merging two agreeing graphs should find zero conflicts."""
    engine = MergeEngine(settings=_settings())
    _, report = engine.merge(two_agreeing_graphs)
    assert report.conflicts_found == 0


def test_merge_agreeing_graphs_weighted_confidence(two_agreeing_graphs: list[SourceGraph]) -> None:
    """Merged confidence for a→b should be the average of 0.8 and 0.9 = 0.85."""
    engine = MergeEngine(settings=_settings(confidence_threshold=0.0))
    graph, report = engine.merge(two_agreeing_graphs)
    all_edges = report.consensus_edges + report.disputed_edges
    ab_edge = next((e for e in all_edges if e.cause == "a" and e.effect == "b"), None)
    assert ab_edge is not None
    assert ab_edge.merged_confidence == pytest.approx(0.85, abs=1e-3)


def test_merge_agreeing_graphs_source_agreement_is_full(
    two_agreeing_graphs: list[SourceGraph],
) -> None:
    """Edges present in both sources should have source_agreement = 1.0."""
    engine = MergeEngine(settings=_settings(confidence_threshold=0.0))
    _, report = engine.merge(two_agreeing_graphs)
    all_edges = report.consensus_edges + report.disputed_edges
    for edge in all_edges:
        assert edge.source_agreement == pytest.approx(1.0, abs=1e-3)


# ── Conflict resolution ───────────────────────────────────────────────────────


def test_merge_detects_conflict(two_conflicting_graphs: list[SourceGraph]) -> None:
    """A merge with one directional conflict should report exactly 1 conflict."""
    engine = MergeEngine(settings=_settings())
    _, report = engine.merge(two_conflicting_graphs)
    assert report.conflicts_found == 1


def test_merge_resolves_conflict(two_conflicting_graphs: list[SourceGraph]) -> None:
    """The conflict should be resolved (not left unresolved)."""
    engine = MergeEngine(settings=_settings())
    _, report = engine.merge(two_conflicting_graphs)
    assert report.conflicts_resolved == 1


def test_merge_conflict_higher_confidence_wins(two_conflicting_graphs: list[SourceGraph]) -> None:
    """After conflict resolution, only the winning direction (a→b, conf=0.9) should survive."""
    engine = MergeEngine(settings=_settings(confidence_threshold=0.0))
    graph, _ = engine.merge(two_conflicting_graphs)
    assert graph.has_edge("a", "b")
    assert not graph.has_edge("b", "a")


# ── Cycle breaking ────────────────────────────────────────────────────────────


def test_merge_breaks_cycle(cyclic_graphs: list[SourceGraph]) -> None:
    """Merging graphs that form a cycle should produce a DAG."""
    engine = MergeEngine(settings=_settings(confidence_threshold=0.0))
    graph, report = engine.merge(cyclic_graphs)
    assert nx.is_directed_acyclic_graph(graph)


def test_merge_reports_cycles_broken(cyclic_graphs: list[SourceGraph]) -> None:
    """At least one cycle should be reported as broken."""
    engine = MergeEngine(settings=_settings(confidence_threshold=0.0))
    _, report = engine.merge(cyclic_graphs)
    assert report.cycles_broken >= 1


# ── Confidence threshold filtering ───────────────────────────────────────────


def test_merge_threshold_filters_low_confidence(two_agreeing_graphs: list[SourceGraph]) -> None:
    """Raising the threshold to 1.0 should exclude all edges."""
    engine = MergeEngine(settings=_settings(confidence_threshold=1.0))
    graph, _ = engine.merge(two_agreeing_graphs)
    assert graph.number_of_edges() == 0


def test_merge_threshold_zero_includes_all(two_agreeing_graphs: list[SourceGraph]) -> None:
    """Threshold of 0.0 should include all edges above zero confidence."""
    engine = MergeEngine(settings=_settings(confidence_threshold=0.0))
    graph, _ = engine.merge(two_agreeing_graphs)
    assert graph.number_of_edges() == 2


# ── Report fields ─────────────────────────────────────────────────────────────


def test_report_sources_merged(two_agreeing_graphs: list[SourceGraph]) -> None:
    """sources_merged should list all source names."""
    engine = MergeEngine(settings=_settings())
    _, report = engine.merge(two_agreeing_graphs)
    assert set(report.sources_merged) == {"s1", "s2"}


def test_report_total_edges_before(two_agreeing_graphs: list[SourceGraph]) -> None:
    """total_edges_before should count all edges across all sources."""
    engine = MergeEngine(settings=_settings())
    _, report = engine.merge(two_agreeing_graphs)
    assert report.total_edges_before == 4  # 2 edges × 2 sources


def test_report_has_created_at(two_agreeing_graphs: list[SourceGraph]) -> None:
    """created_at should be set."""
    engine = MergeEngine(settings=_settings())
    _, report = engine.merge(two_agreeing_graphs)
    assert report.created_at is not None


# ── Error cases ───────────────────────────────────────────────────────────────


def test_merge_no_graphs_raises() -> None:
    """An empty list should raise EmptyGraphError."""
    engine = MergeEngine(settings=_settings())
    with pytest.raises(EmptyGraphError):
        engine.merge([])


def test_merge_all_empty_edges_raises() -> None:
    """Graphs with zero edges should raise EmptyGraphError."""
    g1 = SourceGraph(source_name="s1", source_weight=1.0, nodes=["a"], edges=[])
    g2 = SourceGraph(source_name="s2", source_weight=1.0, nodes=["b"], edges=[])
    engine = MergeEngine(settings=_settings())
    with pytest.raises(EmptyGraphError):
        engine.merge([g1, g2])


# ── End-to-end fixture graphs ─────────────────────────────────────────────────


def test_merge_fixture_graphs_is_dag(all_three_graphs: list[SourceGraph]) -> None:
    """Merging all three fixture graphs should always produce a DAG."""
    engine = MergeEngine(settings=_settings(confidence_threshold=0.4))
    graph, _ = engine.merge(all_three_graphs)
    assert nx.is_directed_acyclic_graph(graph)


def test_merge_fixture_graphs_report_populated(all_three_graphs: list[SourceGraph]) -> None:
    """The merge report should list all three sources."""
    engine = MergeEngine(settings=_settings(confidence_threshold=0.4))
    _, report = engine.merge(all_three_graphs)
    assert set(report.sources_merged) == {"graph_a", "graph_b", "graph_c"}


def test_merge_fixture_graphs_detects_at_least_one_conflict(
    all_three_graphs: list[SourceGraph],
) -> None:
    """The three fixture graphs contain at least one directional conflict."""
    engine = MergeEngine(settings=_settings(confidence_threshold=0.0))
    _, report = engine.merge(all_three_graphs)
    assert report.conflicts_found >= 1
