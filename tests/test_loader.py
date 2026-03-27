"""Tests for causalmerge.data.loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from causalmerge.data.loader import load_graph, load_graphs
from causalmerge.data.schema import SourceGraph
from causalmerge.exceptions import GraphLoadError

from tests.conftest import FIXTURES_DIR, GRAPH_A_PATH, GRAPH_B_PATH, GRAPH_C_PATH


# ── load_graph — happy path ────────────────────────────────────────────────────


def test_load_graph_returns_source_graph() -> None:
    """load_graph should return a SourceGraph with correct metadata."""
    graph = load_graph(GRAPH_A_PATH, source_name="test_a")
    assert isinstance(graph, SourceGraph)
    assert graph.source_name == "test_a"
    assert graph.source_weight == 1.0


def test_load_graph_edge_count() -> None:
    """graph_a.json should contain exactly 4 edges."""
    graph = load_graph(GRAPH_A_PATH, source_name="test_a")
    assert len(graph.edges) == 4


def test_load_graph_node_count() -> None:
    """graph_a.json should contain exactly 5 nodes."""
    graph = load_graph(GRAPH_A_PATH, source_name="test_a")
    assert len(graph.nodes) == 5


def test_load_graph_custom_weight() -> None:
    """Supplied weight should be stored on the SourceGraph."""
    graph = load_graph(GRAPH_A_PATH, source_name="test_a", weight=0.6)
    assert graph.source_weight == pytest.approx(0.6)


def test_load_graph_edge_fields() -> None:
    """Every edge must have cause, effect, confidence, evidence, and edge_type."""
    graph = load_graph(GRAPH_A_PATH, source_name="test_a")
    for edge in graph.edges:
        assert "cause" in edge
        assert "effect" in edge
        assert "confidence" in edge
        assert isinstance(edge["confidence"], float)


def test_load_graph_metadata_captured(tmp_path: Path) -> None:
    """Extra keys in the JSON (model_used, created_at) go into metadata."""
    graph = load_graph(GRAPH_A_PATH, source_name="test_a")
    assert "model_used" in graph.metadata
    assert graph.metadata["model_used"] == "ollama/llama3.1"


def test_load_graph_b_has_conflict_edge() -> None:
    """graph_b should contain the reversed edge timeout_errors→connection_pool_exhaustion."""
    graph = load_graph(GRAPH_B_PATH, source_name="test_b")
    causes = [e["cause"] for e in graph.edges]
    assert "timeout_errors" in causes


# ── load_graph — error cases ───────────────────────────────────────────────────


def test_load_graph_missing_file_raises() -> None:
    """A path that does not exist should raise GraphLoadError."""
    with pytest.raises(GraphLoadError, match="not found"):
        load_graph(Path("/nonexistent/path/graph.json"), source_name="x")


def test_load_graph_invalid_json_raises(tmp_path: Path) -> None:
    """A file containing malformed JSON should raise GraphLoadError."""
    bad = tmp_path / "bad.json"
    bad.write_text("this is not json", encoding="utf-8")
    with pytest.raises(GraphLoadError, match="Invalid JSON"):
        load_graph(bad, source_name="bad")


def test_load_graph_missing_edges_key_raises(tmp_path: Path) -> None:
    """A JSON object without an 'edges' key should raise GraphLoadError."""
    no_edges = tmp_path / "no_edges.json"
    no_edges.write_text(json.dumps({"nodes": ["a", "b"]}), encoding="utf-8")
    with pytest.raises(GraphLoadError, match="missing required key 'edges'"):
        load_graph(no_edges, source_name="no_edges")


def test_load_graph_edge_missing_cause_raises(tmp_path: Path) -> None:
    """An edge dict without 'cause' should raise GraphLoadError."""
    bad = tmp_path / "bad_edge.json"
    bad.write_text(
        json.dumps({"nodes": [], "edges": [{"effect": "b", "confidence": 0.5}]}),
        encoding="utf-8",
    )
    with pytest.raises(GraphLoadError, match="missing required keys"):
        load_graph(bad, source_name="bad")


def test_load_graph_edge_confidence_out_of_range_raises(tmp_path: Path) -> None:
    """An edge confidence > 1.0 should raise GraphLoadError."""
    bad = tmp_path / "bad_conf.json"
    bad.write_text(
        json.dumps({"nodes": [], "edges": [{"cause": "a", "effect": "b", "confidence": 1.5}]}),
        encoding="utf-8",
    )
    with pytest.raises(GraphLoadError, match="confidence"):
        load_graph(bad, source_name="bad")


def test_load_graph_top_level_not_dict_raises(tmp_path: Path) -> None:
    """A JSON array at the top level should raise GraphLoadError."""
    bad = tmp_path / "array.json"
    bad.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(GraphLoadError):
        load_graph(bad, source_name="array")


# ── load_graphs — happy path ───────────────────────────────────────────────────


def test_load_graphs_returns_list() -> None:
    """load_graphs should return a list of the correct length."""
    graphs = load_graphs([GRAPH_A_PATH, GRAPH_B_PATH])
    assert len(graphs) == 2


def test_load_graphs_with_weights() -> None:
    """Weights should be correctly propagated to each SourceGraph."""
    graphs = load_graphs([GRAPH_A_PATH, GRAPH_B_PATH], weights=[0.3, 0.7])
    assert graphs[0].source_weight == pytest.approx(0.3)
    assert graphs[1].source_weight == pytest.approx(0.7)


def test_load_graphs_source_names_derived_from_filenames() -> None:
    """Source names should default to the filename stem."""
    graphs = load_graphs([GRAPH_A_PATH, GRAPH_B_PATH])
    assert graphs[0].source_name == "graph_a"
    assert graphs[1].source_name == "graph_b"


def test_load_graphs_wrong_weights_length_raises() -> None:
    """A mismatch between paths and weights lengths should raise GraphLoadError."""
    with pytest.raises(GraphLoadError, match="weights"):
        load_graphs([GRAPH_A_PATH, GRAPH_B_PATH], weights=[1.0])


def test_load_graphs_empty_paths_raises() -> None:
    """An empty path list should raise ValueError."""
    with pytest.raises(ValueError):
        load_graphs([])
