"""CLI tests for causalmerge using typer.testing.CliRunner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from causalmerge.cli import app

runner = CliRunner(mix_stderr=False)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
GRAPH_A = str(FIXTURES_DIR / "graph_a.json")
GRAPH_B = str(FIXTURES_DIR / "graph_b.json")
GRAPH_C = str(FIXTURES_DIR / "graph_c.json")


# ── Version ───────────────────────────────────────────────────────────────────


def test_version_flag() -> None:
    """--version should print the version string and exit 0."""
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "CausalMerge" in result.output
    assert "0.1.0" in result.output


# ── merge command ─────────────────────────────────────────────────────────────


def test_merge_two_graphs(tmp_path: Path) -> None:
    """Merging two fixture graphs should succeed with exit code 0."""
    output = tmp_path / "merged.json"
    result = runner.invoke(app, ["merge", GRAPH_A, GRAPH_B, "--output", str(output)])
    assert result.exit_code == 0, result.output


def test_merge_three_graphs(tmp_path: Path) -> None:
    """Merging all three fixture graphs should succeed."""
    output = tmp_path / "merged.json"
    result = runner.invoke(app, ["merge", GRAPH_A, GRAPH_B, GRAPH_C, "--output", str(output)])
    assert result.exit_code == 0, result.output


def test_merge_output_file_created(tmp_path: Path) -> None:
    """The --output file should be created after a successful merge."""
    output = tmp_path / "merged.json"
    runner.invoke(app, ["merge", GRAPH_A, GRAPH_B, "--output", str(output)])
    assert output.exists()


def test_merge_output_valid_json(tmp_path: Path) -> None:
    """The --output file should contain valid JSON with 'nodes' and 'edges' keys."""
    output = tmp_path / "merged.json"
    runner.invoke(app, ["merge", GRAPH_A, GRAPH_B, "--output", str(output)])
    data = json.loads(output.read_text())
    assert "nodes" in data
    assert "edges" in data


def test_merge_output_edges_have_required_fields(tmp_path: Path) -> None:
    """Each merged edge should have cause, effect, and confidence fields."""
    output = tmp_path / "merged.json"
    runner.invoke(app, ["merge", GRAPH_A, GRAPH_B, "--output", str(output)])
    data = json.loads(output.read_text())
    for edge in data["edges"]:
        assert "cause" in edge
        assert "effect" in edge
        assert "confidence" in edge


def test_merge_with_weights(tmp_path: Path) -> None:
    """--weights option should be accepted without error."""
    output = tmp_path / "merged.json"
    result = runner.invoke(
        app, ["merge", GRAPH_A, GRAPH_B, "--output", str(output), "--weights", "0.3", "0.7"]
    )
    assert result.exit_code == 0, result.output


def test_merge_with_threshold(tmp_path: Path) -> None:
    """--threshold option should filter edges."""
    output_high = tmp_path / "merged_high.json"
    output_low = tmp_path / "merged_low.json"

    runner.invoke(app, ["merge", GRAPH_A, GRAPH_B, "--output", str(output_high), "--threshold", "0.9"])
    runner.invoke(app, ["merge", GRAPH_A, GRAPH_B, "--output", str(output_low), "--threshold", "0.1"])

    high = json.loads(output_high.read_text())
    low = json.loads(output_low.read_text())
    # Low threshold should include more (or equal) edges
    assert len(low["edges"]) >= len(high["edges"])


def test_merge_save_report(tmp_path: Path) -> None:
    """--save-report should create a separate JSON report file."""
    output = tmp_path / "merged.json"
    report_path = tmp_path / "report.json"
    result = runner.invoke(
        app,
        ["merge", GRAPH_A, GRAPH_B, "--output", str(output), "--save-report", str(report_path)],
    )
    assert result.exit_code == 0, result.output
    assert report_path.exists()
    report_data = json.loads(report_path.read_text())
    assert "sources_merged" in report_data
    assert "conflicts_found" in report_data


def test_merge_too_few_graphs() -> None:
    """Providing only one graph should fail with exit code 1."""
    result = runner.invoke(app, ["merge", GRAPH_A])
    assert result.exit_code == 1


def test_merge_wrong_weights_count(tmp_path: Path) -> None:
    """Providing wrong number of weights should fail with exit code 1."""
    output = tmp_path / "merged.json"
    result = runner.invoke(
        app,
        ["merge", GRAPH_A, GRAPH_B, "--output", str(output), "--weights", "0.5"],
    )
    assert result.exit_code == 1


def test_merge_missing_file() -> None:
    """A non-existent file path should fail with exit code 1."""
    result = runner.invoke(app, ["merge", "/nonexistent/graph.json", GRAPH_B])
    assert result.exit_code == 1


# ── conflicts command ─────────────────────────────────────────────────────────


def test_conflicts_command_two_graphs() -> None:
    """conflicts command with two graphs should exit 0."""
    result = runner.invoke(app, ["conflicts", GRAPH_A, GRAPH_B])
    assert result.exit_code == 0, result.output


def test_conflicts_command_finds_conflict() -> None:
    """The known conflict in graph_a and graph_b should appear in the output."""
    result = runner.invoke(app, ["conflicts", GRAPH_A, GRAPH_B])
    assert result.exit_code == 0
    # The conflict involves connection_pool_exhaustion ↔ timeout_errors
    assert "connection_pool_exhaustion" in result.output or "timeout_errors" in result.output


def test_conflicts_command_too_few_graphs() -> None:
    """Providing only one graph should fail with exit code 1."""
    result = runner.invoke(app, ["conflicts", GRAPH_A])
    assert result.exit_code == 1


def test_conflicts_command_agreeing_graphs() -> None:
    """Two graphs that fully agree should report no conflicts."""
    # graph_a vs itself — no conflicts
    result = runner.invoke(app, ["conflicts", GRAPH_A, GRAPH_A])
    assert result.exit_code == 0
    assert "No directional conflicts" in result.output


# ── report command ────────────────────────────────────────────────────────────


def test_report_command(tmp_path: Path) -> None:
    """report command should display edge table for a merged graph."""
    output = tmp_path / "merged.json"
    runner.invoke(app, ["merge", GRAPH_A, GRAPH_B, "--output", str(output)])
    result = runner.invoke(app, ["report", "--graph", str(output)])
    assert result.exit_code == 0, result.output
    assert "Edges" in result.output


def test_report_command_missing_file() -> None:
    """report on a non-existent file should fail."""
    result = runner.invoke(app, ["report", "--graph", "/nonexistent/merged.json"])
    # Typer will catch the exists=True check or our custom error
    assert result.exit_code != 0


# ── show command ──────────────────────────────────────────────────────────────


def test_show_command_creates_html(tmp_path: Path) -> None:
    """show command should create an HTML visualisation file."""
    output = tmp_path / "merged.json"
    vis = tmp_path / "viz.html"
    runner.invoke(app, ["merge", GRAPH_A, GRAPH_B, "--output", str(output)])
    result = runner.invoke(app, ["show", "--graph", str(output), "--output", str(vis)])
    assert result.exit_code == 0, result.output
    assert vis.exists()


# ── help ──────────────────────────────────────────────────────────────────────


def test_help_shows_commands() -> None:
    """--help should list available commands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "merge" in result.output
    assert "conflicts" in result.output
    assert "show" in result.output
    assert "report" in result.output
    assert "serve" in result.output


def test_merge_help() -> None:
    """merge --help should describe the command."""
    result = runner.invoke(app, ["merge", "--help"])
    assert result.exit_code == 0
    assert "--output" in result.output
    assert "--threshold" in result.output
