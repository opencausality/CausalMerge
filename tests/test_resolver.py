"""Tests for causalmerge.resolution.resolver."""

from __future__ import annotations

import pytest

from causalmerge.data.schema import EdgeConflict
from causalmerge.exceptions import ConflictResolutionError
from causalmerge.resolution.resolver import resolve_all_conflicts, resolve_conflict


def _make_conflict(
    cause: str,
    effect: str,
    conf_a: float,
    sources_a: list[str],
    conf_b: float,
    sources_b: list[str],
) -> EdgeConflict:
    """Build an unresolved EdgeConflict for testing."""
    return EdgeConflict(
        cause=cause,
        effect=effect,
        sources_for_direction_a=sources_a,
        confidence_direction_a=conf_a,
        sources_for_direction_b=sources_b,
        confidence_direction_b=conf_b,
        resolution="",
        resolution_confidence=0.0,
    )


# ── Resolution direction ──────────────────────────────────────────────────────


def test_higher_conf_a_wins() -> None:
    """Direction A should win when its confidence sum is higher."""
    conflict = _make_conflict("a", "b", conf_a=0.8, sources_a=["s1"], conf_b=0.3, sources_b=["s2"])
    resolved = resolve_conflict(conflict)
    assert resolved.resolution == "DIRECTION_A"


def test_higher_conf_b_wins() -> None:
    """Direction B should win when its confidence sum is higher."""
    conflict = _make_conflict("a", "b", conf_a=0.3, sources_a=["s1"], conf_b=0.9, sources_b=["s2"])
    resolved = resolve_conflict(conflict)
    assert resolved.resolution == "DIRECTION_B"


def test_tie_favours_direction_a() -> None:
    """Ties should be broken in favour of direction A (first-encountered)."""
    conflict = _make_conflict("a", "b", conf_a=0.5, sources_a=["s1"], conf_b=0.5, sources_b=["s2"])
    resolved = resolve_conflict(conflict)
    assert resolved.resolution == "DIRECTION_A"


def test_both_zero_confidence_dropped() -> None:
    """When both directions have zero confidence the edge is dropped."""
    conflict = _make_conflict("a", "b", conf_a=0.0, sources_a=[], conf_b=0.0, sources_b=[])
    resolved = resolve_conflict(conflict)
    assert resolved.resolution == "DROPPED"
    assert resolved.resolution_confidence == 0.0


# ── Resolution confidence ─────────────────────────────────────────────────────


def test_resolution_confidence_set_for_direction_a() -> None:
    """resolution_confidence should match direction A's confidence when A wins."""
    conflict = _make_conflict("a", "b", conf_a=0.75, sources_a=["s1"], conf_b=0.4, sources_b=["s2"])
    resolved = resolve_conflict(conflict)
    assert resolved.resolution_confidence == pytest.approx(0.75, abs=1e-4)


def test_resolution_confidence_set_for_direction_b() -> None:
    """resolution_confidence should match direction B's confidence when B wins."""
    conflict = _make_conflict("a", "b", conf_a=0.2, sources_a=["s1"], conf_b=0.85, sources_b=["s2"])
    resolved = resolve_conflict(conflict)
    assert resolved.resolution_confidence == pytest.approx(0.85, abs=1e-4)


# ── Error cases ───────────────────────────────────────────────────────────────


def test_negative_confidence_raises() -> None:
    """Negative confidence values are invalid and should raise ConflictResolutionError."""
    conflict = _make_conflict("a", "b", conf_a=-0.1, sources_a=["s1"], conf_b=0.5, sources_b=["s2"])
    with pytest.raises(ConflictResolutionError):
        resolve_conflict(conflict)


# ── resolve_all_conflicts ─────────────────────────────────────────────────────


def test_resolve_all_conflicts_mutates_in_place() -> None:
    """resolve_all_conflicts should populate every conflict's resolution field."""
    conflicts = [
        _make_conflict("a", "b", 0.8, ["s1"], 0.3, ["s2"]),
        _make_conflict("c", "d", 0.2, ["s1"], 0.9, ["s2"]),
    ]
    result = resolve_all_conflicts(conflicts)
    assert all(c.resolution != "" for c in result)


def test_resolve_all_conflicts_returns_same_list() -> None:
    """The return value should be the same list object (mutation, not copy)."""
    conflicts = [_make_conflict("a", "b", 0.8, ["s1"], 0.3, ["s2"])]
    result = resolve_all_conflicts(conflicts)
    assert result is conflicts


def test_resolve_all_conflicts_empty_list() -> None:
    """An empty conflict list should return an empty list."""
    assert resolve_all_conflicts([]) == []


# ── Integration with conflict data from fixtures ──────────────────────────────


def test_resolve_fixture_conflict() -> None:
    """The known conflict in graph_b (timeout_errors→pool vs pool→timeout) resolves correctly.

    graph_a says: connection_pool_exhaustion → timeout_errors (conf 0.9)
    graph_b says: timeout_errors → connection_pool_exhaustion (conf 0.45)
    Direction A (pool→timeout, conf 0.9) should win.
    """
    conflict = _make_conflict(
        cause="connection_pool_exhaustion",
        effect="timeout_errors",
        conf_a=0.9,
        sources_a=["graph_a"],
        conf_b=0.45,
        sources_b=["graph_b"],
    )
    resolved = resolve_conflict(conflict)
    assert resolved.resolution == "DIRECTION_A"
    assert resolved.resolution_confidence == pytest.approx(0.9, abs=1e-4)
