"""Tests for causalmerge.resolution.conflicts."""

from __future__ import annotations

from causalmerge.data.schema import SourceEdge
from causalmerge.resolution.conflicts import detect_conflicts


def _aggregated(
    *edges: tuple[str, str, str, float],
) -> dict[tuple[str, str], list[SourceEdge]]:
    """Build an aggregated dict from (cause, effect, source, confidence) tuples."""
    agg: dict[tuple[str, str], list[SourceEdge]] = {}
    for cause, effect, source, conf in edges:
        key = (cause, effect)
        if key not in agg:
            agg[key] = []
        agg[key].append(
            SourceEdge(cause=cause, effect=effect, confidence=conf, source_name=source)
        )
    return agg


# ── No conflicts ──────────────────────────────────────────────────────────────


def test_no_conflicts_when_no_reverse_edges() -> None:
    """No conflicts when all edges point in the same direction."""
    agg = _aggregated(
        ("a", "b", "s1", 0.8),
        ("b", "c", "s1", 0.7),
    )
    assert detect_conflicts(agg) == []


def test_no_conflicts_empty_aggregated() -> None:
    """An empty aggregated map should return no conflicts."""
    assert detect_conflicts({}) == []


def test_no_conflicts_single_edge() -> None:
    """A single edge cannot conflict with itself."""
    agg = _aggregated(("x", "y", "s1", 0.9))
    assert detect_conflicts(agg) == []


# ── Conflict detection ────────────────────────────────────────────────────────


def test_detects_simple_conflict() -> None:
    """A→B in s1 and B→A in s2 must be detected as a conflict."""
    agg = _aggregated(
        ("a", "b", "s1", 0.8),
        ("b", "a", "s2", 0.4),
    )
    conflicts = detect_conflicts(agg)
    assert len(conflicts) == 1


def test_conflict_not_reported_twice() -> None:
    """A single A↔B conflict should appear once, not as both A→B and B→A."""
    agg = _aggregated(
        ("smoking", "cancer", "s1", 0.9),
        ("cancer", "smoking", "s2", 0.3),
    )
    conflicts = detect_conflicts(agg)
    assert len(conflicts) == 1


def test_conflict_cause_and_effect_fields() -> None:
    """Conflict cause/effect should correspond to direction A (the first-encountered key)."""
    agg = _aggregated(
        ("a", "b", "s1", 0.8),
        ("b", "a", "s2", 0.4),
    )
    conflict = detect_conflicts(agg)[0]
    # The canonical direction is the lexicographically smaller tuple
    assert set([conflict.cause, conflict.effect]) == {"a", "b"}


def test_conflict_sources_for_direction_a() -> None:
    """sources_for_direction_a should list all sources that express that direction."""
    agg = _aggregated(
        ("a", "b", "s1", 0.8),
        ("a", "b", "s2", 0.7),
        ("b", "a", "s3", 0.4),
    )
    conflicts = detect_conflicts(agg)
    assert len(conflicts) == 1
    c = conflicts[0]
    # direction_a is a→b (expressed by s1, s2); direction_b is b→a (expressed by s3)
    if (c.cause, c.effect) == ("a", "b"):
        assert set(c.sources_for_direction_a) == {"s1", "s2"}
        assert c.sources_for_direction_b == ["s3"]
    else:
        assert set(c.sources_for_direction_b) == {"s1", "s2"}
        assert c.sources_for_direction_a == ["s3"]


def test_conflict_confidence_sums() -> None:
    """Confidence values should be weight-summed for each direction."""
    agg = _aggregated(
        ("a", "b", "s1", 0.8),
        ("b", "a", "s2", 0.3),
    )
    conflict = detect_conflicts(agg)[0]
    # One direction has conf 0.8, the other 0.3
    confidences = {
        conflict.confidence_direction_a,
        conflict.confidence_direction_b,
    }
    assert pytest.approx(0.8, abs=1e-4) in confidences
    assert pytest.approx(0.3, abs=1e-4) in confidences


def test_multiple_independent_conflicts() -> None:
    """Multiple independent conflict pairs are all detected."""
    agg = _aggregated(
        ("a", "b", "s1", 0.8),
        ("b", "a", "s2", 0.4),
        ("c", "d", "s1", 0.7),
        ("d", "c", "s3", 0.5),
    )
    conflicts = detect_conflicts(agg)
    assert len(conflicts) == 2


def test_resolution_field_starts_empty() -> None:
    """Newly detected conflicts should have an empty resolution field."""
    agg = _aggregated(
        ("a", "b", "s1", 0.8),
        ("b", "a", "s2", 0.4),
    )
    conflict = detect_conflicts(agg)[0]
    assert conflict.resolution == ""


import pytest
