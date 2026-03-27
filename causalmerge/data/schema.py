"""Pydantic data models for CausalMerge.

These models represent the core data flowing through the merge pipeline:
- ``SourceEdge``    — a single causal edge from one source graph
- ``EdgeConflict``  — a detected directional conflict between two sources
- ``MergedEdge``    — a resolved edge in the final unified graph
- ``MergeReport``   — full provenance report for a merge operation
- ``SourceGraph``   — a causal graph loaded from a JSON file
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class SourceEdge(BaseModel):
    """A single causal edge as read from one source graph.

    Carries the originating source name and its assigned weight so that
    downstream aggregation can apply weighted confidence averaging.
    """

    cause: str
    effect: str
    confidence: float = Field(ge=0.0, le=1.0)
    source_name: str
    source_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence: str = ""
    edge_type: str = "direct"


class EdgeConflict(BaseModel):
    """A detected directional conflict between source graphs.

    A conflict exists when some sources say A→B and other sources say B→A.
    After resolution, the ``resolution`` field records which direction won
    (or whether the edge was dropped entirely).
    """

    cause: str
    effect: str
    # Direction A: cause→effect
    sources_for_direction_a: list[str]  # source names
    confidence_direction_a: float
    # Direction B: effect→cause (reversed)
    sources_for_direction_b: list[str]
    confidence_direction_b: float
    resolution: str  # "DIRECTION_A", "DIRECTION_B", "DROPPED"
    resolution_confidence: float


class MergedEdge(BaseModel):
    """A causal edge in the final merged graph.

    ``merged_confidence`` is the weighted average of all source confidences.
    ``source_agreement`` is the fraction of sources that include this edge.
    Edges present in only one source get a lower agreement score.
    """

    cause: str
    effect: str
    merged_confidence: float
    source_agreement: float  # fraction of sources that agree on this edge
    contributing_sources: list[str]
    is_disputed: bool = False
    edge_type: str = "direct"


class MergeReport(BaseModel):
    """Full provenance report for a single merge operation.

    Captures statistics, conflict summaries, and the full lists of edges
    included, disputed, and dropped — enabling complete audit trails.
    """

    sources_merged: list[str]
    source_weights: dict[str, float]
    total_edges_before: int
    total_edges_after: int
    conflicts_found: int
    conflicts_resolved: int
    cycles_broken: int
    consensus_edges: list[MergedEdge]
    disputed_edges: list[MergedEdge]
    dropped_edges: list[tuple[str, str]]
    created_at: datetime


class SourceGraph(BaseModel):
    """A causal graph loaded from a JSON source file.

    Compatible with WhyNet's output format:
    ``{"nodes": [...], "edges": [{"cause": ..., "effect": ..., "confidence": ...}]}``
    """

    source_name: str
    source_weight: float = Field(default=1.0, ge=0.0, le=1.0)
    nodes: list[str]
    edges: list[dict[str, Any]]  # raw edge dicts from JSON
    metadata: dict[str, Any] = Field(default_factory=dict)
