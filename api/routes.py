"""REST endpoints for the CausalMerge API.

Endpoints:
    GET  /health         — liveness probe
    POST /merge          — merge a list of graph payloads into a unified graph

All endpoints return JSON.  The ``/merge`` endpoint accepts the same graph
format that WhyNet writes (``{"nodes": [...], "edges": [...], ...}``), making
it easy to pipe WhyNet output directly into CausalMerge over HTTP.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from causalmerge import __version__
from causalmerge.config import get_settings
from causalmerge.data.schema import SourceGraph
from causalmerge.merge.engine import MergeEngine

logger = logging.getLogger("causalmerge.routes")

router = APIRouter()


# ── Request / response models ─────────────────────────────────────────────────


class GraphPayload(BaseModel):
    """A single causal graph submitted to the merge endpoint."""

    source_name: str = Field(..., description="Human-readable label for this source graph.")
    source_weight: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Relative importance of this source (0.0–1.0).",
    )
    nodes: list[str] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(..., description="List of causal edge objects.")
    metadata: dict[str, Any] = Field(default_factory=dict)


class MergeRequest(BaseModel):
    """Request body for the ``POST /merge`` endpoint."""

    graphs: list[GraphPayload] = Field(
        ...,
        min_length=2,
        description="At least two source graphs to merge.",
    )
    confidence_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Override the default confidence threshold for this merge.",
    )


class MergeResponse(BaseModel):
    """Response from the ``POST /merge`` endpoint."""

    nodes: list[str]
    edges: list[dict[str, Any]]
    sources_merged: list[str]
    total_edges_before: int
    total_edges_after: int
    conflicts_found: int
    conflicts_resolved: int
    cycles_broken: int
    merged_at: str


class HealthResponse(BaseModel):
    """Response from ``GET /health``."""

    status: str
    version: str
    timestamp: str


# ── Endpoints ─────────────────────────────────────────────────────────────────


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness probe",
    tags=["meta"],
)
async def health() -> HealthResponse:
    """Return service health status and version.

    Always returns HTTP 200 if the service is running.
    """
    return HealthResponse(
        status="ok",
        version=__version__,
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
    )


@router.post(
    "/merge",
    response_model=MergeResponse,
    status_code=status.HTTP_200_OK,
    summary="Merge multiple causal graphs",
    tags=["merge"],
)
async def merge_graphs(request: MergeRequest) -> MergeResponse:
    """Merge two or more causal graphs into a single unified graph.

    Accepts an array of graph payloads (compatible with WhyNet output format),
    runs the full merge pipeline, and returns the merged graph along with a
    summary of the operation.

    Raises HTTP 422 if fewer than two graphs are provided.
    Raises HTTP 500 if the merge pipeline encounters an internal error.
    """
    if len(request.graphs) < 2:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least two source graphs are required for merging.",
        )

    settings = get_settings()
    if request.confidence_threshold is not None:
        # Create a modified settings copy — we can't mutate the cached singleton
        from causalmerge.config import Settings
        settings = Settings(
            confidence_threshold=request.confidence_threshold,
            cycle_resolution=settings.cycle_resolution,
            log_level=settings.log_level,
        )

    # Convert GraphPayload → SourceGraph
    source_graphs: list[SourceGraph] = []
    for payload in request.graphs:
        source_graphs.append(
            SourceGraph(
                source_name=payload.source_name,
                source_weight=payload.source_weight,
                nodes=payload.nodes,
                edges=payload.edges,
                metadata=payload.metadata,
            )
        )

    try:
        engine = MergeEngine(settings=settings)
        graph, report = engine.merge(source_graphs)
    except Exception as exc:
        logger.exception("Merge pipeline failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Merge failed: {exc}",
        ) from exc

    # Serialise graph edges for the response
    response_edges: list[dict[str, Any]] = []
    for u, v, data in graph.edges(data=True):
        response_edges.append(
            {
                "cause": u,
                "effect": v,
                "confidence": data.get("merged_confidence", 0.0),
                "source_agreement": data.get("source_agreement", 0.0),
                "contributing_sources": data.get("contributing_sources", []),
                "is_disputed": data.get("is_disputed", False),
                "edge_type": data.get("edge_type", "direct"),
            }
        )

    return MergeResponse(
        nodes=list(graph.nodes()),
        edges=response_edges,
        sources_merged=report.sources_merged,
        total_edges_before=report.total_edges_before,
        total_edges_after=report.total_edges_after,
        conflicts_found=report.conflicts_found,
        conflicts_resolved=report.conflicts_resolved,
        cycles_broken=report.cycles_broken,
        merged_at=report.created_at.isoformat(),
    )
