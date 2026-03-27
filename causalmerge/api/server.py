"""FastAPI application factory for the CausalMerge REST API.

Usage (via CLI):
    causalmerge serve --port 8000

Usage (direct):
    uvicorn causalmerge.api.server:create_app --factory --reload

The API is intentionally minimal.  It exposes a single merge endpoint plus a
health check.  All heavy lifting is delegated to the same ``MergeEngine`` used
by the CLI.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from causalmerge import __version__

logger = logging.getLogger("causalmerge.api")


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: RUF029
    """Application lifespan handler.

    Runs startup and shutdown logic.  Currently just logs start/stop events;
    extend this to manage connection pools or model loading if needed.
    """
    logger.info("CausalMerge API v%s starting up", __version__)
    yield
    logger.info("CausalMerge API shutting down")


def create_app() -> FastAPI:
    """Application factory — create and configure the FastAPI instance.

    Returns
    -------
    FastAPI
        A fully configured application instance ready for uvicorn.
    """
    from causalmerge.api.routes import router

    app = FastAPI(
        title="CausalMerge API",
        description=(
            "Fuse multiple causal graphs from different sources into a unified causal model. "
            "Resolves directional conflicts via confidence-weighted voting and enforces DAG constraints."
        ),
        version=__version__,
        lifespan=_lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Allow all origins by default — lock down in production via env vars
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    logger.info("CausalMerge API routes registered")
    return app
