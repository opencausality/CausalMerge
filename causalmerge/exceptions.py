"""CausalMerge exception hierarchy.

All exceptions raised by CausalMerge are subclasses of ``CausalMergeError``
so callers can catch the full family with a single ``except`` clause.
"""

from __future__ import annotations


class CausalMergeError(Exception):
    """Base exception for all CausalMerge errors."""


class GraphLoadError(CausalMergeError):
    """Raised when a source graph file cannot be loaded or parsed.

    Common causes: file not found, invalid JSON, missing required keys,
    or a value that fails schema validation.
    """


class ConflictResolutionError(CausalMergeError):
    """Raised when a directional conflict cannot be resolved.

    This should be rare — the resolver always picks a winner — but it is
    raised if the resolution data is structurally invalid.
    """


class CycleBreakingError(CausalMergeError):
    """Raised when the DAG enforcer cannot break all cycles.

    This indicates a bug in the enforcer logic; cycles must always be
    breakable by removing a single edge.
    """


class EmptyGraphError(CausalMergeError):
    """Raised when a source graph contains no edges.

    A graph with nodes but no edges provides no causal information and
    is likely the result of a failed extraction or an empty input file.
    """
