"""Conflict resolver — pick a winning direction for each detected conflict.

Resolution algorithm:
    1. Compare the sum of weighted confidences for direction A (cause→effect)
       against direction B (effect→cause).
    2. The direction with the higher total wins.
    3. Ties are broken in favour of direction A (the first-encountered direction).
    4. If both totals are zero (should not occur in practice), the edge is
       dropped with resolution ``"DROPPED"``.

The winning direction is written into the ``EdgeConflict.resolution`` field
so that the merge engine can remove the losing direction from the aggregated
map before computing consensus.
"""

from __future__ import annotations

import logging

from causalmerge.data.schema import EdgeConflict
from causalmerge.exceptions import ConflictResolutionError

logger = logging.getLogger("causalmerge.resolver")


def resolve_conflict(conflict: EdgeConflict) -> EdgeConflict:
    """Resolve a directional conflict by confidence-weighted voting.

    Parameters
    ----------
    conflict:
        An ``EdgeConflict`` whose ``resolution`` field is still empty (as
        returned by ``detect_conflicts``).

    Returns
    -------
    EdgeConflict
        The same conflict object, mutated in-place with:
        - ``resolution`` set to ``"DIRECTION_A"``, ``"DIRECTION_B"``, or
          ``"DROPPED"``
        - ``resolution_confidence`` set to the winning direction's confidence
          sum (or 0.0 if dropped)

    Raises
    ------
    ConflictResolutionError
        If the conflict object's confidence values are negative (invalid state).
    """
    conf_a = conflict.confidence_direction_a
    conf_b = conflict.confidence_direction_b

    if conf_a < 0.0 or conf_b < 0.0:
        raise ConflictResolutionError(
            f"Negative confidence values in conflict ({conflict.cause!r} ↔ {conflict.effect!r}): "
            f"direction_a={conf_a}, direction_b={conf_b}"
        )

    if conf_a == 0.0 and conf_b == 0.0:
        # No evidence for either direction — drop the edge
        conflict.resolution = "DROPPED"
        conflict.resolution_confidence = 0.0
        logger.warning(
            "Conflict (%s ↔ %s): both directions have zero confidence — dropping edge",
            conflict.cause,
            conflict.effect,
        )
        return conflict

    if conf_a >= conf_b:
        conflict.resolution = "DIRECTION_A"
        conflict.resolution_confidence = round(conf_a, 4)
        logger.info(
            "Conflict resolved DIRECTION_A: (%s → %s) wins [%.3f vs %.3f]",
            conflict.cause,
            conflict.effect,
            conf_a,
            conf_b,
        )
    else:
        conflict.resolution = "DIRECTION_B"
        conflict.resolution_confidence = round(conf_b, 4)
        logger.info(
            "Conflict resolved DIRECTION_B: (%s → %s) wins [%.3f vs %.3f]",
            conflict.effect,
            conflict.cause,
            conf_b,
            conf_a,
        )

    return conflict


def resolve_all_conflicts(conflicts: list[EdgeConflict]) -> list[EdgeConflict]:
    """Resolve every conflict in the provided list.

    Parameters
    ----------
    conflicts:
        List of ``EdgeConflict`` objects from ``detect_conflicts``.

    Returns
    -------
    list[EdgeConflict]
        The same list with every conflict's ``resolution`` field populated.
    """
    for conflict in conflicts:
        resolve_conflict(conflict)
    return conflicts
