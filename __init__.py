"""CausalMerge — fuse multiple causal graphs into a unified causal model.

When you run WhyNet on several incident reports and obtain multiple causal
graphs, CausalMerge merges them: it resolves directional conflicts via
confidence-weighted voting and enforces DAG constraints (no cycles).
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "CausalMerge Contributors"
__license__ = "MIT"

__all__ = ["__version__", "__author__", "__license__"]
