"""Graph visualiser — render the merged causal graph using pyvis.

Visual conventions:
- Consensus edges (not disputed): green, width proportional to confidence
- Disputed edges (directional conflict resolved): orange, dashed-style label
- Node size proportional to degree (in-degree + out-degree)
- Hovering over an edge shows confidence, agreement, and contributing sources
"""

from __future__ import annotations

import logging
from pathlib import Path

import networkx as nx

from causalmerge.data.schema import MergedEdge

logger = logging.getLogger("causalmerge.visualizer")

# Colour constants
_COLOUR_CONSENSUS = "#27ae60"   # green
_COLOUR_DISPUTED = "#e67e22"    # orange
_COLOUR_NODE = "#2980b9"        # blue
_COLOUR_NODE_BORDER = "#1a5276"


def visualize_graph(
    graph: nx.DiGraph,
    merged_edges: list[MergedEdge],
    output_path: Path | None = None,
    show: bool = False,
) -> Path:
    """Render the merged causal graph as an interactive HTML file via pyvis.

    Parameters
    ----------
    graph:
        The merged ``DiGraph`` to render.
    merged_edges:
        Full edge list used to enrich hover tooltips with provenance data.
    output_path:
        Where to write the HTML file.  Defaults to ``merged_graph.html``
        in the current working directory.
    show:
        If ``True``, attempt to open the rendered file in the default browser.

    Returns
    -------
    Path
        Absolute path of the generated HTML file.

    Notes
    -----
    Requires ``pyvis`` to be installed (included in the default dependencies).
    Falls back to a ``matplotlib`` static PNG if pyvis is unavailable.
    """
    if output_path is None:
        output_path = Path("merged_graph.html")

    output_path = output_path.resolve()

    try:
        from pyvis.network import Network  # type: ignore[import-untyped]
        return _render_pyvis(graph, merged_edges, output_path, show)
    except ImportError:
        logger.warning("pyvis not available — falling back to matplotlib static rendering")
        return _render_matplotlib(graph, merged_edges, output_path.with_suffix(".png"), show)


def _render_pyvis(
    graph: nx.DiGraph,
    merged_edges: list[MergedEdge],
    output_path: Path,
    show: bool,
) -> Path:
    """Render the graph as an interactive pyvis HTML page."""
    from pyvis.network import Network  # type: ignore[import-untyped]

    # Build a quick lookup for edge metadata
    edge_meta: dict[tuple[str, str], MergedEdge] = {
        (e.cause, e.effect): e for e in merged_edges
    }

    net = Network(
        height="750px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="#ecf0f1",
        directed=True,
        notebook=False,
    )
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -80,
          "centralGravity": 0.01,
          "springLength": 150,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": { "iterations": 150 }
      },
      "edges": {
        "smooth": { "type": "curvedCW", "roundness": 0.2 }
      }
    }
    """)

    # Add nodes — size by degree
    degree = dict(graph.degree())
    for node in graph.nodes():
        deg = degree.get(node, 1)
        size = 15 + deg * 5
        net.add_node(
            node,
            label=node,
            title=f"<b>{node}</b><br>Degree: {deg}",
            size=size,
            color={"background": _COLOUR_NODE, "border": _COLOUR_NODE_BORDER},
            font={"size": 14, "color": "#ecf0f1"},
        )

    # Add edges
    for u, v, data in graph.edges(data=True):
        meta = edge_meta.get((u, v))
        conf = data.get("merged_confidence", 0.0)
        agreement = data.get("source_agreement", 0.0)
        is_disputed = data.get("is_disputed", False)
        sources = data.get("contributing_sources", [])

        colour = _COLOUR_DISPUTED if is_disputed else _COLOUR_CONSENSUS
        width = max(1.0, conf * 6)

        sources_str = ", ".join(sources) if sources else "unknown"
        disputed_label = " [disputed]" if is_disputed else ""
        tooltip = (
            f"<b>{u} → {v}</b>{disputed_label}<br>"
            f"Confidence: {conf:.1%}<br>"
            f"Agreement: {agreement:.1%}<br>"
            f"Sources: {sources_str}"
        )
        if meta:
            tooltip += f"<br>Type: {meta.edge_type}"

        net.add_edge(
            u,
            v,
            title=tooltip,
            width=width,
            color=colour,
            arrows="to",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(output_path))

    logger.info("Rendered interactive graph to %s", output_path)

    if show:
        import webbrowser
        webbrowser.open(output_path.as_uri())
        logger.debug("Opened graph in browser: %s", output_path)

    return output_path


def _render_matplotlib(
    graph: nx.DiGraph,
    merged_edges: list[MergedEdge],
    output_path: Path,
    show: bool,
) -> Path:
    """Fallback static renderer using matplotlib and networkx."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edge_meta: dict[tuple[str, str], MergedEdge] = {
        (e.cause, e.effect): e for e in merged_edges
    }

    fig, ax = plt.subplots(figsize=(14, 9))
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#1a1a2e")

    pos = nx.spring_layout(graph, k=2.5, seed=42)

    # Separate disputed and consensus edges
    consensus_edges = [
        (u, v) for u, v in graph.edges()
        if not edge_meta.get((u, v), MergedEdge(
            cause=u, effect=v, merged_confidence=0, source_agreement=0,
            contributing_sources=[], is_disputed=False
        )).is_disputed
    ]
    disputed_edges = [
        (u, v) for u, v in graph.edges()
        if edge_meta.get((u, v), MergedEdge(
            cause=u, effect=v, merged_confidence=0, source_agreement=0,
            contributing_sources=[], is_disputed=False
        )).is_disputed
    ]

    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=_COLOUR_NODE, node_size=800, alpha=0.9)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_color="#ecf0f1", font_size=9)
    nx.draw_networkx_edges(
        graph, pos, ax=ax, edgelist=consensus_edges,
        edge_color=_COLOUR_CONSENSUS, arrows=True, width=2, arrowsize=20
    )
    nx.draw_networkx_edges(
        graph, pos, ax=ax, edgelist=disputed_edges,
        edge_color=_COLOUR_DISPUTED, arrows=True, width=2, arrowsize=20, style="dashed"
    )

    ax.set_title("CausalMerge — Merged Causal Graph", color="#ecf0f1", fontsize=14, pad=16)
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    logger.info("Rendered static graph PNG to %s", output_path)

    if show:
        import webbrowser
        webbrowser.open(output_path.as_uri())

    return output_path
