"""CausalMerge CLI — Typer-based command-line interface.

Commands:
    merge     Merge two or more causal graph JSON files into a unified graph
    conflicts Show directional conflicts between source graphs before merging
    show      Visualise a previously merged graph
    report    Print the merge report for a saved merged graph
    serve     Run the optional REST API server
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

from causalmerge import __version__

app = typer.Typer(
    name="causalmerge",
    help="Fuse multiple causal graphs from different sources into a unified causal model.",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()
logger = logging.getLogger("causalmerge.cli")


# ── Callbacks ─────────────────────────────────────────────────────────────────


def _version_callback(value: bool) -> None:
    if value:
        console.print(f"[bold cyan]CausalMerge[/] v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-V",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """CausalMerge — unify causal graphs from multiple sources."""
    from causalmerge.config import Settings, configure_logging

    settings = Settings(log_level="DEBUG" if verbose else "INFO")
    configure_logging(settings)


# ── merge ──────────────────────────────────────────────────────────────────────


@app.command()
def merge(
    graph_paths: list[Path] = typer.Argument(
        ...,
        help="Two or more causal graph JSON files to merge.",
        metavar="GRAPH...",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Save the merged graph JSON to this path.",
    ),
    weights: Optional[list[float]] = typer.Option(
        None,
        "--weights",
        "-w",
        help="Source weights, one per graph (e.g. --weights 0.3 0.7). Defaults to 1.0 for all.",
    ),
    show: bool = typer.Option(
        False,
        "--show",
        "-s",
        help="Open an interactive visualisation after merging.",
    ),
    threshold: float = typer.Option(
        0.4,
        "--threshold",
        "-t",
        min=0.0,
        max=1.0,
        help="Minimum merged confidence to include an edge (0.0–1.0).",
    ),
    save_report: Optional[Path] = typer.Option(
        None,
        "--save-report",
        help="Save the merge report JSON to this path.",
    ),
) -> None:
    """Merge two or more causal graph JSON files into a single unified graph.

    Example:

        causalmerge merge graph_a.json graph_b.json graph_c.json --output merged.json --show
    """
    from causalmerge.config import Settings
    from causalmerge.data.loader import load_graphs
    from causalmerge.exceptions import CausalMergeError
    from causalmerge.graph.builder import save_graph
    from causalmerge.graph.visualizer import visualize_graph
    from causalmerge.merge.engine import MergeEngine
    from causalmerge.reporting.report import print_merge_report, save_merge_report

    if len(graph_paths) < 2:
        console.print("[red]Error:[/] At least two graph files are required.")
        raise typer.Exit(code=1)

    for p in graph_paths:
        if not p.exists():
            console.print(f"[red]Error:[/] File not found: {p}")
            raise typer.Exit(code=1)

    if weights is not None and len(weights) != len(graph_paths):
        console.print(
            f"[red]Error:[/] Number of weights ({len(weights)}) must match "
            f"number of graph files ({len(graph_paths)})."
        )
        raise typer.Exit(code=1)

    console.print(
        f"[dim]Merging {len(graph_paths)} graph(s) with confidence threshold {threshold:.0%}…[/]"
    )

    try:
        source_graphs = load_graphs(graph_paths, weights=weights)
        settings = Settings(confidence_threshold=threshold)
        engine = MergeEngine(settings=settings)
        graph, report = engine.merge(source_graphs)
    except CausalMergeError as exc:
        console.print(f"[red]Merge failed:[/] {exc}")
        raise typer.Exit(code=1) from exc

    print_merge_report(report)

    if output:
        # Collect merged edges from the report for serialisation
        all_merged = report.consensus_edges + report.disputed_edges
        save_graph(graph, all_merged, output)
        console.print(f"\n[green]✓[/] Merged graph saved to [bold]{output}[/]")

    if save_report:
        save_merge_report(report, save_report)
        console.print(f"[green]✓[/] Report saved to [bold]{save_report}[/]")

    if show:
        all_merged = report.consensus_edges + report.disputed_edges
        vis_path = output.with_suffix(".html") if output else Path("merged_graph.html")
        rendered = visualize_graph(graph, all_merged, output_path=vis_path, show=True)
        console.print(f"[green]✓[/] Interactive graph → [bold]{rendered}[/]")


# ── conflicts ──────────────────────────────────────────────────────────────────


@app.command()
def conflicts(
    graph_paths: list[Path] = typer.Argument(
        ...,
        help="Two or more causal graph JSON files to inspect for conflicts.",
        metavar="GRAPH...",
    ),
    weights: Optional[list[float]] = typer.Option(
        None,
        "--weights",
        "-w",
        help="Source weights, one per graph.",
    ),
) -> None:
    """Show directional conflicts between source graphs without merging.

    Example:

        causalmerge conflicts graph_a.json graph_b.json
    """
    from rich.table import Table

    from causalmerge.data.loader import load_graphs
    from causalmerge.exceptions import CausalMergeError
    from causalmerge.merge.aggregator import aggregate_edges
    from causalmerge.resolution.conflicts import detect_conflicts

    if len(graph_paths) < 2:
        console.print("[red]Error:[/] At least two graph files are required.")
        raise typer.Exit(code=1)

    try:
        source_graphs = load_graphs(graph_paths, weights=weights)
        aggregated = aggregate_edges(source_graphs)
        found = detect_conflicts(aggregated)
    except CausalMergeError as exc:
        console.print(f"[red]Error:[/] {exc}")
        raise typer.Exit(code=1) from exc

    if not found:
        console.print(
            Panel(
                "[green]No directional conflicts found.[/] All source graphs agree on edge directions.",
                border_style="green",
            )
        )
        return

    table = Table(
        title=f"{len(found)} Directional Conflict(s) Detected",
        border_style="yellow",
    )
    table.add_column("Direction A", style="bold yellow")
    table.add_column("Conf A", justify="right")
    table.add_column("Sources A")
    table.add_column("vs")
    table.add_column("Direction B", style="bold magenta")
    table.add_column("Conf B", justify="right")
    table.add_column("Sources B")
    table.add_column("Winner", justify="center")

    for conflict in found:
        from causalmerge.resolution.resolver import resolve_conflict

        resolved = resolve_conflict(conflict)
        if resolved.resolution == "DIRECTION_A":
            winner = f"[green]{conflict.cause} → {conflict.effect}[/]"
        elif resolved.resolution == "DIRECTION_B":
            winner = f"[green]{conflict.effect} → {conflict.cause}[/]"
        else:
            winner = "[red]DROPPED[/]"

        table.add_row(
            f"{conflict.cause} → {conflict.effect}",
            f"{conflict.confidence_direction_a:.2f}",
            ", ".join(conflict.sources_for_direction_a),
            "[dim]vs[/]",
            f"{conflict.effect} → {conflict.cause}",
            f"{conflict.confidence_direction_b:.2f}",
            ", ".join(conflict.sources_for_direction_b),
            winner,
        )

    console.print(table)


# ── show ───────────────────────────────────────────────────────────────────────


@app.command()
def show(
    graph_file: Path = typer.Option(
        ...,
        "--graph",
        "-g",
        help="Path to a merged graph JSON file.",
        exists=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Save the visualisation HTML to this path. Defaults to <graph>.html.",
    ),
) -> None:
    """Visualise a previously merged causal graph in the browser.

    Example:

        causalmerge show --graph merged.json
    """
    from causalmerge.exceptions import CausalMergeError
    from causalmerge.graph.builder import load_merged_graph
    from causalmerge.graph.visualizer import visualize_graph

    try:
        graph, merged_edges = load_merged_graph(graph_file)
    except CausalMergeError as exc:
        console.print(f"[red]Error loading graph:[/] {exc}")
        raise typer.Exit(code=1) from exc

    vis_path = output or graph_file.with_suffix(".html")
    rendered = visualize_graph(graph, merged_edges, output_path=vis_path, show=True)
    console.print(f"[green]✓[/] Graph opened: [bold]{rendered}[/]")


# ── report ─────────────────────────────────────────────────────────────────────


@app.command()
def report(
    graph_file: Path = typer.Option(
        ...,
        "--graph",
        "-g",
        help="Path to a merged graph JSON file.",
        exists=True,
    ),
) -> None:
    """Print a summary report for a saved merged graph.

    Displays node count, edge count, and per-edge confidence and agreement
    scores for the merged graph.

    Example:

        causalmerge report --graph merged.json
    """
    from rich.table import Table

    from causalmerge.exceptions import CausalMergeError
    from causalmerge.graph.builder import load_merged_graph

    try:
        graph, merged_edges = load_merged_graph(graph_file)
    except CausalMergeError as exc:
        console.print(f"[red]Error loading graph:[/] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        Panel(
            f"[bold cyan]Graph:[/] {graph_file}\n"
            f"[bold cyan]Nodes:[/] {graph.number_of_nodes()}\n"
            f"[bold cyan]Edges:[/] {graph.number_of_edges()}",
            title="Merged Graph Summary",
            border_style="cyan",
        )
    )

    table = Table(title="Edges", border_style="blue")
    table.add_column("Cause", style="bold yellow")
    table.add_column("", justify="center")
    table.add_column("Effect", style="bold magenta")
    table.add_column("Confidence", justify="right")
    table.add_column("Agreement", justify="right")
    table.add_column("Disputed", justify="center")
    table.add_column("Type")
    table.add_column("Sources")

    for edge in sorted(merged_edges, key=lambda e: e.merged_confidence, reverse=True):
        conf_colour = "green" if edge.merged_confidence >= 0.7 else (
            "yellow" if edge.merged_confidence >= 0.4 else "red"
        )
        disputed_marker = "[orange1]✓[/]" if edge.is_disputed else "[dim]—[/]"
        table.add_row(
            edge.cause,
            "[dim]→[/]",
            edge.effect,
            f"[{conf_colour}]{edge.merged_confidence:.1%}[/]",
            f"{edge.source_agreement:.0%}",
            disputed_marker,
            edge.edge_type,
            ", ".join(edge.contributing_sources),
        )

    console.print(table)


# ── serve ──────────────────────────────────────────────────────────────────────


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Listen address."),
    port: int = typer.Option(8000, "--port", "-p", help="Listen port."),
) -> None:
    """Run the CausalMerge REST API server.

    Example:

        causalmerge serve --port 8000
    """
    import uvicorn

    console.print(
        f"[bold cyan]CausalMerge API[/] v{__version__} starting on [bold]http://{host}:{port}[/]"
    )
    console.print(f"  [dim]Docs:[/] http://{host}:{port}/docs")
    uvicorn.run("causalmerge.api.server:create_app", host=host, port=port, factory=True)
