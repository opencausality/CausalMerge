"""Reporting — print rich console summaries and save JSON reports.

``print_merge_report`` renders a structured, colourful summary to the terminal
using the ``rich`` library.  ``save_merge_report`` serialises the full
``MergeReport`` object to JSON for archiving and downstream tooling.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from causalmerge.data.schema import MergeReport

logger = logging.getLogger("causalmerge.reporting")

console = Console()


def print_merge_report(report: MergeReport) -> None:
    """Print a formatted merge report to the terminal using rich.

    The report is divided into:
    - Summary statistics panel
    - Source weights table
    - Consensus edges table
    - Disputed edges table (if any)
    - Dropped edges list (if any)

    Parameters
    ----------
    report:
        A ``MergeReport`` returned by ``MergeEngine.merge``.
    """
    # ── Summary panel ─────────────────────────────────────────────────────────
    summary_lines = [
        f"[bold cyan]Sources merged:[/]  {len(report.sources_merged)}",
        f"[bold cyan]Edges before:[/]    {report.total_edges_before}",
        f"[bold cyan]Edges after:[/]     {report.total_edges_after}",
        f"[bold cyan]Conflicts found:[/] {report.conflicts_found}",
        f"[bold cyan]Conflicts resolved:[/] {report.conflicts_resolved}",
        f"[bold cyan]Cycles broken:[/]   {report.cycles_broken}",
        f"[bold cyan]Dropped edges:[/]   {len(report.dropped_edges)}",
        f"[dim]Created:[/]          {report.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
    ]
    console.print(
        Panel(
            "\n".join(summary_lines),
            title="[bold]CausalMerge Report[/]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    # ── Source weights table ──────────────────────────────────────────────────
    src_table = Table(title="Source Graphs", border_style="blue", show_header=True)
    src_table.add_column("Source", style="bold")
    src_table.add_column("Weight", justify="right")
    for source in report.sources_merged:
        weight = report.source_weights.get(source, 1.0)
        src_table.add_row(source, f"{weight:.2f}")
    console.print(src_table)

    # ── Consensus edges table ─────────────────────────────────────────────────
    if report.consensus_edges:
        ce_table = Table(title="Consensus Edges", border_style="green", show_header=True)
        ce_table.add_column("Cause", style="bold yellow")
        ce_table.add_column("", justify="center")
        ce_table.add_column("Effect", style="bold magenta")
        ce_table.add_column("Confidence", justify="right")
        ce_table.add_column("Agreement", justify="right")
        ce_table.add_column("Sources")

        for edge in sorted(report.consensus_edges, key=lambda e: e.merged_confidence, reverse=True):
            conf_colour = _conf_colour(edge.merged_confidence)
            ce_table.add_row(
                edge.cause,
                "[dim]→[/]",
                edge.effect,
                f"[{conf_colour}]{edge.merged_confidence:.1%}[/]",
                f"{edge.source_agreement:.0%}",
                ", ".join(edge.contributing_sources),
            )
        console.print(ce_table)
    else:
        console.print("[dim]No consensus edges.[/]")

    # ── Disputed edges table ──────────────────────────────────────────────────
    if report.disputed_edges:
        de_table = Table(title="Disputed Edges (conflict resolved)", border_style="yellow")
        de_table.add_column("Cause", style="bold yellow")
        de_table.add_column("", justify="center")
        de_table.add_column("Effect", style="bold magenta")
        de_table.add_column("Confidence", justify="right")
        de_table.add_column("Agreement", justify="right")
        de_table.add_column("Sources")

        for edge in sorted(report.disputed_edges, key=lambda e: e.merged_confidence, reverse=True):
            conf_colour = _conf_colour(edge.merged_confidence)
            de_table.add_row(
                edge.cause,
                "[yellow]→[/]",
                edge.effect,
                f"[{conf_colour}]{edge.merged_confidence:.1%}[/]",
                f"{edge.source_agreement:.0%}",
                ", ".join(edge.contributing_sources),
            )
        console.print(de_table)

    # ── Dropped edges ─────────────────────────────────────────────────────────
    if report.dropped_edges:
        console.print(Text("\nDropped edges:", style="bold red"))
        for cause, effect in report.dropped_edges:
            console.print(f"  [dim]✗[/] {cause} → {effect}")


def save_merge_report(report: MergeReport, path: Path) -> None:
    """Serialise the merge report to a JSON file.

    The output is a pretty-printed JSON file using Pydantic's
    ``model_dump`` serialisation, with datetime objects converted to ISO
    strings.  This file can be archived alongside the merged graph JSON for
    full provenance.

    Parameters
    ----------
    report:
        The ``MergeReport`` to serialise.
    path:
        Destination file path (will be created or overwritten).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    data = report.model_dump(mode="json")
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    logger.info("Merge report saved to %s", path)


# ── Internal helpers ──────────────────────────────────────────────────────────


def _conf_colour(confidence: float) -> str:
    """Return a rich colour tag string based on confidence level."""
    if confidence >= 0.7:
        return "green"
    if confidence >= 0.4:
        return "yellow"
    return "red"
