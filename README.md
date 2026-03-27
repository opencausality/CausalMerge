<div align="center">

# CausalMerge

**Fuse multiple causal graphs from different sources into a single unified model.**

When three teams each run WhyNet on the same incident and produce three different graphs,
CausalMerge merges them — resolving directional conflicts by evidence, not by gut feeling.

[![CI](https://github.com/opencausality/causalmerge/actions/workflows/ci.yml/badge.svg)](https://github.com/opencausality/causalmerge/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

## What is CausalMerge?

CausalMerge is a Python library and CLI tool that takes multiple causal graphs — each extracted
from a different document, team, or time period — and fuses them into a single authoritative
directed acyclic graph (DAG).

```
graph_a.json ─┐
graph_b.json ─┼─► CausalMerge ──► merged_graph.json  +  merge_report.json
graph_c.json ─┘
```

It answers the question: *"Given that different sources disagree on what caused what, what is
the most defensible unified causal model?"*

---

## The Graph Merging Problem

Imagine running WhyNet on three incident post-mortems from three different teams:

**Database team (graph_a):**
```
traffic_spike → slow_queries → connection_pool_exhaustion → timeout_errors
```

**Platform team (graph_b):**
```
deploy_event → slow_queries → connection_pool_exhaustion
timeout_errors → connection_pool_exhaustion   ← reversed! conflict with graph_a
```

**SRE team (graph_c):**
```
missing_index → slow_queries → connection_pool_exhaustion
high_cpu → slow_queries   ← is this a cause or an effect?
```

**Why simple union fails:**
A naive union of all three graphs would produce a cycle
(`connection_pool_exhaustion → timeout_errors → connection_pool_exhaustion`)
and include contradictory edges, making the graph unusable for causal reasoning.

**What CausalMerge does:**
1. Detects the directional conflict (`pool → timeout` vs `timeout → pool`)
2. Resolves it via confidence-weighted voting across sources
3. Breaks any remaining cycles by removing the weakest edge
4. Returns a clean, acyclic, annotated graph with full provenance

---

## How Conflicts Are Resolved

CausalMerge uses **confidence-weighted voting** to resolve directional conflicts.

For a conflict between A→B and B→A:

```
score(A→B) = Σ confidence_i × source_weight_i  for all sources supporting A→B
score(B→A) = Σ confidence_i × source_weight_i  for all sources supporting B→A

winner = argmax(score(A→B), score(B→A))
```

**Example:**
```
A→B:  graph_a (conf=0.90, weight=1.0)           → score = 0.90
B→A:  graph_b (conf=0.45, weight=0.7)           → score = 0.315

Winner: A→B  (0.90 > 0.315)
```

The losing direction is removed from the aggregated map before consensus computation.
Surviving edges carry an `is_disputed=True` flag so consumers know the edge had
disagreement in the source data.

---

## Comparison: Simple Union vs CausalMerge

| | Simple Union | CausalMerge |
|---|---|---|
| **Directional conflicts** | Both directions included — graph has cycles | Resolved by confidence-weighted voting |
| **Cycle detection** | None — cycles silently corrupt queries | Enforced: weakest cycle edge is removed |
| **Confidence merging** | Pick one source's confidence | Weighted average across all sources |
| **Source agreement** | Not tracked | Stored as `source_agreement` on each edge |
| **Provenance** | Lost | Full `MergeReport` with all decisions |
| **Disputed edges** | Not flagged | Marked `is_disputed=True` |
| **Output format** | Undefined | WhyNet-compatible JSON — pipe directly back in |
| **Usable for do-calculus** | Rarely | Always (guaranteed DAG) |

---

## Integration with OpenCausality Tools

CausalMerge is designed to work alongside **WhyNet** and other tools in the
OpenCausality ecosystem:

```
# Extract causal graphs from multiple documents using WhyNet
whynet extract --input report_a.txt --output graph_a.json
whynet extract --input report_b.txt --output graph_b.json
whynet extract --input report_c.txt --output graph_c.json

# Merge them with CausalMerge
causalmerge merge graph_a.json graph_b.json graph_c.json --output merged.json

# Ask counterfactual questions against the merged graph using WhyNet
whynet ask --graph merged.json "What would have happened if the index had not been missing?"
```

CausalMerge reads and writes the same JSON format WhyNet uses
(`{"nodes": [...], "edges": [{"cause": ..., "effect": ..., "confidence": ...}]}`),
so merged graphs can be fed directly back into WhyNet for counterfactual queries.

---

## Installation

### With pip

```bash
pip install causalmerge
```

### With uv (recommended — faster)

```bash
uv add causalmerge
```

### From source

```bash
git clone https://github.com/opencausality/causalmerge.git
cd causalmerge

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# Install with dev dependencies
pip install -e ".[dev]"
```

---

## Quick Start

### Python API

```python
from pathlib import Path
from causalmerge.config import Settings
from causalmerge.data.loader import load_graphs
from causalmerge.merge.engine import MergeEngine
from causalmerge.graph.builder import save_graph
from causalmerge.reporting.report import print_merge_report

# Load source graphs (WhyNet output format)
source_graphs = load_graphs(
    [Path("graph_a.json"), Path("graph_b.json"), Path("graph_c.json")],
    weights=[1.0, 0.7, 0.9],  # optional: weight each source
)

# Configure and run the merge
settings = Settings(confidence_threshold=0.4)
engine = MergeEngine(settings=settings)
merged_graph, report = engine.merge(source_graphs)

# Inspect the result
print_merge_report(report)
print(f"Merged graph: {merged_graph.number_of_nodes()} nodes, {merged_graph.number_of_edges()} edges")

# Save to JSON (WhyNet-compatible)
all_edges = report.consensus_edges + report.disputed_edges
save_graph(merged_graph, all_edges, Path("merged.json"))
```

### CLI

```bash
# Merge two graphs and save the result
causalmerge merge graph_a.json graph_b.json --output merged.json

# Merge three graphs with custom source weights and open a browser visualisation
causalmerge merge graph_a.json graph_b.json graph_c.json \
    --weights 1.0 0.7 0.9 \
    --output merged.json \
    --show

# Preview conflicts before committing to a merge
causalmerge conflicts graph_a.json graph_b.json

# Print a summary of a previously merged graph
causalmerge report --graph merged.json

# Visualise a merged graph in the browser
causalmerge show --graph merged.json

# Start the REST API server
causalmerge serve --port 8000
```

---

## Example Output

Running `causalmerge merge graph_a.json graph_b.json graph_c.json` on the
incident report fixture graphs:

```
╭──────────────────────────── CausalMerge Report ────────────────────────────╮
│ Sources merged:     3                                                        │
│ Edges before:       11                                                       │
│ Edges after:        6                                                        │
│ Conflicts found:    2                                                        │
│ Conflicts resolved: 2                                                        │
│ Cycles broken:      0                                                        │
│ Dropped edges:      0                                                        │
│ Created:            2026-03-24 10:15:42 UTC                                 │
╰─────────────────────────────────────────────────────────────────────────────╯

  Source Graphs
  ┌─────────┬────────┐
  │ Source  │ Weight │
  ├─────────┼────────┤
  │ graph_a │  1.00  │
  │ graph_b │  0.70  │
  │ graph_c │  0.90  │
  └─────────┴────────┘

  Consensus Edges
  ┌───────────────────────────────┬──────────┬─────────────────────────────────┐
  │ Cause                         │ Effect   │ Confidence │ Agreement │ Sources │
  ├───────────────────────────────┼──────────┼────────────┼───────────┼─────────┤
  │ slow_queries                  │ conn_... │     85%    │   100%    │ all 3   │
  │ connection_pool_exhaustion    │ timeout  │     90%    │    67%    │ a, c    │
  │ traffic_spike                 │ slow_... │     70%    │    67%    │ a, c    │
  │ missing_index                 │ slow_... │     88%    │    33%    │ c       │
  └───────────────────────────────┴──────────┴────────────┴───────────┴─────────┘

  Disputed Edges (conflict resolved)
  ┌──────────────────┬──────────────────┬────────────┬───────────┬──────────────┐
  │ Cause            │ Effect           │ Confidence │ Agreement │ Sources      │
  ├──────────────────┼──────────────────┼────────────┼───────────┼──────────────┤
  │ deploy_event     │ slow_queries     │     80%    │    33%    │ graph_b      │
  └──────────────────┴──────────────────┴────────────┴───────────┴──────────────┘
```

**Final merged graph (topological order):**
```
missing_index ──────────────────────────────┐
traffic_spike ──► high_cpu ──► slow_queries ──► connection_pool_exhaustion ──► timeout_errors
deploy_event  ──────────────────────────────┘
```

---

## CLI Reference

### `causalmerge merge`

Merge two or more causal graph JSON files.

```
Usage: causalmerge merge [OPTIONS] GRAPH...

Arguments:
  GRAPH...  Two or more causal graph JSON files to merge.

Options:
  -o, --output PATH           Save the merged graph JSON to this path.
  -w, --weights FLOAT...      Source weights, one per graph. [default: 1.0 each]
  -t, --threshold FLOAT       Minimum confidence to include an edge. [default: 0.4]
  -s, --show                  Open interactive visualisation after merging.
      --save-report PATH      Save the merge report JSON to this path.
```

### `causalmerge conflicts`

Preview directional conflicts between source graphs without merging.

```
Usage: causalmerge conflicts [OPTIONS] GRAPH...

Arguments:
  GRAPH...  Two or more causal graph JSON files to inspect.

Options:
  -w, --weights FLOAT...   Source weights, one per graph.
```

### `causalmerge show`

Visualise a merged graph in the browser (interactive pyvis HTML).

```
Usage: causalmerge show [OPTIONS]

Options:
  -g, --graph PATH    Path to a merged graph JSON file. [required]
  -o, --output PATH   Save the HTML to this path. [default: <graph>.html]
```

### `causalmerge report`

Print a summary table of all edges in a merged graph.

```
Usage: causalmerge report [OPTIONS]

Options:
  -g, --graph PATH   Path to a merged graph JSON file. [required]
```

### `causalmerge serve`

Start the optional REST API server.

```
Usage: causalmerge serve [OPTIONS]

Options:
  -h, --host TEXT    Listen address. [default: 127.0.0.1]
  -p, --port INT     Listen port. [default: 8000]
```

**API endpoints:**
- `GET  /health`  — liveness probe
- `POST /merge`   — merge graph payloads, returns unified graph
- `GET  /docs`    — Swagger UI

---

## Architecture

```
Input JSON Files (graph_a, graph_b, graph_c)
    │
    ▼
┌───────────────────────┐
│  Loader               │  ← Validates JSON, checks required keys,
│  data/loader.py       │    normalises source names
└───────────────────────┘
    │
    ▼ list[SourceGraph]
┌───────────────────────┐
│  Aggregator           │  ← Groups edges by (cause, effect) key,
│  merge/aggregator.py  │    normalises node names to lowercase
└───────────────────────┘
    │
    ▼ dict[(cause, effect), list[SourceEdge]]
┌───────────────────────┐
│  Conflict Detector    │  ← Finds all A→B vs B→A disagreements,
│  resolution/          │    computes weighted confidence sums
│  conflicts.py         │
└───────────────────────┘
    │
    ▼ list[EdgeConflict]
┌───────────────────────┐
│  Resolver             │  ← Picks winning direction, removes loser
│  resolution/          │    from the aggregated map
│  resolver.py          │
└───────────────────────┘
    │
    ▼ (cleaned aggregated map)
┌───────────────────────┐
│  Consensus Engine     │  ← Weighted-average confidence per edge,
│  merge/consensus.py   │    computes source_agreement fraction
└───────────────────────┘
    │
    ▼ list[MergedEdge]
┌───────────────────────┐
│  Graph Builder        │  ← Filters by confidence threshold,
│  graph/builder.py     │    constructs NetworkX DiGraph
└───────────────────────┘
    │
    ▼ nx.DiGraph (may have cycles)
┌───────────────────────┐
│  DAG Enforcer         │  ← Iteratively removes lowest-confidence
│  resolution/          │    cycle edge until graph is acyclic
│  dag_enforcer.py      │
└───────────────────────┘
    │
    ▼
merged_graph.json  +  MergeReport
```

---

## Configuration

Set options in `.env` or as environment variables (prefixed `CAUSALMERGE_`):

```env
# Minimum merged confidence to include an edge in the output
# Default: 0.4
CAUSALMERGE_CONFIDENCE_THRESHOLD=0.4

# Cycle resolution strategy
# Options: "min_weight" (remove lowest-confidence edge in cycle)
#          "oldest_source" (reserved for future use)
# Default: min_weight
CAUSALMERGE_CYCLE_RESOLUTION=min_weight

# Log level
# Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
# Default: INFO
CAUSALMERGE_LOG_LEVEL=INFO
```

You can also pass `Settings(...)` directly in code:

```python
from causalmerge.config import Settings
settings = Settings(confidence_threshold=0.6, log_level="DEBUG")
```

---

## Data Models

```python
# A causal edge from one source graph
SourceEdge(
    cause="slow_queries",
    effect="connection_pool_exhaustion",
    confidence=0.85,
    source_name="graph_a",
    source_weight=1.0,
    evidence="Slow queries held connections longer, exhausting the pool",
    edge_type="direct",
)

# An edge in the merged graph
MergedEdge(
    cause="slow_queries",
    effect="connection_pool_exhaustion",
    merged_confidence=0.8333,  # weighted average across sources
    source_agreement=1.0,       # all three sources agree
    contributing_sources=["graph_a", "graph_b", "graph_c"],
    is_disputed=False,
    edge_type="direct",
)
```

---

## Running Tests

```bash
# Run the full test suite
pytest

# With coverage
pytest --cov=causalmerge --cov-report=term-missing

# Run only a specific module
pytest tests/test_conflict_detector.py -v
```

---

## Contributing

CausalMerge is free for personal, educational, and research use (MIT license).

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Write tests for your change
4. Run `ruff check` and `mypy causalmerge/` before committing
5. Open a pull request with a clear description

If you are building a commercial product on top of CausalMerge, please consider
contributing upstream improvements or sponsoring the project.

---

*"The truth about causality emerges from the intersection of many incomplete perspectives."*
