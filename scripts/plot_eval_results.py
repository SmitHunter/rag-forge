#!/usr/bin/env python3
"""Plot grouped retrieval metrics from `scripts/run_eval.py` JSON output.

Reads the harness JSON (`results[].retrieval_metrics` plus `avg_latency_ms`) and
writes a PNG: Recall@1 / Recall@5 / MRR as grouped bars, latency as a second axis.

    python scripts/run_eval.py --offline --output eval_results.json
    python scripts/plot_eval_results.py --input eval_results.json \\
        --output docs/offline_retrieval_metrics.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_rows(path: Path) -> list[dict[str, object]]:
    """Load one row per config from EvalHarness.save_results JSON."""
    with path.open() as f:
        payload = json.load(f)

    rows: list[dict[str, object]] = []
    for result in payload["results"]:
        metrics = result["retrieval_metrics"]
        rows.append(
            {
                "name": result["config_name"],
                "recall@1": float(metrics["recall@1"]),
                "recall@5": float(metrics["recall@5"]),
                "mrr": float(metrics["mrr"]),
                "latency_ms": float(result["avg_latency_ms"]),
            }
        )
    if not rows:
        raise ValueError(f"No results in {path}")
    return rows


def plot_rows(rows: list[dict[str, object]], output: Path, title: str) -> None:
    """Write a grouped-bar chart with latency on a second y-axis."""
    names = [str(r["name"]) for r in rows]
    recall_1 = np.array([float(r["recall@1"]) for r in rows])
    recall_5 = np.array([float(r["recall@5"]) for r in rows])
    mrr = np.array([float(r["mrr"]) for r in rows])
    latency = np.array([float(r["latency_ms"]) for r in rows])

    x = np.arange(len(names))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=140)
    ax.bar(x - width, recall_1, width, label="Recall@1", color="#1f4e79")
    ax.bar(x, recall_5, width, label="Recall@5", color="#4f81bd")
    ax.bar(x + width, mrr, width, label="MRR", color="#9dc3e6")

    ax.set_ylabel("Score")
    ax.set_ylim(0.6, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=18, ha="right")
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    ax2 = ax.twinx()
    ax2.plot(x, latency, color="#c45911", marker="o", linewidth=1.8, label="Latency")
    ax2.set_ylabel("Avg latency (ms)", color="#c45911")
    ax2.tick_params(axis="y", colors="#c45911")
    ax2.set_ylim(0, max(latency) * 1.25)

    for xi, ms in zip(x, latency, strict=True):
        ax2.annotate(
            f"{ms:.0f} ms",
            (xi, ms),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=8,
            color="#c45911",
        )

    handles, labels = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(
        handles + h2,
        labels + l2,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot retrieval metrics from eval JSON")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("eval_results.json"),
        help="JSON written by run_eval.py / EvalHarness.save_results",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/offline_retrieval_metrics.png"),
        help="PNG path",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Offline retrieval (76 questions, 5 novels, MiniLM embeddings)",
        help="Chart title",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}. Run scripts/run_eval.py first.")

    rows = load_rows(args.input)
    plot_rows(rows, args.output, args.title)
    print(f"Wrote {args.output} ({args.output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
