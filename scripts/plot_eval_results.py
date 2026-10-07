#!/usr/bin/env python3
"""Plot retrieval metrics from `scripts/run_eval.py` JSON output.

Reads the harness JSON (`results[].retrieval_metrics` plus `avg_latency_ms`) and
writes theme-aware PNGs: Recall@1 / Recall@5 / MRR as a dot plot, latency as
log-scale bars.

    python scripts/plot_eval_results.py --input results/offline_eval.json --outdir docs
    python scripts/plot_eval_results.py --theme light --output docs/retrieval-light.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

THEMES: dict[str, dict[str, str]] = {
    "light": {
        "text": "#0F172A",
        "muted": "#475569",
        "grid": "#E2E8F0",
        "accent": "#0F766E",
        "accent_2": "#5EEAD4",
        "neutral": "#334155",
        "latency": "#94A3B8",
    },
    "dark": {
        "text": "#E2E8F0",
        "muted": "#94A3B8",
        "grid": "#1E293B",
        "accent": "#2DD4BF",
        "accent_2": "#99F6E4",
        "neutral": "#CBD5E1",
        "latency": "#64748B",
    },
}

HIGHLIGHT = "Hybrid (BM25=0.3)"


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


def plot_rows(
    rows: list[dict[str, object]],
    output: Path,
    title: str | None = None,
    *,
    theme: str = "light",
) -> None:
    """Write a two-panel dot plot + log-latency chart. `title` is ignored."""
    del title
    if theme not in THEMES:
        raise ValueError(f"Unknown theme {theme!r}; expected one of {sorted(THEMES)}")

    colors = THEMES[theme]
    names = [str(r["name"]) for r in rows]
    recall_1 = [float(r["recall@1"]) for r in rows]
    recall_5 = [float(r["recall@5"]) for r in rows]
    mrr = [float(r["mrr"]) for r in rows]
    latency = [float(r["latency_ms"]) for r in rows]
    y = list(range(len(rows)))[::-1]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "text.color": colors["text"],
            "axes.labelcolor": colors["muted"],
            "xtick.color": colors["muted"],
            "ytick.color": colors["text"],
        }
    )
    fig, (ax, ax2) = plt.subplots(
        1,
        2,
        figsize=(11, 4.6),
        dpi=150,
        sharey=True,
        gridspec_kw={"width_ratios": [2.4, 1], "wspace": 0.14},
    )
    fig.patch.set_alpha(0)
    for axis in (ax, ax2):
        axis.set_facecolor("none")
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.tick_params(length=0)
        axis.grid(axis="x", color=colors["grid"], linewidth=1)
        axis.minorticks_off()
        axis.set_axisbelow(True)

    if HIGHLIGHT in names:
        hi = names.index(HIGHLIGHT)
        for axis in (ax, ax2):
            axis.axhspan(
                y[hi] - 0.42,
                y[hi] + 0.42,
                color=colors["accent"],
                alpha=0.10,
                lw=0,
                zorder=0,
            )

    for i in range(len(rows)):
        lo = min(recall_1[i], recall_5[i], mrr[i])
        hi_v = max(recall_1[i], recall_5[i], mrr[i])
        ax.plot(
            [lo, hi_v],
            [y[i], y[i]],
            color=colors["grid"],
            lw=3,
            solid_capstyle="round",
            zorder=1,
        )

    ax.scatter(recall_1, y, s=90, color=colors["accent"], label="Recall@1", zorder=3)
    ax.scatter(mrr, y, s=90, color=colors["neutral"], marker="D", label="MRR", zorder=3)
    ax.scatter(
        recall_5,
        y,
        s=90,
        facecolor="none",
        edgecolor=colors["accent"],
        linewidth=2,
        label="Recall@5",
        zorder=3,
    )
    ax.set_xlim(0.70, 1.0)
    ax.set_xticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    for lbl in ax.get_yticklabels():
        if lbl.get_text() == HIGHLIGHT:
            lbl.set_fontweight("bold")
    ax.set_xlabel("Score (higher is better)")
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
        labelcolor=colors["text"],
        handletextpad=0.3,
        columnspacing=1.4,
    )

    ax2.barh(y, latency, height=0.5, color=colors["latency"])
    ax2.set_xscale("log")
    ax2.set_xlim(1, 3000)
    ax2.set_xlabel("Avg latency, ms (log)")
    for yi, ms in zip(y, latency):
        ax2.text(ms * 1.25, yi, f"{ms:.0f} ms", va="center", fontsize=10, color=colors["text"])
    ax2.set_xticks([1, 10, 100, 1000])
    ax2.set_xticklabels(["1", "10", "100", "1k"])

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", transparent=True, pad_inches=0.15)
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
        default=None,
        help="PNG path (single --theme only; ignored when --theme both)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("docs"),
        help="Directory for retrieval-{light,dark}.png when --theme both",
    )
    parser.add_argument(
        "--theme",
        choices=("light", "dark", "both"),
        default="both",
        help="Colour theme (default: both)",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Ignored; charts have no in-image title (kept for CLI compatibility)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}. Run scripts/run_eval.py first.")

    rows = load_rows(args.input)
    themes = ("light", "dark") if args.theme == "both" else (args.theme,)
    for theme in themes:
        if args.theme != "both" and args.output is not None:
            dest = args.output
        else:
            dest = args.outdir / f"retrieval-{theme}.png"
        plot_rows(rows, dest, args.title, theme=theme)
        print(f"Wrote {dest} ({dest.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
