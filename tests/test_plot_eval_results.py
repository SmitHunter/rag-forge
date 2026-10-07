"""Tests for scripts/plot_eval_results.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "plot_eval_results.py"


def _load_plot_module():
    spec = importlib.util.spec_from_file_location("plot_eval_results", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plot_rows_writes_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    plot = _load_plot_module()
    rows = [
        {
            "name": "BM25 Only",
            "recall@1": 0.72,
            "recall@5": 0.92,
            "mrr": 0.83,
            "latency_ms": 3.0,
        },
        {
            "name": "Hybrid (BM25=0.3)",
            "recall@1": 0.86,
            "recall@5": 0.96,
            "mrr": 0.94,
            "latency_ms": 8.0,
        },
    ]
    out = tmp_path / "retrieval-light.png"
    plot.plot_rows(rows, out, theme="light")
    assert out.is_file()
    assert out.stat().st_size > 0
