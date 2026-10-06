"""Tests for the Gutenberg download script helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_download_data():
    path = Path(__file__).resolve().parents[1] / "scripts" / "download_data.py"
    spec = importlib.util.spec_from_file_location("download_data", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preserve_qa_dataset_does_not_overwrite(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    download_data = _load_download_data()
    qa_path = tmp_path / "qa_dataset.json"
    qa_path.write_text('{"name": "curated", "examples": []}', encoding="utf-8")

    download_data.preserve_qa_dataset(qa_path)

    assert qa_path.read_text(encoding="utf-8") == '{"name": "curated", "examples": []}'
    assert "not overwriting" in capsys.readouterr().out


def test_preserve_qa_dataset_warns_when_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    download_data = _load_download_data()
    qa_path = tmp_path / "qa_dataset.json"

    download_data.preserve_qa_dataset(qa_path)

    assert not qa_path.exists()
    assert "git checkout" in capsys.readouterr().out
