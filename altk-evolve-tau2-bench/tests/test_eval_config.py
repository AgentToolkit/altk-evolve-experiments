"""Tests for eval_tau2's split resolution and run-name/dir derivation (no LLM needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.tau2 import eval_tau2


@pytest.mark.parametrize(
    "value,expected",
    [
        ("test", "test"),
        ("train", "train"),
        ("base", "base"),
        ("  small  ", "small"),  # trimmed
        (None, None),
        ("", None),
        ("none", None),
        ("None", None),
        ("all", None),
        ("ALL", None),
    ],
)
def test_resolve_split(value, expected):
    assert eval_tau2._resolve_split(value) == expected


def test_default_run_name_includes_split(monkeypatch):
    monkeypatch.setenv("EVAL_RUN_ID", "20260810_120000")
    assert eval_tau2._default_run_name("retail", "test") == "react_retail_test_20260810_120000"
    # A None split (no filter) is labelled 'all' so the name stays slash-free and descriptive.
    assert eval_tau2._default_run_name("mock", None) == "react_mock_all_20260810_120000"


def test_split_default_is_test():
    assert eval_tau2._DEFAULTS["split"] == "test"


def test_llm_agent_in_defaults():
    # The ReAct agent's model IS tau2's llm_agent, so the eval defaults expose it
    # as a wiring key.
    assert "llm_agent" in eval_tau2._DEFAULTS


def test_resolve_run_dir_defaults_into_repo_runs():
    # No runs_dir => <repo>/runs/<run_name>, absolute, and NOT under tau2-bench.
    run_dir = eval_tau2._resolve_run_dir(None, "react_retail_test_x")
    assert run_dir.is_absolute()
    assert run_dir == eval_tau2._DEFAULT_RUNS_DIR / "react_retail_test_x"
    assert run_dir.parent == eval_tau2._REPO_ROOT / "runs"
    assert "tau2-bench" not in str(run_dir)


def test_resolve_run_dir_honours_custom_base(tmp_path):
    run_dir = eval_tau2._resolve_run_dir(str(tmp_path), "myrun")
    assert run_dir == (tmp_path / "myrun").resolve()


def test_resolve_run_dir_honours_absolute_run_name(tmp_path):
    # An absolute run_name (e.g. absolute --save-to) is used verbatim.
    abs_name = str(tmp_path / "explicit")
    assert eval_tau2._resolve_run_dir("ignored", abs_name) == Path(abs_name)


def test_results_path_is_run_dir_results_json():
    run_dir = eval_tau2._resolve_run_dir(None, "r")
    assert (run_dir / "results.json").name == "results.json"


def test_runs_dir_default_is_none():
    assert eval_tau2._DEFAULTS["runs_dir"] is None
