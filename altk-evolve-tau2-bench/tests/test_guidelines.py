"""Tests for playbook-JSON guideline loading and special_instructions merge.

This repo ships no playbook (ReAct-specific ones are learned later), so all cases
use ``tmp_path``.
"""

from __future__ import annotations

import json

import pytest

from benchmarks.tau2.guidelines import (
    GUIDELINES_HEADING,
    load_guidelines_section,
    merge_special_instructions,
)


def _write_playbook(tmp_path, entries) -> str:
    p = tmp_path / "pb.json"
    p.write_text(json.dumps({"entries": entries}))
    return str(p)


def test_support_thresholds_select_correctly(tmp_path):
    entries = [
        {"r": "a", "n": 1},
        {"r": "b", "n": 2},
        {"r": "c", "n": 3},
    ]
    path = _write_playbook(tmp_path, entries)

    lossless = load_guidelines_section(path, "lossless")
    assert "- a" in lossless and "- b" in lossless and "- c" in lossless

    sup2 = load_guidelines_section(path, "sup2")
    assert "- a" not in sup2 and "- b" in sup2 and "- c" in sup2

    sup3 = load_guidelines_section(path, "sup3")
    assert "- a" not in sup3 and "- b" not in sup3 and "- c" in sup3


def test_missing_n_defaults_to_1(tmp_path):
    path = _write_playbook(tmp_path, [{"r": "x"}])
    assert "- x" in load_guidelines_section(path, "lossless")
    assert load_guidelines_section(path, "sup2") is None  # n defaults to 1, below 2


def test_empty_selection_returns_none(tmp_path):
    path = _write_playbook(tmp_path, [{"r": "y", "n": 1}])
    assert load_guidelines_section(path, "sup3") is None


def test_unknown_type_raises(tmp_path):
    path = _write_playbook(tmp_path, [{"r": "z", "n": 1}])
    with pytest.raises(ValueError):
        load_guidelines_section(path, "bogus")


def test_heading_present(tmp_path):
    path = _write_playbook(tmp_path, [{"r": "q", "n": 1}])
    section = load_guidelines_section(path, "lossless")
    assert section.startswith(GUIDELINES_HEADING)


def test_merge_orders_base_then_guidelines():
    merged = merge_special_instructions("BASE POLICY", GUIDELINES_HEADING + "- rule\n")
    assert merged.index("BASE POLICY") < merged.index("- rule")


def test_merge_handles_none():
    assert merge_special_instructions(None, None) is None
    assert merge_special_instructions("only base", None) == "only base"
    section = GUIDELINES_HEADING + "- r\n"
    assert merge_special_instructions(None, section) == section
