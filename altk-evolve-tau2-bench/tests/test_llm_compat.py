"""Tests for the tau2 temperature-compat shim.

These verify the name-based drop, the retry safety net, and idempotent install,
without any real litellm call: the shim wraps ``tau2.utils.llm_utils.completion``,
so we swap in a fake ``completion`` and assert on the kwargs it receives.
"""

from __future__ import annotations

import pytest

from benchmarks.tau2 import llm_compat


def test_rejects_temperature_matches_gpt5_family():
    assert llm_compat._rejects_temperature("litellm_proxy/azure/gpt-5.5")
    assert llm_compat._rejects_temperature("Azure/GPT-5-mini")
    assert llm_compat._rejects_temperature("gpt-5.3-codex")
    assert not llm_compat._rejects_temperature("litellm_proxy/Azure/gpt-4.1")
    assert not llm_compat._rejects_temperature("claude-opus-4-5")
    assert not llm_compat._rejects_temperature("")


@pytest.fixture
def fresh_shim(monkeypatch):
    """Reset install state and give llm_utils a recording fake completion."""
    from tau2.utils import llm_utils

    calls = []

    def fake_completion(*args, **kwargs):
        calls.append(dict(kwargs))
        return "ok"

    monkeypatch.setattr(llm_utils, "completion", fake_completion)
    monkeypatch.setattr(llm_compat, "_installed", False)
    llm_compat.install()
    return llm_utils, calls


def test_drops_temperature_for_gpt5(fresh_shim):
    llm_utils, calls = fresh_shim
    out = llm_utils.completion(model="litellm_proxy/azure/gpt-5.5", temperature=0.0, messages=[])
    assert out == "ok"
    assert "temperature" not in calls[0]
    assert calls[0]["model"] == "litellm_proxy/azure/gpt-5.5"


def test_keeps_temperature_for_non_gpt5(fresh_shim):
    llm_utils, calls = fresh_shim
    llm_utils.completion(model="litellm_proxy/Azure/gpt-4.1", temperature=0.0, messages=[])
    assert calls[0]["temperature"] == 0.0


def test_retry_safety_net_on_temperature_error(monkeypatch):
    """A temperature-strict model not matched by name is retried without temperature."""
    from tau2.utils import llm_utils

    calls = []

    def flaky_completion(*args, **kwargs):
        calls.append(dict(kwargs))
        if "temperature" in kwargs:
            raise RuntimeError("400: temperature is not supported for this model")
        return "recovered"

    monkeypatch.setattr(llm_utils, "completion", flaky_completion)
    monkeypatch.setattr(llm_compat, "_installed", False)
    llm_compat.install()

    # "o3-mini" is temperature-strict but NOT matched by _rejects_temperature.
    out = llm_utils.completion(model="litellm_proxy/o3-mini", temperature=0.0, messages=[])
    assert out == "recovered"
    assert len(calls) == 2  # first attempt (with temp) failed, retry (without) succeeded
    assert "temperature" in calls[0]
    assert "temperature" not in calls[1]


def test_unrelated_error_is_not_retried(monkeypatch):
    from tau2.utils import llm_utils

    calls = []

    def failing_completion(*args, **kwargs):
        calls.append(dict(kwargs))
        raise RuntimeError("429: rate limit exceeded")

    monkeypatch.setattr(llm_utils, "completion", failing_completion)
    monkeypatch.setattr(llm_compat, "_installed", False)
    llm_compat.install()

    with pytest.raises(RuntimeError, match="rate limit"):
        llm_utils.completion(model="litellm_proxy/Azure/gpt-4.1", temperature=0.0, messages=[])
    assert len(calls) == 1  # not retried


def test_resolve_nl_judge_precedence(monkeypatch):
    # explicit arg wins over everything
    monkeypatch.setenv("TAU2_LLM_NL_ASSERTIONS", "litellm_proxy/env-model")
    monkeypatch.setenv("MODEL_NAME", "litellm_proxy/agent-model")
    assert llm_compat.resolve_nl_judge("litellm_proxy/arg-model") == "litellm_proxy/arg-model"
    # then env var
    assert llm_compat.resolve_nl_judge() == "litellm_proxy/env-model"
    # then MODEL_NAME
    monkeypatch.delenv("TAU2_LLM_NL_ASSERTIONS", raising=False)
    assert llm_compat.resolve_nl_judge() == "litellm_proxy/agent-model"
    # then the hardcoded default
    monkeypatch.delenv("MODEL_NAME", raising=False)
    assert llm_compat.resolve_nl_judge() == llm_compat.DEFAULT_NL_JUDGE


def test_route_nl_judge_rebinds_evaluator(monkeypatch):
    from tau2.evaluator import evaluator_nl_assertions as ev

    monkeypatch.delenv("TAU2_LLM_NL_ASSERTIONS", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    used = llm_compat.route_nl_judge("litellm_proxy/azure/gpt-5.5")
    assert used == "litellm_proxy/azure/gpt-5.5"
    assert ev.DEFAULT_LLM_NL_ASSERTIONS == "litellm_proxy/azure/gpt-5.5"


def test_install_is_idempotent(monkeypatch):
    from tau2.utils import llm_utils

    monkeypatch.setattr(llm_compat, "_installed", False)
    llm_compat.install()
    wrapped_once = llm_utils.completion
    llm_compat.install()  # second call must not double-wrap
    assert llm_utils.completion is wrapped_once
