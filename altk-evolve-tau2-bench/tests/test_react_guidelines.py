"""Tests for guideline injection into the ReAct agent's system prompt.

These exercise ``GuidedReActAgent.system_prompt`` composition and the
``create_guided_react_agent`` factory without any LLM call — construction and
prompt assembly are pure. They assert the ordering contract: policy first, then
optional extra instructions, then learned guidelines.
"""

from __future__ import annotations

import json

from benchmarks.tau2 import react_agent
from benchmarks.tau2.guidelines import GUIDELINES_HEADING
from benchmarks.tau2.react_agent import (
    _POLICY_OMITTED,
    GuidedReActAgent,
    ReactTau2Config,
    create_guided_react_agent,
)

_POLICY = "RETAIL POLICY: authenticate before acting."


def _agent(**kwargs) -> GuidedReActAgent:
    return GuidedReActAgent(tools=[], domain_policy=_POLICY, **kwargs)


def _write_playbook(tmp_path, entries) -> str:
    p = tmp_path / "pb.json"
    p.write_text(json.dumps({"entries": entries}))
    return str(p)


def test_policy_present_in_system_prompt():
    sp = _agent().system_prompt
    assert _POLICY in sp
    assert "<policy>" in sp and "</policy>" in sp


def test_no_guidelines_leaves_base_prompt_untouched():
    sp = _agent(guidelines_section=None).system_prompt
    assert GUIDELINES_HEADING not in sp


def test_guidelines_appended_after_policy():
    section = GUIDELINES_HEADING + "- authenticate the user first\n"
    sp = _agent(guidelines_section=section).system_prompt
    # policy comes before the guidelines section
    assert sp.index(_POLICY) < sp.index(GUIDELINES_HEADING)
    assert "- authenticate the user first" in sp


def test_extra_instructions_sit_between_policy_and_guidelines():
    section = GUIDELINES_HEADING + "- rule\n"
    sp = _agent(
        guidelines_section=section,
        extra_special_instructions="EXTRA NOTE",
    ).system_prompt
    assert sp.index(_POLICY) < sp.index("EXTRA NOTE") < sp.index(GUIDELINES_HEADING)


def test_no_domain_policy_blanks_policy_body():
    sp = _agent(include_domain_policy=False).system_prompt
    assert _POLICY not in sp
    assert _POLICY_OMITTED in sp


def test_no_domain_policy_still_injects_guidelines():
    section = GUIDELINES_HEADING + "- only rule\n"
    sp = _agent(include_domain_policy=False, guidelines_section=section).system_prompt
    assert _POLICY not in sp
    assert "- only rule" in sp


def test_get_init_state_uses_composed_system_prompt():
    section = GUIDELINES_HEADING + "- r\n"
    agent = _agent(guidelines_section=section)
    state = agent.get_init_state()
    assert len(state.system_messages) == 1
    assert state.system_messages[0].content == agent.system_prompt
    assert state.messages == []


def test_factory_loads_playbook_from_config(tmp_path, monkeypatch):
    path = _write_playbook(
        tmp_path, [{"r": "recurring rule", "n": 2}, {"r": "rare rule", "n": 1}]
    )
    monkeypatch.setattr(
        react_agent,
        "CONFIG",
        ReactTau2Config(playbook_path=path, guidelines_type="sup2"),
    )
    agent = create_guided_react_agent(
        tools=[], domain_policy=_POLICY, llm="litellm_proxy/Azure/gpt-4.1"
    )
    sp = agent.system_prompt
    assert "- recurring rule" in sp
    assert "- rare rule" not in sp  # n=1 filtered out by sup2
    assert agent.llm == "litellm_proxy/Azure/gpt-4.1"


def test_factory_without_playbook_injects_no_guidelines(monkeypatch):
    monkeypatch.setattr(react_agent, "CONFIG", ReactTau2Config(playbook_path=None))
    agent = create_guided_react_agent(tools=[], domain_policy=_POLICY, llm="m")
    assert GUIDELINES_HEADING not in agent.system_prompt


def test_factory_defaults_llm_when_none(monkeypatch):
    monkeypatch.setattr(react_agent, "CONFIG", ReactTau2Config())
    agent = create_guided_react_agent(tools=[], domain_policy=_POLICY, llm=None)
    assert agent.llm == "openai/gpt-4.1-mini"


def test_register_is_idempotent(monkeypatch):
    monkeypatch.setattr(react_agent, "_REGISTERED", False)
    from tau2.registry import registry

    react_agent.register("react_agent")
    react_agent.register("react_agent")  # second call must not raise
    assert registry.get_agent_factory("react_agent") is not None
