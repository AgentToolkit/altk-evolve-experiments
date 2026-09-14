"""tau2's native ReAct agent + playbook-guideline injection.

The ``ReActAgent`` below is **vendored verbatim** from tau2-bench's shipped
example ``examples/agents/react_agent.py`` (two-phase THINK -> ACT loop). That
file is a standalone script, not part of tau2's importable package
(``examples/`` has no ``__init__`` and is not built), and the agent is not in
tau2's default registry, so we copy it here to depend on it cleanly. Keep the
``ReActAgent`` body in sync with upstream if tau2 updates it.

On top of the stock agent we add:

* ``GuidedReActAgent`` — a subclass that appends a **playbook-guidelines**
  section to the system prompt, *after* the policy. The guidelines text is
  produced by ``guidelines.py``.
* ``ReactTau2Config`` / ``CONFIG`` — process-level injection knobs the factory
  reads.
* ``create_guided_react_agent`` + ``register`` — the tau2 factory/registration.

This IS a native tau2 agent driven by ``tau2.utils.llm_utils.generate``, so its
model is tau2's ``llm_agent`` (a ``litellm_proxy/`` string), set by the eval
runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from loguru import logger

from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
    UserMessage,
)
from tau2.environment.toolkit import Tool
from tau2.utils.llm_utils import generate

# =============================================================================
# Prompts (verbatim from tau2-bench examples/agents/react_agent.py)
# =============================================================================

SYSTEM_PROMPT = """\
You are a customer service agent that helps users according to the policy below.
Always follow the policy strictly. Use the provided tools when needed.

<policy>
{domain_policy}
</policy>"""

THINK_PROMPT = """\
Before responding, reason step-by-step about what to do next.

Consider:
- What is the user asking for?
- What information do I already have?
- What tool should I call, or should I respond directly?
- Does the policy require any specific steps?

Write your reasoning concisely."""

ACT_PROMPT = """\
Based on your reasoning above, now take action:
- If you need information or need to perform an operation, call the appropriate tool.
- If you have all the information needed, respond to the user directly.

Your reasoning was:
{reasoning}"""


# =============================================================================
# ReAct Agent (verbatim from tau2-bench examples/agents/react_agent.py)
# =============================================================================


class ReActAgentState:
    """Conversation state for the ReAct agent."""

    def __init__(
        self,
        system_messages: list[SystemMessage],
        messages: list[APICompatibleMessage],
    ):
        self.system_messages = system_messages
        self.messages = messages


class ReActAgent(HalfDuplexAgent[ReActAgentState]):
    """A ReAct agent that reasons before acting.

    Each turn follows the pattern:
        1. THINK: Ask the LLM to reason about the situation (no tools).
        2. ACT: Ask the LLM to take action based on its reasoning (with tools).

    The reasoning trace is injected into the action prompt so the LLM
    can refer to its own analysis when choosing a tool or composing a response.
    """

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str = "openai/gpt-4.1-mini",
        llm_args: Optional[dict] = None,
    ):
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.llm = llm
        self.llm_args = llm_args or {}

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> ReActAgentState:
        system_content = SYSTEM_PROMPT.format(domain_policy=self.domain_policy)
        system_messages = [SystemMessage(role="system", content=system_content)]
        messages = list(message_history) if message_history else []

        return ReActAgentState(
            system_messages=system_messages,
            messages=messages,
        )

    def generate_next_message(
        self,
        message: ValidAgentInputMessage,
        state: ReActAgentState,
    ) -> tuple[AssistantMessage, ReActAgentState]:
        # Add incoming message(s) to state
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        # Phase 1: THINK -- reason about the situation (no tools)
        reasoning = self._think(state)
        logger.debug(f"[ReAct] Reasoning: {reasoning[:200]}")

        # Phase 2: ACT -- choose an action based on reasoning (with tools)
        assistant_message = self._act(state, reasoning)

        # Store the action in conversation history
        # (reasoning is ephemeral -- not stored in history)
        state.messages.append(assistant_message)

        return assistant_message, state

    def _think(self, state: ReActAgentState) -> str:
        """Phase 1: Generate a reasoning trace without tools."""
        think_messages = (
            state.system_messages
            + state.messages
            + [UserMessage(role="user", content=THINK_PROMPT)]
        )

        # Call LLM without tools -- forces text-only reasoning
        response = generate(
            model=self.llm,
            tools=[],  # No tools available during thinking
            messages=think_messages,
            call_name="react_think",
            **self.llm_args,
        )

        return str(response.content) if response.content else ""

    def _act(self, state: ReActAgentState, reasoning: str) -> AssistantMessage:
        """Phase 2: Choose an action based on the reasoning trace."""
        act_instruction = ACT_PROMPT.format(reasoning=reasoning)
        act_messages = (
            state.system_messages
            + state.messages
            + [UserMessage(role="user", content=act_instruction)]
        )

        # Call LLM with tools -- can choose to call a tool or respond
        response = generate(
            model=self.llm,
            tools=self.tools,
            messages=act_messages,
            call_name="react_act",
            **self.llm_args,
        )

        return response


# =============================================================================
# Runtime configuration (guidelines / policy injection)
# =============================================================================

# Shown in place of the policy body when the domain-policy ablation is on. The
# ReAct system prompt hard-codes a <policy> block, so we blank the body rather
# than delete the tags.
_POLICY_OMITTED = "(Domain policy intentionally omitted for this ablation.)"


@dataclass
class ReactTau2Config:
    """Process-level knobs the factory reads when building the agent.

    ``build_agent`` (tau2's builder) does not forward a guidelines argument, so
    the eval runner sets these before launching a run. Fields also fall back to
    environment variables so a bare ``tau2 run`` still works.
    """

    playbook_path: Optional[str] = None
    guidelines_type: str = "lossless"
    include_domain_policy: bool = True
    extra_special_instructions: Optional[str] = None

    @classmethod
    def from_env(cls) -> "ReactTau2Config":
        import os

        return cls(
            playbook_path=os.getenv("REACT_TAU2_PLAYBOOK") or None,
            guidelines_type=os.getenv("REACT_TAU2_GUIDELINES_TYPE", "lossless"),
            include_domain_policy=os.getenv("REACT_TAU2_INCLUDE_POLICY", "1") != "0",
            extra_special_instructions=os.getenv("REACT_TAU2_EXTRA_INSTRUCTIONS") or None,
        )


# Mutable process-level config; the eval runner may overwrite this.
CONFIG = ReactTau2Config.from_env()


# =============================================================================
# Guideline-injecting ReAct agent
# =============================================================================


class GuidedReActAgent(ReActAgent):
    """A ReAct agent whose system prompt carries an injected playbook.

    Identical THINK/ACT machinery as the vendored ``ReActAgent``; the only
    difference is the composed system prompt: the stock ReAct system prompt
    (which already contains ``<policy>...</policy>``) plus, appended after it, an
    optional ``## Guidelines learned from past task attempts...`` section built
    from a playbook JSON. Guidelines therefore ride in the system prompt for both
    the THINK and ACT phases.

    The ordering is policy, then optional extra instructions, then learned
    guidelines, joined via ``guidelines.merge_special_instructions``.
    """

    def __init__(
        self,
        tools: list[Tool],
        domain_policy: str,
        llm: str = "openai/gpt-4.1-mini",
        llm_args: Optional[dict] = None,
        guidelines_section: Optional[str] = None,
        include_domain_policy: bool = True,
        extra_special_instructions: Optional[str] = None,
    ):
        super().__init__(
            tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args
        )
        self._guidelines_section = guidelines_section
        self._include_domain_policy = include_domain_policy
        self._extra_special_instructions = extra_special_instructions

    @property
    def system_prompt(self) -> str:
        from benchmarks.tau2.guidelines import merge_special_instructions

        policy = self.domain_policy if self._include_domain_policy else _POLICY_OMITTED
        parts: List[str] = [SYSTEM_PROMPT.format(domain_policy=policy)]
        if self._extra_special_instructions:
            parts.append(self._extra_special_instructions.strip())
        base = "\n\n".join(parts)
        return merge_special_instructions(base, self._guidelines_section) or base

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> ReActAgentState:
        system_messages = [SystemMessage(role="system", content=self.system_prompt)]
        messages = list(message_history) if message_history else []
        return ReActAgentState(system_messages=system_messages, messages=messages)


# =============================================================================
# Factory + registration
# =============================================================================


def create_guided_react_agent(tools, domain_policy, **kwargs):
    """tau2 agent factory: ``factory(tools, domain_policy, **kwargs) -> agent``.

    Reads the process-level ``CONFIG`` for playbook/guideline injection. The
    ``llm`` kwarg is tau2's ``llm_agent`` (a real ``litellm_proxy/`` model set by
    the eval runner).
    """
    from benchmarks.tau2.guidelines import load_guidelines_section

    config = CONFIG
    guidelines_section = None
    if config.playbook_path:
        guidelines_section = load_guidelines_section(
            config.playbook_path, config.guidelines_type
        )

    llm = kwargs.get("llm") or "openai/gpt-4.1-mini"
    return GuidedReActAgent(
        tools=tools,
        domain_policy=domain_policy,
        llm=llm,
        llm_args=kwargs.get("llm_args"),
        guidelines_section=guidelines_section,
        include_domain_policy=config.include_domain_policy,
        extra_special_instructions=config.extra_special_instructions,
    )


_REGISTERED = False


def register(name: str = "react_agent") -> None:
    """Register the guided ReAct agent factory in tau2's registry (idempotent)."""
    global _REGISTERED
    if _REGISTERED:
        return
    from tau2.registry import registry

    if registry.get_agent_factory(name) is None:
        registry.register_agent_factory(create_guided_react_agent, name)
    _REGISTERED = True
