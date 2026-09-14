# Architecture & configuration: ReAct on τ²-bench

This document records **every architectural decision** in how tau2-bench's own
**native ReAct agent** is evaluated on τ²-bench (tau2) by this repo: what the
agent is, how its turn works, exactly where the playbook is injected, and how the
models are wired.

This file is descriptive of the code as it stands; file/line references point at
the authoritative source. If you change `config/tau2.env`, `eval.sh`, or
`react_agent.py`, update this file.

---

## 1. The agent: tau2's native ReAct, vendored

tau2 ships a ReAct agent as a **standalone example** at
`tau2-bench/examples/agents/react_agent.py`. It is *not* usable as a dependency:
`examples/` has no `__init__.py` and is not part of tau2's built package, and the
agent self-registers only under `if __name__ == "__main__"`. So we **vendor the
`ReActAgent` class verbatim** into `benchmarks/tau2/react_agent.py` (prompts +
`ReActAgentState` + THINK/ACT machinery), with a header noting the source. Keep
it in sync with upstream if tau2 revises it.

Because it is a native tau2 `HalfDuplexAgent`, its tool calls, trajectory, and
reward all flow through tau2's ordinary runner and trajectory-replay evaluator —
there is **no bridge**, no threading, no rendezvous queue, no per-round event
loop. The agent plugs directly into tau2's turn-based protocol.

### The ReAct turn (2 LLM calls)

`generate_next_message` runs two phases per turn (`react_agent.py`):

1. **THINK** (`_think`): calls `generate(model=self.llm, tools=[], ...)` — no
   tools, forcing a text-only reasoning trace. The trace is *ephemeral*: it is
   not stored in conversation history.
2. **ACT** (`_act`): calls `generate(model=self.llm, tools=self.tools, ...)` with
   the reasoning injected into the ACT prompt. The model may emit tool calls or a
   text reply; that `AssistantMessage` is appended to history and returned.

So every turn costs **two completions**. A run therefore costs roughly twice the
completions of a single-call agent for the same number of turns.

---

## 2. Model & credential wiring

Set in `benchmarks/tau2/config/tau2.env` (non-secret) + `../.env` (secrets),
assembled by `eval.sh`.

| Role | Model | Notes |
|---|---|---|
| **ReAct agent** | `litellm_proxy/Azure/gpt-4.1` (`TAU2_LLM_AGENT`) | The agent's model **is** tau2's `llm_agent`; the eval runner sets `TextRunConfig(llm_agent=...)` and tau2's `build_agent` forwards it to the factory as `llm=`. |
| **τ² user simulator** | `litellm_proxy/azure/gpt-5.5` (`TAU2_LLM_USER`) | Drives the conversation as the "customer". |
| **NL-assertion judge** | `litellm_proxy/azure/gpt-5.5` (`TAU2_LLM_NL_ASSERTIONS`) | Rerouted by `llm_compat` off tau2's bare-OpenAI default. |

**Model wiring.** The ReAct agent is native, so its model is exactly tau2's
`llm_agent`. We set it to `litellm_proxy/Azure/gpt-4.1`; whatever is passed as
`--agent-llm` (or `TAU2_LLM_AGENT`) is the model the agent runs on, with no
separate agent-side model selection.

`eval.sh` derives the provider variables and, importantly, does **not** set
`OPENAI_BASE_URL` / `LITELLM_API_BASE` (that would force litellm to treat the
`litellm_proxy/...` string as an OpenAI model id, which the proxy rejects). Model
casing (`Azure/gpt-4.1`, capital A) is enforced by the proxy allowlist. See the
README for both pitfalls.

---

## 3. Where the playbook is injected

The stock ReAct `SYSTEM_PROMPT` already embeds the domain policy in a
`<policy>...</policy>` block, and the system prompt is present in **both** the
THINK and ACT phases. So the natural injection point for learned guidelines is
the system prompt, appended **after** the policy.

`GuidedReActAgent(ReActAgent)` overrides only the system-prompt composition
(`system_prompt` property + `get_init_state`). It builds:

1. the stock `SYSTEM_PROMPT.format(domain_policy=policy)` (policy body blanked to
   `_POLICY_OMITTED` when `include_domain_policy=False` — the `--no-domain-policy`
   ablation),
2. optional `extra_special_instructions`,
3. then `merge_special_instructions(base, guidelines_section)` appends the
   `## Guidelines learned from past task attempts...` section.

This ordering (policy → extra → guidelines) is implemented in `guidelines.py`
(`load_guidelines_section`, `merge_special_instructions`). Guidelines land as a
distinct section, never inside `<policy>`.

`ReactTau2Config` / module-level `CONFIG` carry the injection knobs
(`playbook_path`, `guidelines_type`, `include_domain_policy`,
`extra_special_instructions`), settable from env (`REACT_TAU2_*`) or overwritten
by the eval runner before a run. `create_guided_react_agent(tools, domain_policy,
**kwargs)` is the tau2 factory: it reads `CONFIG`, loads the guidelines section if
a playbook is set, takes `llm` from `kwargs` (tau2's `llm_agent`), and constructs
the agent. `register("react_agent")` installs it idempotently into tau2's
registry.

**No playbook ships.** `playbooks/` is empty on purpose; ReAct-specific playbooks
are learned later. The `--playbook` path is fully wired and tested against
temp-file playbooks; baseline runs (no `--playbook`) inject no guidelines and
work immediately.

---

## 4. Reward / scoring path

Identical to any native tau2 run: tau2 executes
the agent's tool calls against a live `Environment`, and the domain evaluator
rebuilds a fresh env, replays the trajectory's tool-call messages to reconstruct
the predicted DB, and hash-compares against the gold DB — plus NL-assertion and
communicate checks. Nothing is mocked. Because the ReAct agent is native, its
`AssistantMessage.tool_calls` are already in the trajectory with no adaptation.

---

## 5. Run-level parameters

`eval_tau2.py` builds a `TextRunConfig` from `eval_config.toml` defaults (CLI
overrides). Notable values:

- `agent="react_agent"`, `llm_agent` from `$TAU2_LLM_AGENT`,
  `llm_user` from `$TAU2_LLM_USER`.
- `max_concurrency=8` (default). The native ReAct path runs in tau2's ordinary
  synchronous flow with no per-round event loops, so tau2's runner parallelises
  simulations cleanly.
- Results written **inside this repo** under `runs/react_<domain>_<split>_<run_id>/results.json`
  via an absolute `--save-to`, not in `tau2-bench/data`.

---

## 6. Evaluation-infra shims (not agent behavior)

`llm_compat.install()` applies two proxy-compatibility patches, both required and
neither specific to the ReAct agent:

- **Temperature shim**: wraps tau2's litellm `completion` to drop an explicit
  `temperature` for the gpt-5.x reasoning family (which 400s on `temperature != 1`),
  with a retry safety net for unmatched temperature-strict models.
- **NL-judge routing**: reroutes tau2's NL-assertion judge off its bare-OpenAI
  default (`gpt-4.1-2025-04-14`, which hits `api.openai.com`) onto a
  `litellm_proxy/` model (`TAU2_LLM_NL_ASSERTIONS` → `MODEL_NAME` → default).

No async success-logging-worker patch is needed here: the native ReAct path runs
in tau2's normal synchronous flow and never produces the event-loop noise that
would require one.
