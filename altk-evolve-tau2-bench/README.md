# react-eval-tau2

Evaluate [tau2-bench](https://github.com/sierra-research/tau2-bench)'s **native
ReAct agent** on tau2-bench with optional **guideline injection**: this repo
integrates [altk-evolve](https://github.com/AgentToolkit/altk-evolve)
guidelines — a learned "playbook" appended to the agent's system prompt.

The ReAct agent is vendored verbatim from tau2's shipped example
(`tau2-bench/examples/agents/react_agent.py`): a two-phase **THINK → ACT** loop
that reasons without tools, then acts with tools (2 LLM calls per turn). It is a
native tau2 agent, so its tool calls, trajectory, and reward all flow through
tau2's own runner and trajectory-replay evaluator — nothing about the reward
path is mocked.

> **See [`ARCHITECTURE.md`](ARCHITECTURE.md)** for the full record of every
> decision: why the agent is vendored, the THINK/ACT loop, the exact guideline
> injection point, how the models are wired, and the concurrency note.

## Layout

```
benchmarks/tau2/
  react_agent.py      # vendored native ReActAgent + GuidedReActAgent + `react_agent` factory
  guidelines.py       # playbook-JSON loader + system-prompt merge
  llm_compat.py       # temperature-compat shim + NL-judge routing for the litellm proxy
  run_spike.py        # single/few-task runner (spike; run_single_task)
  eval_tau2.py        # batch runner over tau2's run_domain (avg_reward, pass^k)
  eval_config.toml    # defaults for eval_tau2 (the [eval] table)
  config/tau2.env     # non-secret model/runtime wiring
  playbooks/          # EMPTY — ReAct-specific playbooks are learned later
eval.sh               # reproducible launcher: env bootstrap -> eval_tau2
tests/                # guideline-injection / guidelines / llm_compat / config tests (no LLM needed)
```

## Setup

```bash
uv sync                 # installs tau2 (editable sibling) + dev tools; Python 3.12 (see .python-version)
uv run pytest -q        # 44 tests, no credentials required
```

> **Python 3.12 is pinned** (`.python-version`). tau2 imports `audioop`, which
> was removed from the stdlib in Python 3.13; on 3.13 the import chain fails at
> collection.

## Credentials & model wiring

The eval needs a live LLM via the IBM litellm proxy. Secrets live **only** in
the repo-group env at `../.env` (`LITELLM_URL`, `LITELLM_API_KEY`) — never in
this repo. Non-secret wiring is in `benchmarks/tau2/config/tau2.env`:

| Var | Value | Used by |
|---|---|---|
| `TAU2_LLM_AGENT` | `litellm_proxy/Azure/gpt-4.1` | **the ReAct agent** (its model *is* tau2's `llm_agent`) |
| `TAU2_LLM_USER` | `litellm_proxy/azure/gpt-5.5` | tau2 user simulator model |
| `TAU2_LLM_NL_ASSERTIONS` | `litellm_proxy/azure/gpt-5.5` | tau2 NL-assertion judge |

**Model wiring:** the ReAct agent is a native tau2 agent using
`tau2.utils.llm_utils.generate`, so its model **is** tau2's `llm_agent` — set
here to `litellm_proxy/Azure/gpt-4.1`. Whatever you pass as `--agent-llm` (or
`TAU2_LLM_AGENT`) is the model the agent runs on; there is no separate
agent-side model selection.

`eval.sh` sources both env files and derives the provider variables:

```bash
export OPENAI_API_KEY="$LITELLM_API_KEY"
export LITELLM_PROXY_API_BASE="${LITELLM_URL%/}/v1"
export LITELLM_PROXY_API_KEY="$LITELLM_API_KEY"
unset OPENAI_BASE_URL LITELLM_API_BASE
```

> **Pitfall (do not "fix"):** for `litellm_proxy/...` model strings you must
> **not** set `OPENAI_BASE_URL` / `LITELLM_API_BASE`. litellm treats a base URL
> as a signal to force `custom_llm_provider="openai"` and pass the whole
> `litellm_proxy/...` string as an OpenAI model id, which the proxy rejects. The
> `litellm_proxy` provider reads `LITELLM_PROXY_API_BASE` /
> `LITELLM_PROXY_API_KEY` natively instead.

> **Model casing** is enforced by the proxy allowlist: `Azure/gpt-4.1` (capital
> A) is accepted; `azure/gpt-4.1` is not.

### Temperature-strict models

The Azure **gpt-5.x** reasoning family rejects any explicit `temperature != 1`
with a 400, but tau2's user sim defaults to `temperature=0.0`. `llm_compat.py`
wraps tau2's litellm `completion` to drop `temperature` for the gpt-5.x family
(with a retry safety net for anything unmatched), so no manual override is
needed. Both runners install it automatically.

### NL-assertion judge routing

tau2's NL-assertion judge (`RewardType.NL_ASSERTION`) defaults to a **bare**
OpenAI model id (`gpt-4.1-2025-04-14`), which litellm sends to `api.openai.com`
— where the proxy key is rejected. It only fires once a task with NL assertions
*runs to completion*, so it hides behind assertion-free tasks.
`llm_compat.install()` reroutes the judge to a `litellm_proxy/` model, chosen as
`TAU2_LLM_NL_ASSERTIONS` → `MODEL_NAME` → `litellm_proxy/Azure/gpt-4.1`. The
shipped `tau2.env` sets it to `litellm_proxy/azure/gpt-5.5`.

### Concurrency

`--max-concurrency` fans out simulations. The native ReAct agent runs in tau2's
ordinary synchronous flow, so simulations parallelise cleanly; `eval_config.toml`
defaults to `8`. Under high concurrency the proxy may occasionally return a
transient `All connection attempts failed`; litellm's retries usually recover
within the run. If you see infrastructure errors in results, lower
`--max-concurrency`.

## Running

### Batch (recommended)

Runs are selected by tau2's built-in **task split**, not a raw task count. Every
domain ships `train` / `test` / `base` (all) splits:

| Domain | train | test | base (all) |
|---|---|---|---|
| retail | 74 | 40 | 114 |
| airline | 30 | 20 | 50 |
| telecom | 74 | 40 | 114 (also `small`=20, `full`=2285) |

```bash
# Held-out evaluation (default split is `test`)
./eval.sh --domain retail --split test

# test split, 8-way concurrency, with a playbook (once one exists)
./eval.sh --domain retail --split test --max-concurrency 8 \
    --playbook benchmarks/tau2/playbooks/retail.json --guidelines-type sup2

# pass^k over trials
./eval.sh --domain retail --split test --num-trials 4 --max-concurrency 8

# Quick smoke: first 3 tasks of the split
./eval.sh --domain retail --split test --num-tasks 3

# Domains without splits (e.g. mock): disable split filtering
./eval.sh --domain mock --split none
```

`--split` defaults to `test` (set in `eval_config.toml`). `--num-tasks N` is a
smoke limiter — it takes the first N tasks *within* the chosen split. `--split
none` (or `all`) disables split filtering and runs the full task set.

`eval.sh` stamps an `EVAL_RUN_ID` (override by exporting it) and passes all extra
flags through to `eval_tau2.py`, whose defaults come from `eval_config.toml`
(CLI overrides the file). Results are written **inside this repo** under
`runs/<run_name>/results.json` (not in `tau2-bench/data`); the run name is
prefixed `react_` and includes the split, e.g.
`runs/react_retail_test_<run_id>/results.json`. Override the base dir with
`--runs-dir DIR` (or an absolute `--save-to`). The run prints `avg_reward` and
`pass^k`. Inspect a run with:

```bash
uv run tau2 view --file runs/react_retail_test_<run_id>/results.json
# or browse all runs in the dir:  uv run tau2 view --dir runs
```

(tau2's plain `tau2 view` only scans `tau2-bench/data/simulations`, so point it
at `runs/` with `--file`/`--dir`.)

### Spike (single task, verbose)

```bash
uv run python -m benchmarks.tau2.run_spike --domain retail --num-tasks 1 \
    --llm-agent litellm_proxy/Azure/gpt-4.1 \
    --llm-user litellm_proxy/azure/gpt-5.5
```

(Run it under the same env bootstrap as `eval.sh` — or just use `./eval.sh` for
convenience.)

## Guidelines / playbooks

A **playbook** is a JSON file of rules learned from past task attempts. It is
distinct from the domain policy: the policy is tau2's ground-truth rules; the
playbook is extra guidance appended to the ReAct agent's **system prompt** after
the policy (so it rides in both the THINK and ACT phases).

```json
{
  "entries": [
    {"r": "Authenticate the user before any account-specific action.", "n": 3},
    {"r": "Confirm the exact item and action before any irreversible change.", "n": 2}
  ]
}
```

`n` is the support count (how many attempts corroborated the rule).
`--guidelines-type` selects a threshold:

| Type | Keeps entries with |
|---|---|
| `lossless` | `n >= 1` (all) |
| `sup2` | `n >= 2` |
| `sup3` | `n >= 3` |

Use `--no-domain-policy` to inject **only** the playbook (ablation).

> **No playbook ships yet.** `playbooks/` is intentionally empty — ReAct-specific
> playbooks will be built/learned later. The `--playbook` flag and injection path
> are fully wired; baseline runs (no `--playbook`) work immediately and establish
> the no-playbook `avg_reward`. Once a playbook exists, the A/B is
> `--playbook <path> --guidelines-type sup2` on the same split.

## Notes

- The ReAct agent makes **2 LLM calls per turn** (a tool-free THINK call then a
  tool-enabled ACT call), so a run costs roughly twice the completions of a
  single-call agent for the same number of turns.
- The vendored `ReActAgent` should be kept in sync with upstream
  `tau2-bench/examples/agents/react_agent.py` if tau2 updates it; only the
  system-prompt composition is overridden (in `GuidedReActAgent`).
