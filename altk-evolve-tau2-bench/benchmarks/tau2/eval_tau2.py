"""Batch evaluation of tau2's native ReAct agent on tau2 domains.

A thin, reproducible wrapper over tau2's ``run_domain``. It:

  1. installs the temperature-compat shim (gpt-5.x user-sim models),
  2. configures playbook/guideline injection into the ReAct agent's system prompt,
  3. registers the ``react_agent`` factory into tau2's registry,
  4. builds a ``TextRunConfig`` (with ``llm_agent`` set) and calls ``run_domain``,
  5. prints ``avg_reward`` and ``pass^k`` and the results path.

The ReAct agent is a *native* tau2 agent, so its model IS tau2's ``llm_agent`` —
set here from ``--llm-agent`` / ``$TAU2_LLM_AGENT``.

Defaults may be supplied by a TOML file via ``--config`` (an ``[eval]`` table or
top-level keys); any command-line flag overrides the file. Credentials and the
proxy wiring are provided by the environment (see ``eval.sh``); this module
never reads secrets from disk.

Examples
--------
    # 20 retail tasks, 8-way concurrency, baseline (no playbook)
    uv run python -m benchmarks.tau2.eval_tau2 \
        --domain retail --num-tasks 20 --max-concurrency 8 \
        --llm-agent litellm_proxy/Azure/gpt-4.1

    # Everything from a config file
    uv run python -m benchmarks.tau2.eval_tau2 --config benchmarks/tau2/eval_config.toml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tomllib
from pathlib import Path
from typing import Any


# Results live inside THIS repo, not tau2-bench's data dir. tau2 always computes
# save_dir = DATA_DIR/"simulations"/<save_to>; but pathlib discards the left side
# when the right side is absolute, so passing an absolute run directory as
# `save_to` relocates results here instead of polluting tau2-bench/data.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_RUNS_DIR = _REPO_ROOT / "runs"


# Hardcoded fallbacks (lowest precedence: TOML overrides these, CLI overrides TOML).
_DEFAULTS: dict[str, Any] = {
    "domain": "retail",
    "split": "test",  # tau2 task split: train | test | base (all). "none"/"all" => no split filter.
    "runs_dir": None,  # base dir for results; None => <repo>/runs (see _resolve_run_dir)
    "task_ids": None,
    "num_tasks": None,  # None => all tasks in the split; else first N of the split (smoke limiter)
    "num_trials": 1,
    "max_concurrency": 1,
    "llm_agent": None,  # ReAct agent model; None => tau2 default; usually set via TAU2_LLM_AGENT
    "llm_args_agent": None,
    "llm_user": None,  # None => tau2 default; usually set via TAU2_LLM_USER in eval.sh
    "llm_args_user": None,
    "max_steps": 200,  # tau2's DEFAULT_MAX_STEPS; keep for leaderboard comparability
    "seed": 42,
    "save_to": None,
    "log_level": "INFO",
    "playbook": None,
    "guidelines_type": "lossless",
    "no_domain_policy": False,
}


def _load_config_defaults(argv) -> dict[str, Any]:
    """Read --config TOML (if any) and return a defaults dict merged over _DEFAULTS."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(argv)

    defaults = dict(_DEFAULTS)
    if known.config:
        with open(known.config, "rb") as f:
            data = tomllib.load(f)
        table = data.get("eval", data)  # accept [eval] table or top-level keys
        unknown = set(table) - set(_DEFAULTS)
        if unknown:
            raise SystemExit(f"Unknown key(s) in {known.config}: {sorted(unknown)}")
        defaults.update({k: v for k, v in table.items() if k in _DEFAULTS})
    return defaults


def _parse_args(argv=None) -> argparse.Namespace:
    defaults = _load_config_defaults(argv)

    p = argparse.ArgumentParser(description="Batch-evaluate tau2's ReAct agent on a domain.")
    p.add_argument("--config", default=None, help="TOML file of defaults ([eval] table or top-level)")
    p.add_argument("--domain", help="tau2 domain (mock, retail, airline, telecom, ...)")
    p.add_argument("--split", help="tau2 task split: train | test | base (all). "
                                   "Use 'none'/'all' for domains without splits (e.g. mock). "
                                   "tau2 validates the name and lists valid splits on error.")
    p.add_argument("--task-id", action="append", dest="task_ids", help="Specific task id(s); repeatable")
    p.add_argument("--num-tasks", type=int, help="Limit to first N tasks of the split (smoke; ignored if --task-id given)")
    p.add_argument("--num-trials", type=int, help="Trials per task (for pass^k)")
    p.add_argument("--max-concurrency", type=int, help="Concurrent simulations")
    p.add_argument("--llm-agent", help="litellm model for the ReAct agent (e.g. litellm_proxy/Azure/gpt-4.1)")
    p.add_argument("--llm-args-agent", help='JSON dict of agent LLM args (usually unnecessary)')
    p.add_argument("--llm-user", help="litellm model for the user simulator")
    p.add_argument("--llm-args-user", help='JSON dict of user-sim LLM args (usually unnecessary)')
    p.add_argument("--max-steps", type=int, help="Max conversation turns per task")
    p.add_argument("--seed", type=int)
    p.add_argument("--save-to", help="Run NAME (results go to <runs-dir>/<name>/results.json); "
                                     "an absolute path is honoured verbatim")
    p.add_argument("--runs-dir", help="Base directory for run results (default: <repo>/runs)")
    p.add_argument("--log-level", help="loguru level (INFO, DEBUG, WARNING, ...)")
    p.add_argument("--playbook", help="Path to playbook JSON for guideline injection")
    p.add_argument(
        "--guidelines-type",
        choices=["lossless", "sup2", "sup3"],
        help="Support-count threshold for playbook entries",
    )
    p.add_argument("--no-domain-policy", action="store_true", default=None,
                   help="Do NOT inject tau2's domain policy into the system prompt")

    p.set_defaults(**defaults)
    return p.parse_args(argv)


def _resolve_split(value: Any) -> str | None:
    """Map the ``--split`` flag to tau2's ``task_split_name``.

    Returns ``None`` (no split filter -> full task set) for empty/``none``/``all``;
    otherwise the split name verbatim (tau2 validates it against the domain).
    """
    if value is None:
        return None
    v = str(value).strip()
    return None if v.lower() in ("", "none", "all") else v


def _default_run_name(domain: str, split: str | None) -> str:
    """Reproducible run NAME (a single path segment, not a file path).

    Used as the leaf directory under the runs dir; keep it slash-free.
    """
    run_id = os.environ.get("EVAL_RUN_ID", "local")
    return f"react_{domain}_{split or 'all'}_{run_id}"


def _resolve_run_dir(runs_dir: Any, run_name: str) -> Path:
    """Absolute directory for this run's results (inside this repo by default).

    tau2 computes ``save_dir = DATA_DIR/"simulations"/save_to``; passing an
    absolute directory as ``save_to`` makes pathlib discard the left side, so
    results land here instead of in tau2-bench/data. An already-absolute
    ``run_name`` (e.g. an absolute ``--save-to``) is honoured verbatim.
    """
    if os.path.isabs(run_name):
        return Path(run_name)
    base = Path(runs_dir).expanduser() if runs_dir else _DEFAULT_RUNS_DIR
    return (base / run_name).resolve()


def main(argv=None) -> int:
    args = _parse_args(argv)

    # Tolerate temperature-strict user-sim models (gpt-5.x) before any tau2 LLM call.
    from benchmarks.tau2.llm_compat import install as install_llm_compat

    install_llm_compat()

    # Configure guideline/policy injection, then register the factory.
    from benchmarks.tau2 import react_agent

    react_agent.CONFIG = react_agent.ReactTau2Config(
        playbook_path=args.playbook,
        guidelines_type=args.guidelines_type,
        include_domain_policy=not args.no_domain_policy,
    )
    react_agent.register("react_agent")

    from tau2.data_model.simulation import TextRunConfig
    from tau2.metrics.agent_metrics import compute_metrics
    from tau2.run import run_domain

    # Results are written inside this repo (see _resolve_run_dir), not tau2-bench.
    split = _resolve_split(args.split)
    run_name = args.save_to or _default_run_name(args.domain, split)
    run_dir = _resolve_run_dir(args.runs_dir, run_name)
    results_path = run_dir / "results.json"

    config_kwargs: dict[str, Any] = dict(
        domain=args.domain,
        agent="react_agent",
        # Pass the split explicitly (incl. None) so it overrides RunConfig's "base"
        # default: --split none/all then means the full, unfiltered task set.
        task_split_name=split,
        max_steps=args.max_steps,
        num_trials=args.num_trials,
        max_concurrency=args.max_concurrency,
        seed=args.seed,
        # Absolute save_to => tau2's DATA_DIR/simulations/<save_to> collapses to
        # this path, so results land in <runs-dir>/<run-name>/ inside this repo.
        save_to=str(run_dir),
        log_level=args.log_level,
    )
    if args.task_ids:
        config_kwargs["task_ids"] = args.task_ids
    elif args.num_tasks is not None:
        config_kwargs["num_tasks"] = args.num_tasks
    # The ReAct agent is a native tau2 agent: its model IS tau2's llm_agent.
    if args.llm_agent:
        config_kwargs["llm_agent"] = args.llm_agent
    if args.llm_args_agent:
        config_kwargs["llm_args_agent"] = json.loads(args.llm_args_agent)
    if args.llm_user:
        config_kwargs["llm_user"] = args.llm_user
    if args.llm_args_user:
        config_kwargs["llm_args_user"] = json.loads(args.llm_args_user)

    config = TextRunConfig(**config_kwargs)

    print(f"[eval_tau2] domain={args.domain} split={split or 'all'} agent=react_agent "
          f"model={config.llm_agent} trials={args.num_trials} "
          f"concurrency={args.max_concurrency} max_steps={args.max_steps} seed={args.seed}")
    if args.playbook:
        print(f"[eval_tau2] playbook={args.playbook} (type={args.guidelines_type})")
    print(f"[eval_tau2] run_name={run_name}")
    print(f"[eval_tau2] results -> {results_path}")

    results = run_domain(config)

    metrics = compute_metrics(results)
    n = len(results.simulations)
    print("\n=== react-on-tau2 results ===")
    print(f"domain        : {args.domain}")
    print(f"split         : {split or 'all'}")
    print(f"simulations   : {n}")
    print(f"avg_reward    : {metrics.avg_reward:.4f}")
    for k, v in sorted(metrics.pass_hat_ks.items()):
        print(f"pass^{k}        : {v:.4f}")
    print(f"results file  : {results_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
