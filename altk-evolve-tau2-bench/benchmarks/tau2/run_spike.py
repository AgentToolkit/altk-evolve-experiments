"""Spike runner: evaluate tau2's ReAct agent on one or a few tau2 tasks.

This is the thin end-to-end harness for the spike phase. It registers the ReAct
agent factory into tau2's registry, configures guideline/playbook injection, and
runs tasks through tau2's native ``run_single_task`` so scoring uses tau2's real
trajectory-replay evaluator.

Credentials
-----------
Both sides need a live LLM:
  * The ReAct agent is a native tau2 agent: its model is tau2's ``llm_agent``,
    passed via ``--llm-agent`` (a litellm model string).
  * The tau2 user simulator uses ``--llm-user`` (a litellm model string).

Examples
--------
    # Mock domain, one task, default models
    uv run python -m benchmarks.tau2.run_spike --domain mock --num-tasks 1

    # One retail task with a playbook, on a litellm proxy
    uv run python -m benchmarks.tau2.run_spike \
        --domain retail --num-tasks 1 \
        --llm-agent litellm_proxy/Azure/gpt-4.1 \
        --llm-user litellm_proxy/azure/gpt-5.5 \
        --playbook benchmarks/tau2/playbooks/retail.json --guidelines-type sup2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run tau2's ReAct agent on tau2 tasks (spike).")
    p.add_argument("--domain", default="retail", help="tau2 domain / task set (e.g. mock, retail)")
    p.add_argument("--split", default=None,
                   help="tau2 task split: train | test | base. Default (unset) loads the full "
                        "task set; use 'none'/'all' to force no split filter (e.g. mock).")
    p.add_argument("--task-id", action="append", dest="task_ids", help="Specific task id(s); repeatable")
    p.add_argument("--num-tasks", type=int, default=1, help="Limit to first N tasks of the split (ignored if --task-id given)")
    p.add_argument("--llm-agent", default=None, help="litellm model for the ReAct agent")
    p.add_argument("--llm-user", default=None, help="litellm model for the user simulator")
    p.add_argument(
        "--llm-args-user",
        default=None,
        help='JSON dict of user-sim LLM args, e.g. \'{"top_p": 0.9}\'. '
        "Usually unnecessary: the temperature-compat shim already drops "
        "temperature for gpt-5.x reasoning models.",
    )
    p.add_argument("--max-steps", type=int, default=200, help="Max conversation turns (tau2 default 200)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-dir", default=None, help="Directory for per-task logs/results")
    # Guidelines / playbook
    p.add_argument("--playbook", default=None, help="Path to playbook JSON for guideline injection")
    p.add_argument(
        "--guidelines-type",
        default="lossless",
        choices=["lossless", "sup2", "sup3"],
        help="Support-count threshold for playbook entries",
    )
    p.add_argument(
        "--no-domain-policy",
        action="store_true",
        help="Do NOT inject tau2's domain policy into the agent system prompt",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)

    # Tolerate temperature-strict user-sim models (e.g. Azure gpt-5.x) before
    # any tau2 LLM call is made. Makes --llm-args-user temperature workarounds
    # unnecessary for the gpt-5.x family.
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
    from tau2.run import get_tasks, run_single_task

    # "none"/"all"/empty -> no split filter (full task set); else the split verbatim.
    _split = (args.split or "").strip()
    split = None if _split.lower() in ("", "none", "all") else _split
    tasks = get_tasks(
        args.domain,
        task_split_name=split,
        task_ids=args.task_ids,
        num_tasks=None if args.task_ids else args.num_tasks,
    )
    if not tasks:
        print(f"No tasks found for domain {args.domain!r}", file=sys.stderr)
        return 1

    config_kwargs = dict(domain=args.domain, agent="react_agent", max_steps=args.max_steps)
    if args.llm_agent:
        config_kwargs["llm_agent"] = args.llm_agent
    if args.llm_user:
        config_kwargs["llm_user"] = args.llm_user
    if args.llm_args_user:
        import json

        config_kwargs["llm_args_user"] = json.loads(args.llm_args_user)
    config = TextRunConfig(**config_kwargs)

    save_dir = Path(args.save_dir) if args.save_dir else None

    print(f"Running {len(tasks)} task(s) from {args.domain!r} with agent 'react_agent' "
          f"(model={config.llm_agent})")
    if args.playbook:
        print(f"  playbook={args.playbook} (type={args.guidelines_type})")

    for task in tasks:
        print(f"\n=== task {task.id} ===")
        sim = run_single_task(config, task, seed=args.seed, save_dir=save_dir)
        reward = getattr(sim, "reward_info", None)
        if reward is not None:
            print(f"reward: {reward.reward}")
            if getattr(reward, "reward_breakdown", None):
                print(f"breakdown: {reward.reward_breakdown}")
        else:
            print("no reward_info on SimulationRun")
        if getattr(sim, "termination_reason", None):
            print(f"termination: {sim.termination_reason}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
