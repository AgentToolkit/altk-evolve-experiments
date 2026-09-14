#!/usr/bin/env python3
"""Reconstruct COMPLETE, untruncated trajectories for every simulation in a
tau2 results.json, one text file per task under <run>/trajectories/.

Each file contains, with no truncation anywhere:
  1. Run/sim metadata (reward, breakdown, termination, cost, timing)
  2. The user scenario (reason for call, known/unknown info, persona)
  3. The ground-truth evaluation criteria (expected actions, assertions)
  4. The domain policy handed to the agent (the persisted agent-prompt policy)
  5. The full conversation: every user/assistant turn, every tool call with
     its exact arguments, every tool result verbatim
  6. The judge output: DB check, per-action checks, NL assertions with
     justifications, communicate checks, and the reward breakdown
"""
import json
import os
import sys

RUN = sys.argv[1] if len(sys.argv) > 1 else "runs/train"
RES = os.path.join(RUN, "results.json")
OUT = os.path.join(RUN, "trajectories")

BAR = "=" * 90
SUB = "-" * 90


def j(obj):
    """Pretty JSON, never truncated, stable ordering."""
    return json.dumps(obj, indent=2, ensure_ascii=False)


def fmt_scenario(task):
    lines = ["USER SCENARIO", SUB]
    us = (task or {}).get("user_scenario") or {}
    persona = us.get("persona")
    if persona:
        lines += ["persona:", persona, ""]
    instr = us.get("instructions") or {}
    if isinstance(instr, dict):
        for k in ("domain", "reason_for_call", "known_info", "unknown_info", "task_instructions"):
            if instr.get(k) is not None:
                lines.append(f"{k}: {instr[k]}")
    elif instr:
        lines.append(str(instr))
    desc = (task or {}).get("description") or {}
    for k in ("purpose", "relevant_policies", "notes"):
        if isinstance(desc, dict) and desc.get(k) is not None:
            lines.append(f"{k}: {desc[k]}")
    ticket = (task or {}).get("ticket")
    if ticket:
        lines += ["", "ticket:", str(ticket)]
    return "\n".join(lines)


def fmt_criteria(task):
    lines = ["GROUND-TRUTH EVALUATION CRITERIA", SUB]
    ec = (task or {}).get("evaluation_criteria") or {}
    lines.append(f"reward_basis: {ec.get('reward_basis')}")
    actions = ec.get("actions") or []
    lines.append(f"\nexpected actions ({len(actions)}):")
    for a in actions:
        lines.append(f"  [{a.get('action_id')}] {a.get('name')}({j(a.get('arguments'))})")
    if ec.get("nl_assertions"):
        lines.append("\nexpected nl_assertions:")
        for x in ec["nl_assertions"]:
            lines.append(f"  - {x}")
    if ec.get("communicate_info"):
        lines.append(f"\ncommunicate_info: {ec.get('communicate_info')}")
    if ec.get("env_assertions"):
        lines.append("\nenv_assertions:")
        for x in ec["env_assertions"]:
            lines.append(f"  - {j(x)}")
    return "\n".join(lines)


def fmt_conversation(sim):
    lines = ["CONVERSATION", SUB]
    for i, m in enumerate(sim.get("messages") or []):
        role = m.get("role")
        content = m.get("content")
        ts = m.get("timestamp")
        stamp = f"  @{ts}" if ts else ""
        if role == "assistant":
            if content:
                lines.append(f"[ASSISTANT]{stamp}")
                lines.append(content)
            for tc in (m.get("tool_calls") or []):
                lines.append(f"    -> CALL {tc.get('name')}(id={tc.get('id')})")
                lines.append(f"       args: {j(tc.get('arguments'))}")
            if not content and not (m.get("tool_calls")):
                lines.append(f"[ASSISTANT]{stamp} (empty)")
        elif role == "tool":
            err = " ERROR" if m.get("error") else ""
            lines.append(f"    <- RESULT{err} (id={m.get('id')})")
            # tool content is often a JSON string; keep it verbatim, no truncation
            lines.append(f"       {content}")
        elif role == "user":
            lines.append(f"[USER]{stamp}")
            lines.append(content if content is not None else "(none)")
        else:
            lines.append(f"[{str(role).upper()}]{stamp}")
            lines.append(content if content is not None else "(none)")
        lines.append("")
    return "\n".join(lines)


def fmt_judge(sim):
    ri = sim.get("reward_info") or {}
    lines = ["JUDGE / REWARD DETAILS", SUB]
    lines.append(f"reward: {ri.get('reward')}")
    lines.append(f"reward_breakdown: {ri.get('reward_breakdown')}")
    lines.append(f"reward_basis: {ri.get('reward_basis')}")
    lines.append("")
    lines.append(f"DB check: {j(ri.get('db_check'))}")
    lines.append("")
    acs = ri.get("action_checks") or []
    lines.append(f"action_checks ({len(acs)}):")
    for ac in acs:
        a = ac.get("action") or {}
        lines.append(
            f"  [{a.get('action_id')}] {a.get('name')}  "
            f"match={ac.get('action_match')} reward={ac.get('action_reward')} "
            f"type={ac.get('tool_type')}"
        )
        lines.append(f"      expected args: {j(a.get('arguments'))}")
    lines.append("")
    nls = ri.get("nl_assertions") or []
    lines.append(f"nl_assertions ({len(nls)}):")
    for x in nls:
        lines.append(f"  - assertion: {x.get('nl_assertion')}")
        lines.append(f"    met: {x.get('met')}")
        lines.append(f"    justification: {x.get('justification')}")
    lines.append("")
    ccs = ri.get("communicate_checks") or []
    lines.append(f"communicate_checks ({len(ccs)}):")
    for x in ccs:
        lines.append(f"  - info: {x.get('info')}  met: {x.get('met')}")
        lines.append(f"    justification: {x.get('justification')}")
    if ri.get("env_assertions"):
        lines.append("")
        lines.append(f"env_assertions: {j(ri.get('env_assertions'))}")
    return "\n".join(lines)


def final_answer(sim):
    for m in reversed(sim.get("messages") or []):
        if m.get("role") == "assistant" and m.get("content"):
            return m["content"]
    return "(no final assistant text; task ended on tool call or transfer/stop)"


def main():
    d = json.load(open(RES))
    sims = d["simulations"]
    tasks_by_id = {t["id"]: t for t in d.get("tasks", [])}
    os.makedirs(OUT, exist_ok=True)
    written = []
    for sim in sims:
        tid = sim["task_id"]
        task = tasks_by_id.get(tid)
        ri = sim.get("reward_info") or {}
        header = [
            BAR,
            f"TASK {tid}   reward={ri.get('reward')}   breakdown={ri.get('reward_breakdown')}",
            f"sim_id={sim.get('id')}  trial={sim.get('trial')}  seed={sim.get('seed')}",
            f"termination_reason={sim.get('termination_reason')}  duration={sim.get('duration')}s",
            f"agent_cost={sim.get('agent_cost')}  user_cost={sim.get('user_cost')}  "
            f"messages={len(sim.get('messages') or [])}",
            f"start={sim.get('start_time')}  end={sim.get('end_time')}",
            BAR,
        ]
        parts = [
            "\n".join(header),
            fmt_scenario(task),
            BAR,
            fmt_criteria(task),
            BAR,
            "AGENT DOMAIN POLICY (persisted policy handed to the agent)",
            SUB,
            sim.get("policy") or "(no policy stored)",
            BAR,
            fmt_conversation(sim),
            BAR,
            "FINAL ANSWER (last assistant utterance)",
            SUB,
            final_answer(sim),
            BAR,
            fmt_judge(sim),
            BAR,
        ]
        text = "\n\n".join(parts) + "\n"
        path = os.path.join(OUT, f"task_{tid}.txt")
        with open(path, "w") as f:
            f.write(text)
        written.append((int(tid) if str(tid).isdigit() else tid, path, ri.get("reward")))

    written.sort(key=lambda x: (isinstance(x[0], str), x[0]))
    print(f"Wrote {len(written)} trajectory files to {OUT}")
    solved = sum(1 for _, _, r in written if (r or 0) >= 1.0)
    print(f"pass^1 = {solved}/{len(written)}")
    print("failures (reward 0):",
          ", ".join(str(t) for t, _, r in written if (r or 0) < 1.0))


if __name__ == "__main__":
    main()
