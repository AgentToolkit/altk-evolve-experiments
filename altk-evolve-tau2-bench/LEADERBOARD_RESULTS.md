# τ²-bench Results — Guideline-Support Variants (baseline / sup3 / lossless)

Compares the tau2 native ReAct agent (gpt-4.1) with **no
guidelines** (baseline) against two ALTK-Evolve guideline sets filtered by
support count. On the full retail **base** split, **lossless is the strongest
variant** and is the one **we will be submitting** to the leaderboard.

> **Leaderboard standing:** lossless's **Pass^4 = 67.5%** would take **#1** on the
> τ²-bench retail leaderboard. The current #1 is a model **fine-tuned on the
> τ²-bench retail dataset**, scoring **61.4% Pass^4** — lossless beats it by
> **+6.1 %**.

**Agent:** tau2's own implementation of ReAct agent (cloned from
`tau2-bench/examples/agents/react_agent.py`), driven by `gpt-4.1`, a
two-phase THINK→ACT loop. Variants differ **only** in the ALTK-Evolve guideline
set appended to the agent's system prompt after tau2's domain policy.

**Domain:** retail - customer service tasks - (114 tasks - 74 train / 40 test)

**Trials:** 4

**User simulator:** `gpt-5.5`

### Variants (guideline set)

The playbook `benchmarks/tau2/playbooks/tau_retail_react_playbook.json` holds 292
learned guidelines. Support-count thresholds keep a subset:

| Variant | Keeps | # guidelines |
|---|---|---|
| **baseline** | — (no playbook) | **0** |
| **sup3** | n ≥ 3 | **31** |
| **lossless** | n ≥ 1 (all) | **292** |

---

## Headline: Pass^k (retail, base split, all 114 tasks)

Computed by tau2's own `compute_metrics` (combinatorial estimator
`C(c,k)/C(n,k)` averaged over tasks). Rewards are binary (0/1), so `avg_reward` =
`Pass^1`.

| Variant | # guidelines | Pass^1 | Pass^2 | Pass^3 | **Pass^4** |
|---|---|---|---|---|---|
| baseline | 0 | 0.7763 | 0.6813 | 0.6316 | 0.5965 |
| sup3 | 31 | 0.7895 | 0.7018 | 0.6447 | 0.6053 |
| **lossless** | 292 | **0.8114** | **0.7500** | **0.7083** | **0.6754** |

**Pass^4 is the value the leaderboard records.** lossless (0.6754) beats sup3
(0.6053) and baseline (0.5965), and its lead **widens with k** — i.e. it is not
just more accurate but more *consistent* across repeated trials. W.r.t Pass^1, ALTK Evolve lossless agent is #4 on the leaderboard, but since it is more consistent than the other agents, it climbs to #1 when k > 1 trials.

## Seen vs. held-out breakdown

The `base` split is exactly `train ∪ test` (train=74, test=40, disjoint,
union=114). Guidelines were learned from the **train** tasks, so train tasks were
*seen* during guideline learning and test tasks were *held out*.

| Variant | Subset | Tasks | Pass^1 | Pass^2 | Pass^3 | Pass^4 |
|---|---|---|---|---|---|---|
| baseline | train (seen) | 74 | 0.7736 | 0.6847 | 0.6419 | 0.6081 |
| baseline | test (held-out) | 40 | 0.7812 | 0.6750 | 0.6125 | 0.5750 |
| sup3 | train (seen) | 74 | 0.7905 | 0.7072 | 0.6486 | 0.6081 |
| sup3 | test (held-out) | 40 | 0.7875 | 0.6917 | 0.6375 | 0.6000 |
| **lossless** | train (seen) | 74 | 0.8041 | 0.7477 | 0.7095 | 0.6757 |
| **lossless** | test (held-out) | 40 | 0.8250 | 0.7542 | 0.7063 | 0.6750 |

**Key point:** for ALTK Evolve agents, held-out (test) performance is essentially
identical to seen (train). Thus the guidelines generalize and it is not just memorization.

## Effort & cost

| Variant | Assistant turns / task | Duration / sim | Agent $ / conv | User-sim $ / conv | **Total $ / conv** |
|---|---|---|---|---|---|
| baseline | 11.02 (min 4, max 25) | 107.2 s | $0.0505 | $0.0337 | **$0.0842** |
| sup3 | 10.96 (min 5, max 22) | 152.7 s | $0.0619 | $0.0327 | **$0.0947** |
| **lossless** | 10.67 (min 5, max 24) | 119.8 s | **$0.1084** | $0.0309 | **$0.1393** |

lossless is the **most expensive** (~1.75× sup3, ~2.1× baseline on agent cost)
despite the *fewest* assistant turns — the 292-guideline prompt inflates
per-call input tokens rather than conversation length.
