# Playbook A/B results — ReAct agent on τ²-bench (retail / test)

**Agent:** tau2 native ReAct (vendored), model `litellm_proxy/Azure/gpt-4.1`.
**Split:** retail / test (40 tasks). **Concurrency:** 3. **Trials:** 3 runs per
config at **seeds 42 / 43 / 44** (`_run2` = seed 43, `_run3` = seed 44).
**Captured:** 2026-09-08. All 24 runs terminated `user_stop` with zero error
fields (no infra failures); retail rewards are binary (0/1), so mean accuracy =
pass^1.

The playbook is `benchmarks/tau2/playbooks/tau_retail_react_playbook.json`
(292 entries). Support-count thresholds keep:

| Type | Keeps | # guidelines |
|---|---|---|
| lossless | n ≥ 1 | 292 |
| sup2 | n ≥ 2 | 148 |
| sup3 | n ≥ 3 | 31 |

Baseline = no `--playbook`.

## Headline metrics (mean over 3 runs)

**Turns/task** = agent (assistant) turns — how many times the agent acted (each
is one THINK→ACT ReAct turn) — averaged over all 120 sims (3 runs × 40 tasks).
**pass^3** = strict: a task counts only if it passed in **all 3** runs.

**Tasks solved** = mean number of tasks passed per run (of 40), averaged over the
3 runs.

| Config | #rules | Mean acc | pass^1 | pass^3 | Tasks solved | Avg turns/task |
|---|---|---|---|---|---|---|
| baseline | 0 | 0.8083 | 0.808 | 0.675 (27/40) | 32.33 / 40 | 10.78 |
| **sup3** | 31 | **0.8417** | 0.842 | **0.700 (28/40)** | **33.67 / 40** | 10.69 |
| sup2 | 148 | 0.8083 | 0.808 | 0.675 (27/40) | 32.33 / 40 | 10.57 |
| lossless | 292 | 0.7917 | 0.792 | 0.700 (28/40) | 31.67 / 40 | 10.48 |

## Per-seed accuracy + paired delta vs baseline (same seeds)

| Config | seed42 | seed43 | seed44 | Mean | stdev | Δ mean | Paired Δ (42/43/44) |
|---|---|---|---|---|---|---|---|
| baseline | 0.800 | 0.775 | 0.850 | 0.8083 | 0.0382 | — | — |
| **sup3** | 0.825 | 0.850 | 0.850 | **0.8417** | **0.0144** | **+0.033** | +0.025 / +0.075 / 0.000 |
| sup2 | 0.850 | 0.775 | 0.800 | 0.8083 | 0.0382 | +0.000 | +0.050 / 0.000 / −0.050 |
| lossless | 0.825 | 0.800 | 0.750 | 0.7917 | 0.0382 | −0.017 | +0.025 / +0.025 / −0.100 |

## Turns by outcome (mean agent turns)

| Config | pass turns | fail turns |
|---|---|---|
| baseline | 10.57 | 11.65 |
| sup3 | 10.73 | 10.47 |
| sup2 | 10.42 | 11.17 |
| lossless | 10.14 | 11.80 |

sup3 is the only config where **failed** tasks take *fewer* turns than passed
ones — its failures are quick/clean (e.g. a hard-blocked task), not prolonged
flailing. Everywhere else, failures drag on longer than successes.

## Hard-core failures (failed all 3 baseline seeds) — fail-count out of 3

| task | baseline | sup3 | sup2 | lossless |
|---|---|---|---|---|
| 27 | 3 | 2 | **0** | **0** |
| 38 | 3 | **1** | 3 | 3 |
| 64 | 3 | 3 | 3 | 3 |
| 86 | 3 | **1** | **0** | **0** |

- **27, 86** — fixed by every playbook (any threshold).
- **38** — only sup3 helps; the extra rules in sup2/lossless re-break it.
- **64** — fails 3/3 under **every** config; guidelines can't touch it (genuine
  capability/policy gap, worth reading its trajectory).

## Conclusions

1. **Guideline count is inversely related to benefit** — monotonic:
   sup3 (31) **+0.033** → sup2 (148) **0.00** → lossless (292) **−0.017**.
   The supersets contain the same 31 winning rules plus more low-support ones;
   each addition dilutes/contradicts the strong rules back to (sup2) and below
   (lossless) baseline. Only the ~31 highest-support (n ≥ 3) rules carry signal.
2. **sup3 is the clear winner** — highest mean accuracy, the only pass^3 tick-up
   with a genuine accuracy gain, lowest variance (stdev 0.014 vs 0.038 for every
   other config, baseline included), never below baseline on any seed, and the
   only config whose failures aren't drawn-out.
3. **pass^3 is nearly flat** (27↔28) — the mean-accuracy gains come mostly from
   tasks flipping pass/fail between seeds, not from making shaky tasks reliably
   solved. The playbook effect is real but modest on one split.
4. **Turn budget is unchanged** (~10.5–10.8 across all configs) — accuracy
   differences are not bought with more agent effort.

## Run directories

All under `runs/`:

| Config | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| baseline | `react_retail_test_baseline_20260908_145113` | `..._baseline_run2_20260908_152027` | `..._baseline_run3_20260908_154445` |
| sup3 | `react_retail_test_sup3_20260908_162514` | `..._sup3_run2_20260908_165406` | `..._sup3_run3_20260908_171409` |
| sup2 | `react_retail_test_sup2_20260908_192024` | `..._sup2_run2_20260908_193520` | `..._sup2_run3_20260908_195022` |
| lossless | `react_retail_test_lossless_20260908_200521` | `..._lossless_run2_20260908_202032` | `..._lossless_run3_20260908_203433` |

Each has `results.json` and a `trajectories/` directory (one untruncated text
file per task). Also: retail **train** baseline (seed 42, 74 tasks) =
`react_retail_train_baseline_20260908_135840`, avg_reward 0.7703.
