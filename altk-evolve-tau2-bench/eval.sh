#!/usr/bin/env bash
# Reproducible launcher for the ReAct-on-tau2 evaluation.
#
# Sources secrets from the repo-group .env and non-secret wiring from tau2.env,
# derives the litellm_proxy provider variables, stamps an EVAL_RUN_ID, and runs
# eval_tau2.py. Extra arguments pass straight through, e.g.:
#
#   ./eval.sh --domain retail --split test --num-tasks 20 --max-concurrency 8 \
#       --playbook benchmarks/tau2/playbooks/retail.json --guidelines-type sup2
#
# Override the secrets path with GROUP_ENV=/path/to/.env ./eval.sh ...
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"              # react-eval-tau2/
GROUP_ENV="${GROUP_ENV:-$(cd "$HERE/.." && pwd)/.env}"            # ../.env  (secrets)
TAU2_ENV="$HERE/benchmarks/tau2/config/tau2.env"                  # non-secret wiring

[ -f "$GROUP_ENV" ] || { echo "Missing secrets env: $GROUP_ENV" >&2; exit 1; }
[ -f "$TAU2_ENV" ]  || { echo "Missing wiring env:  $TAU2_ENV"  >&2; exit 1; }

set -a
# shellcheck disable=SC1090
. "$GROUP_ENV"    # LITELLM_URL, LITELLM_API_KEY
# shellcheck disable=SC1090
. "$TAU2_ENV"     # TAU2_LLM_AGENT, TAU2_LLM_USER, TAU2_LLM_NL_ASSERTIONS
set +a

: "${LITELLM_URL:?LITELLM_URL not set in $GROUP_ENV}"
: "${LITELLM_API_KEY:?LITELLM_API_KEY not set in $GROUP_ENV}"

# Derive the litellm_proxy provider vars used by the agent, the user sim, and the
# NL judge. Do NOT set OPENAI_BASE_URL / LITELLM_API_BASE: that makes litellm
# force custom_llm_provider="openai" and pass the whole "litellm_proxy/..."
# string as an OpenAI model id, which the proxy rejects.
export OPENAI_API_KEY="$LITELLM_API_KEY"
export LITELLM_PROXY_API_BASE="${LITELLM_URL%/}/v1"
export LITELLM_PROXY_API_KEY="$LITELLM_API_KEY"
unset OPENAI_BASE_URL LITELLM_API_BASE 2>/dev/null || true

# Reproducibility stamp; eval_tau2 derives the results dir from it.
export EVAL_RUN_ID="${EVAL_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

echo "[eval.sh] EVAL_RUN_ID=$EVAL_RUN_ID  agent=$TAU2_LLM_AGENT  user=$TAU2_LLM_USER"

cd "$HERE"
exec uv run python -m benchmarks.tau2.eval_tau2 \
    --config "$HERE/benchmarks/tau2/eval_config.toml" \
    --llm-agent "$TAU2_LLM_AGENT" \
    --llm-user "$TAU2_LLM_USER" \
    "$@"
