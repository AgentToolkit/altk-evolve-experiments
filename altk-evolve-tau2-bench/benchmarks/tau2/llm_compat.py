"""Compatibility shims for running tau2 through a litellm proxy.

``install()`` applies two fixes *without* editing the tau2 or litellm packages,
so the eval stays portable:

1. **Temperature-strict models.** The Azure gpt-5.x reasoning family (gpt-5,
   gpt-5-mini, gpt-5.3-codex, gpt-5.4, gpt-5.5, gpt-5.6, ...) returns a 400 on
   any explicit ``temperature`` other than the default (1). tau2's user
   simulator defaults to ``temperature=0.0``. We wrap the ``completion`` symbol
   used inside ``tau2.utils.llm_utils`` to drop ``temperature`` proactively for
   the known family, with a retry safety net for anything unmatched.

2. **Proxy-routed NL-assertion judge.** tau2's ``DEFAULT_LLM_NL_ASSERTIONS`` is
   a bare OpenAI model id (``gpt-4.1-2025-04-14``); litellm sends that straight
   to ``api.openai.com``, where the proxy key is rejected with a 401. This only
   bites once a task with NL assertions runs to completion (so it hides behind
   truncated or assertion-free tasks). We rebind the judge model in the
   evaluator module to a ``litellm_proxy/`` model (``$TAU2_LLM_NL_ASSERTIONS``,
   else ``$MODEL_NAME``, else ``litellm_proxy/Azure/gpt-4.1``).

(No litellm async-LoggingWorker shim is needed: this ReAct agent runs in tau2's
normal synchronous flow, so the event-loop noise that would require one never
arises.)

Note: ``DEFAULT_LLM_ENV_INTERFACE`` (used only by LLM-backed env domains, e.g.
telecom) has the same bare-model issue but is captured as a default argument, so
it is not patched here; set it explicitly if you evaluate such a domain.
"""

from __future__ import annotations

import os
from typing import Any, Callable

from loguru import logger


DEFAULT_NL_JUDGE = "litellm_proxy/Azure/gpt-4.1"


def _rejects_temperature(model: str) -> bool:
    """True for models that 400 on any explicit temperature != default (1).

    Covers the Azure gpt-5.x reasoning family (gpt-5, gpt-5-mini, gpt-5.3-codex,
    gpt-5.4, gpt-5.5, gpt-5.6, ...) regardless of provider prefix or casing. The
    retry safety net in the wrapper is the backstop for anything not matched
    here; this just avoids the wasteful fail-then-retry on every call for the
    known family.
    """
    return "gpt-5" in (model or "").lower()


def _is_temperature_error(exc: Exception) -> bool:
    """Heuristic: does this exception look like a temperature-not-supported 400?"""
    return "temperature" in str(exc).lower()


def resolve_nl_judge(model: str | None = None) -> str:
    """Pick the NL-assertion judge model: arg, then env, then MODEL_NAME, then default."""
    return (
        model
        or os.environ.get("TAU2_LLM_NL_ASSERTIONS")
        or os.environ.get("MODEL_NAME")
        or DEFAULT_NL_JUDGE
    )


def route_nl_judge(model: str | None = None) -> str:
    """Rebind tau2's NL-assertion judge model to a proxy-routed model. Returns it."""
    from tau2.evaluator import evaluator_nl_assertions as ev

    judge = resolve_nl_judge(model)
    ev.DEFAULT_LLM_NL_ASSERTIONS = judge
    logger.info(f"Routed tau2 NL-assertion judge through proxy: {judge}")
    return judge


_installed = False


def install() -> None:
    """Idempotently apply the tau2 proxy-compat shims (temperature + NL judge).

    Safe to call more than once; only the first call patches.
    """
    global _installed
    if _installed:
        return

    from tau2.utils import llm_utils

    original: Callable[..., Any] = llm_utils.completion

    def _completion(*args: Any, **kwargs: Any) -> Any:
        model = kwargs.get("model") or (args[0] if args else "")
        # Azure gpt-5.x reasoning models (gpt-5-mini, gpt-5.3/5.4/5.5/5.6, ...)
        # reject any explicit `temperature` other than the default (1) with a
        # 400. Drop it proactively for the whole family so we don't fail-then-
        # retry on every call.
        if "temperature" in kwargs and _rejects_temperature(model):
            logger.warning(f"Model {model} doesn't support temperature. Dropping param.")
            kwargs.pop("temperature", None)
        try:
            return original(*args, **kwargs)
        except Exception as e:  # noqa: BLE001 - safety net, re-raised below if unrelated
            # Backstop for a temperature-strict model we didn't match by name:
            # retry once without the offending param, otherwise re-raise as-is.
            if "temperature" in kwargs and _is_temperature_error(e):
                logger.warning(f"Retrying {model} without temperature after error: {e}")
                kwargs.pop("temperature", None)
                return original(*args, **kwargs)
            raise

    llm_utils.completion = _completion
    route_nl_judge()
    _installed = True
    logger.info("Installed tau2 temperature-compat shim (drops temperature for gpt-5.x family).")
