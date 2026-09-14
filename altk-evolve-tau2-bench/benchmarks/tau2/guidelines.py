"""Playbook-JSON guideline loading for the tau2 ReAct agent.

Guidelines are learned lessons from past task attempts, kept in a *playbook JSON
file* that is **separate from the domain policy**. They are rendered into a
markdown section and appended to the agent's system prompt after the policy.

Playbook file shape::

    {"entries": [{"r": "<guideline text>", "n": <support count>, "e": <ignored>}, ...]}

Entries are selected purely by their support count ``n`` against a threshold
chosen by ``gtype``:

    lossless -> n >= 1   (everything)
    sup2     -> n >= 2   (recurring)
    sup3     -> n >= 3   (tight core)
"""

from __future__ import annotations

import json
from typing import List, Optional

from loguru import logger

# Support-count thresholds per guideline "type", matching the AppWorld harness.
GUIDELINES_MIN_SUPPORT = {"lossless": 1, "sup2": 2, "sup3": 3}

GUIDELINES_HEADING = (
    "## Guidelines learned from past task attempts (follow these carefully):\n"
)


def format_guidelines_section(rules: List[str]) -> Optional[str]:
    """Render a list of guideline strings as a markdown system-prompt section.

    Returns ``None`` when *rules* is empty so callers can skip injection.
    """
    if not rules:
        return None
    bullets = "\n".join(f"- {g}" for g in rules)
    return GUIDELINES_HEADING + bullets + "\n"


def load_guidelines_section(path: str, gtype: str = "lossless") -> Optional[str]:
    """Build a markdown guidelines section from a playbook JSON file.

    Args:
        path: Path to the playbook JSON file (shape ``{"entries": [{"r", "n"}]}``).
        gtype: Which entries to include by support count ``n``: ``lossless``
            (n>=1, all), ``sup2`` (n>=2), or ``sup3`` (n>=3).

    Returns:
        A markdown section (heading + ``- `` bullets) to append to the agent's
        system prompt, or ``None`` if the file yields no guidelines.

    Raises:
        ValueError: If *gtype* is not one of the known thresholds.
    """
    min_n = GUIDELINES_MIN_SUPPORT.get(gtype)
    if min_n is None:
        raise ValueError(
            f"Unknown guidelines type {gtype!r}; expected one of "
            f"{sorted(GUIDELINES_MIN_SUPPORT)}"
        )
    with open(path) as f:
        entries = json.load(f).get("entries", [])
    chosen = [x["r"] for x in entries if int(x.get("n", 1)) >= min_n]
    logger.info(
        f"Guidelines: selected {len(chosen)}/{len(entries)} entries from {path} "
        f"(type={gtype})"
    )
    return format_guidelines_section(chosen)


def merge_special_instructions(
    base: Optional[str], guidelines_section: Optional[str]
) -> Optional[str]:
    """Append a guidelines section to base instructions.

    Either argument may be ``None``. The guidelines section is appended after
    the base instructions with a blank-line separator (base first, learned
    guidelines after).
    """
    base = (base or "").rstrip()
    if not guidelines_section:
        return base or None
    if not base:
        return guidelines_section
    return base + "\n\n" + guidelines_section
