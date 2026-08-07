"""Pre-match strategy tool selection (automated harness exposure surface).

Flow (once per match, before the first Commander decision):

1. Deterministically map ``# Resource Costs`` labels → Action.py tools.
2. Ask the LLM only to **add** missing macro tools (never remove required ones).
3. Always keep army/meta tools. Validate names against the full registry.
4. Cache the filtered action space for every later decision cycle.

The model never sees the full race catalog during play; selection is automated
from the strategy document with a single optional LLM add-pass.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from llm.caller import call_openai_detailed

from commander.tools import NON_MACRO_TOOL_NAMES

logger = logging.getLogger("commander.tool_selection")

_RESOURCE_COSTS_HEADER_RE = re.compile(
    r"^\s*#\s*Resource\s+Costs\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_NEXT_HEADER_RE = re.compile(r"^\s*#\s+\S+", re.MULTILINE)
_COST_LINE_RE = re.compile(r"^\s*[-*]\s+([^:\n]+)\s*:", re.MULTILINE)

# Display name (strategy Resource Costs label) → Action.py key(s).
# Keep aliases short and race-agnostic where possible; unknown labels are logged.
_LABEL_TO_TOOLS: Dict[str, Tuple[str, ...]] = {
    "scv": ("train_scv",),
    "marine": ("train_marine",),
    "marauder": ("train_marauder",),
    "reaper": ("train_reaper",),
    "ghost": ("train_ghost",),
    "hellion": ("train_hellion",),
    "hellbat": ("train_hellbat",),
    "widow mine": ("train_widow_mine",),
    "cyclone": ("train_cyclone",),
    "siege tank": ("train_siege_tank",),
    "thor": ("train_thor",),
    "viking": ("train_viking",),
    "medivac": ("train_medivac",),
    "liberator": ("train_liberator",),
    "raven": ("train_raven",),
    "banshee": ("train_banshee",),
    "battlecruiser": ("train_battlecruiser",),
    "supply depot": ("build_supply_depot",),
    "refinery": ("build_gas",),
    "barracks": ("build_barracks",),
    "factory": ("build_factory",),
    "starport": ("build_starport",),
    "engineering bay": ("build_engineering_bay",),
    "armory": ("build_armory",),
    "ghost academy": ("build_ghost_academy",),
    "fusion core": ("build_fusion_core",),
    "bunker": ("build_bunker",),
    "missile turret": ("build_missile_turret",),
    "sensor tower": ("build_sensor_tower",),
    "barracks tech lab": ("build_barracks_techlab",),
    "barracks reactor": ("build_barracks_reactor",),
    "factory tech lab": ("build_factory_techlab",),
    "factory reactor": ("build_factory_reactor",),
    "starport tech lab": ("build_starport_techlab",),
    "starport reactor": ("build_starport_reactor",),
    "command center": ("expand",),
    "orbital command": ("morph_orbital_command",),
    "planetary fortress": ("morph_planetary_fortress",),
    "yamato cannon": ("research_yamato_cannon",),
    "combat shield": ("research_shieldwall",),
    "stimpack": ("research_stimpack",),
    "concussive shells": ("research_concussive_shells",),
    "scanner sweep": ("scanner_sweep",),
}

# Always useful for Terran macro even when the cost line is terse.
_DEFAULT_MACRO_EXTRAS: Tuple[str, ...] = (
    "build_supply_depot",
    "expand",
    "train_scv",
)


def extract_resource_cost_labels(strategy_text: str) -> List[str]:
    """Return ordered Resource Costs labels from strategy markdown."""
    text = strategy_text or ""
    header = _RESOURCE_COSTS_HEADER_RE.search(text)
    if not header:
        return []
    rest = text[header.end() :]
    next_header = _NEXT_HEADER_RE.search(rest)
    block = rest[: next_header.start()] if next_header else rest
    labels: List[str] = []
    seen: Set[str] = set()
    for match in _COST_LINE_RE.finditer(block):
        label = " ".join(match.group(1).strip().split())
        key = label.lower()
        if not label or key in seen:
            continue
        seen.add(key)
        labels.append(label)
    return labels


def tools_from_resource_cost_labels(
    labels: Sequence[str],
    *,
    full_action_space: Dict[str, str],
) -> Tuple[Set[str], List[str]]:
    """Map cost labels to registry tools. Returns (tools, unmatched_labels)."""
    known = set(full_action_space)
    selected: Set[str] = set()
    unmatched: List[str] = []
    for label in labels:
        mapped = _LABEL_TO_TOOLS.get(label.strip().lower())
        if not mapped:
            unmatched.append(label)
            continue
        for name in mapped:
            if name in known:
                selected.add(name)
            else:
                unmatched.append(f"{label}->{name}")
    return selected, unmatched


def required_tools_from_strategy(
    strategy_text: str,
    *,
    full_action_space: Dict[str, str],
) -> Dict[str, Any]:
    """Deterministic required set: Resource Costs + defaults + army/meta."""
    labels = extract_resource_cost_labels(strategy_text)
    mapped, unmatched = tools_from_resource_cost_labels(
        labels, full_action_space=full_action_space
    )
    known = set(full_action_space)
    required = set(mapped)
    for name in _DEFAULT_MACRO_EXTRAS:
        if name in known:
            required.add(name)
    for name in NON_MACRO_TOOL_NAMES:
        if name in known:
            required.add(name)
    return {
        "labels": labels,
        "required_tools": sorted(required),
        "unmatched_labels": unmatched,
    }


def _parse_add_list(content: str) -> List[str]:
    text = (content or "").strip()
    if not text:
        return []
    # Prefer fenced or raw JSON object.
    candidates = [text]
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        candidates.insert(0, fence.group(1))
    brace = re.search(r"\{.*\}", text, re.DOTALL)
    if brace:
        candidates.append(brace.group(0))
    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except Exception:
            continue
        if not isinstance(data, dict):
            continue
        raw = data.get("add", data.get("tools", data.get("tool_names")))
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            continue
        out: List[str] = []
        for item in raw:
            name = str(item or "").strip()
            if name:
                out.append(name)
        return out
    return []


def _format_catalog_grouped(action_space: Dict[str, str]) -> str:
    groups = {
        "train": [],
        "build/morph": [],
        "research": [],
        "army/meta": [],
        "other": [],
    }
    for name in sorted(action_space):
        desc = " ".join((action_space.get(name) or "").split())
        line = f"- {name}: {desc}" if desc else f"- {name}"
        if name.startswith("train_"):
            groups["train"].append(line)
        elif name.startswith("research_"):
            groups["research"].append(line)
        elif name.startswith("build_") or name.startswith("morph_") or name == "expand":
            groups["build/morph"].append(line)
        elif name in NON_MACRO_TOOL_NAMES:
            groups["army/meta"].append(line)
        else:
            groups["other"].append(line)
    parts: List[str] = []
    for title, lines in groups.items():
        if not lines:
            continue
        parts.append(f"## {title}\n" + "\n".join(lines))
    return "\n\n".join(parts) if parts else "(none)"


def build_tool_selection_messages(
    *,
    strategy_text: str,
    full_action_space: Dict[str, str],
    required_tools: Sequence[str],
) -> List[Dict[str, str]]:
    required_sorted = sorted(required_tools)
    catalog = _format_catalog_grouped(full_action_space)
    system = (
        "You prepare the Commander tool exposure surface for one StarCraft II "
        "match. The harness already derived a REQUIRED tool set from the "
        "strategy Resource Costs. Your only job is to ADD macro tools that are "
        "clearly needed to execute the strategy but are missing from REQUIRED.\n\n"
        "Rules:\n"
        "- Do not remove or replace REQUIRED tools.\n"
        "- Prefer fewer tools; do not add unused race tech/upgrades.\n"
        "- Army/meta tools are handled by the harness; you may omit them.\n"
        "- Output ONE JSON object only: {\"add\":[\"tool_name\",...]} "
        "(use [] if nothing to add). No markdown fences, no prose."
    )
    user = (
        f"[Strategy]\n{(strategy_text or '').strip()}\n\n"
        f"[REQUIRED tools — keep all]\n"
        + "\n".join(f"- {name}" for name in required_sorted)
        + "\n\n[Full Action catalog]\n"
        + catalog
        + "\n\nReturn JSON {\"add\":[...]} now."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def merge_selected_tools(
    *,
    full_action_space: Dict[str, str],
    required_tools: Iterable[str],
    added_tools: Iterable[str] = (),
) -> Dict[str, str]:
    """Union required+added, drop unknown names, always keep army/meta."""
    known = set(full_action_space)
    names: Set[str] = set()
    for name in required_tools:
        if name in known:
            names.add(name)
    for name in added_tools:
        if name in known:
            names.add(name)
    for name in NON_MACRO_TOOL_NAMES:
        if name in known:
            names.add(name)
    return {name: full_action_space[name] for name in sorted(names)}


def select_tools_for_strategy(
    *,
    strategy_text: str,
    full_action_space: Dict[str, str],
    model_key: str = "",
    use_llm: bool = True,
) -> Dict[str, Any]:
    """Run the automated selection pipeline. Safe if the LLM call fails."""
    seed = required_tools_from_strategy(
        strategy_text, full_action_space=full_action_space
    )
    required = list(seed["required_tools"])
    added: List[str] = []
    llm_error = ""
    llm_content = ""
    llm_latency = None
    messages: List[Dict[str, str]] = []

    if use_llm and (model_key or "").strip():
        messages = build_tool_selection_messages(
            strategy_text=strategy_text,
            full_action_space=full_action_space,
            required_tools=required,
        )
        result = call_openai_detailed(
            messages, model_key=model_key.strip(), timeout=90.0
        )
        llm_content = str(result.get("content") or "")
        llm_error = str(result.get("error") or "")
        llm_latency = result.get("latency_seconds")
        if not llm_error:
            parsed = _parse_add_list(llm_content)
            known = set(full_action_space)
            required_set = set(required)
            added = [
                name
                for name in parsed
                if name in known and name not in required_set
            ]
        else:
            logger.warning(
                "tool selection LLM failed (%s); using required set only",
                llm_error,
            )
    elif use_llm:
        llm_error = "missing_model_key"
        logger.warning("tool selection skipped LLM: missing model_key")

    action_space = merge_selected_tools(
        full_action_space=full_action_space,
        required_tools=required,
        added_tools=added,
    )
    return {
        "action_space": action_space,
        "required_tools": required,
        "added_tools": added,
        "selected_tools": sorted(action_space),
        "labels": seed["labels"],
        "unmatched_labels": seed["unmatched_labels"],
        "llm_error": llm_error,
        "llm_content": llm_content,
        "llm_latency_seconds": llm_latency,
        "messages": messages,
        "full_tool_count": len(full_action_space),
        "selected_tool_count": len(action_space),
    }
