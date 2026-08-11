"""Select a compact tool surface from the natural-language strategy.

Once per match, a selector reads the whole strategy and chooses strategic macro
tools from the full race catalog. Race-specific metadata then expands those
choices to their transitive production and technology prerequisites. If semantic
selection is unavailable or invalid, the safe fallback is the full catalog.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

from llm.caller import call_openai_detailed

from commander.tools import NON_MACRO_TOOL_NAMES

logger = logging.getLogger("commander.tool_selection")

# Safe universal Terran surface. Strategic tools and their prerequisites are added
# by semantic selection and the race-specific dependency resolver.
_DEFAULT_MACRO_EXTRAS: tuple[str, ...] = (
    "build_supply_depot",
    "expand",
    "morph_orbital_command",
    "train_scv",
)

def _parse_select_list(content: str) -> List[str]:
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
        raw = data.get("select", data.get("tools", data.get("tool_names")))
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
) -> List[Dict[str, str]]:
    catalog = _format_catalog_grouped(full_action_space)
    system = (
        "You select the strategic Commander macro tools for one StarCraft II "
        "match. Read the entire natural-language strategy and choose the tools "
        "that express its intended units, production structures, add-ons, "
        "upgrades, defenses, and economy.\n\n"
        "Rules:\n"
        "- The strategy is authoritative for intent; catalog descriptions are "
        "authoritative for costs, producers, research locations, and prerequisites.\n"
        "- Select strategic goals, not every noun mentioned in scouting, enemy, "
        "cleanup, examples, or negative constraints. A statement such as 'no "
        "upgrades' must not select upgrade tools.\n"
        "- Do not select structural prerequisites merely because another chosen "
        "tool needs them; the harness computes the dependency closure.\n"
        "- Basic SCV, supply, expansion, Orbital, army and scheduling tools are "
        "added by the harness; omit them unless they are themselves a special "
        "strategic focus.\n"
        "- The harness also adds the Refinery tool automatically when selected "
        "actions consume vespene gas.\n"
        "- Prefer a compact set, but include every explicitly intended combat "
        "unit, production addon, upgrade, and static defense.\n"
        "- Output ONE JSON object only: {\"select\":[\"tool_name\",...]} "
        "with exact catalog names. No markdown fences, no prose."
    )
    user = (
        f"[Strategy]\n{(strategy_text or '').strip()}\n\n"
        "[Full Action catalog]\n"
        + catalog
        + "\n\nReturn JSON {\"select\":[...]} now."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def merge_selected_tools(
    *,
    full_action_space: Dict[str, str],
    selected_tools: Iterable[str],
) -> Dict[str, str]:
    """Drop unknown names and always keep army/meta tools."""
    known = set(full_action_space)
    names = {name for name in selected_tools if name in known}
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
    dependency_resolver: Optional[Callable[..., Set[str]]] = None,
) -> Dict[str, Any]:
    """Select semantic tools, expand dependencies, or safely expose everything."""
    known = set(full_action_space)
    baseline = {
        name
        for name in (*_DEFAULT_MACRO_EXTRAS, *NON_MACRO_TOOL_NAMES)
        if name in known
    }
    semantic: List[str] = []
    dependencies: Set[str] = set()
    llm_error = ""
    llm_content = ""
    llm_latency = None
    messages: List[Dict[str, str]] = []
    fallback_reason = ""
    dependency_error = ""

    if use_llm and (model_key or "").strip():
        messages = build_tool_selection_messages(
            strategy_text=strategy_text,
            full_action_space=full_action_space,
        )
        result = call_openai_detailed(
            messages, model_key=model_key.strip(), timeout=90.0
        )
        llm_content = str(result.get("content") or "")
        llm_error = str(result.get("error") or "")
        llm_latency = result.get("latency_seconds")
        if not llm_error:
            parsed = _parse_select_list(llm_content)
            semantic = list(dict.fromkeys(name for name in parsed if name in known))
            strategic_macro = [
                name
                for name in semantic
                if name not in baseline and name not in NON_MACRO_TOOL_NAMES
            ]
            if not strategic_macro:
                fallback_reason = "empty_or_invalid_selection"
        else:
            fallback_reason = "llm_error"
            logger.warning(
                "tool selection LLM failed (%s); exposing full catalog",
                llm_error,
            )
    elif use_llm:
        llm_error = "missing_model_key"
        fallback_reason = "missing_model_key"
        logger.warning("tool selection skipped LLM: exposing full catalog")
    else:
        fallback_reason = "llm_disabled"

    fallback_used = bool(fallback_reason)
    if fallback_used:
        action_space = dict(full_action_space)
    else:
        seed = baseline | set(semantic)
        expanded = set(seed)
        if dependency_resolver is not None:
            try:
                expanded = set(
                    dependency_resolver(seed, known_action_names=known)
                )
            except Exception as exc:
                dependency_error = f"{type(exc).__name__}: {exc}"
                fallback_reason = "dependency_resolver_error"
                fallback_used = True
                logger.exception(
                    "tool dependency expansion failed; exposing full catalog"
                )
        else:
            dependency_error = "missing dependency resolver"
            fallback_reason = "dependency_resolver_error"
            fallback_used = True
            logger.error(
                "tool dependency resolver missing; exposing full catalog"
            )
        if fallback_used:
            action_space = dict(full_action_space)
        else:
            dependencies = expanded - seed
            action_space = merge_selected_tools(
                full_action_space=full_action_space,
                selected_tools=expanded,
            )
    return {
        "action_space": action_space,
        "baseline_tools": sorted(baseline),
        "semantic_tools": semantic,
        "dependency_tools": sorted(dependencies),
        "selected_tools": sorted(action_space),
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason,
        "dependency_error": dependency_error,
        "llm_error": llm_error,
        "llm_content": llm_content,
        "llm_latency_seconds": llm_latency,
        "messages": messages,
        "full_tool_count": len(full_action_space),
        "selected_tool_count": len(action_space),
    }
