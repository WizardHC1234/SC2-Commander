from __future__ import annotations

import importlib
import re
from typing import Optional

from .core.config import REQUIRED_TOP_HEADINGS
from .core.types import ValidationResult
from .optimization.strategy_document import StrategyDocument


# Only reject instructions that the current strategy interface genuinely
# cannot express. Strategic quality is evaluated by playing the candidate.
_UNEXECUTABLE_STRATEGY_PATTERNS = (
    (
        re.compile(r"\bzone_\d+\b", re.IGNORECASE),
        "strategy.md cannot contain literal zone_id values; use semantic locations",
    ),
    (
        re.compile(r"\(\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*\)"),
        "strategy.md cannot command coordinates; use semantic locations",
    ),
    (
        re.compile(r"\bunit[_\s-]?tags?\b", re.IGNORECASE),
        "strategy.md cannot address units by tag",
    ),
    (
        re.compile(
            r"\b(?:use|fire|order|cast|activate|click)\s+yamato\b|"
            r"\byamato\s+(?:the|cannon\s+on|at|into)\b",
            re.IGNORECASE,
        ),
        "strategy.md cannot order Yamato; Sharpy handles that ability",
    ),
    (
        re.compile(
            r"\b(?:use|order|cast|activate)\s+tactical\s+jump\b|"
            r"\btactical\s+jump\s+(?:to|toward|onto)\b",
            re.IGNORECASE,
        ),
        "strategy.md cannot order Tactical Jump",
    ),
    (
        re.compile(
            r"\b(?:load|unload)\b.{0,40}\b(?:medivac|bunker)s?\b|"
            r"\b(?:medivac|bunker)s?\b.{0,40}\b(?:load|unload)\b",
            re.IGNORECASE,
        ),
        "strategy.md cannot order transport or bunker loading/unloading",
    ),
    (
        re.compile(
            r"\b(?:enter|toggle|activate|use)\s+siege\s+mode\b|"
            r"\bsiege\s+up\b|"
            r"\b(?:order|command|force|manually)\s+(?:to\s+)?unsiege\b|"
            r"\bunsiege\s+(?:the\s+)?(?:tanks?|units?|army)\b",
            re.IGNORECASE,
        ),
        "strategy.md cannot toggle Siege Mode; Sharpy handles unit micro",
    ),
    (
        re.compile(
            r"\bresearch\s+siege\s+mode\b|\bsiege\s+mode\s+research\b|"
            r"\bresearch\b.{0,20}\bsiege\s+tech\b",
            re.IGNORECASE,
        ),
        "current multiplayer Siege Tanks do not require Siege Mode research",
    ),
)

_DETAIL_BULLET = re.compile(r"^\* [A-Za-z][^:\n]{0,80}: \S")
_BAD_LIST_PREFIX = re.compile(r"^(\s*[-+] |\s*\d+[.)]\s+|\s*\[[ xX]\]\s+)")
_END_STATE_BULLET = re.compile(
    r"^\*\s*(?:Ultimate Goal|End State|Final Composition)\s*:\s*(.+)$",
    re.IGNORECASE,
)
_COUNTED_ENTITY = re.compile(
    r"\b(\d+)\s+([A-Za-z][A-Za-z -]*?)"
    r"(?=\s*(?:,|\band\b|\bwhile\b|\bto\b|\buntil\b|[.;]|$))",
    re.IGNORECASE,
)


def _section_body(text: str, heading: str, next_heading: str | None) -> str:
    start = text.find(heading)
    if start < 0:
        return ""
    start += len(heading)
    end = text.find(next_heading, start) if next_heading else len(text)
    return text[start:] if end < 0 else text[start:end]


def _nonempty_content_lines(section: str) -> list[str]:
    return [
        line.rstrip()
        for line in section.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _entity_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def _singular_entity_keys(value: str) -> list[str]:
    key = _entity_key(value)
    variants = [key]
    if key.endswith("ies") and len(key) > 3:
        variants.append(key[:-3] + "y")
    if key.endswith("s") and len(key) > 1:
        variants.append(key[:-1])
    return list(dict.fromkeys(variants))


def _unit_supply_by_name(race: str) -> dict[str, float]:
    normalized_race = str(race or "").strip().lower()
    if not normalized_race:
        return {}
    try:
        module = importlib.import_module(f"commander.races.{normalized_race}.actions")
    except (ImportError, ModuleNotFoundError):
        return {}
    supply_by_name: dict[str, float] = {}
    for action_name, spec in getattr(module, "ACTION_SPECS", {}).items():
        if not str(action_name).startswith("train_"):
            continue
        if getattr(spec, "action_type", "") != "unit":
            continue
        supply = float(getattr(spec, "supply", 0) or 0)
        if supply > 0:
            supply_by_name[_entity_key(str(action_name)[len("train_"):])] = supply
    return supply_by_name


def validate_strategy_supply_budget(content: str, *, race: str) -> Optional[str]:
    supply_by_name = _unit_supply_by_name(race)
    if not supply_by_name:
        return None
    for raw_line in str(content or "").splitlines():
        match = _END_STATE_BULLET.match(raw_line.strip())
        if not match:
            continue
        total = 0.0
        breakdown: list[str] = []
        for count_text, entity_text in _COUNTED_ENTITY.findall(match.group(1)):
            supply = next(
                (
                    supply_by_name[key]
                    for key in _singular_entity_keys(entity_text)
                    if key in supply_by_name
                ),
                None,
            )
            if supply is None:
                continue
            count = int(count_text)
            subtotal = count * supply
            total += subtotal
            subtotal_text = str(int(subtotal)) if subtotal.is_integer() else str(subtotal)
            breakdown.append(f"{count} {entity_text.strip()}={subtotal_text}")
        if total > 200:
            total_text = str(int(total)) if total.is_integer() else str(total)
            return (
                f"strategy.md explicit end-state requires {total_text} supply "
                f"({', '.join(breakdown)}), exceeding the hard 200 supply cap"
            )
    return None


def validate_strategy_house_style(content: str) -> Optional[str]:
    text = str(content or "")
    if re.search(r"^#{2,}\s", text, re.MULTILINE):
        return "strategy.md must keep only # Summary and # Details headings"
    if "```" in text:
        return "strategy.md must not contain code fences"
    if re.search(r"^\|.+\|$", text, re.MULTILINE):
        return "strategy.md must not contain Markdown tables"

    summary = _section_body(text, "# Summary", "# Details")
    details = _section_body(text, "# Details", None)
    for line in _nonempty_content_lines(summary):
        if line.lstrip().startswith("*") or _BAD_LIST_PREFIX.match(line):
            return "# Summary must be prose without bullets"
    detail_lines = _nonempty_content_lines(details)
    if not detail_lines:
        return "# Details section is empty"
    for line in detail_lines:
        if not _DETAIL_BULLET.match(line):
            return "# Details lines must use the form '* Title: instruction'"
    return None


def validate_strategy_markdown(content: str, *, race: str = "") -> Optional[str]:
    text = str(content or "").strip()
    if not text:
        return "strategy.md is empty"
    headings = [line.strip() for line in text.splitlines() if line.startswith("# ")]
    if headings != REQUIRED_TOP_HEADINGS:
        return (
            "strategy.md must contain exactly these non-empty level-one headings "
            f"in order: {', '.join(REQUIRED_TOP_HEADINGS)}"
        )
    for index, heading in enumerate(REQUIRED_TOP_HEADINGS):
        start = text.index(heading) + len(heading)
        end = (
            text.index(REQUIRED_TOP_HEADINGS[index + 1])
            if index + 1 < len(REQUIRED_TOP_HEADINGS)
            else len(text)
        )
        if not text[start:end].strip():
            return f"{heading} section is empty"
    style_error = validate_strategy_house_style(text)
    if style_error:
        return style_error
    try:
        StrategyDocument.parse(text)
    except ValueError as exc:
        return str(exc)
    for pattern, message in _UNEXECUTABLE_STRATEGY_PATTERNS:
        if pattern.search(text):
            return message
    return validate_strategy_supply_budget(text, race=race)


def validate_improvement(
    *,
    files: dict[str, str],
    race: str = "",
) -> ValidationResult:
    if set(files) != {"strategy.md"}:
        return ValidationResult(ok=False, error="files must contain only strategy.md")
    content = files.get("strategy.md", "")
    error = validate_strategy_markdown(content, race=race)
    if error:
        return ValidationResult(ok=False, error=error)
    return ValidationResult(ok=True, files={"strategy.md": content.strip() + "\n"})


__all__ = [
    "validate_improvement",
    "validate_strategy_markdown",
    "validate_strategy_supply_budget",
]

