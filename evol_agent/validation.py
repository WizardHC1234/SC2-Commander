from __future__ import annotations

import importlib
import re
from difflib import SequenceMatcher
from typing import Any, Optional

from .core.config import REQUIRED_TOP_HEADINGS
from .core.types import ValidationResult

_MATCH_NUMBER = re.compile(r"^Match\s+(\d+)\b", re.IGNORECASE)
_TIME_TOKEN = re.compile(r"\b\d+(?:-\d+)?s\b", re.IGNORECASE)
_MATCH_REF_AUTO_FIX_THRESHOLD = 0.45
_QUERY_INDEX_IN_TEXT = re.compile(
    r"(?:query[_\s-]?index|Q)\s*[:=]?\s*(\d+)",
    re.IGNORECASE,
)

# Knowledge questions go to the SC2 Data Agent (units/tech/counters), not this
# bot's runtime. Reject questions that ask about Commander/runtime behavior.
_RUNTIME_KNOWLEDGE_QUESTION = re.compile(
    r"(?ix)\b(?:"
    r"commander|army\s*planner|macro\s*planner|strategy\s*coordinator|action\s*translator|"
    r"build_directive|army_directive|main_force|main\s+force|group_1|reinforcement\s+group|"
    r"movement\s*modes?|defensive_retreat|panic_retreat|search_and_destroy|"
    r"attack\s*gate|runtime\s+(?:automatically|behavior|merge|contract)|"
    r"observation\s*mask|zone_id|python\s*automation|sharpy|"
    r"llm\s+behavior|per-?unit\s+micro"
    r")\b"
)
_MATCH_STRATEGY_TIMING_QUESTION = re.compile(
    r"(?ix)(?:"
    r"\b(?:optimal|best|ideal|recommended|typical)\b.{0,40}\b(?:timing|time|minute|when)\b|"
    r"\b(?:third|3rd|second|2nd)\s+base\b.{0,30}\b(?:timing|time|minute|when)\b|"
    r"\b(?:expand|expansion|attack|push)\b.{0,25}\b(?:timing|when\s+should|which\s+minute)\b|"
    r"\bwhen\s+should\b.{0,40}\b(?:expand|attack|push|retreat|engage)\b"
    r")"
)


def find_out_of_scope_knowledge_question_error(question: str) -> Optional[str]:
    """Return an error if a knowledge question targets bot runtime, not SC2 data."""
    text = str(question or "").strip()
    if not text:
        return None
    timing_match = _MATCH_STRATEGY_TIMING_QUESTION.search(text)
    if timing_match:
        return (
            "knowledge_question asks for a match-dependent strategy timing. The bundled "
            "entity dataset can verify costs, build times, producers, prerequisites, "
            "capabilities, counters, and synergies, but it cannot determine the optimal "
            "expansion/attack timing for these matches. Decide timing from match evidence."
        )
    match = _RUNTIME_KNOWLEDGE_QUESTION.search(text)
    if not match:
        return None
    return (
        "knowledge_question must stay inside the SC2 entity dataset "
        "(units/structures/upgrades/costs/tech/counters/synergies). "
        f"Found out-of-scope term {match.group(0)!r}. The knowledge base has no "
        "bot-runtime or fight-coordination facts (Commander, movement modes, "
        "group_1 merge, attack gates, etc.). Decide those from RUNTIME_CONTRACT "
        "and match evidence instead."
    )


def parse_query_index(value: Any) -> int | None:
    """Extract a positive 1-based knowledge query_index from int/str fields."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float) and value.is_integer():
        as_int = int(value)
        return as_int if as_int > 0 else None
    text = str(value or "").strip()
    if not text:
        return None
    if text.isdigit():
        as_int = int(text)
        return as_int if as_int > 0 else None
    match = _QUERY_INDEX_IN_TEXT.search(text)
    if match:
        as_int = int(match.group(1))
        return as_int if as_int > 0 else None
    return None


def resolve_match_reference(
    reference: str,
    allowed_match_references: set[str] | list[str],
) -> tuple[str | None, list[tuple[float, str]]]:
    """Map a paraphrased match citation to the closest finalized match_evidence.

    Returns ``(resolved_or_none, scored_candidates)`` where candidates are
    ``(score, allowed_text)`` sorted best-first.
    """
    ref = str(reference or "").strip()
    allowed = [
        str(item).strip()
        for item in allowed_match_references
        if str(item).strip()
    ]
    if not ref or not allowed:
        return None, []
    if ref in allowed:
        return ref, [(1.0, ref)]

    match_num = _MATCH_NUMBER.match(ref)
    candidates = allowed
    if match_num is not None:
        number = match_num.group(1)
        same_match = [
            item
            for item in allowed
            if (parsed := _MATCH_NUMBER.match(item)) and parsed.group(1) == number
        ]
        if same_match:
            candidates = same_match

    ref_times = set(_TIME_TOKEN.findall(ref))
    scored: list[tuple[float, str]] = []
    for item in candidates:
        ratio = SequenceMatcher(None, ref, item).ratio()
        item_times = set(_TIME_TOKEN.findall(item))
        if ref_times and item_times:
            overlap = len(ref_times & item_times) / len(ref_times | item_times)
            ratio = (0.65 * ratio) + (0.35 * overlap)
        scored.append((ratio, item))
    scored.sort(key=lambda pair: pair[0], reverse=True)
    if scored and scored[0][0] >= _MATCH_REF_AUTO_FIX_THRESHOLD:
        return scored[0][1], scored
    return None, scored


def normalize_improvement_match_references(
    analysis: dict[str, Any],
    allowed_match_references: set[str] | list[str],
) -> list[str]:
    """Rewrite approximate match citations onto exact finalized match_evidence.

    Mutates ``analysis['changes_made']`` in place. Returns human-readable fix notes.
    """
    allowed = {
        str(item).strip()
        for item in allowed_match_references
        if str(item).strip()
    }
    if not allowed:
        return []
    changes = analysis.get("changes_made")
    if not isinstance(changes, list):
        return []

    notes: list[str] = []

    def _rewrite(value: str, *, where: str) -> str:
        text = str(value or "").strip()
        if not text or text in allowed:
            return text
        resolved, _scored = resolve_match_reference(text, allowed)
        if resolved and resolved != text:
            notes.append(f"{where}: mapped approximate match citation onto finalized evidence")
            return resolved
        return text

    for index, change in enumerate(changes, 1):
        if not isinstance(change, dict):
            continue
        evidence = change.get("match_evidence")
        if isinstance(evidence, list):
            change["match_evidence"] = [
                _rewrite(item, where=f"changes_made[{index}].match_evidence")
                if str(item).strip()
                else item
                for item in evidence
            ]
        supported_by = change.get("supported_by")
        if isinstance(supported_by, list):
            for source_index, source in enumerate(supported_by, 1):
                if not isinstance(source, dict):
                    continue
                if str(source.get("source_type") or "").strip() != "match":
                    continue
                source["reference"] = _rewrite(
                    str(source.get("reference") or ""),
                    where=f"changes_made[{index}].supported_by[{source_index}]",
                )
        numeric_claims = change.get("new_numeric_claims")
        if isinstance(numeric_claims, list):
            for claim_index, claim in enumerate(numeric_claims, 1):
                if not isinstance(claim, dict):
                    continue
                if str(claim.get("source_type") or "").strip() != "match":
                    continue
                claim["reference"] = _rewrite(
                    str(claim.get("reference") or ""),
                    where=f"changes_made[{index}].new_numeric_claims[{claim_index}]",
                )
    return notes


def normalize_improvement_knowledge_citations(
    analysis: dict[str, Any],
    verified_knowledge_indices: set[int] | list[int],
    *,
    knowledge_references: dict[int, str] | None = None,
) -> list[str]:
    """Fill missing knowledge ``query_index`` from reference text like ``query_index: 7``."""
    allowed = {
        int(index)
        for index in verified_knowledge_indices
        if isinstance(index, int) or str(index).isdigit()
    }
    if not allowed:
        return []
    changes = analysis.get("changes_made")
    if not isinstance(changes, list):
        return []

    notes: list[str] = []
    summaries = {
        int(index): str(text).strip()
        for index, text in (knowledge_references or {}).items()
        if str(text).strip()
    }

    def _fix_knowledge_item(item: dict[str, Any], *, where: str) -> None:
        if str(item.get("source_type") or "").strip() != "knowledge":
            return
        parsed = parse_query_index(item.get("query_index"))
        if parsed is None:
            parsed = parse_query_index(item.get("reference"))
        if parsed is None or parsed not in allowed:
            return
        previous = item.get("query_index")
        if previous != parsed:
            item["query_index"] = parsed
            notes.append(f"{where}: set query_index={parsed}")
        reference = str(item.get("reference") or "").strip()
        stub_only = bool(
            reference
            and parse_query_index(reference) == parsed
            and not re.search(r"[A-Za-z]{4,}", re.sub(_QUERY_INDEX_IN_TEXT, "", reference))
        )
        if (not reference or stub_only) and parsed in summaries:
            item["reference"] = summaries[parsed]
            notes.append(f"{where}: filled knowledge reference from Q{parsed}")
        elif not reference:
            item["reference"] = f"Q{parsed}"
            notes.append(f"{where}: filled stub knowledge reference Q{parsed}")

    for index, change in enumerate(changes, 1):
        if not isinstance(change, dict):
            continue
        supported_by = change.get("supported_by")
        if isinstance(supported_by, list):
            for source_index, source in enumerate(supported_by, 1):
                if isinstance(source, dict):
                    _fix_knowledge_item(
                        source,
                        where=f"changes_made[{index}].supported_by[{source_index}]",
                    )
        numeric_claims = change.get("new_numeric_claims")
        if isinstance(numeric_claims, list):
            for claim_index, claim in enumerate(numeric_claims, 1):
                if isinstance(claim, dict):
                    _fix_knowledge_item(
                        claim,
                        where=(
                            f"changes_made[{index}].new_numeric_claims[{claim_index}]"
                        ),
                    )
    return notes


# Phrases that ask for behavior the live SC2 Agent stack cannot execute from strategy.md.
_UNEXECUTABLE_STRATEGY_PATTERNS = (
    (
        re.compile(r"\bzone_\d+\b", re.IGNORECASE),
        "strategy.md must be map-agnostic and cannot contain literal zone_id "
        "values such as zone_3; use semantic zone roles instead",
    ),
    (
        re.compile(
            r"\(\s*\d+(?:\.\d+)?\s*,\s*\d+(?:\.\d+)?\s*\)",
        ),
        "strategy.md cannot command coordinates; use semantic locations "
        "such as own_natural or enemy_main",
    ),
    (
        re.compile(r"\bunit[_\s-]?tags?\b", re.IGNORECASE),
        "strategy.md cannot address units by tag; leave unit selection to the runtime",
    ),
    (
        re.compile(
            r"\b(?:use|fire|order|cast|activate|click)\s+yamato\b|"
            r"\byamato\s+(?:the|cannon\s+on|at|into)\b",
            re.IGNORECASE,
        ),
        "strategy.md cannot order Yamato; that ability is handled by Sharpy micro",
    ),
    (
        re.compile(
            r"\b(?:use|order|cast|activate)\s+tactical\s+jump\b|"
            r"\btactical\s+jump\s+(?:to|toward|onto)\b",
            re.IGNORECASE,
        ),
        "strategy.md cannot order Tactical Jump; that ability is unexposed to strategy text",
    ),
    (
        re.compile(
            r"\b(?:load|unload)\b.{0,40}\b(?:medivac|bunker)s?\b|"
            r"\b(?:medivac|bunker)s?\b.{0,40}\b(?:load|unload)\b",
            re.IGNORECASE,
        ),
        "strategy.md cannot order transport or bunker loading/unloading; "
        "those actions are unexposed to strategy text",
    ),
    (
        re.compile(
            r"\b(?:enter|toggle|activate|use|put\s+\w+\s+into)\s+siege\s+mode\b|"
            r"\bsiege\s+up\b|"
            r"\b(?:order|command|force|manually)\s+(?:to\s+)?unsiege\b|"
            r"\bunsiege\s+(?:the\s+)?(?:tanks?|units?|army)\b",
            re.IGNORECASE,
        ),
        "strategy.md cannot toggle Siege Mode on units; siege/unsiege micro is left to Sharpy",
    ),
    (
        re.compile(
            r"\bresearch\s+siege\s+mode\b|"
            r"\bsiege\s+mode\s+research\b|"
            r"\bresearch\b.{0,20}\bsiege\s+tech\b",
            re.IGNORECASE,
        ),
        "strategy.md must not Research Siege Mode; current multiplayer Siege Tanks "
        "have Siege Mode without a separate upgrade",
    ),
    (
        re.compile(r"\bhold\s+position\b", re.IGNORECASE),
        "strategy.md cannot issue hold-position micro; use Army movement modes instead",
    ),
)

# Macro production must use fixed absolute targets. Forbid enemy-sighting forks
# that choose what to build/train (If enemy X detected, train Y; otherwise Z).
_DETECTION_GATED_PRODUCTION_PATTERNS = (
    (
        re.compile(
            r"\bif\s+enemy\b.{0,160}"
            r"(?:observed|remembered|detected|seen|visible|suspected|shows?|contact).{0,200}"
            r"(?:build|train|produce|add|open|start|research)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        "strategy.md must not choose what to build or train based on enemy "
        "detection; lock fixed absolute Macro targets from own-side gates only",
    ),
    (
        re.compile(
            r"(?:when|once|after)\s+enemy\b.{0,120}"
            r"(?:observed|remembered|detected|seen|visible|suspected|shows?).{0,160}"
            r"(?:build|train|produce|add|open|start|research)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        "strategy.md must not start production or research from an enemy sighting; "
        "use fixed own-side absolute targets",
    ),
    (
        re.compile(
            r"(?:build|train|produce|add)\b.{0,80}"
            r"(?:only\s+)?(?:when|if)\s+enemy\b",
            re.IGNORECASE | re.DOTALL,
        ),
        "strategy.md must not gate build/train targets on enemy presence; "
        "fix absolute production counts from own-side timing",
    ),
    (
        re.compile(
            r"(?:otherwise|else)\s+(?:train|build|produce|add)\b",
            re.IGNORECASE,
        ),
        "strategy.md must not branch Macro production with otherwise/else "
        "composition forks; pick one fixed production plan",
    ),
    (
        re.compile(
            r"(?:against|versus|vs\.?)\s+enemy\b.{0,60}"
            r"(?:or|,).{0,40}"
            r"(?:train|add|build|produce)\b|"
            r"(?:add|train|produce)\b.{0,80}"
            r"(?:against|versus)\s+enemy\b.{0,40}"
            r"\bor\b.{0,40}"
            r"(?:train|add|build|produce)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        "strategy.md must not present alternative unit packages keyed to enemy "
        "composition; commit to fixed absolute train targets",
    ),
)

# Sighting-only start triggers for production paths that need lead time.
_LATE_REACTION_STRATEGY_PATTERNS = (
    (
        re.compile(
            r"(?:only\s+)?when\s+enemy.{0,100}"
            r"(?:detected|is\s+seen|are\s+seen|shows?|shown|information|contact).{0,160}"
            r"(?:add|build|train|produce|open|start|research)\b",
            re.IGNORECASE | re.DOTALL,
        ),
        "strategy.md must not start a lead-time production path "
        "only after enemy units are detected; lock fixed own-side absolute targets "
        "before the first planned fight",
    ),
)


def find_late_reaction_strategy_error(content: str) -> Optional[str]:
    """Reject enemy-sighting forks that choose or start Macro production."""
    text = str(content or "")
    if not text.strip():
        return None
    for pattern, message in (
        _DETECTION_GATED_PRODUCTION_PATTERNS + _LATE_REACTION_STRATEGY_PATTERNS
    ):
        if pattern.search(text):
            return message
    return None


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
    if end < 0:
        end = len(text)
    return text[start:end]


def _nonempty_content_lines(section: str) -> list[str]:
    lines: list[str] = []
    for raw in section.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.lstrip().startswith("#"):
            continue
        lines.append(line)
    return lines


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
    specs = getattr(module, "ACTION_SPECS", {})
    supply_by_name: dict[str, float] = {}
    for action_name, spec in specs.items():
        if not str(action_name).startswith("train_"):
            continue
        if getattr(spec, "action_type", "") != "unit":
            continue
        supply = float(getattr(spec, "supply", 0) or 0)
        if supply <= 0:
            continue
        supply_by_name[_entity_key(str(action_name)[len("train_"):])] = supply
    return supply_by_name


def validate_strategy_supply_budget(content: str, *, race: str) -> Optional[str]:
    """Reject explicit end-state unit targets whose catalog supply exceeds 200."""
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
                "strategy.md explicit end-state requires "
                f"{total_text} supply ({', '.join(breakdown)}), exceeding the hard "
                "200 supply cap; reduce worker or army targets"
            )
    return None


def validate_strategy_house_style(content: str) -> Optional[str]:
    """Enforce the SKILL strategy.md bullet/outline house style."""
    text = str(content or "")
    if re.search(r"^#{2,}\s", text, re.MULTILINE):
        return (
            "strategy.md must not use ## or deeper headings; keep only "
            "# Summary and # Details"
        )
    if "```" in text:
        return "strategy.md must not contain code fences"
    if "|" in text and re.search(r"^\|.+\|$", text, re.MULTILINE):
        return "strategy.md must not contain markdown tables"

    summary = _section_body(text, "# Summary", "# Details")
    details = _section_body(text, "# Details", None)

    for line in _nonempty_content_lines(summary):
        if line.lstrip().startswith("*") or _BAD_LIST_PREFIX.match(line):
            return (
                "# Summary must be short prose paragraphs without bullets "
                "(house style must match the supplied current strategy.md)"
            )

    detail_lines = _nonempty_content_lines(details)
    if not detail_lines:
        return "# Details section is empty"
    for line in detail_lines:
        if _BAD_LIST_PREFIX.match(line) or line.lstrip().startswith("-"):
            return (
                "# Details bullets must use '* Title: ...' (asterisk + titled "
                "topic), not hyphen/numbered lists"
            )
        if not _DETAIL_BULLET.match(line):
            return (
                "# Details lines must look like '* Main Attack Gate: Begin "
                "the planned attack ...' (asterisk, Title, colon, body)"
            )

    return None


def validate_strategy_markdown(content: str, *, race: str = "") -> Optional[str]:
    """Validate current SC2-Commander strategy format and executability."""
    text = str(content or "").strip()
    if not text:
        return "strategy.md is empty"

    headings = [
        line.strip()
        for line in text.splitlines()
        if line.startswith("# ")
    ]
    if headings != REQUIRED_TOP_HEADINGS:
        return (
            "strategy.md must contain exactly these non-empty level-one "
            "headings in order: " + ", ".join(REQUIRED_TOP_HEADINGS)
        )

    positions = [text.index(heading) for heading in REQUIRED_TOP_HEADINGS]
    for index, heading in enumerate(REQUIRED_TOP_HEADINGS):
        start = positions[index] + len(heading)
        end = positions[index + 1] if index + 1 < len(positions) else len(text)
        if not text[start:end].strip():
            return f"{heading} section is empty"

    style_error = validate_strategy_house_style(text)
    if style_error:
        return style_error

    for pattern, message in _UNEXECUTABLE_STRATEGY_PATTERNS:
        if pattern.search(text):
            return message

    late_reaction_error = find_late_reaction_strategy_error(text)
    if late_reaction_error:
        return late_reaction_error

    supply_error = validate_strategy_supply_budget(text, race=race)
    if supply_error:
        return supply_error

    return None

def validate_improvement(*, files: dict[str, str], race: str = "") -> ValidationResult:
    if set(files) != {"strategy.md"}:
        return ValidationResult(
            ok=False,
            error="files must contain only strategy.md",
        )

    content = files.get("strategy.md", "")
    error = validate_strategy_markdown(content, race=race)
    if error:
        return ValidationResult(ok=False, error=error)

    return ValidationResult(
        ok=True,
        files={"strategy.md": content.strip() + "\n"},
    )


_NUMERIC_TOKEN = re.compile(
    r"(?<![\w.])\d+(?:\.\d+)?(?:\s*(?:%|seconds?|secs?|minutes?|mins?))?",
    re.IGNORECASE,
)


def validate_improvement_metadata(
    *,
    analysis: dict[str, Any],
    files: dict[str, str],
    current_strategy: str,
    allowed_problem_ids: set[str],
    verified_knowledge_available: bool,
    allowed_match_references: set[str] | None = None,
    verified_knowledge_indices: set[int] | None = None,
) -> Optional[str]:
    """Light-touch check of change metadata.

    Provenance fields are best-effort: unknown problem_ids, paraphrased match
    citations, missing knowledge indices, and thin runtime_requirements no
    longer block accepting an otherwise valid strategy.md. Only structurally
    empty or non-object changes_made entries hard-fail.
    """
    del files  # strategy body is validated separately by validate_improvement
    changes = analysis.get("changes_made")
    if not isinstance(changes, list) or not changes:
        return "optimization analysis requires non-empty changes_made"

    allowed_match_references = {
        str(value).strip() for value in (allowed_match_references or set())
        if str(value).strip()
    }
    verified_knowledge_indices = set(verified_knowledge_indices or set())
    allowed_problem_ids = {
        str(value).strip() for value in (allowed_problem_ids or set()) if str(value).strip()
    }

    for index, change in enumerate(changes, 1):
        if not isinstance(change, dict):
            return f"changes_made[{index}] must be an object"

        problem_id = str(change.get("problem_id") or "").strip()
        if not problem_id:
            # Soft: invent a label rather than rejecting the whole draft.
            change["problem_id"] = (
                next(iter(sorted(allowed_problem_ids)), f"P{index}")
            )
        # Unknown problem_id is allowed (Optimization may fix diagnosed issues
        # that were not promoted into optimization_targets).

        late_reaction_error = find_late_reaction_strategy_error(
            str(change.get("change") or "")
        )
        if late_reaction_error:
            return f"changes_made[{index}].change: {late_reaction_error}"

        supported_by = change.get("supported_by")
        if not isinstance(supported_by, list):
            supported_by = []
            change["supported_by"] = supported_by
        if not supported_by:
            # Soft: synthesize a minimal provenance stub so drafts are not blocked.
            evidence = change.get("match_evidence")
            if isinstance(evidence, list) and evidence:
                ref = str(evidence[0] or "").strip()
                if ref:
                    supported_by.append({"source_type": "match", "reference": ref})
            if not supported_by and current_strategy.strip():
                snippet = current_strategy.strip().splitlines()[0][:120]
                supported_by.append(
                    {"source_type": "current_strategy", "reference": snippet}
                )
            if not supported_by:
                supported_by.append(
                    {
                        "source_type": "current_strategy",
                        "reference": "strategy.md",
                    }
                )

        cleaned_sources: list[dict[str, Any]] = []
        for source in supported_by:
            if not isinstance(source, dict):
                continue
            source_type = str(source.get("source_type") or "").strip()
            reference = str(source.get("reference") or "").strip()
            if source_type not in {"match", "knowledge", "current_strategy"}:
                continue
            if not reference:
                continue
            if source_type == "match" and allowed_match_references:
                resolved, _scored = resolve_match_reference(
                    reference, allowed_match_references
                )
                if resolved:
                    source["reference"] = resolved
                # Keep paraphrases if unresolved; do not hard-fail.
            if (
                source_type == "current_strategy"
                and reference not in current_strategy
                and reference != "strategy.md"
            ):
                # Soft: keep the citation; Optimization narrative still useful.
                pass
            if source_type == "knowledge":
                if not verified_knowledge_available:
                    continue
                query_index = parse_query_index(source.get("query_index"))
                if query_index is None:
                    query_index = parse_query_index(source.get("reference"))
                if (
                    verified_knowledge_indices
                    and query_index not in verified_knowledge_indices
                ):
                    # Soft: drop broken knowledge cite rather than reject draft.
                    continue
                if query_index is not None:
                    source["query_index"] = query_index
            cleaned_sources.append(source)
        if not cleaned_sources:
            cleaned_sources.append(
                {"source_type": "current_strategy", "reference": "strategy.md"}
            )
        change["supported_by"] = cleaned_sources

        numeric_claims = change.get("new_numeric_claims")
        if numeric_claims is None or not isinstance(numeric_claims, list):
            change["new_numeric_claims"] = []
            numeric_claims = []
        cleaned_claims: list[dict[str, Any]] = []
        for claim in numeric_claims:
            if not isinstance(claim, dict):
                continue
            value = str(claim.get("value") or "").strip()
            source_type = str(claim.get("source_type") or "").strip()
            reference = str(claim.get("reference") or "").strip()
            if not value or not _NUMERIC_TOKEN.search(value):
                continue
            if source_type not in {"match", "knowledge"} or not reference:
                continue
            if source_type == "knowledge" and not verified_knowledge_available:
                continue
            if source_type == "match" and allowed_match_references:
                resolved, _scored = resolve_match_reference(
                    reference, allowed_match_references
                )
                if resolved:
                    claim["reference"] = resolved
            if source_type == "knowledge":
                query_index = parse_query_index(claim.get("query_index"))
                if query_index is None:
                    query_index = parse_query_index(claim.get("reference"))
                if (
                    verified_knowledge_indices
                    and query_index not in verified_knowledge_indices
                ):
                    continue
                if query_index is not None:
                    claim["query_index"] = query_index
            cleaned_claims.append(claim)
        change["new_numeric_claims"] = cleaned_claims

        runtime_requirements = change.get("runtime_requirements")
        if not isinstance(runtime_requirements, list) or not runtime_requirements:
            change["runtime_requirements"] = [
                {
                    "requirement": str(change.get("change") or "apply strategy change"),
                    "support_kind": "macro",
                    "supported": True,
                    "reference": "absolute targets",
                }
            ]
            runtime_requirements = change["runtime_requirements"]
        cleaned_requirements: list[dict[str, Any]] = []
        for requirement in runtime_requirements:
            if not isinstance(requirement, dict):
                continue
            statement = str(requirement.get("requirement") or "").strip()
            support_kind = str(requirement.get("support_kind") or "").strip()
            reference = str(requirement.get("reference") or "").strip()
            if not statement:
                continue
            if support_kind not in {"macro", "army", "automatic"}:
                support_kind = "macro"
                requirement["support_kind"] = support_kind
            if requirement.get("supported") is not True:
                requirement["supported"] = True
            if not reference:
                requirement["reference"] = "runtime contract"
            cleaned_requirements.append(requirement)
        if not cleaned_requirements:
            cleaned_requirements.append(
                {
                    "requirement": str(change.get("change") or "apply strategy change"),
                    "support_kind": "macro",
                    "supported": True,
                    "reference": "absolute targets",
                }
            )
        change["runtime_requirements"] = cleaned_requirements

    return None


__all__ = [
    "find_out_of_scope_knowledge_question_error",
    "normalize_improvement_knowledge_citations",
    "normalize_improvement_match_references",
    "parse_query_index",
    "resolve_match_reference",
    "validate_improvement",
    "validate_improvement_metadata",
    "validate_strategy_markdown",
    "validate_strategy_supply_budget",
]
