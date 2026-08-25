from __future__ import annotations

import json
import re
from typing import Any

from .config import DEFAULT_OPTIMIZATION_MODEL, OPTIMIZATION_ENABLE_REASONING
from .context import json_compact_block, render_knowledge_results, render_optimizer_decision
from .llm import call_json_llm
from ..optimization.strategy_document import StrategyDocument
from commander.wake_events import ALLOWED_CONDITION_TYPES, DISABLED_WAKE_TYPES


_GENERIC_WHY = {
    "improves the strategy",
    "better scouting",
    "more consistent",
    "improves scouting",
    "better scouting is useful",
}

_COMMANDER_OWNED_DETAIL_IDS = frozenset({"scouting", "scans"})
_CONSISTENCY_ONLY_DETAIL_IDS = frozenset(
    {"engagement_and_reinforcement", "recovery_and_cleanup"}
)
_CONSISTENCY_REASON = re.compile(
    r"\b(?:stale|old|consisten|reference|align|dependency|引用|一致|同步)\w*\b",
    re.IGNORECASE,
)

_RUNTIME_FORBIDDEN_PHRASES = (
    "per-unit kiting",
    "kiting",
    "focus-fire",
    "focus fire",
    "exact focus-fire",
    "exact coordinate",
    "per-unit",
    "unit-level micro",
    "manual formation",
    "precise focus-fire",
    "multiple independent autonomous combat groups",
    "scanning is unsafe",
    "scan is unsafe",
    "unsafe to scan",
    "scan safety",
    "at maximum range",
    "enemy movement out of position",
    "enemy_truth",
    "replay-only",
    "last_seen_enemy_contents",
    "seconds_since_last_seen",
)

_CROSS_CYCLE_STATE_PATTERNS = (
    (
        re.compile(r"\b(?:cycle|step)\s*[12]\b", re.IGNORECASE),
        "encodes a multi-cycle state machine in strategy prose",
    ),
    (
        re.compile(
            r"\bafter\s+(?:the\s+)?scan_ready\s+(?:fires|fired|triggers|triggered)\b",
            re.IGNORECASE,
        ),
        "depends on remembering that scan_ready fired in an earlier decision cycle",
    ),
    (
        re.compile(r"\b(?:later|next)\s+scan\s+cycle\b", re.IGNORECASE),
        "depends on an implicit scan-cycle state",
    ),
    (
        re.compile(
            r"\b(?:after|once)\s+(?:the\s+)?(?:scan|scanner sweep)\s+"
            r"(?:completes|completed|has completed)\b",
            re.IGNORECASE,
        ),
        "depends on persistent scan-completion state",
    ),
)

_ATTACK_WORDS = re.compile(r"\b(?:attack|assault|push|commit|launch)\b", re.IGNORECASE)
_HARD_GATE_WORDS = re.compile(
    r"\b(?:do\s+not|must\s+not|only\s+(?:after|if|when)|hold(?:\s+the)?|until|unless)\b",
    re.IGNORECASE,
)
_COUNT_WORDS = re.compile(r"\b\d+\s+(?:completed\s+and\s+living\s+|living\s+)?[A-Za-z]", re.IGNORECASE)


def _decision_policy_errors(
    patches: list[dict[str, Any]],
    candidate_text: str,
) -> list[str]:
    """Reject paragraph patches that try to implement a hidden state machine."""
    errors: list[str] = []
    replacements = {
        str(item.get("target") or "").strip(): str(item.get("replacement") or "").strip()
        for item in patches
        if isinstance(item, dict)
    }
    complete_details: dict[str, str] = {}
    try:
        complete_details = {
            item.id: item.value for item in StrategyDocument.parse(candidate_text).details
        }
    except ValueError as exc:
        errors.append(f"internal_inconsistency — strategy.md — {exc}")

    for target, text in complete_details.items():
        for pattern, reason in _CROSS_CYCLE_STATE_PATTERNS:
            if pattern.search(text):
                errors.append(f"runtime_boundary — {target} — {reason}")

    scans = complete_details.get("scans", "")
    if scans and _ATTACK_WORDS.search(scans) and _HARD_GATE_WORDS.search(scans):
        errors.append(
            "internal_inconsistency — scans — Scans may request information but "
            "must not grant, deny, or delay first-attack permission"
        )

    posture = replacements.get("pre_attack_army_posture", "")
    if posture and _ATTACK_WORDS.search(posture) and _COUNT_WORDS.search(posture):
        errors.append(
            "internal_inconsistency — pre_attack_army_posture — staging may "
            "reference Main Attack Gate but must not copy a numerical attack gate"
        )

    recovery = replacements.get("recovery_and_cleanup", "")
    if recovery and _ATTACK_WORDS.search(recovery) and _COUNT_WORDS.search(recovery):
        errors.append(
            "internal_inconsistency — recovery_and_cleanup — recovery must "
            "reapply Main Attack Gate instead of copying numerical launch thresholds"
        )

    gate = complete_details.get("main_attack_gate", "")
    if gate and re.search(
        r"\b(?:no|none|fewer\s+than|below)\b.{0,70}\benemy\b.{0,70}"
        r"\b(?:visible|observed|seen)\b",
        gate,
        re.IGNORECASE,
    ):
        errors.append(
            "runtime_boundary — main_attack_gate — absence of a currently visible "
            "enemy unit is not a reliable hard permission to attack"
        )

    return list(dict.fromkeys(errors))


def _attack_gate_counts(text: str) -> dict[str, int]:
    try:
        document = StrategyDocument.parse(text)
    except ValueError:
        return {}
    gate = next((item.value for item in document.details if item.id == "main_attack_gate"), "")
    counts: dict[str, int] = {}
    pattern = re.compile(
        r"\b(\d+)\s+(?:completed\s+and\s+living\s+|completed\s+|living\s+)?"
        r"([A-Z][A-Za-z-]*(?:\s+[A-Z][A-Za-z-]*)?)",
    )
    for count_text, unit_text in pattern.findall(gate):
        unit = re.sub(r"s$", "", unit_text.replace(" ", "").casefold())
        counts[unit] = int(count_text)
    return counts


def _core_timing_errors(
    decision: dict[str, Any],
    parent_text: str,
    candidate_text: str,
) -> list[str]:
    guard = decision.get("core_mechanism_guard")
    if not isinstance(guard, dict):
        return []
    intended = str(guard.get("first_commitment_effect") or "").strip().lower()
    if intended not in {"earlier", "same"}:
        return []
    parent_counts = _attack_gate_counts(parent_text)
    candidate_counts = _attack_gate_counts(candidate_text)
    increases = [
        f"{unit}: {count}->{candidate_counts[unit]}"
        for unit, count in parent_counts.items()
        if unit in candidate_counts and candidate_counts[unit] > count
    ]
    if not increases:
        return []
    return [
        "strategy_identity — Main Attack Gate — candidate raises first-commitment "
        "counts despite core_mechanism_guard declaring an earlier or unchanged "
        f"commitment ({', '.join(increases)})"
    ]


def build_strategy_patch_validation_prompt(
    *,
    decision: dict[str, Any],
    parent_text: str,
    candidate_text: str,
    patches: list[dict[str, Any]],
    capability_manifest: dict[str, Any] | None = None,
    knowledge_runs: list[dict[str, Any]] | None = None,
    inheritance: dict[str, Any] | None = None,
) -> str:
    compact_patches = [
        {
            "target": str(item.get("target") or ""),
            "why_required": str(item.get("why_required") or ""),
            "replacement": str(item.get("replacement") or ""),
        }
        for item in patches
        if isinstance(item, dict)
    ]
    from .prompts import RUNTIME_CONTRACT

    return f"""You are validating a strategy patch.

You are NOT evaluating whether the hypothesis is strategically correct.
You are NOT choosing a better strategy.
You are NOT analyzing the matches.
You are NOT judging whether another causal hypothesis would have been better.

The Cross-Match Decision has already selected one primary failure mode and one
coherent intervention package. strategy_contract owns strategy identity,
priority_problem owns the problem definition, and plan.coordinated_changes owns
the requested modifications.

Check only whether the candidate strategy patch is a clean implementation of
that hypothesis.

Validate:

Allowed modification scope
The candidate may change only economy/expansion targets, production-building and
unit-count targets, technology/upgrades, army composition, or attack readiness and
strategic objective. Scouting, scanning, wake events, decision-cycle protocols,
reinforcement routing, retreat, recovery, and cleanup are not optimization domains.
Scouting and Scans must remain unchanged. Reinforcement or Recovery may change only
to replace a stale reference created by an allowed change with a reference to Main
Attack Gate or the selected objective; reject any new behavior in those paragraphs.

0. Decision grounding precondition
Do not re-rank strategic hypotheses, but reject the candidate when the selected plan depends on a factual or numerical premise that contradicts the supplied deterministic knowledge. This includes misreading production slots, time, cost, throughput, total resource demand, supply totals, prerequisites, producer availability, base or geyser availability, and upgrade effects. Reject rather than quietly patching around a false premise; the analysis must be rerun with the verified facts. A wake condition only requests a new high-level decision and never grants permission to attack or overrides the strategy's attack gate.

Use information_grounding only when an allowed decision directly uses currently
available enemy information. Replay-only enemy_truth may explain a result but cannot
become a live condition. The candidate must not add a scan/scout acquisition flow or
wake implementation to obtain that information.

1. Intervention-package scope and coverage
One failure mode is not one paragraph or one strategy category. Multiple coordinated
paragraph changes are expected when required by the supplied package. Every item in
plan.coordinated_changes must be implemented or already satisfied by the parent
strategy, and the complete candidate must be capable of producing
plan.material_behavior_change. For
every patch ask: "If this patch were removed, would the selected hypothesis become
incomplete, internally inconsistent, non-executable, or materially different?"
Reject a patch that fails this test because it introduces an unrelated second
strategic objective.

2. Dependency completeness and internal consistency
Reject when a required prerequisite, resource/production dependency, execution
dependency, or consistency dependency is missing. In particular, reject when a
global target changes but another paragraph retains a stale target or contradictory
rule. Do not require a redundant patch when the parent strategy already satisfies
the dependency.

When the candidate depends on production throughput, timing, cost, or sustained resource demand, validate it against the supplied deterministic knowledge calculations. Recompute totals by summing every concurrently required production line and explicit end-state unit count; do not validate each line in isolation. Reject numerical feasibility claims that contradict those calculations, and reject a package whose required production demand is unsupported by its own economy/resource rules. Check that gas extraction, producer construction, prerequisites, and expansions become available before—not after—the timing they are supposed to support. Do not invent missing income rates.

3. Selected-package coverage
Verify that the candidate implements every plan.coordinated_changes item or that
the parent already satisfies it. Do not re-evaluate whether the selected hypothesis
is strategically strong enough; match evaluation tests that question. A concern
about expected effect size is non-blocking unless a requested change is absent.

The complete candidate strategy must not contain contradictory thresholds,
production targets, priorities, technology requirements, attack conditions,
recovery conditions, or information requirements.

Check changed targets across Production, Technology, Main Attack Gate, Recovery,
and Ultimate Goal for direct contradictions. Numerical production bounds and final
supply are checked deterministically and must not be guessed from prose.

4. Analysis-optimization alignment
Check only that the candidate implements the supplied plan without adding a
different objective. Do not re-rank the hypothesis, reinterpret match evidence,
compare alternative strategies, or decide whether the selected army will win.
Those judgments belong to Cross-Match Analysis and later match evaluation.

5. Preserved strengths, strategy mechanism, and identity
Treat Cross-match Decision.strategy_contract as the binding interpretation of the
parent strategy. Verify its style, core_win_mechanism,
critical_timing_or_power_spike, and core commitments against the complete candidate.

Infer the defining army concept and win plan from the complete parent strategy,
not from a hard-coded strategy-family template. A candidate that turns support,
scan, scout, transformation state, and core unit counts into one accumulated hard
attack gate is over-constrained.

6. Runtime boundary
The strategy must not require unavailable micro, runtime behavior, or controls.

7. Concision and ownership
Prefer one clear observable rule over repeated warnings and narrow exceptions.
Main Attack Gate exclusively owns first-attack permission. Scans may request
information but cannot grant, deny, or delay attack permission. Pre-Attack Army
Posture owns gathering and staging and may only reference Main Attack Gate.
Recovery owns withdrawal and rebuilding and must reapply Main Attack Gate rather
than copying its thresholds. No paragraph may encode Cycle 1/Cycle 2, remember that
a scan_ready wake fired, or otherwise use natural-language prose as cross-cycle
state. Reject material strategy bloat that does not add a required dependency of
the selected hypothesis.

Exclusive if/else branches are consistent. Do not treat "fresh intel OR request
a scan, else use a fallback threshold" as an AND of mutually exclusive states.
Scan availability is not acquired enemy information. Scans may request current
information, and a later Commander decision may use its current observation, but
strategy.md must not encode the transition or remember scan completion.

Restating the same gate in a posture paragraph is non-blocking duplication.
A weak but internally consistent implementation of the hypothesis is not a
blocking failure.

Do not reject a candidate because you personally prefer another strategy.
Do not propose alternative strategy changes.
Do not generate replacement patches.

{RUNTIME_CONTRACT}

Capability summary:
{json.dumps(capability_manifest or {}, ensure_ascii=False, indent=2)}

Cross-match Decision:
{render_optimizer_decision(decision)}

Verified knowledge and deterministic feasibility calculations:
{render_knowledge_results(knowledge_runs or [])}

Patches:
{json_compact_block(compact_patches)}

Parent strategy.md:
{parent_text}

Candidate strategy.md:
{candidate_text}

Return JSON only:
{{
  "valid": true,
  "errors": [
    {{
      "type": "decision_grounding|unrelated_patch|missing_dependency|underpowered_implementation|internal_inconsistency|preserved_strengths|strategy_identity|runtime_boundary",
      "location": "paragraph title",
      "description": "what is wrong",
      "severity": "blocking|non-blocking"
    }}
  ]
}}

Set valid=true when there are no blocking issues. Include non-blocking notes
only as non-blocking errors; they must not make valid=false.
"""


def validate_strategy_patch_structure(
    *,
    decision: dict[str, Any],
    patches: list[dict[str, Any]],
    parent_document: StrategyDocument,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(patches, list) or not patches:
        errors.append("strategy patch must contain at least one paragraph change")
        return errors

    hypothesis = str(decision.get("hypothesis") or "").strip()
    if not hypothesis:
        errors.append("cross-match hypothesis is missing")

    detail_ids = {item.id for item in parent_document.details}
    seen: set[str] = set()
    for patch in patches:
        if not isinstance(patch, dict):
            errors.append("each strategy patch must be an object")
            continue
        target = str(patch.get("target") or "").strip()
        replacement = str(patch.get("replacement") or patch.get("value") or "").strip()
        why = str(patch.get("why_required") or "").strip()
        if not target:
            errors.append("patch target is required")
            continue
        if target == "summary":
            errors.append("summary modification is not allowed")
        if target in _COMMANDER_OWNED_DETAIL_IDS:
            errors.append(
                f"{target}: scouting and scanning behavior is Commander-owned "
                "and not evolvable"
            )
        if target not in detail_ids and target != "summary":
            errors.append(f"unknown strategy paragraph: {target}")
        if target in seen:
            errors.append(f"duplicate patch target: {target}")
        seen.add(target)
        if not replacement:
            errors.append(f"{target}: replacement is required")
        else:
            current = next(
                (item for item in parent_document.details if item.id == target),
                None,
            )
            if current is not None and replacement == current.value:
                errors.append(f"{target}: replacement is unchanged")
        if "\n" in str(patch.get("replacement") or patch.get("value") or ""):
            errors.append(
                f"{target}: replacement must be one complete Detail instruction"
            )
        if not why:
            errors.append(f"{target}: why_required is required")
        elif why.lower() in _GENERIC_WHY:
            errors.append(
                f"{target}: why_required must explain a direct hypothesis dependency"
            )
        elif target in _CONSISTENCY_ONLY_DETAIL_IDS and not _CONSISTENCY_REASON.search(
            why
        ):
            errors.append(
                f"{target}: this paragraph may change only for a stale-reference "
                "consistency repair caused by an allowed strategy change"
            )
        if target in _CONSISTENCY_ONLY_DETAIL_IDS and _COUNT_WORDS.search(
            replacement
        ):
            errors.append(
                f"{target}: consistency repair must reference Main Attack Gate or "
                "the selected objective instead of copying numerical targets"
            )
    return errors


def _runtime_boundary_errors(
    patches: list[dict[str, Any]],
    candidate_text: str,
    capability_manifest: dict[str, Any],
) -> list[str]:
    haystack = " ".join(
        [
            candidate_text,
            *[
                str(item.get("replacement") or "")
                for item in patches
                if isinstance(item, dict)
            ],
        ]
    ).lower()
    errors: list[str] = []
    for phrase in _RUNTIME_FORBIDDEN_PHRASES:
        if phrase in haystack:
            errors.append(
                f"strategy requires unavailable runtime behavior: {phrase}"
            )
    for item in capability_manifest.get("strategy_must_not_require") or []:
        text = str(item or "").strip().lower()
        if text and text in haystack:
            errors.append(f"strategy requires unavailable runtime behavior: {item}")
    supported_wake_types = set(ALLOWED_CONDITION_TYPES) - set(DISABLED_WAKE_TYPES)
    allowed_tokens = supported_wake_types | {
        "set_wake_event",
        *(
            str(name)
            for name in (capability_manifest.get("control_actions") or {}).keys()
        ),
        *(
            str(name)
            for name in (
                (capability_manifest.get("macro_contract") or {}).get(
                    "available_actions"
                )
                or []
            )
        ),
    }
    for clause in re.split(r"[.;\n]", haystack):
        if "wake" not in clause:
            continue
        tokens = set(re.findall(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b", clause))
        for token in sorted(tokens - allowed_tokens):
            errors.append(f"unsupported wake condition in strategy: {token}")
    return list(dict.fromkeys(errors))


def _semantic_issue(item: Any) -> tuple[str, str]:
    if isinstance(item, dict):
        severity = str(item.get("severity") or "blocking").strip().lower()
        if severity not in {"blocking", "non-blocking"}:
            severity = "blocking"
        parts = [
            str(item.get("type") or "").strip(),
            str(item.get("location") or "").strip(),
            str(item.get("description") or item.get("error") or "").strip(),
        ]
        message = " — ".join(part for part in parts if part)
        if not message:
            message = json.dumps(item, ensure_ascii=False)
        return severity, message
    return "blocking", str(item).strip()


_MISSING_TARGET_MARKERS = (
    "no explicit",
    "not explicit",
    "not specified",
    "not listed",
    "not stated",
    "missing",
    "unspecified",
    "unknown",
    "none",
    "n/a",
)


def _audit_value_is_missing(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return not text or any(marker in text for marker in _MISSING_TARGET_MARKERS)


def _normalized_entity(value: Any) -> str:
    text = re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())
    return text[:-1] if text.endswith("s") and len(text) > 2 else text


def _trainable_units(capability_manifest: dict[str, Any]) -> dict[str, str]:
    macro = capability_manifest.get("macro_contract")
    actions = macro.get("available_actions") if isinstance(macro, dict) else []
    result: dict[str, str] = {}
    for action in actions or []:
        name = str(action or "")
        if not name.startswith("train_"):
            continue
        stem = _normalized_entity(name[len("train_") :])
        if stem:
            result[stem] = name[len("train_") :]
    return result


def _text_has_numeric_target(text: str, unit_stem: str) -> bool:
    compact = re.sub(r"[^a-z0-9]+", "", str(text or "").casefold())
    singular = _normalized_entity(unit_stem)
    return bool(
        singular
        and re.search(rf"\d+[a-z]{{0,36}}{re.escape(singular)}s?", compact)
    )


def _deterministic_production_target_errors(
    *,
    candidate_text: str,
    capability_manifest: dict[str, Any],
) -> list[str]:
    """Check only actual trainable units named in continuing-production rules."""
    try:
        document = StrategyDocument.parse(candidate_text)
    except ValueError:
        return []
    trainable = _trainable_units(capability_manifest)
    if not trainable:
        return []
    continuation_terms = re.compile(
        r"\b(?:continue|continuously|resume|restart|re-enable|reenable|return to|"
        r"replace losses|keep producing|keep training)\b",
        re.IGNORECASE,
    )
    required: dict[str, str] = {}
    for detail in document.details:
        if not continuation_terms.search(detail.value):
            continue
        compact = _normalized_entity(detail.value)
        for stem, display in trainable.items():
            if stem and stem in compact:
                required[stem] = display

    ultimate = next(
        (item.value for item in document.details if item.id == "ultimate_goal"),
        "",
    )
    full_text = document.render()
    errors: list[str] = []
    for stem, display in sorted(required.items()):
        if not _text_has_numeric_target(full_text, stem):
            errors.append(
                f"missing_dependency — production target for {display} — "
                "explicit numerical stage target is required"
            )
        if not _text_has_numeric_target(ultimate, stem):
            errors.append(
                f"missing_dependency — Ultimate Goal for {display} — "
                "continuing production requires an explicit final count or cap"
            )
    return errors


def _production_contract_errors(
    payload: dict[str, Any],
    *,
    candidate_text: str = "",
    capability_manifest: dict[str, Any] | None = None,
) -> list[str]:
    """Use deterministic candidate text for blocking production-bound checks.

    The semantic model's audit remains useful diagnostic output, but it is not
    authoritative: it previously treated structures as trainable units and missed
    numerical targets that were present elsewhere in the complete strategy.
    """
    capability_manifest = capability_manifest or {}
    if candidate_text and capability_manifest:
        return _deterministic_production_target_errors(
            candidate_text=candidate_text,
            capability_manifest=capability_manifest,
        )
    errors: list[str] = []
    audit = payload.get("production_target_audit")
    if not isinstance(audit, list) or not audit:
        errors.append(
            "missing_dependency — production_target_audit — "
            "semantic validator must return the candidate-wide production audit"
        )
    else:
        seen_units: set[str] = set()
        continuing_markers = (
            "continue",
            "continuously",
            "resume",
            "restart",
            "re-enable",
            "reenable",
            "return to",
        )
        for index, row in enumerate(audit, start=1):
            if not isinstance(row, dict):
                errors.append(
                    f"missing_dependency — production_target_audit[{index}] — "
                    "audit row must be an object"
                )
                continue
            unit = str(row.get("unit") or "").strip()
            label = unit or f"row {index}"
            unit_key = unit.casefold()
            if not unit:
                errors.append(
                    f"missing_dependency — production_target_audit[{index}] — "
                    "unit is required"
                )
            elif unit_key in seen_units:
                errors.append(
                    f"internal_inconsistency — production_target_audit — "
                    f"duplicate audit row for {unit}"
                )
            else:
                seen_units.add(unit_key)

            instruction = str(row.get("instruction") or "").strip().lower()
            stage_missing = _audit_value_is_missing(row.get("stage_target"))
            ultimate_missing = _audit_value_is_missing(
                row.get("ultimate_goal_target")
            )
            stop_missing = _audit_value_is_missing(row.get("temporary_stop_rule"))
            continues_or_resumes = any(
                marker in instruction for marker in continuing_markers
            )
            verdict = str(row.get("verdict") or "").strip().lower()

            if stage_missing:
                errors.append(
                    f"missing_dependency — production target for {label} — "
                    "explicit numerical stage target is required"
                )
            if ultimate_missing and continues_or_resumes:
                errors.append(
                    f"missing_dependency — Ultimate Goal for {label} — "
                    "resumed or continuing production requires an explicit final count or cap"
                )
            elif ultimate_missing and stop_missing:
                errors.append(
                    f"missing_dependency — production bound for {label} — "
                    "provide an Ultimate Goal count or an explicit temporary stop rule"
                )
            if verdict != "bounded":
                errors.append(
                    f"missing_dependency — production_target_audit for {label} — "
                    f"verdict must be bounded, got {verdict or 'empty'}"
                )

    final_supply = payload.get("final_supply")
    if not isinstance(final_supply, dict):
        errors.append(
            "missing_dependency — final_supply — semantic validator must return "
            "the complete final supply audit"
        )
    else:
        total = final_supply.get("total")
        if isinstance(total, bool) or not isinstance(total, (int, float)):
            errors.append(
                "missing_dependency — final_supply.total — numeric total is required"
            )
        elif total < 0 or total > 200:
            errors.append(
                f"internal_inconsistency — final_supply.total — {total} exceeds "
                "the valid 0-200 supply range"
            )
        calculation = str(final_supply.get("calculation") or "").strip()
        if not calculation:
            errors.append(
                "missing_dependency — final_supply.calculation — complete supply "
                "calculation is required"
            )
        verdict = str(final_supply.get("verdict") or "").strip().lower()
        if verdict != "valid":
            errors.append(
                "internal_inconsistency — final_supply.verdict — "
                f"expected valid, got {verdict or 'empty'}"
            )
    return list(dict.fromkeys(errors))


def _blocking_semantic_errors(
    payload: dict[str, Any],
    *,
    candidate_text: str = "",
    capability_manifest: dict[str, Any] | None = None,
) -> list[str]:
    reported = payload.get("errors") or []
    blocking: list[str] = []
    has_non_blocking = False
    if not isinstance(reported, list):
        reported = [reported]
    for item in reported:
        if isinstance(item, dict) and str(item.get("type") or "").strip().lower() == (
            "underpowered_implementation"
        ):
            # Effect size is tested by candidate matches. It is not a compile error.
            has_non_blocking = True
            continue
        if candidate_text and capability_manifest:
            audit_text = (
                " ".join(
                    str(item.get(key) or "")
                    for key in ("type", "location", "description", "error")
                )
                if isinstance(item, dict)
                else str(item)
            ).casefold()
            if any(
                marker in audit_text
                for marker in (
                    "production target for",
                    "ultimate goal for",
                    "production_target_audit",
                    "final_supply",
                )
            ):
                # The complete candidate and live train_* catalog are the
                # authority for these checks; ignore semantic-auditor guesses.
                continue
        severity, message = _semantic_issue(item)
        if not message:
            continue
        if severity == "non-blocking":
            has_non_blocking = True
            continue
        blocking.append(message)
    valid = payload.get("valid")
    blocking.extend(
        _production_contract_errors(
            payload,
            candidate_text=candidate_text,
            capability_manifest=capability_manifest,
        )
    )
    if blocking:
        return list(dict.fromkeys(blocking))
    if valid is True or (valid is False and has_non_blocking):
        return []
    if valid is False:
        return ["semantic validator rejected the patch"]
    if valid is not True:
        return ["semantic validator must return valid=true or errors"]
    return []


def validate_strategy_patch_semantics(
    *,
    decision: dict[str, Any],
    parent_text: str,
    candidate_text: str,
    patches: list[dict[str, Any]],
    capability_manifest: dict[str, Any] | None = None,
    knowledge_runs: list[dict[str, Any]] | None = None,
    inheritance: dict[str, Any] | None = None,
    model: str = "",
) -> list[str]:
    capability_manifest = capability_manifest or {}
    errors = _runtime_boundary_errors(patches, candidate_text, capability_manifest)
    errors.extend(_decision_policy_errors(patches, candidate_text))
    result = call_json_llm(
        build_strategy_patch_validation_prompt(
            decision=decision,
            parent_text=parent_text,
            candidate_text=candidate_text,
            patches=patches,
            capability_manifest=capability_manifest,
            knowledge_runs=knowledge_runs,
            inheritance=inheritance,
        ),
        model=str(model or "").strip() or DEFAULT_OPTIMIZATION_MODEL,
        is_reasoning=OPTIMIZATION_ENABLE_REASONING,
    )
    if not isinstance(result, dict):
        errors.append("semantic validator returned no JSON object")
        return errors
    payload = result.get("validation") if isinstance(result.get("validation"), dict) else result
    if not isinstance(payload, dict):
        errors.append("semantic validator returned no JSON object")
        return errors
    errors.extend(
        _blocking_semantic_errors(
            payload,
            candidate_text=candidate_text,
            capability_manifest=capability_manifest,
        )
    )
    return errors


def validate_strategy_patch(
    *,
    decision: dict[str, Any],
    patches: list[dict[str, Any]],
    parent_document: StrategyDocument,
    capability_manifest: dict[str, Any] | None = None,
    parent_text: str = "",
    candidate_text: str = "",
    model: str = "",
    knowledge_runs: list[dict[str, Any]] | None = None,
) -> list[str]:
    errors = validate_strategy_patch_structure(
        decision=decision,
        patches=patches,
        parent_document=parent_document,
    )
    if errors or not candidate_text:
        return errors
    errors.extend(
        validate_strategy_patch_semantics(
            decision=decision,
            parent_text=parent_text or parent_document.render(),
            candidate_text=candidate_text,
            patches=patches,
            capability_manifest=capability_manifest or {},
            knowledge_runs=knowledge_runs,
            model=model,
        )
    )
    return errors


__all__ = [
    "build_strategy_patch_validation_prompt",
    "validate_strategy_patch",
    "validate_strategy_patch_semantics",
    "validate_strategy_patch_structure",
]
