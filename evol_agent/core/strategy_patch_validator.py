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
)


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
    from .prompts import RUNTIME_CONTRACT, SC2_STRATEGIC_PRIORITY

    return f"""You are validating a strategy patch.

You are NOT evaluating whether the hypothesis is strategically correct.
You are NOT choosing a better strategy.
You are NOT analyzing the matches.
You are NOT judging whether another causal hypothesis would have been better.

The Cross-Match Decision has already selected one primary failure mode and one
coherent intervention package in plan.

Check only whether the candidate strategy patch is a clean implementation of
that hypothesis.

Validate:

0. Decision grounding precondition
Do not re-rank strategic hypotheses, but reject the candidate when the selected plan depends on a factual or numerical premise that contradicts the supplied deterministic knowledge. This includes misreading production slots, time, cost, throughput, total resource demand, supply totals, prerequisites, producer availability, base or geyser availability, and upgrade effects. Reject rather than quietly patching around a false premise; the analysis must be rerun with the verified facts. A wake condition only requests a new high-level decision and never grants permission to attack or overrides the strategy's attack gate.

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

3. Test strength
The candidate must be structurally capable of producing the pre-registered
mechanism_prediction.expected_change at or beyond minimum_material_change. Reject
a cosmetic, token, isolated, or clearly underpowered implementation that cannot materially
test the supplied hypothesis. Judge intervention strength from the declared
mechanism and parent-to-candidate strategy difference, never from patch count.
This validates designed test strength only; do not claim that runtime execution or
the realized match mechanism has already been observed.

The complete candidate strategy must not contain contradictory thresholds,
production targets, priorities, technology requirements, attack conditions,
recovery conditions, or information requirements.

4. Analysis-optimization priority alignment
The candidate must preserve the same strategic priority used by Cross-Match
Analysis. Reject a candidate that replaces a required combat-package, survival,
matchup, or relative power-window change with an easier lower-priority surrogate
such as more scouting, a later gate, more production, more economy, or an isolated
upgrade. Information is sufficient only when it causes the named higher-priority
composition, readiness, commitment, defense, or recovery decision. The optimized
strategy must realize the complete plan rather than merely its easiest item.

If the package targets a delayed technology, upgrade, composition, or power spike,
reject it when the candidate does not implement the supplied survival prerequisite
needed to reach that state under the observed pressure pattern. Check matchup and
support changes as part of the whole package rather than treating one upgrade as a
complete implementation.

5. Preserved strengths and identity
Unrelated supported strengths must remain intact. The candidate must preserve the
parent strategy's defining army concept and win plan unless the Cross-match
Decision explicitly justifies changing it.

Use the inheritance ledger as an audit claim, then verify it against the complete parent and candidate texts. Reject any material parent mechanism that disappears without appearing under remove with an evidence-based reason. Reject a keep entry that is not actually retained and a revise entry that does not describe the corresponding material change.

Infer the defining army concept and win plan from the complete parent strategy,
not from a hard-coded strategy-family template. A candidate that turns support,
scan, scout, transformation state, and core unit counts into one accumulated hard
attack gate is over-constrained.

6. Runtime boundary
The strategy must not require unavailable micro, runtime behavior, or controls.

7. Concision and ownership
Prefer one clear observable rule over repeated warnings and narrow exceptions.
One paragraph owns the complete attack gate. Dependent paragraphs may reference
that gate but must not copy its full condition. Reject material strategy bloat
that does not add a required dependency of the selected hypothesis.

Exclusive if/else branches are consistent. Do not treat "fresh intel OR request
a scan, else use a fallback threshold" as an AND of mutually exclusive states.
Scan availability is not acquired enemy information. Reject an intel gate that
allows commitment merely because a scan can be requested; the strategy must scan,
hold, wake on a supported condition, and then re-decide from the later observation.

Do not require strategy.md to name a wake predicate for every clause. Requesting
a scan or scout on this cycle and deciding after a later observation is allowed.

Restating the same gate in a posture paragraph is non-blocking duplication.
A weak but internally consistent implementation of the hypothesis is not a
blocking failure.

Do not reject a candidate because you personally prefer another strategy.
Do not propose alternative strategy changes.
Do not generate replacement patches.

{RUNTIME_CONTRACT}
{SC2_STRATEGIC_PRIORITY}

Capability summary:
{json.dumps(capability_manifest or {}, ensure_ascii=False, indent=2)}

Cross-match Decision:
{render_optimizer_decision(decision)}

Verified knowledge and deterministic feasibility calculations:
{render_knowledge_results(knowledge_runs or [])}

Patches:
{json_compact_block(compact_patches)}

Inheritance ledger:
{json_compact_block(inheritance or {})}

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


def _blocking_semantic_errors(payload: dict[str, Any]) -> list[str]:
    reported = payload.get("errors") or []
    blocking: list[str] = []
    has_non_blocking = False
    if not isinstance(reported, list):
        reported = [reported]
    for item in reported:
        severity, message = _semantic_issue(item)
        if not message:
            continue
        if severity == "non-blocking":
            has_non_blocking = True
            continue
        blocking.append(message)
    valid = payload.get("valid")
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
    errors.extend(_blocking_semantic_errors(payload))
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
