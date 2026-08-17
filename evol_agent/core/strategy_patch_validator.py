from __future__ import annotations

import json
from typing import Any

from .config import DEFAULT_OPTIMIZATION_MODEL, OPTIMIZATION_ENABLE_REASONING
from .context import json_compact_block, render_optimizer_decision
from .llm import call_json_llm
from ..optimization.strategy_document import StrategyDocument


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
)


def build_strategy_patch_validation_prompt(
    *,
    decision: dict[str, Any],
    parent_text: str,
    candidate_text: str,
    patches: list[dict[str, Any]],
    capability_manifest: dict[str, Any] | None = None,
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

The Cross-Match Decision has already selected one hypothesis.

Check only whether the candidate strategy patch is a clean implementation of
that hypothesis.

Validate:

1. Scope
Every modified paragraph must be necessary for the supplied hypothesis or for
keeping dependent strategy rules internally consistent.

2. Internal consistency
The complete candidate strategy must not contain contradictory thresholds,
production targets, priorities, technology requirements, attack conditions,
recovery conditions, or information requirements.

3. Preserved strengths
Unrelated supported strengths must remain intact.

4. Runtime boundary
The strategy must not require unavailable micro, runtime behavior, or controls.

Do not reject a candidate because you personally prefer another strategy.
Do not propose alternative strategy changes.
Do not generate replacement patches.

{RUNTIME_CONTRACT}

Capability summary:
{json.dumps(capability_manifest or {}, ensure_ascii=False, indent=2)}

Cross-match Decision:
{render_optimizer_decision(decision)}

Patches:
{json_compact_block(compact_patches)}

Parent strategy.md:
{parent_text}

Candidate strategy.md:
{candidate_text}

Return JSON only:
{{
  "valid": true,
  "errors": []
}}
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
    return list(dict.fromkeys(errors))


def validate_strategy_patch_semantics(
    *,
    decision: dict[str, Any],
    parent_text: str,
    candidate_text: str,
    patches: list[dict[str, Any]],
    capability_manifest: dict[str, Any] | None = None,
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
    reported = [
        str(item).strip()
        for item in (payload.get("errors") or [])
        if str(item).strip()
    ]
    valid = payload.get("valid")
    if valid is False or reported:
        errors.extend(reported or ["semantic validator rejected the patch"])
    elif valid is not True and not reported:
        errors.append("semantic validator must return valid=true or errors")
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
