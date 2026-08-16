from __future__ import annotations

from typing import Any


def critique_candidate_contract(
    rationale: dict[str, Any],
    *,
    capability_manifest: dict[str, Any],
    selected_plan: dict[str, Any] | None = None,
) -> list[str]:
    """Return deterministic experiment-contract errors before a candidate is saved."""
    errors: list[str] = []
    hypothesis = str(rationale.get("hypothesis") or "").strip()
    primary_lever = str(rationale.get("primary_lever") or "").strip()
    predictions = [str(item).strip() for item in rationale.get("predictions") or [] if str(item).strip()]
    disproof = [
        str(item).strip()
        for item in rationale.get("disproof_conditions") or []
        if str(item).strip()
    ]
    mapping = rationale.get("capability_mapping")
    if not hypothesis:
        errors.append("candidate rationale.hypothesis is required")
    if not primary_lever:
        errors.append("candidate rationale.primary_lever is required")
    if not predictions:
        errors.append("candidate rationale.predictions must contain observable predictions")
    if not disproof:
        errors.append("candidate rationale.disproof_conditions must be non-empty")
    if not isinstance(mapping, dict):
        errors.append("candidate rationale.capability_mapping must be an object")
        return errors

    known_actions = set(
        capability_manifest.get("macro_contract", {}).get("available_actions") or []
    )
    requested_actions = [
        str(item).strip()
        for item in mapping.get("macro_actions") or []
        if str(item).strip()
    ]
    changed_actions = [
        str(item).strip()
        for item in mapping.get("changed_macro_actions") or []
        if str(item).strip()
    ]
    unknown = sorted(set(requested_actions) - known_actions)
    if unknown:
        errors.append(f"capability_mapping contains unknown macro actions: {', '.join(unknown)}")
    unknown_changed = sorted(set(changed_actions) - known_actions)
    if unknown_changed:
        errors.append(
            "capability_mapping contains unknown changed macro actions: "
            + ", ".join(unknown_changed)
        )
    missing_from_complete = sorted(set(changed_actions) - set(requested_actions))
    if missing_from_complete:
        errors.append(
            "changed_macro_actions must also appear in macro_actions: "
            + ", ".join(missing_from_complete)
        )
    unsupported = [
        str(item).strip()
        for item in mapping.get("unsupported_dependencies") or []
        if str(item).strip()
    ]
    if unsupported:
        errors.append(
            "candidate depends on unsupported execution behavior: " + "; ".join(unsupported)
        )
    if isinstance(selected_plan, dict):
        selected_plan_id = str(selected_plan.get("id") or "").strip()
        change_plan_ids = {
            str(item.get("source_plan_id") or "").strip()
            for item in rationale.get("selected_changes") or []
            if isinstance(item, dict) and str(item.get("source_plan_id") or "").strip()
        }
        if selected_plan_id and change_plan_ids != {selected_plan_id}:
            errors.append(
                "all selected_changes must reference only the selected plan "
                f"{selected_plan_id}"
            )
        planned_lever = str(selected_plan.get("primary_lever") or "").strip().lower()
        if planned_lever and planned_lever != "other" and primary_lever.lower() != planned_lever:
            errors.append(
                f"candidate primary_lever={primary_lever!r} does not match selected plan "
                f"primary_lever={planned_lever!r}"
            )
        plan_mapping = selected_plan.get("capability_mapping")
        if changed_actions and isinstance(plan_mapping, dict):
            planned_actions = {
                str(item).strip()
                for item in plan_mapping.get("macro_actions") or []
                if str(item).strip()
            }
            extra_changed_actions = sorted(set(changed_actions) - planned_actions)
            if planned_actions and extra_changed_actions:
                errors.append(
                    "candidate changes macro actions outside the selected plan: "
                    + ", ".join(extra_changed_actions)
                )

        # macro_actions is the complete executable strategy; the selected plan
        # is only the experimental delta. Existing foundations such as workers,
        # gas, supply, expansion, and production are intentionally not compared
        # with that delta. When present, changed_macro_actions carries the
        # narrower set that is safe to compare above.
    return errors


__all__ = ["critique_candidate_contract"]
