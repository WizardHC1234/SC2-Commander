"""Read-only semantic audit for an EvolAgent optimization direction.

The CLI in this module is read-only and never performs evolution state
transitions.  Its normalized verdict schema is also reused by the in-process
package selector before candidate generation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from ..core.config import DEFAULT_ANALYSIS_MODEL
from ..core.llm import call_json_llm


AUDIT_VERDICTS = frozenset(
    {
        "approve_for_trial",
        "revise_before_trial",
        "reject_repeated_direction",
        "inspect_runtime",
    }
)
ROOT_CAUSES = frozenset(
    {
        "production_readiness",
        "attack_gate_timing",
        "engagement_position",
        "matchup_composition",
        "post_defense_conversion",
        "runtime_execution",
        "mixed",
    }
)


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return data


def _read_strategy(run_dir: Path, strategy_name: str) -> str:
    if not strategy_name:
        return ""
    path = run_dir / "strategies" / strategy_name / "strategy.md"
    return path.read_text(encoding="utf-8-sig") if path.is_file() else ""


def _history_rows(history: Any, *, exclude_candidate: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in history if isinstance(history, list) else []:
        if not isinstance(item, dict):
            continue
        if exclude_candidate and str(item.get("candidate") or "").strip() == exclude_candidate:
            continue
        rows.append(
            {
                key: item.get(key)
                for key in (
                    "experiment_id",
                    "mutation_parent",
                    "candidate",
                    "difficulty",
                    "decision",
                    "score_delta",
                    "implementation_verdict",
                    "hypothesis_verdict",
                    "mechanism_family",
                    "primary_change",
                    "mechanism_prediction",
                    "first_commitment_timing",
                    "failed_dependencies",
                    "salvageable_changes",
                    "lesson",
                    "selected_history_assessment",
                )
                if item.get(key) not in (None, "", [], {})
            }
        )
    return rows[-12:]


def _candidate_name(
    state: dict[str, Any],
    parent_name: str,
    explicit_candidate: str,
) -> str:
    if explicit_candidate:
        return explicit_candidate
    for item in reversed(state.get("experiment_history") or []):
        if not isinstance(item, dict):
            continue
        if str(item.get("mutation_parent") or item.get("parent") or "") != parent_name:
            continue
        candidate = str(item.get("candidate") or "").strip()
        if candidate:
            return candidate
    return ""


def build_audit_context(
    *,
    run_dir: Path,
    analysis_dir: Path,
    candidate_name: str = "",
) -> dict[str, Any]:
    state = _read_json(run_dir / "state.json")
    analysis = _read_json(analysis_dir / "analysis.json")
    discovery_path = analysis_dir / "cross_match_discovery.json"
    discovery = _read_json(discovery_path) if discovery_path.is_file() else {}
    parent_name = str(analysis.get("strategy_name") or state.get("champion") or "").strip()
    resolved_candidate = _candidate_name(state, parent_name, candidate_name)
    analysis_fields = (
        "record_mix",
        "strengths_to_preserve",
        "outcome_contrast",
        "priority_problem",
        "hypothesis",
        "failure_mode_analysis",
        "mechanism_prediction",
        "selected_package_id",
        "selected_timing_budget",
        "selected_package_budget",
        "candidate_packages",
        "package_budget_reports",
        "selected_engagement_assessment",
        "selected_history_assessment",
        "action_reason",
        "expected_effect",
        "main_risk",
        "evidence_limits",
    )
    discovery_fields = (
        "strengths",
        "weaknesses",
        "opponent_pressure_patterns",
        "engagement_initiative_patterns",
        "defense_counterattack_patterns",
        "matchup_patterns",
        "unknowns",
    )
    return {
        "run_status": {
            "style": state.get("style"),
            "champion": state.get("champion"),
            "difficulty": state.get("difficulty"),
            "generation": state.get("generation"),
        },
        "parent_strategy_name": parent_name,
        "parent_strategy_md": _read_strategy(run_dir, parent_name),
        "candidate_strategy_name": resolved_candidate,
        "candidate_strategy_md": _read_strategy(run_dir, resolved_candidate),
        "analysis": {
            key: analysis.get(key)
            for key in analysis_fields
            if analysis.get(key) not in (None, "", [], {})
        },
        "cross_match_evidence": {
            key: discovery.get(key)
            for key in discovery_fields
            if discovery.get(key) not in (None, "", [], {})
        },
        "experiment_history": _history_rows(
            state.get("experiment_history"),
            exclude_candidate=resolved_candidate,
        ),
    }


def build_direction_audit_prompt(context: dict[str, Any]) -> str:
    return f"""You are an independent, read-only reviewer of an SC2 language-strategy optimization direction. Do not generate a replacement strategy and do not assume that a candidate must be accepted. Judge the proposed change from observed trajectory behavior, timing, production realization, engagement conditions, post-contact continuation, and experiment history rather than mechanism names or wording.

First identify the earliest supported cause of the observed losses. Distinguish production_readiness, attack_gate_timing, engagement_position, matchup_composition, post_defense_conversion, runtime_execution, and mixed causes. If the planned package was feasible on paper but the required units repeatedly did not exist by the observed contact window, do not treat a new holding instruction as a production fix. If useful combat power already existed but the strategy kept waiting for a larger gate, distinguish that from insufficient production. If a force won a defense but did not convert the remaining advantage, treat that as post-defense conversion rather than another reason to raise the opening gate.

Compare the candidate semantically with rejected history. Different buildings, units, thresholds, or labels are still the same failed direction when they preserve the same observable trajectory, such as remaining passive until the full gate while adding another pre-gate defensive requirement. A material repair must change the failed dependency itself and show how the changed dependency alters the observed contact window. Judge semantic equivalence with reasoning, not character matching.

Audit three links separately: pre_contact must state whether production or timing changes make a useful force available by the evidence-supported opponent window; engagement must state whether force matchup, target, or contact position changes; post_contact must state whether a held defense or surviving advantage is converted into continued pressure. The direction need not change all three links, but an approval requires a material change to the supported root cause, preserved proven gains, and no repetition of a contradicted intervention. A statement that merely holds existing units somewhere is not evidence that those units will be produced earlier.

Respect the existing Commander capability boundary. Do not require new wake events, custom fixed squads, exact per-unit assignments, unavailable persistent enemy memory, or map-specific scripted paths. This audit must not edit any file or evolution state.

Audit context:
{json.dumps(context, ensure_ascii=False, indent=2)}

Return one JSON object only:
{{
  "verdict":"approve_for_trial|revise_before_trial|reject_repeated_direction|inspect_runtime",
  "root_cause":{{"primary":"production_readiness|attack_gate_timing|engagement_position|matchup_composition|post_defense_conversion|runtime_execution|mixed","reason":"earliest supported causal explanation","evidence_refs":["Game N @ Ts"]}},
  "behavioral_change":{{
    "pre_contact":{{"changes_observable_trajectory":true,"description":"what becomes observably different before contact","deadline_supported":true}},
    "engagement":{{"changes_observable_trajectory":false,"description":"what changes in matchup, target, or position"}},
    "post_contact":{{"changes_observable_trajectory":false,"description":"what changes after a held defense or surviving engagement"}}
  }},
  "history_comparison":{{"semantic_family":"behavior-level family","equivalent_rejected_experiment_ids":["exact experiment id"],"why_new_or_repeated":"semantic comparison"}},
  "preserved_gain_experiment_ids":["exact accepted experiment id"],
  "blocking_reasons":["reason the direction should not enter a trial"],
  "recommended_revision":{{"change":"smallest evidence-supported revision, or empty when approved","preserve":["proven strength"],"avoid":["failed behavior to avoid"]}},
  "confidence":"high|medium|low"
}}"""


def _strings(value: Any, *, limit: int = 12) -> list[str]:
    return [str(item).strip() for item in (value or []) if str(item).strip()][:limit]


def normalize_audit_result(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("direction audit returned no JSON object")
    verdict = str(raw.get("verdict") or "").strip().lower()
    if verdict not in AUDIT_VERDICTS:
        raise ValueError("invalid direction-audit verdict")
    root_raw = raw.get("root_cause") if isinstance(raw.get("root_cause"), dict) else {}
    primary = str(root_raw.get("primary") or "").strip().lower()
    if primary not in ROOT_CAUSES:
        raise ValueError("invalid direction-audit root cause")
    behavior_raw = raw.get("behavioral_change") if isinstance(raw.get("behavioral_change"), dict) else {}
    behavior: dict[str, dict[str, Any]] = {}
    for stage in ("pre_contact", "engagement", "post_contact"):
        item = behavior_raw.get(stage) if isinstance(behavior_raw.get(stage), dict) else {}
        changed = item.get("changes_observable_trajectory")
        if not isinstance(changed, bool):
            raise ValueError(f"behavioral_change.{stage}.changes_observable_trajectory must be boolean")
        behavior[stage] = {
            "changes_observable_trajectory": changed,
            "description": str(item.get("description") or "").strip(),
        }
        if stage == "pre_contact":
            deadline_supported = item.get("deadline_supported")
            if not isinstance(deadline_supported, bool):
                raise ValueError("behavioral_change.pre_contact.deadline_supported must be boolean")
            behavior[stage]["deadline_supported"] = deadline_supported
    history_raw = raw.get("history_comparison") if isinstance(raw.get("history_comparison"), dict) else {}
    revision_raw = raw.get("recommended_revision") if isinstance(raw.get("recommended_revision"), dict) else {}
    return {
        "verdict": verdict,
        "root_cause": {
            "primary": primary,
            "reason": str(root_raw.get("reason") or "").strip(),
            "evidence_refs": _strings(root_raw.get("evidence_refs")),
        },
        "behavioral_change": behavior,
        "history_comparison": {
            "semantic_family": str(history_raw.get("semantic_family") or "").strip(),
            "equivalent_rejected_experiment_ids": _strings(history_raw.get("equivalent_rejected_experiment_ids")),
            "why_new_or_repeated": str(history_raw.get("why_new_or_repeated") or "").strip(),
        },
        "preserved_gain_experiment_ids": _strings(raw.get("preserved_gain_experiment_ids")),
        "blocking_reasons": _strings(raw.get("blocking_reasons")),
        "recommended_revision": {
            "change": str(revision_raw.get("change") or "").strip(),
            "preserve": _strings(revision_raw.get("preserve")),
            "avoid": _strings(revision_raw.get("avoid")),
        },
        "confidence": str(raw.get("confidence") or "low").strip().lower(),
    }


def audit_direction(
    *,
    context: dict[str, Any],
    model: str,
    is_reasoning: bool = True,
    llm_call: Callable[..., Any] = call_json_llm,
) -> dict[str, Any]:
    raw = llm_call(
        build_direction_audit_prompt(context),
        model=model,
        is_reasoning=is_reasoning,
        system="You are an independent SC2 optimization-direction auditor. Return valid JSON only.",
    )
    return normalize_audit_result(raw)


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only semantic audit of one EvolAgent optimization direction")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--analysis-dir", required=True, type=Path)
    parser.add_argument("--candidate", default="")
    parser.add_argument("--model", default=DEFAULT_ANALYSIS_MODEL)
    parser.add_argument("--no-reasoning", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help="Optional report path; omitted means stdout only")
    args = parser.parse_args()
    context = build_audit_context(run_dir=args.run_dir, analysis_dir=args.analysis_dir, candidate_name=args.candidate)
    result = audit_direction(context=context, model=args.model, is_reasoning=not args.no_reasoning)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
