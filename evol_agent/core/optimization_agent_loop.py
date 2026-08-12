from __future__ import annotations

from typing import Any

from .config import (
    DEFAULT_OPTIMIZATION_MODEL,
    MAX_VALIDATION_RETRIES,
    OPTIMIZATION_ENABLE_REASONING,
)
from .llm import call_json_llm
from .simple_prompts import build_candidate_prompt
from .types import BattleAnalysis, EvolImprovement, ToolObservation, ValidationResult
from ..validation import validate_improvement


def run_optimization_agent_loop(
    *,
    strategy_name: str,
    race: str,
    battle_analysis: BattleAnalysis,
    skill_texts: dict[str, str],
    initial_tool_observations: list[ToolObservation],
    knowledge_mode: str = "enabled",
    model: str = "",
    prefix: str = "  ",
) -> tuple[
    ValidationResult,
    EvolImprovement | None,
    list[ToolObservation],
    list[str],
    list[dict[str, Any]],
]:
    """Generate one strategy candidate and retry only basic file validation.

    Strategic quality is intentionally not validated here. It is measured by
    playing the candidate; this loop checks only that Commander can load and
    execute the strategy document.
    """
    model = str(model or "").strip() or DEFAULT_OPTIMIZATION_MODEL
    observations = list(initial_tool_observations)
    validation_errors: list[str] = []
    events: list[dict[str, Any]] = []
    candidate: dict[str, Any] | None = None
    last_improvement: EvolImprovement | None = None

    print(
        f"{prefix}OptimizationAgent: generating one candidate for "
        f"{race}/{strategy_name}",
        flush=True,
    )
    for attempt in range(1, MAX_VALIDATION_RETRIES + 2):
        action = call_json_llm(
            build_candidate_prompt(
                strategy_name=strategy_name,
                race=race,
                battle_analysis=battle_analysis,
                skill_texts=skill_texts,
                tool_observations=observations,
                validation_errors=validation_errors,
                candidate=candidate,
                knowledge_mode=knowledge_mode,
            ),
            model=model,
            is_reasoning=OPTIMIZATION_ENABLE_REASONING,
        )
        if not isinstance(action, dict):
            error = "OptimizationAgent returned no JSON object"
            validation_errors.append(error)
            events.append({"attempt": attempt, "action": "invalid", "error": error})
            continue

        name = str(action.get("action") or "").strip()
        files = action.get("files") if isinstance(action.get("files"), dict) else {}
        rationale = (
            action.get("rationale")
            if isinstance(action.get("rationale"), dict)
            else action.get("analysis")
            if isinstance(action.get("analysis"), dict)
            else {}
        )
        normalized = {
            "action": name or "draft_candidate",
            "rationale": dict(rationale),
            "files": {str(key): str(value) for key, value in files.items()},
        }
        candidate = normalized
        last_improvement = EvolImprovement(
            analysis=normalized["rationale"],
            files=normalized["files"],
            raw=normalized,
        )

        result = validate_improvement(files=last_improvement.files, race=race)
        events.append(
            {
                "attempt": attempt,
                "action": name or "draft_candidate",
                "valid": result.ok,
                "error": result.error,
            }
        )
        if result.ok:
            last_improvement.files = result.files or last_improvement.files
            last_improvement.raw["files"] = dict(last_improvement.files)
            print(
                f"{prefix}OptimizationAgent: candidate passed basic validation",
                flush=True,
            )
            return result, last_improvement, observations, validation_errors, events

        validation_errors.append(result.error)
        if attempt <= MAX_VALIDATION_RETRIES:
            print(
                f"{prefix}OptimizationAgent: basic validation failed; "
                f"retrying ({attempt}/{MAX_VALIDATION_RETRIES}): {result.error}",
                flush=True,
            )

    error = validation_errors[-1] if validation_errors else "OptimizationAgent exhausted"
    return (
        ValidationResult(ok=False, error=error),
        last_improvement,
        observations,
        validation_errors,
        events,
    )
