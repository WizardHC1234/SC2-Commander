from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core.config import OPTIMIZATION_LOG_DIR
from ..core.run_recorder import get_run_events
from ..core.types import BattleAnalysis, EvolImprovement, GameDigest, ToolObservation


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path


def _digest_payload(game_digests: list[GameDigest]) -> list[dict[str, Any]]:
    return [d.raw or d.__dict__ for d in game_digests]


def _analysis_payload(battle_analysis: BattleAnalysis) -> dict[str, Any]:
    return battle_analysis.raw or battle_analysis.__dict__


def _tool_payload(tool_observations: list[ToolObservation]) -> list[dict[str, Any]]:
    return [obs.__dict__ for obs in tool_observations]


def _improvement_payload(improvement: EvolImprovement | None, changes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "improvement": improvement.raw if improvement else None,
        "changes": changes,
    }


def _completed_outputs(
    *,
    game_digests: list[GameDigest],
    battle_analysis: BattleAnalysis,
    tool_observations: list[ToolObservation],
    improvement: EvolImprovement | None,
    changes: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "game_digests": _digest_payload(game_digests),
        "battle_analysis": _analysis_payload(battle_analysis),
        "tool_observations": _tool_payload(tool_observations),
        "improvement": improvement.raw if improvement else None,
        "changes": changes,
    }


def _context_payload(
    *,
    strategy_name: str,
    game_digests: list[GameDigest],
    battle_analysis: BattleAnalysis,
    tool_observations: list[ToolObservation],
    improvement: EvolImprovement | None,
    changes: list[dict[str, Any]],
    run_context: dict[str, Any] | None,
) -> dict[str, Any]:
    context = dict(run_context or {})
    context.setdefault("strategy_name", strategy_name)
    context["completed_outputs"] = _completed_outputs(
        game_digests=game_digests,
        battle_analysis=battle_analysis,
        tool_observations=tool_observations,
        improvement=improvement,
        changes=changes,
    )
    return context


def save_evol_logs(
    *,
    strategy_name: str,
    game_digests: list[GameDigest],
    battle_analysis: BattleAnalysis,
    tool_observations: list[ToolObservation],
    improvement: EvolImprovement | None,
    changes: list[dict[str, Any]],
    run_context: dict[str, Any] | None = None,
    run_dir: Path | None = None,
) -> dict[str, Path]:
    if run_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = OPTIMIZATION_LOG_DIR / strategy_name / ts
    else:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    context = _context_payload(
        strategy_name=strategy_name,
        game_digests=game_digests,
        battle_analysis=battle_analysis,
        tool_observations=tool_observations,
        improvement=improvement,
        changes=changes,
        run_context=run_context,
    )
    analysis_pipeline = context.get("analysis_pipeline")
    single_game_analyses = (
        analysis_pipeline.get("single_game_analyses", [])
        if isinstance(analysis_pipeline, dict)
        else []
    )
    paths = {
        "digests": _write_json(run_dir / "digests.json", _digest_payload(game_digests)),
        "single_game_analyses": _write_json(
            run_dir / "single_game_analyses.json",
            single_game_analyses,
        ),
        "analysis": _write_json(run_dir / "analysis.json", _analysis_payload(battle_analysis)),
        "knowledge_trace": _write_json(
            run_dir / "knowledge_trace.json",
            analysis_pipeline.get("knowledge_trace", {})
            if isinstance(analysis_pipeline, dict)
            else {},
        ),
        "tools": _write_json(run_dir / "tool_observations.json", _tool_payload(tool_observations)),
        "improvement": _write_json(run_dir / "improvement.json", _improvement_payload(improvement, changes)),
        "context": _write_json(run_dir / "context.json", context),
        "run_record": _write_json(
            run_dir / "run_record.json",
            {
                "schema": "evol_agent_run_record.v1",
                "created": datetime.now().isoformat(),
                "strategy_name": strategy_name,
                "run_dir": str(run_dir),
                "conversation_events": get_run_events(),
                "context": context,
                "completed_outputs": context["completed_outputs"],
            },
        ),
    }
    return paths