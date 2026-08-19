from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

from .analysis_agent_loop import run_analysis_agent_loop
from .capabilities import build_executor_capability_manifest
from .checkpoint import (
    create_checkpoint,
    load_checkpoint,
    normalize_record_files,
    stage_reached,
    validate_analysis_seed_checkpoint,
    validate_checkpoint_fingerprint,
)
from .loop_helpers import fallback_analysis
from .config import KNOWLEDGE_MODES, resolve_model
from .optimization_agent_loop import run_optimization_agent_loop
from .run_recorder import reset_run_events
from .types import EvolRunRequest, EvolRunResult, ValidationResult
from ..analysis.record_reader import find_record_jsons, group_records_by_strategy, load_skill_texts
from ..optimization.logs import save_evol_logs
from ..optimization.snapshot import output_dir_for_strategy, save_snapshot


def _record_context(records: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "file": record.file,
            "result": record.result,
            "duration": record.duration,
            "timeline": record.timeline,
            "meta": record.meta,
        }
        for record in records
    ]


def _validation_context(validation: ValidationResult | None, validation_errors: list[str]) -> dict[str, Any]:
    if validation is None:
        return {"validation_errors": validation_errors}
    return {
        "ok": validation.ok,
        "error": validation.error,
        "validation_errors": validation_errors,
    }


def _base_run_context(
    *,
    request: EvolRunRequest,
    strategy_name: str,
    race: str,
    skill_dir: Path,
    skill_texts: dict[str, str],
    records: list[Any],
    model: str,
) -> dict[str, Any]:
    return {
        "request": {
            "record_paths": [str(path) for path in request.record_paths],
            "batch_dir": str(request.batch_dir) if request.batch_dir else "",
            "strategy_name": request.strategy_name,
            "race": request.race,
            "skill_dir": str(request.skill_dir) if request.skill_dir else "",
            "output_dir": str(request.output_dir) if request.output_dir else "",
            "model": model,
            "knowledge_mode": request.knowledge_mode,
            "dry_run": request.dry_run,
            "resume_dir": str(request.resume_dir) if request.resume_dir else "",
            "analysis_seed_dir": (
                str(request.analysis_seed_dir) if request.analysis_seed_dir else ""
            ),
            "prior_experiences": list(request.prior_experiences),
        },
        "selected_group": {
            "strategy_name": strategy_name,
            "race": race,
            "skill_dir": str(skill_dir),
            "records": _record_context(records),
        },
        "skill_context": skill_texts,
    }


class EvolAgent:
    def __init__(self, *, model: str = "") -> None:
        # Empty means "use per-role defaults from core.config".
        self.model = str(model or "").strip()

    def run(self, request: EvolRunRequest) -> EvolRunResult:
        checkpoint = None
        if request.resume_dir:
            try:
                checkpoint = load_checkpoint(request.resume_dir)
            except (OSError, ValueError) as exc:
                return EvolRunResult(ok=False, message=f"failed to load checkpoint: {exc}")
            meta = checkpoint.meta
            if request.strategy_name and request.strategy_name != str(meta.get("strategy_name") or ""):
                return EvolRunResult(
                    ok=False,
                    message=(
                        "strategy mismatch with checkpoint: "
                        f"cli={request.strategy_name} checkpoint={meta.get('strategy_name')}"
                    ),
                )
            # Resume identity comes from the checkpoint fingerprint.
            request.strategy_name = str(meta.get("strategy_name") or request.strategy_name)
            request.race = str(meta.get("race") or request.race)
            request.knowledge_mode = str(meta.get("knowledge_mode") or request.knowledge_mode)
            print(
                f"  EvolAgent resume: stage={checkpoint.stage} dir={checkpoint.run_dir}",
                flush=True,
            )

        if request.knowledge_mode not in KNOWLEDGE_MODES:
            return EvolRunResult(
                ok=False,
                message=f"knowledge_mode must be one of: {', '.join(KNOWLEDGE_MODES)}",
            )
        record_paths = list(request.record_paths)
        if request.batch_dir:
            record_paths.extend(find_record_jsons(request.batch_dir))
        record_paths = list(dict.fromkeys(record_paths))
        if not record_paths and checkpoint is not None:
            record_paths = [
                Path(path) for path in (checkpoint.meta.get("record_files") or []) if str(path).strip()
            ]
        if not record_paths:
            return EvolRunResult(ok=False, message="no record paths provided")

        grouped = group_records_by_strategy(record_paths)
        if not grouped:
            return EvolRunResult(ok=False, message="no records could be processed")

        if request.strategy_name:
            key = (request.race.lower(), request.strategy_name)
            if key not in grouped:
                return EvolRunResult(ok=False, message=f"strategy not found in records: {key}")
            group = grouped[key]
        elif len(grouped) == 1:
            group = next(iter(grouped.values()))
        else:
            names = ", ".join(f"{group_race}/{name}" for group_race, name in grouped)
            return EvolRunResult(ok=False, message=f"multiple strategies found; select one with --strategy: {names}")

        return self._run_group(request, group, checkpoint=checkpoint)

    def _run_group(
        self,
        request: EvolRunRequest,
        group: dict,
        *,
        checkpoint=None,
    ) -> EvolRunResult:
        strategy_name = group["strategy_name"]
        race = group["race"]
        skill_dir = Path(request.skill_dir) if request.skill_dir else group["skill_dir"]
        skill_texts = load_skill_texts(skill_dir)
        if not skill_texts.get("strategy.md", "").strip():
            return EvolRunResult(
                ok=False,
                message=f"strategy.md not found in {skill_dir}",
                strategy_name=strategy_name,
                race=race,
            )
        if request.output_dir and request.output_dir.exists() and any(
            request.output_dir.iterdir()
        ):
            return EvolRunResult(
                ok=False,
                message=(
                    "output candidate directory is immutable and already contains files: "
                    f"{request.output_dir}"
                ),
                strategy_name=strategy_name,
                race=race,
            )

        records = group["records"]
        record_files = normalize_record_files(records)
        reset_run_events()
        model_override = str(request.model or self.model or "").strip()
        analysis_model = resolve_model(model_override, role="analysis")
        optimization_model = resolve_model(model_override, role="optimization")
        models = {
            "override": model_override,
            "analysis": analysis_model,
            "optimization": optimization_model,
            "knowledge": "deterministic",
        }
        capability_manifest = build_executor_capability_manifest(race)

        analysis_seed = None
        if request.analysis_seed_dir:
            try:
                loaded_seed = load_checkpoint(request.analysis_seed_dir)
                validate_analysis_seed_checkpoint(
                    loaded_seed,
                    strategy_name=strategy_name,
                    race=race,
                    knowledge_mode=request.knowledge_mode,
                    record_files=record_files,
                    analysis_model=analysis_model,
                )
                if checkpoint is None or loaded_seed.run_dir != checkpoint.run_dir:
                    analysis_seed = loaded_seed
            except (OSError, ValueError) as exc:
                print(
                    f"  EvolAgent: analysis seed ignored: {exc}",
                    flush=True,
                )

        if checkpoint is not None:
            try:
                validate_checkpoint_fingerprint(
                    checkpoint,
                    strategy_name=strategy_name,
                    race=race,
                    knowledge_mode=request.knowledge_mode,
                    record_files=record_files,
                )
            except ValueError as exc:
                return EvolRunResult(
                    ok=False,
                    message=f"checkpoint fingerprint mismatch: {exc}",
                    strategy_name=strategy_name,
                    race=race,
                )
        else:
            checkpoint = create_checkpoint(
                strategy_name=strategy_name,
                race=race,
                knowledge_mode=request.knowledge_mode,
                models=models,
                record_files=record_files,
            )
            print(f"  EvolAgent checkpoint: {checkpoint.run_dir}", flush=True)

        run_context = _base_run_context(
            request=request,
            strategy_name=strategy_name,
            race=race,
            skill_dir=skill_dir,
            skill_texts=skill_texts,
            records=records,
            model=analysis_model,
        )
        run_context["request"]["models"] = models
        run_context["executor_capability_manifest"] = capability_manifest
        run_context["checkpoint"] = {
            "run_dir": str(checkpoint.run_dir),
            "stage": checkpoint.stage,
        }
        if analysis_seed is not None:
            run_context["analysis_seed"] = {
                "run_dir": str(analysis_seed.run_dir),
                "stage": analysis_seed.stage,
                "record_count": len(analysis_seed.meta.get("record_files") or []),
            }

        if stage_reached(checkpoint.stage, "candidate"):
            message = f"checkpoint already produced a candidate at {checkpoint.run_dir}"
            print(f"  EvolAgent: {message}", flush=True)
            return EvolRunResult(
                ok=True,
                message=message,
                checkpoint_dir=checkpoint.run_dir,
                strategy_name=strategy_name,
                race=race,
            )

        def _save_logs(**kwargs: Any) -> dict[str, Path]:
            return save_evol_logs(run_dir=checkpoint.run_dir, **kwargs)

        if stage_reached(checkpoint.stage, "analysis_complete") and not request.dry_run:
            print(
                f"  EvolAgent: resume skip analysis (stage={checkpoint.stage}); "
                "continuing from optimization",
                flush=True,
            )
            (
                battle_analysis,
                analysis_observations,
                knowledge_trace,
                analysis_events,
                analysis_errors,
            ) = checkpoint.load_analysis_complete()
            digests, single_game_analyses, _completed, match_events, match_errors = (
                checkpoint.load_match_summaries()
                if (checkpoint.run_dir / "match_summaries.json").is_file()
                else ([], [], 0, [], [])
            )
            run_context["analysis_pipeline"] = {
                "events": [*match_events, *analysis_events],
                "single_game_analyses": [
                    analysis.raw or analysis.__dict__ for analysis in single_game_analyses
                ],
                "aggregate_battle_analysis": battle_analysis.raw or battle_analysis.__dict__,
                "tool_observations": [obs.__dict__ for obs in analysis_observations],
                "knowledge_trace": knowledge_trace,
                "errors": [*match_errors, *analysis_errors],
                "resumed_from": "analysis_complete",
            }
        else:
            analysis_prior_experiences = list(request.prior_experiences)
            if analysis_seed is not None and stage_reached(
                analysis_seed.stage, "analysis_complete"
            ):
                seeded_analysis, _observations, _trace, _events, _errors = (
                    analysis_seed.load_analysis_complete()
                )
                analysis_prior_experiences.append(
                    {
                        "kind": "parent_analysis_seed",
                        "source_checkpoint": str(analysis_seed.run_dir),
                        "source_record_count": len(
                            analysis_seed.meta.get("record_files") or []
                        ),
                        "current_record_count": len(records),
                        "analysis": seeded_analysis.raw
                        or seeded_analysis.__dict__,
                    }
                )
            print(
                f"  EvolAgent running match summaries and one batch analysis "
                f"for {len(records)} records: {race}/{strategy_name} "
                f"(analysis={analysis_model}, knowledge=deterministic, "
                f"stage={checkpoint.stage})",
                flush=True,
            )
            analysis_result = run_analysis_agent_loop(
                strategy_name=strategy_name,
                race=race,
                records=records,
                skill_texts=skill_texts,
                model=analysis_model,
                knowledge_mode=request.knowledge_mode,
                prefix="    ",
                checkpoint=checkpoint,
                summary_seed_checkpoint=analysis_seed,
                prior_experiences=analysis_prior_experiences,
                capability_manifest=capability_manifest,
            )
            digests = analysis_result.game_digests
            battle_analysis = analysis_result.battle_analysis
            analysis_observations = analysis_result.tool_observations
            analysis_errors = analysis_result.errors
            run_context["analysis_pipeline"] = {
                "events": analysis_result.events,
                "single_game_analyses": [
                    analysis.raw or analysis.__dict__
                    for analysis in analysis_result.single_game_analyses
                ],
                "aggregate_battle_analysis": (
                    battle_analysis.raw or battle_analysis.__dict__
                    if battle_analysis is not None
                    else None
                ),
                "tool_observations": [obs.__dict__ for obs in analysis_observations],
                "knowledge_trace": analysis_result.knowledge_trace,
                "errors": analysis_errors,
            }

            if not analysis_result.completed or battle_analysis is None:
                detail = (
                    analysis_errors[-1]
                    if analysis_errors
                    else "cross-match BattleAnalysis was not produced"
                )
                message = f"analysis pipeline stopped: {detail}"
                failed_analysis = battle_analysis or fallback_analysis(
                    strategy_name=strategy_name,
                    race=race,
                    records=records,
                    reason=message,
                )
                _save_logs(
                    strategy_name=strategy_name,
                    game_digests=digests,
                    battle_analysis=failed_analysis,
                    tool_observations=analysis_observations,
                    improvement=None,
                    changes=[],
                    run_context={**run_context, "output": {"ok": False, "message": message}},
                )
                return EvolRunResult(
                    ok=False,
                    message=message,
                    checkpoint_dir=checkpoint.run_dir,
                    strategy_name=strategy_name,
                    race=race,
                    game_digests=digests,
                    battle_analysis=failed_analysis,
                    tool_observations=analysis_observations,
                )

        if request.dry_run:
            _save_logs(
                strategy_name=strategy_name,
                game_digests=digests,
                battle_analysis=battle_analysis,
                tool_observations=analysis_observations,
                improvement=None,
                changes=[],
                run_context={
                    **run_context,
                    "output": {"ok": True, "dry_run": True, "message": "dry run complete"},
                },
            )
            return EvolRunResult(
                ok=True,
                message="dry run complete",
                checkpoint_dir=checkpoint.run_dir,
                strategy_name=strategy_name,
                race=race,
                game_digests=digests,
                battle_analysis=battle_analysis,
                tool_observations=analysis_observations,
            )

        decision_action = str(
            battle_analysis.raw.get("next_action") or "propose_strategy_patch"
        )
        action_reason = str(battle_analysis.raw.get("action_reason") or "")
        if decision_action != "propose_strategy_patch":
            message = f"EvolAgent selected {decision_action}"
            if action_reason:
                message += f": {action_reason}"
            _save_logs(
                strategy_name=strategy_name,
                game_digests=digests,
                battle_analysis=battle_analysis,
                tool_observations=analysis_observations,
                improvement=None,
                changes=[],
                run_context={
                    **run_context,
                    "output": {
                        "ok": True,
                        "message": message,
                        "decision_action": decision_action,
                        "action_reason": action_reason,
                    },
                },
            )
            return EvolRunResult(
                ok=True,
                message=message,
                checkpoint_dir=checkpoint.run_dir,
                decision_action=decision_action,
                action_reason=action_reason,
                strategy_name=strategy_name,
                race=race,
                game_digests=digests,
                battle_analysis=battle_analysis,
                tool_observations=analysis_observations,
            )

        print(
            f"  EvolAgent running optimization phase for {race}/{strategy_name} "
            f"(model={optimization_model})",
            flush=True,
        )
        validation, improvement, observations, validation_errors, optimization_events = (
            run_optimization_agent_loop(
                strategy_name=strategy_name,
                race=race,
                battle_analysis=battle_analysis,
                skill_texts=skill_texts,
                initial_tool_observations=analysis_observations,
                knowledge_mode=request.knowledge_mode,
                model=optimization_model,
                prefix="    ",
                capability_manifest=capability_manifest,
            )
        )
        run_context["optimization_agent_loop"] = {
            "events": optimization_events,
            "tool_observations": [obs.__dict__ for obs in observations],
            "validation": _validation_context(validation, validation_errors),
            "improvement": improvement.raw if improvement else None,
        }

        if not validation.ok or improvement is None:
            _save_logs(
                strategy_name=strategy_name,
                game_digests=digests,
                battle_analysis=battle_analysis,
                tool_observations=observations,
                improvement=improvement,
                changes=[],
                run_context={**run_context, "output": {"ok": False, "message": validation.error}},
            )
            return EvolRunResult(
                ok=False,
                message=validation.error,
                checkpoint_dir=checkpoint.run_dir,
                strategy_name=strategy_name,
                race=race,
                game_digests=digests,
                battle_analysis=battle_analysis,
                improvement=improvement,
                tool_observations=observations,
            )

        out_dir = request.output_dir or output_dir_for_strategy(strategy_name, race)
        candidate_hash = hashlib.sha256(
            improvement.files["strategy.md"].encode("utf-8")
        ).hexdigest()[:16]
        changes = save_snapshot(
            source_dir=skill_dir,
            files=improvement.files,
            output_dir=out_dir,
            source_info={
                "parent": strategy_name,
                "created": datetime.now().isoformat(),
                "agent": "EvolAgent",
                "records": len(records),
                "record_mix": battle_analysis.record_mix,
                "knowledge_mode": request.knowledge_mode,
                "main_lesson": improvement.analysis.get("primary_change", ""),
            },
            race=race,
        )
        checkpoint.mark_candidate_complete()
        _save_logs(
            strategy_name=strategy_name,
            game_digests=digests,
            battle_analysis=battle_analysis,
            tool_observations=observations,
            improvement=improvement,
            changes=changes,
            run_context={
                **run_context,
                "output": {
                    "ok": True,
                    "message": "OK",
                    "output_dir": str(out_dir),
                    "candidate_name": out_dir.name,
                    "candidate_hash": candidate_hash,
                    "parent_strategy": strategy_name,
                    "changes": changes,
                },
            },
        )
        return EvolRunResult(
            ok=True,
            message="OK",
            checkpoint_dir=checkpoint.run_dir,
            strategy_name=strategy_name,
            race=race,
            output_dir=out_dir,
            candidate_hash=candidate_hash,
            game_digests=digests,
            battle_analysis=battle_analysis,
            improvement=improvement,
            changes=changes,
            tool_observations=observations,
        )
