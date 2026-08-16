from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from functools import partial
from typing import Any

from .checkpoint import EvolCheckpoint, stage_reached
from .config import (
    ANALYSIS_ENABLE_REASONING,
    DEFAULT_ANALYSIS_MODEL,
    MAX_CONCURRENT_MATCH_SUBAGENTS,
    MAX_KNOWLEDGE_QUERIES,
    MIN_KNOWLEDGE_QUERIES,
)
from .llm import call_json_llm
from .loop_helpers import (
    abandon_executor,
    analysis_from_json,
    evidence_digest,
    exit_on_keyboard_interrupt,
    fallback_analysis,
    normalize_strategy_contract,
)
from .match_summary import run_fixed_match_summary
from .prompts import build_batch_analysis_prompt
from .types import AnalysisPipelineResult, BattleAnalysis, GameDigest, ToolObservation
from ..sc2_data_agent import (
    build_knowledge_query,
    find_knowledge_run_error,
    is_knowledge_run_verified,
    run_knowledge_query,
)


_ANALYSIS_ATTEMPTS = 2
_KNOWLEDGE_NEEDS = {"effects", "synergy", "counters", "requirements"}


def _clean_strings(value: Any, *, limit: int | None = None) -> list[str]:
    if not isinstance(value, list):
        return []
    result = [text for item in value if (text := str(item).strip())]
    return result[:limit] if limit is not None else result


def _normalize_batch_analysis(
    raw: dict[str, Any],
    *,
    strategy_name: str,
    knowledge_mode: str,
) -> tuple[dict[str, Any] | None, str]:
    """Normalize the intentionally small batch-analysis schema.

    Optional bookkeeping is repaired locally. At least one strategy-fixable
    problem and several coherent deterministic plans are required. Knowledge
    questions are optional and capped; semantic quality is guided by the
    prompt rather than rejected here.
    """
    payload = dict(raw)
    valid_next_actions = {
        "propose_strategy_patch",
        "request_more_matches",
        "inspect_runtime",
        "stop",
    }
    next_action = str(payload.get("next_action") or "propose_strategy_patch").strip()
    if next_action not in valid_next_actions:
        next_action = "propose_strategy_patch"
    payload["next_action"] = next_action
    payload["action_reason"] = str(payload.get("action_reason") or "").strip()
    payload["strategy_contract"] = normalize_strategy_contract(
        payload.get("strategy_contract"), strategy_name=strategy_name
    )
    payload["winning_mechanism"] = str(payload.get("winning_mechanism") or "").strip()

    wins: list[dict[str, Any]] = []
    for item in payload.get("wins_to_preserve") or []:
        if not isinstance(item, dict):
            continue
        pattern = str(item.get("pattern") or "").strip()
        if not pattern:
            continue
        wins.append(
            {
                "pattern": pattern,
                "evidence": _clean_strings(item.get("evidence"), limit=3),
                "why": str(item.get("why") or "").strip(),
            }
        )
        if len(wins) >= 5:
            break
    payload["wins_to_preserve"] = wins

    raw_problems = payload.get("problems") or []
    if not raw_problems and isinstance(payload.get("primary_problem"), dict):
        raw_problems = [payload["primary_problem"]]
    problems: list[dict[str, Any]] = []
    valid_control_classes = {
        "strategy_fixable",
        "commander_execution",
        "runtime_execution",
        "observation_limited",
    }
    for item in raw_problems:
        if not isinstance(item, dict):
            continue
        problem = str(item.get("problem") or item.get("cause") or "").strip()
        control_class = str(item.get("control_class") or "").strip().lower()
        fixable = item.get("strategy_fixable")
        if not isinstance(fixable, bool):
            fixable = str(fixable).strip().lower() in {"1", "true", "yes"}
        if control_class not in valid_control_classes:
            control_class = "strategy_fixable" if fixable else "runtime_execution"
        fixable = control_class == "strategy_fixable"
        if not problem:
            continue
        problems.append(
            {
                "problem_id": f"P{len(problems) + 1}",
                "problem": problem,
                "evidence": _clean_strings(item.get("evidence"), limit=4),
                "consequence": str(item.get("consequence") or "").strip(),
                "strategy_fixable": fixable,
                "control_class": control_class,
                "confidence": str(item.get("confidence") or "medium").strip().lower(),
            }
        )
        if len(problems) >= 5:
            break
    payload["problems"] = problems
    strategy_problems = [problem for problem in problems if problem["strategy_fixable"]]
    if next_action == "propose_strategy_patch" and not strategy_problems:
        next_action = (
            "inspect_runtime"
            if any(problem["control_class"] == "runtime_execution" for problem in problems)
            else "request_more_matches"
        )
        payload["next_action"] = next_action
        if not payload["action_reason"]:
            payload["action_reason"] = (
                "No strategy-fixable problem is supported by the current evidence."
            )
    payload["primary_problem"] = (
        strategy_problems[0] if strategy_problems else problems[0] if problems else {}
    )
    valid_problem_ids = {
        item["problem_id"] for item in problems if item["strategy_fixable"]
    }

    raw_plans = (
        payload.get("candidate_plans") or payload.get("candidate_directions") or []
    ) if valid_problem_ids and next_action == "propose_strategy_patch" else []
    plans: list[dict[str, Any]] = []
    for item in raw_plans:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or item.get("direction") or "").strip()
        addressed = [
            problem_id.upper()
            for problem_id in _clean_strings(
                item.get("addresses_problem_ids") or item.get("problem_ids"), limit=5
            )
            if problem_id.upper() in valid_problem_ids
        ]
        if not addressed:
            addressed = [
                next(
                    problem["problem_id"]
                    for problem in problems
                    if problem["strategy_fixable"]
                )
            ]
        changes: list[dict[str, str]] = []
        raw_changes = item.get("changes") or []
        if not raw_changes and (item.get("baseline_rule") or item.get("candidate_rule")):
            raw_changes = [item]
        for change in raw_changes:
            if not isinstance(change, dict):
                continue
            baseline_rule = str(change.get("baseline_rule") or "").strip()
            candidate_rule = str(change.get("candidate_rule") or "").strip()
            if baseline_rule and candidate_rule:
                changes.append(
                    {
                        "target_paragraph_id": str(
                            change.get("target_paragraph_id") or ""
                        ).strip(),
                        "baseline_rule": baseline_rule,
                        "candidate_rule": candidate_rule,
                        "why_required": str(
                            change.get("why_required") or change.get("reason") or ""
                        ).strip(),
                    }
                )
            if len(changes) >= 6:
                break
        if not name or not changes:
            continue
        plans.append(
            {
                "id": f"D{len(plans) + 1}",
                "name": name,
                "hypothesis": str(item.get("hypothesis") or name).strip(),
                "primary_lever": str(item.get("primary_lever") or "other").strip().lower(),
                "addresses_problem_ids": list(dict.fromkeys(addressed)),
                "changes": changes,
                "predictions": _clean_strings(item.get("predictions"), limit=5),
                "disproof_conditions": _clean_strings(
                    item.get("disproof_conditions"), limit=5
                ),
                "capability_mapping": (
                    dict(item.get("capability_mapping"))
                    if isinstance(item.get("capability_mapping"), dict)
                    else {}
                ),
                "expected_benefit": str(item.get("expected_benefit") or "").strip(),
                "risk_to_winning_mechanism": str(
                    item.get("risk_to_winning_mechanism") or ""
                ).strip(),
            }
        )
        if len(plans) >= 1:
            break
    if next_action == "propose_strategy_patch" and not plans:
        return None, "analysis.candidate_plans must contain one complete coherent plan"
    if next_action != "propose_strategy_patch":
        plans = []
    payload["candidate_plans"] = plans
    payload.pop("candidate_directions", None)
    valid_plan_ids = {item["id"] for item in plans}

    evidence_limits = _clean_strings(payload.get("evidence_limits"))
    questions: list[dict[str, Any]] = []
    if knowledge_mode == "enabled":
        for item in payload.get("knowledge_questions") or []:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question") or "").strip()
            evidence_motivation = str(item.get("evidence_motivation") or "").strip()
            decision_use = str(item.get("decision_use") or "").strip()
            if not question or not evidence_motivation or not decision_use:
                continue
            entities = _clean_strings(item.get("entities"), limit=6)
            needs = [
                need.lower()
                for need in _clean_strings(item.get("needs"), limit=4)
                if need.lower() in _KNOWLEDGE_NEEDS
            ]
            plan_ids = [
                plan_id.upper()
                for plan_id in _clean_strings(
                    item.get("plan_ids") or item.get("direction_ids"), limit=5
                )
                if plan_id.upper() in valid_plan_ids
            ]
            if not entities or not needs or not plan_ids:
                continue
            linked_problem_ids = list(
                dict.fromkeys(
                    problem_id
                    for plan in plans
                    if plan["id"] in plan_ids
                    for problem_id in plan["addresses_problem_ids"]
                )
            )
            questions.append(
                {
                    "id": f"Q{len(questions) + 1}",
                    "problem_ids": linked_problem_ids,
                    "plan_ids": list(dict.fromkeys(plan_ids)),
                    "evidence_motivation": evidence_motivation,
                    "decision_use": decision_use,
                    "question": question,
                    "entities": entities,
                    "needs": list(dict.fromkeys(needs)),
                }
            )
            if len(questions) >= MAX_KNOWLEDGE_QUERIES:
                break
        if len(questions) < MIN_KNOWLEDGE_QUERIES:
            return None, (
                f"analysis.knowledge_questions must contain at least "
                f"{MIN_KNOWLEDGE_QUERIES} complete plan-linked questions"
            )
    payload["knowledge_questions"] = questions
    payload["evidence_limits"] = list(dict.fromkeys(evidence_limits))

    payload["repeated_failures"] = [
        {
            "problem_id": problem["problem_id"],
            "cause": problem["problem"],
            "consequence": problem["consequence"],
            "seen_in": problem["evidence"],
            "strategy_fixable": problem["strategy_fixable"],
            "control_class": problem["control_class"],
            "confidence": problem["confidence"],
        }
        for problem in problems
    ]
    payload["optimization_targets"] = [
        {
            "plan_id": plan["id"],
            "addresses_problem_ids": plan["addresses_problem_ids"],
            "strategy_change": plan["name"],
            "changes": plan["changes"],
        }
        for plan in plans
    ]
    payload["cross_outcome_comparison"] = _clean_strings(
        payload.get("cross_outcome_comparison"), limit=5
    )
    return payload, ""


def _observations_from_runs(
    runs: list[dict[str, Any]], *, prefix: str
) -> list[ToolObservation]:
    observations: list[ToolObservation] = []
    for index, run in enumerate(runs, 1):
        verified = is_knowledge_run_verified(run)
        error = "" if verified else (
            find_knowledge_run_error(run)
            or str(run.get("error") or "knowledge query failed")
        )
        run["ok"] = verified
        run["error"] = error
        observations.append(
            ToolObservation(
                tool="sc2_knowledge",
                args={
                    "question_id": run.get("question_id"),
                    "problem_ids": run.get("problem_ids") or ["P1"],
                    "plan_ids": run.get("plan_ids") or [],
                    "query": run.get("query"),
                },
                result={"answer": run.get("answer"), "error": error},
                ok=verified,
                summary=str(run.get("answer") if verified else error),
                status="complete" if verified else "failed",
            )
        )
        print(
            f"{prefix}AnalysisAgent: knowledge {index}/{len(runs)} "
            f"question={run.get('question_id')} status={'ok' if verified else 'failed'}",
            flush=True,
        )
    return observations


def _run_knowledge_queries(
    questions: list[dict[str, Any]],
    *,
    race: str,
    checkpoint: EvolCheckpoint | None,
    prefix: str,
) -> list[dict[str, Any]]:
    if not questions:
        return []
    cached: dict[str, dict[str, Any]] = {}
    if checkpoint is not None:
        for run in checkpoint.load_knowledge_results():
            question_id = str(run.get("question_id") or "").strip()
            if question_id:
                cached[question_id] = run

    runs: list[dict[str, Any]] = []
    print(
        f"{prefix}AnalysisAgent: resolving {len(questions)} deterministic knowledge question(s)",
        flush=True,
    )
    for question in questions:
        question_id = str(question.get("id") or "").strip()
        expected_query = build_knowledge_query(question, race=race)
        previous = cached.get(question_id)
        if (
            previous
            and is_knowledge_run_verified(previous)
            and str(previous.get("query") or "").strip() == expected_query.strip()
        ):
            run = previous
        else:
            run = run_knowledge_query(question, race=race)
            if checkpoint is not None:
                checkpoint.save_knowledge_result(run)
        runs.append(run)
    return runs


def _summarize_matches(
    *,
    strategy_name: str,
    race: str,
    records: list[Any],
    skill_texts: dict[str, str],
    model: str,
    prefix: str,
    checkpoint: EvolCheckpoint | None,
) -> tuple[
    list[GameDigest],
    list[BattleAnalysis],
    int,
    list[dict[str, Any]],
    list[str],
]:
    if checkpoint is not None and stage_reached(checkpoint.stage, "match_summaries"):
        values = checkpoint.load_match_summaries()
        print(
            f"{prefix}AnalysisAgent: resume loaded {len(values[1])} match summaries",
            flush=True,
        )
        return values

    worker_count = min(MAX_CONCURRENT_MATCH_SUBAGENTS, len(records))
    print(
        f"{prefix}AnalysisAgent: summarizing {len(records)} matches "
        f"(max_concurrency={worker_count})",
        flush=True,
    )
    results: dict[
        int,
        tuple[GameDigest, BattleAnalysis, bool, list[str], list[dict[str, Any]]],
    ] = {}
    executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="evol-match")
    futures = {}
    try:
        for game_index, record in enumerate(records, 1):
            task = partial(
                run_fixed_match_summary,
                strategy_name=strategy_name,
                race=race,
                record=record,
                game_index=game_index,
                model=model,
                prefix=prefix,
            )
            futures[executor.submit(copy_context().run, task)] = (game_index, record)
        for future in as_completed(futures):
            game_index, record = futures[future]
            try:
                results[game_index] = future.result()
            except Exception as exc:  # noqa: BLE001 - preserve the remaining batch
                error = f"Match summary crashed: {type(exc).__name__}: {exc}"
                digest = evidence_digest(record, game_index)
                digest.summary = error
                results[game_index] = (
                    digest,
                    fallback_analysis(
                        strategy_name=strategy_name,
                        race=race,
                        records=[record],
                        reason=error,
                    ),
                    False,
                    [error],
                    [{"action": "crashed", "error": error}],
                )
    except KeyboardInterrupt:
        abandon_executor(executor, futures)
        exit_on_keyboard_interrupt("stopped during match summaries")
    else:
        executor.shutdown(wait=True)

    digests: list[GameDigest] = []
    analyses: list[BattleAnalysis] = []
    events: list[dict[str, Any]] = []
    errors: list[str] = []
    completed = 0
    for game_index, record in enumerate(records, 1):
        digest, analysis, ok, item_errors, item_events = results[game_index]
        digests.append(digest)
        analyses.append(analysis)
        completed += int(ok)
        errors.extend(f"match_{game_index:03d}: {error}" for error in item_errors)
        events.append(
            {
                "stage": "match_summary",
                "game_index": game_index,
                "record_path": record.file,
                "completed": ok,
                "events": item_events,
                "errors": item_errors,
            }
        )

    if checkpoint is not None and completed:
        checkpoint.save_match_summaries(
            game_digests=digests,
            single_game_analyses=analyses,
            completed_matches=completed,
            events=events,
            errors=errors,
        )
        print(
            f"{prefix}AnalysisAgent: checkpoint saved match_summaries -> {checkpoint.run_dir}",
            flush=True,
        )
    return digests, analyses, completed, events, errors


def run_analysis_agent_loop(
    *,
    strategy_name: str,
    race: str,
    records: list[Any],
    skill_texts: dict[str, str],
    model: str = "",
    knowledge_mode: str = "enabled",
    prefix: str = "  ",
    checkpoint: EvolCheckpoint | None = None,
    prior_experiences: list[Any] | None = None,
    capability_manifest: dict[str, Any] | None = None,
) -> AnalysisPipelineResult:
    model = str(model or "").strip() or DEFAULT_ANALYSIS_MODEL
    if not records:
        analysis = fallback_analysis(
            strategy_name=strategy_name,
            race=race,
            records=[],
            reason="No match records were supplied.",
        )
        return AnalysisPipelineResult(completed=False, battle_analysis=analysis)

    if checkpoint is not None and stage_reached(checkpoint.stage, "analysis_complete"):
        analysis, observations, trace, events, errors = (
            checkpoint.load_analysis_complete()
        )
        digests, summaries, _completed, match_events, match_errors = (
            checkpoint.load_match_summaries()
            if (checkpoint.run_dir / "match_summaries.json").is_file()
            else ([], [], 0, [], [])
        )
        print(
            f"{prefix}AnalysisAgent: resume loaded completed batch analysis",
            flush=True,
        )
        return AnalysisPipelineResult(
            completed=True,
            game_digests=digests,
            single_game_analyses=summaries,
            battle_analysis=analysis,
            tool_observations=observations,
            knowledge_trace=trace,
            errors=[*match_errors, *errors],
            events=[*match_events, *events],
        )

    digests, summaries, completed, match_events, errors = _summarize_matches(
        strategy_name=strategy_name,
        race=race,
        records=records,
        skill_texts=skill_texts,
        model=model,
        prefix=prefix,
        checkpoint=checkpoint,
    )
    if completed == 0:
        analysis = fallback_analysis(
            strategy_name=strategy_name,
            race=race,
            records=records,
            reason="No match summary produced usable evidence.",
        )
        return AnalysisPipelineResult(
            completed=False,
            game_digests=digests,
            single_game_analyses=summaries,
            battle_analysis=analysis,
            errors=errors,
            events=match_events,
        )

    degraded = len(records) - completed
    if degraded:
        errors.append(
            f"{degraded} of {len(records)} summaries are degraded; trajectory details are uncertain."
        )

    payload: dict[str, Any] | None = None
    analysis_events: list[dict[str, Any]] = []
    if checkpoint is not None and stage_reached(checkpoint.stage, "batch_analysis"):
        loaded = checkpoint.load_batch_analysis()
        payload, resume_error = _normalize_batch_analysis(
            loaded,
            strategy_name=strategy_name,
            knowledge_mode=knowledge_mode,
        )
        if resume_error:
            errors.append(f"old checkpoint analysis was not reusable: {resume_error}")

    if payload is None:
        print(
            f"{prefix}AnalysisAgent: analyzing {len(summaries)} match summaries in one call",
            flush=True,
        )
        schema_errors: list[str] = []
        for attempt in range(1, _ANALYSIS_ATTEMPTS + 1):
            result = call_json_llm(
                build_batch_analysis_prompt(
                    strategy_name=strategy_name,
                    race=race,
                    single_game_analyses=summaries,
                    skill_texts=skill_texts,
                    validation_errors=schema_errors,
                    knowledge_mode=knowledge_mode,
                    prior_experiences=prior_experiences or [],
                    capability_manifest=capability_manifest or {},
                ),
                model=model,
                is_reasoning=ANALYSIS_ENABLE_REASONING,
            )
            raw = result.get("analysis") if isinstance(result, dict) else None
            if not isinstance(raw, dict) and isinstance(result, dict):
                raw = result
            if not isinstance(raw, dict):
                error = "AnalysisAgent returned no analysis object"
            else:
                payload, error = _normalize_batch_analysis(
                    raw,
                    strategy_name=strategy_name,
                    knowledge_mode=knowledge_mode,
                )
            analysis_events.append(
                {"attempt": attempt, "action": "analyze_batch", "error": error}
            )
            if payload is not None:
                break
            schema_errors.append(error)
        errors.extend(f"analysis: {item}" for item in schema_errors)

    if payload is None:
        analysis = fallback_analysis(
            strategy_name=strategy_name,
            race=race,
            records=records,
            reason="Batch Analysis Agent failed to produce one usable optimization hypothesis.",
        )
        return AnalysisPipelineResult(
            completed=False,
            game_digests=digests,
            single_game_analyses=summaries,
            battle_analysis=analysis,
            errors=errors,
            events=[*match_events, *analysis_events],
        )

    if checkpoint is not None:
        checkpoint.save_batch_analysis(payload)
        print(
            f"{prefix}AnalysisAgent: checkpoint saved batch analysis -> {checkpoint.run_dir}",
            flush=True,
        )

    questions = payload.get("knowledge_questions") or []
    runs = _run_knowledge_queries(
        questions,
        race=race,
        checkpoint=checkpoint,
        prefix=prefix,
    ) if knowledge_mode == "enabled" else []
    observations = _observations_from_runs(runs, prefix=prefix)
    failed_questions = [
        str(run.get("question_id") or "")
        for run in runs
        if not is_knowledge_run_verified(run)
    ]
    if failed_questions:
        payload["evidence_limits"] = list(
            dict.fromkeys(
                [
                    *payload.get("evidence_limits", []),
                    f"Knowledge unavailable for: {', '.join(failed_questions)}",
                ]
            )
        )
    payload["knowledge_used"] = [
        {
            "question_id": run.get("question_id"),
            "finding": run.get("answer"),
        }
        for run in runs
        if is_knowledge_run_verified(run)
    ]
    payload["knowledge_queries"] = [
        {
            "question_id": run.get("question_id"),
            "ok": is_knowledge_run_verified(run),
        }
        for run in runs
    ]
    battle_analysis = analysis_from_json(
        strategy_name=strategy_name,
        race=race,
        records=records,
        data=payload,
    )
    knowledge_trace = {
        "knowledge_mode": knowledge_mode,
        "questions": questions,
        "runs": runs,
        "failed_questions": failed_questions,
    }
    analysis_events.append(
        {
            "action": "analysis_complete",
            "llm_cross_match_calls": 1,
            "knowledge_questions": len(questions),
            "analysis": battle_analysis.raw,
        }
    )
    if checkpoint is not None:
        checkpoint.save_analysis_complete(
            battle_analysis=battle_analysis,
            tool_observations=observations,
            knowledge_trace=knowledge_trace,
            events=analysis_events,
            errors=errors,
        )
        print(
            f"{prefix}AnalysisAgent: checkpoint saved analysis_complete -> {checkpoint.run_dir}",
            flush=True,
        )

    return AnalysisPipelineResult(
        completed=True,
        game_digests=digests,
        single_game_analyses=summaries,
        battle_analysis=battle_analysis,
        tool_observations=observations,
        knowledge_trace=knowledge_trace,
        errors=errors,
        events=[*match_events, *analysis_events],
    )
