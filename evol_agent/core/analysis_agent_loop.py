from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from functools import partial
from typing import Any

from .config import (
    ANALYSIS_ENABLE_REASONING,
    DEFAULT_ANALYSIS_MODEL,
    MAX_DIAGNOSED_PROBLEMS,
    MAX_EVOL_AGENT_STEPS,
    MAX_CONCURRENT_MATCH_SUBAGENTS,
    MAX_KNOWLEDGE_QUERIES,
)
from .checkpoint import EvolCheckpoint, stage_reached
from .llm import call_json_llm
from .loop_helpers import (
    action_summary,
    analysis_from_json,
    evidence_digest,
    fallback_analysis,
    normalize_strategy_contract,
)
from .loop_helpers import abandon_executor, exit_on_keyboard_interrupt
from .match_summary import run_fixed_match_summary
from .prompts import build_analysis_agent_prompt
from .types import AnalysisPipelineResult, BattleAnalysis, GameDigest, ToolObservation
from ..sc2_data_agent import (
    build_knowledge_query,
    find_knowledge_run_error,
    is_knowledge_run_verified,
    run_knowledge_query,
)
from ..validation import find_out_of_scope_knowledge_question_error


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
) -> AnalysisPipelineResult:
    model = str(model or "").strip() or DEFAULT_ANALYSIS_MODEL
    game_digests: list[GameDigest] = []
    single_game_analyses: list[BattleAnalysis] = []
    pipeline_errors: list[str] = []
    pipeline_events: list[dict[str, Any]] = []
    completed_matches = 0

    if not records:
        analysis = fallback_analysis(
            strategy_name=strategy_name,
            race=race,
            records=[],
            reason="No match records were supplied to the analysis pipeline.",
        )
        return AnalysisPipelineResult(completed=False, battle_analysis=analysis)

    if checkpoint is not None and stage_reached(checkpoint.stage, "finish_analysis"):
        (
            battle_analysis,
            knowledge_observations,
            knowledge_trace,
            _diagnosis,
            analysis_events,
            analysis_errors,
        ) = checkpoint.load_finish_analysis()
        digests, analyses, completed_matches, match_events, match_errors = (
            checkpoint.load_match_summaries()
            if (checkpoint.run_dir / "match_summaries.json").is_file()
            else ([], [], 0, [], [])
        )
        print(
            f"{prefix}AnalysisAgent: resume skip through finish_analysis "
            f"(checkpoint={checkpoint.run_dir})",
            flush=True,
        )
        return AnalysisPipelineResult(
            completed=True,
            game_digests=digests,
            single_game_analyses=analyses,
            battle_analysis=battle_analysis,
            tool_observations=knowledge_observations,
            knowledge_trace=knowledge_trace,
            errors=[*match_errors, *analysis_errors],
            events=[*match_events, *analysis_events],
        )

    if checkpoint is not None and stage_reached(checkpoint.stage, "match_summaries"):
        game_digests, single_game_analyses, completed_matches, pipeline_events, pipeline_errors = (
            checkpoint.load_match_summaries()
        )
        print(
            f"{prefix}AnalysisAgent: resume loaded {len(single_game_analyses)} match summaries "
            f"from checkpoint",
            flush=True,
        )
    else:
        worker_count = min(MAX_CONCURRENT_MATCH_SUBAGENTS, len(records))
        print(
            f"{prefix}Analysis pipeline: {len(records)} independent match sub-agents "
            f"(max_concurrency={worker_count})",
            flush=True,
        )
        results: dict[
            int,
            tuple[GameDigest, BattleAnalysis, bool, list[str], list[dict[str, Any]]],
        ] = {}
        executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="evol-match",
        )
        futures = {}
        try:
            for game_index, record in enumerate(records, 1):
                task = partial(
                    run_fixed_match_summary,
                    strategy_name=strategy_name,
                    race=race,
                    record=record,
                    game_index=game_index,
                    skill_texts=skill_texts,
                    model=model,
                    prefix=prefix,
                )
                future = executor.submit(copy_context().run, task)
                futures[future] = (game_index, record)

            for future in as_completed(futures):
                game_index, record = futures[future]
                try:
                    results[game_index] = future.result()
                except Exception as exc:
                    error = f"Match Sub-Agent crashed: {type(exc).__name__}: {exc}"
                    digest = evidence_digest(record, game_index)
                    analysis = fallback_analysis(
                        strategy_name=strategy_name,
                        race=race,
                        records=[record],
                        reason=error,
                    )
                    digest.summary = error
                    results[game_index] = (
                        digest,
                        analysis,
                        False,
                        [error],
                        [{"step": 0, "action": "crashed", "error": error}],
                    )
        except KeyboardInterrupt:
            print(
                f"{prefix}Interrupted during match summaries; abandoning in-flight LLM calls...",
                flush=True,
            )
            abandon_executor(executor, futures)
            exit_on_keyboard_interrupt("stopped during match sub-agents")
        else:
            executor.shutdown(wait=True)

        for game_index, record in enumerate(records, 1):
            digest, analysis, completed, errors, events = results[game_index]
            game_digests.append(digest)
            single_game_analyses.append(analysis)
            completed_matches += int(completed)
            pipeline_errors.extend(f"match_{game_index:03d}: {error}" for error in errors)
            pipeline_events.append(
                {
                    "stage": "match_subagent",
                    "game_index": game_index,
                    "record_path": record.file,
                    "completed": completed,
                    "events": events,
                    "errors": errors,
                }
            )

        if checkpoint is not None and completed_matches > 0:
            checkpoint.save_match_summaries(
                game_digests=game_digests,
                single_game_analyses=single_game_analyses,
                completed_matches=completed_matches,
                events=pipeline_events,
                errors=pipeline_errors,
            )
            print(
                f"{prefix}AnalysisAgent: checkpoint saved match_summaries -> {checkpoint.run_dir}",
                flush=True,
            )

    if completed_matches == 0:
        analysis = fallback_analysis(
            strategy_name=strategy_name,
            race=race,
            records=records,
            reason="No Match Summary Sub-Agent produced a usable summary.",
        )
        return AnalysisPipelineResult(
            completed=False,
            game_digests=game_digests,
            single_game_analyses=single_game_analyses,
            battle_analysis=analysis,
            errors=pipeline_errors,
            events=pipeline_events,
        )

    analysis_errors: list[str] = []
    analysis_events: list[dict[str, Any]] = []
    degraded_matches = len(records) - completed_matches
    if degraded_matches:
        analysis_errors.append(
            f"{degraded_matches} of {len(records)} match summaries are metadata-only degraded "
            "fallbacks; do not infer trajectory details from them."
        )

    print(
        f"{prefix}AnalysisAgent: synthesizing {len(single_game_analyses)}/{len(records)} "
        f"match summaries (full={completed_matches}, degraded={degraded_matches}, "
        f"knowledge={knowledge_mode})",
        flush=True,
    )

    knowledge_observations: list[ToolObservation] = []
    diagnosis: dict[str, Any] | None = None
    battle_analysis: BattleAnalysis | None = None
    knowledge_runs: list[dict[str, Any]] = []
    knowledge_trace: dict[str, Any] = {
        "knowledge_mode": knowledge_mode,
        "diagnosed_problems": [],
        "knowledge_questions": [],
        "runs": [],
        "withheld_questions": [],
        "failed_questions": [],
    }
    def observations_from_runs(runs: list[dict[str, Any]]) -> list[ToolObservation]:
        observations: list[ToolObservation] = []
        for index, run in enumerate(runs, 1):
            verified = is_knowledge_run_verified(run)
            effective_error = "" if verified else (
                find_knowledge_run_error(run)
                or str(run.get("error") or "knowledge query failed")
            )
            run["ok"] = verified
            run["error"] = effective_error
            observations.append(
                ToolObservation(
                    tool="sc2_knowledge",
                    args={
                        "question_id": run.get("question_id"),
                        "problem_ids": run.get("problem_ids") or [],
                        "problem_id": run.get("problem_id"),
                        "query": run.get("query"),
                    },
                    result={
                        "answer": run.get("answer"),
                        "error": effective_error,
                    },
                    ok=verified,
                    summary=str(run.get("answer") if verified else effective_error),
                    status="complete" if verified else "failed",
                )
            )
            status = "ok" if verified else "failed"
            print(
                f"{prefix}AnalysisAgent: knowledge query {index}/{len(runs)} "
                f"question={run.get('question_id')} "
                f"problems={run.get('problem_ids') or []} status={status}",
                flush=True,
            )
        return observations

    def validate_diagnosis(payload: dict[str, Any]) -> str:
        """Keep useful diagnosis data and repair optional formatting in place."""

        def clean_strings(value: Any) -> list[str]:
            if not isinstance(value, list):
                return []
            return [text for item in value if (text := str(item).strip())]

        payload["strategy_contract"] = normalize_strategy_contract(
            payload.get("strategy_contract"), strategy_name=strategy_name
        )

        raw_problems = payload.get("problems")
        if not isinstance(raw_problems, list):
            return "diagnosis.problems must be a list"
        problems: list[dict[str, Any]] = []
        used_problem_ids: set[str] = set()
        for raw_problem in raw_problems:
            if not isinstance(raw_problem, dict):
                continue
            problem_text = str(raw_problem.get("problem") or "").strip()
            if not problem_text:
                continue
            problem = dict(raw_problem)
            problem_id = str(problem.get("problem_id") or "").strip()
            if not problem_id or problem_id in used_problem_ids:
                number = len(problems) + 1
                problem_id = f"P{number}"
                while problem_id in used_problem_ids:
                    number += 1
                    problem_id = f"P{number}"
            used_problem_ids.add(problem_id)
            problem["problem_id"] = problem_id
            problem["problem"] = problem_text
            fixable = problem.get("strategy_fixable")
            if not isinstance(fixable, bool):
                fixable = str(fixable).strip().lower() in {"1", "true", "yes"}
            problem["strategy_fixable"] = fixable
            problem["evidence"] = clean_strings(problem.get("evidence"))
            problems.append(problem)
            if len(problems) >= MAX_DIAGNOSED_PROBLEMS:
                break
        if not problems:
            return "diagnosis contains no usable problems"
        payload["problems"] = problems

        raw_wins = payload.get("wins_to_preserve")
        wins: list[dict[str, Any]] = []
        used_win_ids: set[str] = set()
        if isinstance(raw_wins, list):
            for raw_win in raw_wins:
                if not isinstance(raw_win, dict):
                    continue
                pattern = str(raw_win.get("pattern") or "").strip()
                if not pattern:
                    continue
                win = dict(raw_win)
                win_id = str(win.get("win_id") or "").strip()
                if not win_id or win_id in used_win_ids:
                    number = len(wins) + 1
                    win_id = f"W{number}"
                    while win_id in used_win_ids:
                        number += 1
                        win_id = f"W{number}"
                used_win_ids.add(win_id)
                win["win_id"] = win_id
                win["pattern"] = pattern
                win["evidence"] = clean_strings(win.get("evidence"))
                wins.append(win)
                if len(wins) >= 5:
                    break
        payload["wins_to_preserve"] = wins
        payload["cross_outcome_comparison"] = clean_strings(
            payload.get("cross_outcome_comparison")
        )[:5]
        payload.pop("match_coverage", None)

        evidence_limits = clean_strings(payload.get("evidence_limits"))
        raw_questions = payload.get("knowledge_questions")
        questions: list[dict[str, Any]] = []
        used_question_ids: set[str] = set()
        if knowledge_mode != "disabled" and isinstance(raw_questions, list):
            for raw_question in raw_questions:
                if not isinstance(raw_question, dict):
                    continue
                question_text = str(raw_question.get("question") or "").strip()
                if not question_text:
                    continue
                scope_error = find_out_of_scope_knowledge_question_error(question_text)
                if scope_error:
                    evidence_limits.append(
                        f"Knowledge question omitted because it was out of scope: {scope_error}"
                    )
                    continue
                linked_problem_ids = [
                    problem_id
                    for problem_id in clean_strings(raw_question.get("problem_ids"))
                    if problem_id in used_problem_ids
                ]
                if not linked_problem_ids:
                    continue
                question = dict(raw_question)
                question_id = str(question.get("id") or "").strip()
                if not question_id or question_id in used_question_ids:
                    number = len(questions) + 1
                    question_id = f"Q{number}"
                    while question_id in used_question_ids:
                        number += 1
                        question_id = f"Q{number}"
                used_question_ids.add(question_id)
                question["id"] = question_id
                question["question"] = question_text
                question["problem_ids"] = list(dict.fromkeys(linked_problem_ids))
                question["entities"] = clean_strings(question.get("entities"))[:6]
                question["needs"] = clean_strings(question.get("needs"))[:3]
                questions.append(question)
                if len(questions) >= MAX_KNOWLEDGE_QUERIES:
                    break
        payload["knowledge_questions"] = questions
        payload["evidence_limits"] = list(dict.fromkeys(evidence_limits))
        return ""

    def run_knowledge_queries(questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        targets = [
            question for question in questions if isinstance(question, dict)
        ][:MAX_KNOWLEDGE_QUERIES]
        if not targets:
            return []

        cached_by_id: dict[str, dict[str, Any]] = {}
        completed_ids: set[str] = set()
        if checkpoint is not None:
            completed_ids = checkpoint.completed_knowledge_ids()
            for run in checkpoint.load_knowledge_results():
                qid = str(run.get("question_id") or "").strip()
                if qid:
                    cached_by_id[qid] = run

        pending: list[dict[str, Any]] = []
        ordered: dict[str, dict[str, Any]] = {}
        for question in targets:
            question_id = str(question.get("id") or "").strip()
            cached = cached_by_id.get(question_id)
            expected_query = build_knowledge_query(question, race=race)
            if (
                question_id
                and question_id in completed_ids
                and isinstance(cached, dict)
                and is_knowledge_run_verified(cached)
                and str(cached.get("query") or "").strip() == expected_query.strip()
            ):
                ordered[question_id] = cached
            else:
                pending.append(question)

        if ordered and not pending:
            print(
                f"{prefix}AnalysisAgent: resume loaded {len(ordered)} knowledge result(s); "
                "none pending",
                flush=True,
            )
            return [ordered[str(q.get("id") or "")] for q in targets]

        if ordered:
            print(
                f"{prefix}AnalysisAgent: resume skipping {len(ordered)} completed knowledge "
                f"question(s); launching {len(pending)} remaining",
                flush=True,
            )

        print(
            f"{prefix}AnalysisAgent: resolving {len(pending)} deterministic knowledge query(s)",
            flush=True,
        )

        for question in pending:
            run = run_knowledge_query(question, race=race)
            if checkpoint is not None:
                checkpoint.save_knowledge_result(run)
            ordered[str(question.get("id") or "")] = run

        return [ordered[str(question.get("id") or "")] for question in targets]

    def apply_knowledge_after_diagnosis(
        knowledge_questions: list[Any],
    ) -> None:
        nonlocal knowledge_runs, knowledge_observations
        if knowledge_mode == "enabled":
            knowledge_runs = run_knowledge_queries(
                [q for q in knowledge_questions if isinstance(q, dict)]
            )
            knowledge_trace["runs"] = knowledge_runs
            knowledge_observations = observations_from_runs(knowledge_runs)
            failed = [
                str(run.get("question_id") or run.get("problem_id") or "")
                for run in knowledge_runs
                if not is_knowledge_run_verified(run)
            ]
            knowledge_trace["failed_questions"] = failed
            return

        withheld = [
            str(question.get("id") or "")
            for question in knowledge_questions
            if isinstance(question, dict) and str(question.get("id") or "").strip()
        ]
        knowledge_trace["withheld_questions"] = withheld
        knowledge_runs = [
            {
                "question_id": question_id,
                "problem_ids": [],
                "problem_id": question_id,
                "ok": False,
                "query": "",
                "answer": "",
                "error": "knowledge withheld (disabled mode)",
                "status": "withheld",
            }
            for question_id in withheld
        ]
        knowledge_trace["runs"] = knowledge_runs

    if checkpoint is not None and stage_reached(checkpoint.stage, "diagnosis"):
        loaded_diagnosis = checkpoint.load_diagnosis()
        resume_error = validate_diagnosis(loaded_diagnosis)
        if resume_error:
            analysis_errors.append(
                "checkpoint diagnosis has no usable problem data; "
                f"regenerating it: {resume_error}"
            )
            loaded_diagnosis = None
        diagnosis = loaded_diagnosis
    if diagnosis is not None:
        knowledge_trace["diagnosed_problems"] = diagnosis.get("problems") or []
        knowledge_questions = (
            diagnosis.get("knowledge_questions")
            if isinstance(diagnosis.get("knowledge_questions"), list)
            else []
        )
        knowledge_trace["knowledge_questions"] = knowledge_questions
        analysis_events.append(
            {
                "step": 0,
                "action": "diagnose",
                "diagnosis": diagnosis,
                "resumed": True,
            }
        )
        print(
            f"{prefix}AnalysisAgent: resume loaded diagnosis "
            f"(problems={len(knowledge_trace['diagnosed_problems'])}, "
            f"questions={len(knowledge_questions)})",
            flush=True,
        )
        apply_knowledge_after_diagnosis(knowledge_questions)

    for step in range(1, MAX_EVOL_AGENT_STEPS + 1):
        phase = "diagnose" if diagnosis is None else "finish"
        action = call_json_llm(
            build_analysis_agent_prompt(
                strategy_name=strategy_name,
                race=race,
                single_game_analyses=single_game_analyses,
                skill_texts=skill_texts,
                tool_observations=knowledge_observations,
                validation_errors=analysis_errors,
                phase=phase,
                diagnosis=diagnosis,
                knowledge_mode=knowledge_mode,
                knowledge_runs=knowledge_runs,
            ),
            model=model,
            is_reasoning=ANALYSIS_ENABLE_REASONING,
        )
        if not action:
            analysis_errors.append(f"Analysis Agent returned no JSON action during {phase}")
            analysis_events.append(
                {"step": step, "action": "invalid", "phase": phase, "error": analysis_errors[-1]}
            )
            continue

        name = str(action.get("action") or "")
        print(f"{prefix}AnalysisAgent: {action_summary(action)}", flush=True)

        if diagnosis is None:
            if name != "diagnose":
                analysis_errors.append("Analysis Agent must return diagnose before final analysis")
                continue
            payload = action.get("diagnosis") if isinstance(action.get("diagnosis"), dict) else {}
            error = validate_diagnosis(payload)
            if error:
                analysis_errors.append(error)
                analysis_events.append({"step": step, "action": name, "error": error})
                print(f"{prefix}AnalysisAgent: rejected diagnosis: {error}", flush=True)
                continue
            diagnosis = payload
            knowledge_trace["diagnosed_problems"] = payload["problems"]
            knowledge_questions = (
                payload.get("knowledge_questions")
                if isinstance(payload.get("knowledge_questions"), list)
                else []
            )
            knowledge_trace["knowledge_questions"] = knowledge_questions
            analysis_events.append({"step": step, "action": name, "diagnosis": payload})
            if checkpoint is not None:
                checkpoint.save_diagnosis(payload)
                print(
                    f"{prefix}AnalysisAgent: checkpoint saved diagnosis -> {checkpoint.run_dir}",
                    flush=True,
                )
            apply_knowledge_after_diagnosis(knowledge_questions)
            continue

        if name != "finish_analysis":
            analysis_errors.append(
                "Analysis Agent must return finish_analysis after diagnosis and knowledge queries"
            )
            analysis_events.append(
                {"step": step, "action": name or "invalid", "error": analysis_errors[-1]}
            )
            continue

        payload = action.get("analysis") if isinstance(action.get("analysis"), dict) else {}
        targets = payload.get("optimization_targets")
        if not payload:
            analysis_errors.append("finish_analysis requires a non-empty analysis object")
            continue
        if not isinstance(targets, list) or not targets:
            analysis_errors.append("finish_analysis requires at least one optimization target")
            continue
        if len(targets) > 5:
            analysis_errors.append("finish_analysis allows at most five optimization targets")
            continue
        payload["strategy_contract"] = normalize_strategy_contract(
            diagnosis.get("strategy_contract"), strategy_name=strategy_name
        )
        diagnosed_ids = {
            str(problem.get("problem_id"))
            for problem in diagnosis.get("problems", [])
            if isinstance(problem, dict)
        }
        bad_target = next(
            (
                target
                for target in targets
                if not isinstance(target, dict)
                or str(target.get("problem_id") or "") not in diagnosed_ids
            ),
            None,
        )
        if bad_target is not None:
            analysis_errors.append("every optimization target must reference a diagnosed problem_id")
            continue

        external_knowledge = payload.get("knowledge_used")
        target_claims = [
            item
            for target in targets
            if isinstance(target, dict)
            for item in (target.get("knowledge_used") or [])
            if str(item).strip()
        ]
        verified_runs = [run for run in knowledge_runs if is_knowledge_run_verified(run)]
        failed_runs = [
            run
            for run in knowledge_runs
            if not is_knowledge_run_verified(run) and run.get("status") != "withheld"
        ]
        if knowledge_mode == "disabled":
            if (isinstance(external_knowledge, list) and external_knowledge) or target_claims:
                analysis_errors.append(
                    "disabled knowledge mode requires empty knowledge_used fields"
                )
                continue
        elif verified_runs:
            if not isinstance(external_knowledge, list) or not external_knowledge:
                analysis_errors.append(
                    "enabled mode must state how verified knowledge affected the analysis"
                )
                continue
        elif (isinstance(external_knowledge, list) and external_knowledge) or target_claims:
            analysis_errors.append(
                "no verified knowledge is available; knowledge_used fields must be empty"
            )
            continue
        if knowledge_mode == "enabled" and (failed_runs or knowledge_trace["failed_questions"]):
            evidence_limits = payload.get("evidence_limits")
            if not isinstance(evidence_limits, list) or not any(
                str(item).strip() for item in evidence_limits
            ):
                analysis_errors.append(
                    "finish_analysis must record unresolved knowledge gaps in evidence_limits"
                )
                continue

        payload["knowledge_mode"] = knowledge_mode
        payload["knowledge_queries"] = {
            "runs": [
                {
                    "question_id": run.get("question_id"),
                    "problem_ids": run.get("problem_ids") or [],
                    "problem_id": run.get("problem_id"),
                    "ok": run.get("ok"),
                    "status": run.get("status")
                    or ("complete" if run.get("ok") else "failed"),
                }
                for run in knowledge_runs
            ],
            "failed_questions": knowledge_trace["failed_questions"],
            "withheld_questions": knowledge_trace["withheld_questions"],
        }
        payload["diagnosis"] = diagnosis
        knowledge_trace["termination"] = {"reason": "agent_finish", "step": step}
        battle_analysis = analysis_from_json(
            strategy_name=strategy_name,
            race=race,
            records=records,
            data=payload,
        )
        analysis_events.append({"step": step, "action": name, "analysis": battle_analysis.raw})
        if checkpoint is not None:
            checkpoint.save_finish_analysis(
                battle_analysis=battle_analysis,
                tool_observations=knowledge_observations,
                knowledge_trace=knowledge_trace,
                diagnosis=diagnosis,
                events=analysis_events,
                errors=analysis_errors,
            )
            print(
                f"{prefix}AnalysisAgent: checkpoint saved finish_analysis -> {checkpoint.run_dir}",
                flush=True,
            )
        break

    pipeline_events.append(
        {
            "stage": "cross_match_analysis",
            "completed": battle_analysis is not None,
            "events": analysis_events,
            "errors": analysis_errors,
            "knowledge_trace": knowledge_trace,
        }
    )
    pipeline_errors.extend(f"analysis: {error}" for error in analysis_errors)

    if battle_analysis is None:
        battle_analysis = fallback_analysis(
            strategy_name=strategy_name,
            race=race,
            records=records,
            reason="Analysis Agent failed to produce cross-match BattleAnalysis.",
        )
        return AnalysisPipelineResult(
            completed=False,
            game_digests=game_digests,
            single_game_analyses=single_game_analyses,
            battle_analysis=battle_analysis,
            tool_observations=knowledge_observations,
            knowledge_trace=knowledge_trace,
            errors=pipeline_errors,
            events=pipeline_events,
        )

    return AnalysisPipelineResult(
        completed=True,
        game_digests=game_digests,
        single_game_analyses=single_game_analyses,
        battle_analysis=battle_analysis,
        tool_observations=knowledge_observations,
        knowledge_trace=knowledge_trace,
        errors=pipeline_errors,
        events=pipeline_events,
    )
