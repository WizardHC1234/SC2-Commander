"""EvolAgent run checkpoints for interrupted/resume execution."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import OPTIMIZATION_LOG_DIR
from .types import BattleAnalysis, GameDigest, ToolObservation
from ..sc2_data_agent.bridge import is_knowledge_run_verified

CHECKPOINT_SCHEMA = "evol_agent_checkpoint.v2"
PIPELINE_VERSION = "full_timeline_summary_v1_evidence_retrieval_v1"

STAGE_ORDER = (
    "created",
    "match_summaries",
    "batch_discovery",
    "knowledge",
    "batch_analysis",
    "analysis_complete",
    "candidate",
)

STAGE_RANK = {name: index for index, name in enumerate(STAGE_ORDER)}


def stage_reached(current: str, target: str) -> bool:
    return STAGE_RANK.get(str(current or ""), -1) >= STAGE_RANK.get(str(target or ""), 999)


def normalize_record_files(records: list[Any]) -> list[str]:
    files: list[str] = []
    for record in records:
        raw = getattr(record, "file", None)
        if raw is None and isinstance(record, dict):
            raw = record.get("file") or record.get("record_path")
        path = Path(str(raw or "")).resolve()
        files.append(str(path))
    return files


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return path


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def game_digest_from_dict(data: dict[str, Any]) -> GameDigest:
    return GameDigest(
        record_path=str(data.get("record_path") or ""),
        result=str(data.get("result") or ""),
        duration=str(data.get("duration") or ""),
        summary=str(data.get("summary") or ""),
        key_events=list(data.get("key_events") or [])
        if isinstance(data.get("key_events"), list)
        else [],
        failure_modes=list(data.get("failure_modes") or [])
        if isinstance(data.get("failure_modes"), list)
        else [],
        success_patterns=list(data.get("success_patterns") or [])
        if isinstance(data.get("success_patterns"), list)
        else [],
        actionable_signals=list(data.get("actionable_signals") or [])
        if isinstance(data.get("actionable_signals"), list)
        else [],
        uncertain_questions=list(data.get("uncertain_questions") or [])
        if isinstance(data.get("uncertain_questions"), list)
        else [],
        raw=dict(data.get("raw") or data),
    )


def battle_analysis_from_dict(data: dict[str, Any]) -> BattleAnalysis:
    return BattleAnalysis(
        strategy_name=str(data.get("strategy_name") or ""),
        race=str(data.get("race") or ""),
        sample_size=int(data.get("sample_size") or 0),
        record_mix=str(data.get("record_mix") or ""),
        strategy_contract=dict(data.get("strategy_contract") or {})
        if isinstance(data.get("strategy_contract"), dict)
        else {},
        repeated_failures=list(data.get("repeated_failures") or [])
        if isinstance(data.get("repeated_failures"), list)
        else [],
        wins_to_preserve=list(data.get("wins_to_preserve") or [])
        if isinstance(data.get("wins_to_preserve"), list)
        else [],
        cross_outcome_comparison=list(data.get("cross_outcome_comparison") or [])
        if isinstance(data.get("cross_outcome_comparison"), list)
        else [],
        optimization_targets=list(data.get("optimization_targets") or [])
        if isinstance(data.get("optimization_targets"), list)
        else [],
        knowledge_used=list(data.get("knowledge_used") or [])
        if isinstance(data.get("knowledge_used"), list)
        else [],
        evidence_limits=list(data.get("evidence_limits") or [])
        if isinstance(data.get("evidence_limits"), list)
        else [],
        raw=dict(data.get("raw") or data),
    )


def tool_observation_from_dict(data: dict[str, Any]) -> ToolObservation:
    return ToolObservation(
        tool=str(data.get("tool") or ""),
        args=dict(data.get("args") or {}) if isinstance(data.get("args"), dict) else {},
        result=dict(data.get("result") or {}) if isinstance(data.get("result"), dict) else {},
        ok=bool(data.get("ok", True)),
        summary=str(data.get("summary") or ""),
        status=str(data.get("status") or ""),
    )


@dataclass
class EvolCheckpoint:
    run_dir: Path
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def stage(self) -> str:
        return str(self.meta.get("stage") or "created")

    @property
    def knowledge_dir(self) -> Path:
        return self.run_dir / "knowledge"

    def checkpoint_path(self) -> Path:
        return self.run_dir / "checkpoint.json"

    def flush_meta(self) -> None:
        self.meta["updated"] = datetime.now().isoformat()
        _write_json(self.checkpoint_path(), self.meta)

    def set_stage(self, stage: str) -> None:
        if stage not in STAGE_RANK:
            raise ValueError(f"unknown checkpoint stage: {stage}")
        if STAGE_RANK[stage] >= STAGE_RANK.get(self.stage, -1):
            self.meta["stage"] = stage
        self.flush_meta()

    def save_match_summaries(
        self,
        *,
        game_digests: list[GameDigest],
        single_game_analyses: list[BattleAnalysis],
        completed_matches: int,
        events: list[dict[str, Any]] | None = None,
        errors: list[str] | None = None,
    ) -> None:
        payload = {
            "game_digests": [d.raw or d.__dict__ for d in game_digests],
            "single_game_analyses": [a.raw or a.__dict__ for a in single_game_analyses],
            "completed_matches": completed_matches,
            "events": events or [],
            "errors": errors or [],
        }
        _write_json(self.run_dir / "match_summaries.json", payload)
        self.set_stage("match_summaries")

    def load_match_summaries(
        self,
    ) -> tuple[list[GameDigest], list[BattleAnalysis], int, list[dict[str, Any]], list[str]]:
        data = _read_json(self.run_dir / "match_summaries.json")
        digests = [
            game_digest_from_dict(item)
            for item in data.get("game_digests") or []
            if isinstance(item, dict)
        ]
        analyses = [
            battle_analysis_from_dict(item)
            for item in data.get("single_game_analyses") or []
            if isinstance(item, dict)
        ]
        events = list(data.get("events") or []) if isinstance(data.get("events"), list) else []
        errors = [str(item) for item in (data.get("errors") or [])]
        return digests, analyses, int(data.get("completed_matches") or 0), events, errors

    def save_cross_match_discovery(self, discovery: dict[str, Any]) -> None:
        _write_json(self.run_dir / "cross_match_discovery.json", discovery)
        self.set_stage("batch_discovery")

    def load_cross_match_discovery(self) -> dict[str, Any]:
        path = self.run_dir / "cross_match_discovery.json"
        if not path.is_file():
            raise FileNotFoundError(f"cross_match_discovery.json not found in {self.run_dir}")
        data = _read_json(path)
        if not isinstance(data, dict):
            raise ValueError("cross_match_discovery.json must contain an object")
        return data

    def has_cross_match_discovery(self) -> bool:
        return (self.run_dir / "cross_match_discovery.json").is_file()

    def save_batch_analysis(self, analysis: dict[str, Any]) -> None:
        _write_json(self.run_dir / "batch_analysis.json", analysis)
        self.set_stage("batch_analysis")

    def load_batch_analysis(self) -> dict[str, Any]:
        data = _read_json(self.run_dir / "batch_analysis.json")
        if not isinstance(data, dict):
            raise ValueError("batch_analysis.json must contain an object")
        return data

    def knowledge_result_path(self, question_id: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in question_id)
        return self.knowledge_dir / f"{safe or 'Q'}.json"

    def save_knowledge_result(self, run: dict[str, Any]) -> None:
        question_id = str(run.get("question_id") or run.get("id") or "").strip() or "Q"
        _write_json(self.knowledge_result_path(question_id), run)
        completed = list(self.meta.get("completed_knowledge_ids") or [])
        if is_knowledge_run_verified(run) and question_id not in completed:
            completed.append(question_id)
        elif not is_knowledge_run_verified(run) and question_id in completed:
            completed.remove(question_id)
        self.meta["completed_knowledge_ids"] = completed
        # Knowledge stage is reached once at least one result exists; finish marks full set.
        if STAGE_RANK.get(self.stage, -1) < STAGE_RANK["knowledge"]:
            self.meta["stage"] = "knowledge"
        self.flush_meta()

    def load_knowledge_results(self) -> list[dict[str, Any]]:
        if not self.knowledge_dir.exists():
            return []
        runs: list[dict[str, Any]] = []
        for path in sorted(self.knowledge_dir.glob("*.json")):
            data = _read_json(path)
            if isinstance(data, dict):
                runs.append(data)
        return runs

    def completed_knowledge_ids(self) -> set[str]:
        ids: set[str] = set()
        for run in self.load_knowledge_results():
            qid = str(run.get("question_id") or "").strip()
            if qid and is_knowledge_run_verified(run):
                ids.add(qid)
        return ids

    def save_analysis_complete(
        self,
        *,
        battle_analysis: BattleAnalysis,
        tool_observations: list[ToolObservation],
        knowledge_trace: dict[str, Any],
        events: list[dict[str, Any]] | None = None,
        errors: list[str] | None = None,
    ) -> None:
        payload = {
            "battle_analysis": battle_analysis.raw or battle_analysis.__dict__,
            "tool_observations": [obs.__dict__ for obs in tool_observations],
            "knowledge_trace": knowledge_trace,
            "events": events or [],
            "errors": errors or [],
        }
        _write_json(self.run_dir / "analysis_checkpoint.json", payload)
        # Also keep a stable analysis.json early for inspection.
        _write_json(self.run_dir / "analysis.json", payload["battle_analysis"])
        _write_json(self.run_dir / "knowledge_trace.json", knowledge_trace)
        _write_json(
            self.run_dir / "tool_observations.json",
            payload["tool_observations"],
        )
        self.set_stage("analysis_complete")

    def load_analysis_complete(
        self,
    ) -> tuple[BattleAnalysis, list[ToolObservation], dict[str, Any], list[dict[str, Any]], list[str]]:
        data = _read_json(self.run_dir / "analysis_checkpoint.json")
        analysis = battle_analysis_from_dict(
            data.get("battle_analysis") if isinstance(data.get("battle_analysis"), dict) else {}
        )
        observations = [
            tool_observation_from_dict(item)
            for item in data.get("tool_observations") or []
            if isinstance(item, dict)
        ]
        knowledge_trace = (
            dict(data.get("knowledge_trace") or {})
            if isinstance(data.get("knowledge_trace"), dict)
            else {}
        )
        events = list(data.get("events") or []) if isinstance(data.get("events"), list) else []
        errors = [str(item) for item in (data.get("errors") or [])]
        return analysis, observations, knowledge_trace, events, errors

    def mark_candidate_complete(self) -> None:
        self.set_stage("candidate")


def create_checkpoint(
    *,
    strategy_name: str,
    race: str,
    knowledge_mode: str,
    models: dict[str, str],
    record_files: list[str],
    run_id: str = "",
) -> EvolCheckpoint:
    ts = run_id.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OPTIMIZATION_LOG_DIR / strategy_name / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "knowledge").mkdir(parents=True, exist_ok=True)
    meta = {
        "schema": CHECKPOINT_SCHEMA,
        "pipeline_version": PIPELINE_VERSION,
        "stage": "created",
        "created": datetime.now().isoformat(),
        "updated": datetime.now().isoformat(),
        "strategy_name": strategy_name,
        "race": race,
        "knowledge_mode": knowledge_mode,
        "models": dict(models),
        "record_files": list(record_files),
        "completed_knowledge_ids": [],
        "run_dir": str(run_dir),
    }
    checkpoint = EvolCheckpoint(run_dir=run_dir, meta=meta)
    checkpoint.flush_meta()
    return checkpoint


def load_checkpoint(run_dir: str | Path) -> EvolCheckpoint:
    path = Path(run_dir).resolve()
    checkpoint_file = path / "checkpoint.json"
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"checkpoint.json not found in {path}")
    meta = _read_json(checkpoint_file)
    if not isinstance(meta, dict):
        raise ValueError(f"invalid checkpoint.json in {path}")
    if str(meta.get("schema") or "") != CHECKPOINT_SCHEMA:
        raise ValueError(
            f"unsupported checkpoint schema in {path}: {meta.get('schema')}"
        )
    return EvolCheckpoint(run_dir=path, meta=meta)


def validate_checkpoint_fingerprint(
    checkpoint: EvolCheckpoint,
    *,
    strategy_name: str,
    race: str,
    knowledge_mode: str,
    record_files: list[str],
) -> None:
    meta = checkpoint.meta
    errors: list[str] = []
    if str(meta.get("pipeline_version") or "") != PIPELINE_VERSION:
        errors.append(
            "pipeline_version mismatch: checkpoint predates structured evidence "
            "retrieval; start a new EvolAgent run"
        )
    if str(meta.get("strategy_name") or "") != strategy_name:
        errors.append(
            f"strategy_name mismatch: checkpoint={meta.get('strategy_name')} current={strategy_name}"
        )
    if str(meta.get("race") or "").lower() != race.lower():
        errors.append(f"race mismatch: checkpoint={meta.get('race')} current={race}")
    if str(meta.get("knowledge_mode") or "") != knowledge_mode:
        errors.append(
            f"knowledge_mode mismatch: checkpoint={meta.get('knowledge_mode')} current={knowledge_mode}"
        )
    expected = [str(Path(item).resolve()) for item in (meta.get("record_files") or [])]
    current = [str(Path(item).resolve()) for item in record_files]
    if expected != current:
        errors.append(
            "record_files mismatch: checkpoint and current run selected different matches"
        )
    if errors:
        raise ValueError("; ".join(errors))


def validate_analysis_seed_checkpoint(
    checkpoint: EvolCheckpoint,
    *,
    strategy_name: str,
    race: str,
    knowledge_mode: str,
    record_files: list[str],
    analysis_model: str,
) -> None:
    """Validate a completed checkpoint used to seed a larger analysis batch."""
    meta = checkpoint.meta
    errors: list[str] = []
    if str(meta.get("pipeline_version") or "") != PIPELINE_VERSION:
        errors.append("pipeline_version mismatch")
    if not stage_reached(checkpoint.stage, "match_summaries"):
        errors.append("seed has no completed match summaries")
    if str(meta.get("strategy_name") or "") != strategy_name:
        errors.append("strategy_name mismatch")
    if str(meta.get("race") or "").lower() != race.lower():
        errors.append("race mismatch")
    if str(meta.get("knowledge_mode") or "") != knowledge_mode:
        errors.append("knowledge_mode mismatch")
    seed_model = str((meta.get("models") or {}).get("analysis") or "").strip()
    if seed_model and seed_model != str(analysis_model or "").strip():
        errors.append("analysis model mismatch")
    current = {str(Path(item).resolve()) for item in record_files}
    seed = {str(Path(item).resolve()) for item in (meta.get("record_files") or [])}
    if not seed:
        errors.append("seed has no record files")
    elif not seed.issubset(current):
        errors.append("seed records are not a subset of the current batch")
    if errors:
        raise ValueError("; ".join(errors))
