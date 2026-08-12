from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

JsonDict = dict[str, Any]


@dataclass
class ToolObservation:
    tool: str
    args: JsonDict
    result: JsonDict
    ok: bool = True
    summary: str = ""
    status: str = ""


@dataclass
class GameEvidence:
    file: str
    result: str
    duration: str
    timeline: str
    meta: JsonDict = field(default_factory=dict)


@dataclass
class GameDigest:
    record_path: str
    result: str
    duration: str
    summary: str
    key_events: list[JsonDict] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)
    success_patterns: list[str] = field(default_factory=list)
    actionable_signals: list[str] = field(default_factory=list)
    uncertain_questions: list[str] = field(default_factory=list)
    raw: JsonDict = field(default_factory=dict)


@dataclass
class BattleAnalysis:
    strategy_name: str
    race: str
    sample_size: int
    record_mix: str
    strategy_contract: JsonDict = field(default_factory=dict)
    repeated_failures: list[JsonDict] = field(default_factory=list)
    wins_to_preserve: list[JsonDict] = field(default_factory=list)
    cross_outcome_comparison: list[str] = field(default_factory=list)
    optimization_targets: list[str] = field(default_factory=list)
    knowledge_used: list[JsonDict] = field(default_factory=list)
    evidence_limits: list[str] = field(default_factory=list)
    raw: JsonDict = field(default_factory=dict)


@dataclass
class AnalysisPipelineResult:
    completed: bool
    game_digests: list[GameDigest] = field(default_factory=list)
    single_game_analyses: list[BattleAnalysis] = field(default_factory=list)
    battle_analysis: Optional[BattleAnalysis] = None
    tool_observations: list[ToolObservation] = field(default_factory=list)
    knowledge_trace: JsonDict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    events: list[JsonDict] = field(default_factory=list)


@dataclass
class EvolImprovement:
    analysis: JsonDict
    files: dict[str, str]
    raw: JsonDict = field(default_factory=dict)


@dataclass
class ValidationResult:
    ok: bool
    error: str = ""
    files: Optional[dict[str, str]] = None


@dataclass
class EvolRunRequest:
    record_paths: list[Path] = field(default_factory=list)
    batch_dir: Optional[Path] = None
    strategy_name: str = ""
    race: str = "terran"
    skill_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    model: str = ""
    knowledge_mode: str = "enabled"
    dry_run: bool = False
    resume_dir: Optional[Path] = None
    prior_experiences: list[str] = field(default_factory=list)


@dataclass
class EvolRunResult:
    ok: bool
    message: str
    strategy_name: str = ""
    race: str = "terran"
    output_dir: Optional[Path] = None
    candidate_hash: str = ""
    game_digests: list[GameDigest] = field(default_factory=list)
    battle_analysis: Optional[BattleAnalysis] = None
    improvement: Optional[EvolImprovement] = None
    changes: list[JsonDict] = field(default_factory=list)
    tool_observations: list[ToolObservation] = field(default_factory=list)
