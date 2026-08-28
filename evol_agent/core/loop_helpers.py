from __future__ import annotations
from typing import Any
from .types import BattleAnalysis, GameDigest


STRATEGY_DIRECTIONS = frozenset({"preserve", "adjust", "replace"})


def normalize_strategy_contract(value: Any, *, strategy_name: str) -> dict[str, Any]:
    """Normalize the compact strategy identity while accepting old checkpoints."""
    contract = value if isinstance(value, dict) else {}

    def clean_list(raw: Any) -> list[str]:
        if not isinstance(raw, list):
            return []
        return [text for item in raw if (text := str(item).strip())]

    identity = str(
        contract.get("identity") or contract.get("intended_plan") or ""
    ).strip() or f"Current {strategy_name} strategy"
    style = str(contract.get("style") or "").strip()
    core_win_mechanism = str(
        contract.get("core_win_mechanism") or identity
    ).strip()
    critical_power_window = str(
        contract.get("critical_power_window")
        or contract.get("critical_timing_or_power_spike")
        or ""
    ).strip()
    observed_winning_signature = str(
        contract.get("observed_winning_signature")
        or contract.get("winning_signature")
        or ""
    ).strip()
    commitments = clean_list(contract.get("core_commitments"))
    if not commitments:
        commitments = clean_list(contract.get("must_preserve"))
    protected_invariants = clean_list(contract.get("protected_invariants"))
    if not protected_invariants and (
        "intended_plan" in contract or "must_preserve" in contract
    ):
        # Legacy discovery schemas treated every listed commitment as protected.
        # Modern discovery separates defining invariants from adjustable strategy
        # implementation, so an omitted protected list must stay empty.
        protected_invariants = list(commitments)
    boundary = str(contract.get("optimization_boundary") or "").strip()
    if not boundary:
        legacy_boundary = clean_list(contract.get("must_not_break"))
        boundary = "; ".join(legacy_boundary) or (
            "Keep the current strategy identity unless cross-match evidence shows "
            "that its core win plan fails."
        )
    direction = str(contract.get("direction") or "adjust").strip().lower()
    if direction not in STRATEGY_DIRECTIONS:
        direction = "adjust"
    return {
        "identity": identity,
        "style": style,
        "core_win_mechanism": core_win_mechanism,
        "critical_power_window": critical_power_window,
        "observed_winning_signature": observed_winning_signature,
        "winning_evidence": clean_list(contract.get("winning_evidence")),
        "core_commitments": commitments,
        "protected_invariants": protected_invariants,
        "flexible_components": clean_list(contract.get("flexible_components")),
        "optimization_boundary": boundary,
        "direction": direction,
    }


def record_mix(records: list[Any]) -> str:
    wins = sum(1 for record in records if str(record.result).upper() in ("WIN", "VICTORY"))
    return f"{wins}W/{len(records)-wins}L"


def evidence_digest(record: Any, index: int) -> GameDigest:
    return GameDigest(
        record_path=record.file,
        result=record.result,
        duration=record.duration,
        summary="Baseline evidence included in EvolAgent state.",
        raw={
            "game_index": index,
            "record_path": record.file,
            "result": record.result,
            "duration": record.duration,
            "timeline": record.timeline,
            "meta": record.meta,
            "summary": "Baseline evidence included in EvolAgent state.",
        },
    )


def analysis_from_json(*, strategy_name: str, race: str, records: list[Any], data: dict[str, Any]) -> BattleAnalysis:
    return BattleAnalysis(
        strategy_name=strategy_name,
        race=race,
        sample_size=len(records),
        record_mix=record_mix(records),
        strategy_contract=data.get("strategy_contract") if isinstance(data.get("strategy_contract"), dict) else {},
        repeated_failures=data.get("repeated_failures") if isinstance(data.get("repeated_failures"), list) else [],
        wins_to_preserve=data.get("wins_to_preserve") if isinstance(data.get("wins_to_preserve"), list) else [],
        cross_outcome_comparison=(
            data.get("cross_outcome_comparison") if isinstance(data.get("cross_outcome_comparison"), list) else []
        ),
        optimization_targets=(
            data.get("optimization_targets") if isinstance(data.get("optimization_targets"), list) else []
        ),
        knowledge_used=data.get("knowledge_used") if isinstance(data.get("knowledge_used"), list) else [],
        evidence_limits=data.get("evidence_limits") if isinstance(data.get("evidence_limits"), list) else [],
        raw={
            "strategy_name": strategy_name,
            "race": race,
            "sample_size": len(records),
            "record_mix": record_mix(records),
            **data,
        },
    )


def fallback_analysis(*, strategy_name: str, race: str, records: list[Any], reason: str) -> BattleAnalysis:
    return analysis_from_json(
        strategy_name=strategy_name,
        race=race,
        records=records,
        data={
            "strategy_contract": {},
            "repeated_failures": [],
            "wins_to_preserve": [],
            "cross_outcome_comparison": [],
            "optimization_targets": [],
            "knowledge_used": [],
            "evidence_limits": [reason],
        },
    )


def abandon_executor(executor: Any, futures: Any = None) -> None:
    """Stop waiting on worker threads after Ctrl+C.

    ThreadPoolExecutor's context manager always ``shutdown(wait=True)``, which
    blocks until in-flight LLM HTTP calls finish. Cancel pending work and leave
    without waiting so the CLI can exit.
    """
    if futures:
        for future in futures:
            try:
                future.cancel()
            except Exception:
                pass
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except TypeError:
        # Python < 3.9 may lack cancel_futures.
        executor.shutdown(wait=False)


def exit_on_keyboard_interrupt(message: str = "EvolAgent stopped by Ctrl+C") -> None:
    """Force-process exit; non-daemon LLM worker threads would otherwise keep us alive."""
    import os
    import sys

    print(f"\n[INTERRUPTED] {message}", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(130)
