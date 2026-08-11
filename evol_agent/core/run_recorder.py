from __future__ import annotations

from contextvars import ContextVar
from datetime import datetime
from typing import Any


_RUN_EVENTS: ContextVar[list[dict[str, Any]] | None] = ContextVar("evol_agent_run_events", default=None)


def reset_run_events() -> None:
    _RUN_EVENTS.set([])


def append_run_event(event_type: str, payload: dict[str, Any]) -> None:
    events = _RUN_EVENTS.get()
    if events is None:
        return
    events.append(
        {
            "index": len(events) + 1,
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            **payload,
        }
    )


def get_run_events() -> list[dict[str, Any]]:
    events = _RUN_EVENTS.get()
    return list(events or [])