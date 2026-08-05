"""Tolerate SC2 client IDs newer than the installed burnysc2 enums.

StarCraft II may emit BuffId values (e.g. 301) that are absent from burnysc2
5.0.5's BuffId enum. Parsing them via BuffId(id) raises ValueError and aborts
the game loop inside combat grouping (Unit.is_flying -> has_buff -> buffs).

Apply once at process start before the game loop runs.
"""

from __future__ import annotations

_APPLIED = False


def apply() -> None:
    global _APPLIED
    if _APPLIED:
        return

    from sc2.cache import property_immutable_cache
    from sc2.ids.buff_id import BuffId
    from sc2.unit import Unit

    @classmethod  # type: ignore[misc]
    def _buff_missing_(cls, value):  # noqa: ANN001
        # Newer burnysc2 returns NULL; returning None still raises on Py3.9+.
        return cls.NULL

    BuffId._missing_ = _buff_missing_

    @property_immutable_cache
    def _buffs(self):  # noqa: ANN001
        known = BuffId._value2member_map_
        return {
            known[buff_id]
            for buff_id in self._proto.buff_ids
            if buff_id in known
        }

    Unit.buffs = _buffs

    _APPLIED = True
