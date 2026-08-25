"""SC2-Commander application package (flat layout).

Do not import CommanderBot at package import time. Evolution tooling only
needs wake-event constants, and pulling the bot would require sc2pathlib
native bindings before a match even starts.
"""

from __future__ import annotations

from typing import Any

__all__ = ["CommanderBot"]


def __getattr__(name: str) -> Any:
    if name == "CommanderBot":
        from commander.bot import CommanderBot

        return CommanderBot
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
