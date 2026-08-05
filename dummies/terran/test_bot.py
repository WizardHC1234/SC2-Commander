"""Legacy TestBot stub — use ``commander`` bot for real matches."""

from sharpy.knowledges import KnowledgeBot
from sharpy.plans import BuildOrder
from commander.macro_exec import EmptyTactics


class TestBot(KnowledgeBot):
    def __init__(self):
        super().__init__("TestBot")

    async def create_plan(self) -> BuildOrder:
        return EmptyTactics()
