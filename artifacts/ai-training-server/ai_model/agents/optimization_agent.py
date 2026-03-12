from __future__ import annotations
from dataclasses import dataclass
from typing import List
from ..model.creative_model import CreativeModel
from ..boostsheets.boostsheet import BoostSheet


@dataclass
class OptimizationRequest:
    sheet: BoostSheet
    performance: dict
    diffs: list
    platform: str
    goal: str


@dataclass
class OptimizationResponse:
    revised_sheet: BoostSheet
    notes: List[str]


class OptimizationAgent:
    def __init__(self, model: CreativeModel):
        self.model = model

    def run(self, req: OptimizationRequest) -> OptimizationResponse:
        platform_token = f"<PLATFORM_{req.platform.upper()}>"
        goal_token = f"<GOAL_{req.goal.upper()}>"

        prompt = (
            f"{platform_token} {goal_token} <STAGE_HOOK>\n"
            f"Performance: {req.performance}\n"
            f"Diffs: {req.diffs}\n"
            f"Blocks: {req.sheet.blocks}\n"
            f"Suggest optimized revisions.\n"
        )
        suggestion = self.model.generate(prompt)
        req.sheet.add_agent_note(suggestion)
        req.sheet.add_history("OptimizationAgent applied suggestions")
        return OptimizationResponse(
            revised_sheet=req.sheet,
            notes=[suggestion]
        )
