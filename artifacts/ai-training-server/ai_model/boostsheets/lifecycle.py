from __future__ import annotations
from .boostsheet import BoostSheet


class BoostSheetLifecycle:
    VALID_STATES = [
        "draft",
        "generated_by_agent",
        "refined_by_agent",
        "refined_by_user",
        "optimized",
        "ready_for_render",
        "rendered",
        "scheduled",
        "published",
        "performance_logged",
    ]

    def __init__(self, sheet: BoostSheet):
        self.sheet = sheet
        self.state = "draft"
        self.sheet.add_history("State initialized: draft")

    def transition(self, new_state: str):
        if new_state not in self.VALID_STATES:
            raise ValueError(f"Invalid state: {new_state}")
        self.state = new_state
        self.sheet.add_history(f"State changed to {new_state}")
