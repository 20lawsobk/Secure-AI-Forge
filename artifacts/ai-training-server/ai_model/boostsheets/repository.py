from __future__ import annotations
from typing import List
import json
import os
from .boostsheet import BoostSheet


class BoostSheetRepository:
    def __init__(self, path: str = "boostsheets_db"):
        self.path = path
        os.makedirs(path, exist_ok=True)

    def save(self, sheet: BoostSheet):
        with open(f"{self.path}/{sheet.sheet_id}.json", "w") as f:
            json.dump(sheet.__dict__, f, indent=2)

    def load(self, sheet_id: str) -> BoostSheet:
        with open(f"{self.path}/{sheet_id}.json", "r") as f:
            data = json.load(f)
        return BoostSheet(**data)

    def list_ids(self) -> List[str]:
        return [
            f.replace(".json", "")
            for f in os.listdir(self.path)
            if f.endswith(".json")
        ]
