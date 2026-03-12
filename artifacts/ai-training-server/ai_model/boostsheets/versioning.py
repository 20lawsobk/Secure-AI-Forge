from __future__ import annotations
from typing import List, Dict, Any
import copy
import time
from .boostsheet import BoostSheet


class BoostSheetVersioning:
    def __init__(self):
        self.versions: List[Dict[str, Any]] = []

    def snapshot(self, sheet: BoostSheet) -> Dict[str, Any]:
        version = {
            "timestamp": time.time(),
            "sheet": copy.deepcopy(sheet.__dict__)
        }
        self.versions.append(version)
        return version


def diff_sheets(old: BoostSheet, new: BoostSheet) -> List[Dict[str, Any]]:
    diffs = []
    for field_name in vars(old):
        old_val = getattr(old, field_name)
        new_val = getattr(new, field_name)
        if old_val != new_val:
            diffs.append({
                "field": field_name,
                "old": old_val,
                "new": new_val
            })
    return diffs
