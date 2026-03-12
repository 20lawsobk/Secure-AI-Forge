from __future__ import annotations
from typing import Dict, Any
from .boostsheets.boostsheet import BoostSheet


class RenderManager:
    def render_thumbnail(self, sheet: BoostSheet) -> Dict[str, Any]:
        sheet.add_history("Render requested for thumbnail")
        return {"status": "queued", "sheet_id": sheet.sheet_id, "type": "thumbnail"}

    def render_video(self, sheet: BoostSheet) -> Dict[str, Any]:
        sheet.add_history("Render requested for video")
        return {"status": "queued", "sheet_id": sheet.sheet_id, "type": "video"}
