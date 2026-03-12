from __future__ import annotations
from ..boostsheets.boostsheet import BoostSheet
from ..boostsheets.repository import BoostSheetRepository


class UrlToBoostSheetAdapter:
    def __init__(self, repo: BoostSheetRepository):
        self.repo = repo

    def create_from_url_result(self, result: dict) -> BoostSheet:
        sheet_type = "video_sheet" if result.get("format") == "video" else "script_sheet"

        sheet = BoostSheet(
            type=sheet_type,
            platform=result.get("platform", "tiktok"),
            goal="growth",
            tone="default",
            source_url=result.get("url")
        )

        sheet.blocks["text"].append({
            "id": "t1",
            "content": result.get("content", ""),
            "role": "primary_script"
        })

        if result.get("format") == "video":
            sheet.blocks["video"].append({
                "id": "v1",
                "status": "not_generated",
                "engine": "in_browser_canvas",
                "templates_supported": [
                    "teaser", "release_announcement", "quote", "lyric"
                ],
                "aspect_ratios": ["9:16", "1:1", "16:9"]
            })

        sheet.add_history("Created from URL content result")
        self.repo.save(sheet)
        return sheet
