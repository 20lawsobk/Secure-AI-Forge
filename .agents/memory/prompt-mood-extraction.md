---
name: Prompt-mood extraction across routes
description: Shared creative-direction keyword extractor for audio/content/text routes; cache-key and awareness-line rules
---
- One module-level extractor pulls creative-brief mood keywords ("dark", "cinematic", "drill"…) from raw prompt/instruction text; all generation routes (audio, social content, ad/text) rank these ABOVE live trending moods but BELOW explicit req.mood/tone.
- **Why:** trending buffer ("chill") was overriding explicit briefs ("dark phonk"); user confirmed prompt-first precedence is correct.
- **How to apply:** any new generation route that reads free-text creative direction must (1) use the shared extractor, not a local list; (2) surface moods as ONE authoritative `Trending moods:` line — strip pre-existing lines first, since the awareness parser is first-match; (3) include the raw direction text (prompt/instruction/extra_context/mood) in its dedup/coalescer key, or requests differing only in register collapse onto one cached result.
- Pydantic gotcha: extra body fields are silent no-ops — the request model must declare the `prompt` field or `getattr(req,"prompt",None)` is always None.
