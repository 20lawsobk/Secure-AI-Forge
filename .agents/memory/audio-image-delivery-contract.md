---
name: Audio/image client delivery contract
description: Quality requirements the external MaxBooster client enforces on rendered audio and image assets
---

# Audio delivery
- MP3s must be **stereo, 320 kbps CBR** (`-ac 2`, `-b:a 320k`) — VBR `-q:a` produced ~92 kbps mono files the client rejected. Applies to master_export, the plain-encode fallback, AND the ARC re-encode.
- Duration cap is 180 s (leaseable-beat length). The job-level render deadline and ffmpeg per-stage timeouts must scale with duration (deadline = max(120, dur*1.5+60)).
- Every stage in the chain must preserve stereo — ARC spectral clean denoises **per channel** (reshape interleaved → clean each → re-interleave); a single `-ac 1` anywhere collapses the master to mono.
- Fade-out must be musical (~1.5 s, scaled down for short clips), not a 0.3 s hard clip.

# Image delivery
- `render_text: false` on /api/generate/image → `ImageRequest.suppress_text`: skips headline, sub-label, style dots, divider, AND corner platform/artist labels — pure artwork for cover art.
- **Why:** clients pass long prompts that would render as literal typography.
- Gotcha: guarding one draw block can orphan variables used by later blocks (`sub_y` → style dots) — grep downstream uses before wrapping draws in a conditional.

# Auth
- `verify_api_key` and `verify_admin` accept `Authorization: Bearer <key>` alongside X-Api-Key/X-Admin-Key (some client proxies strip custom headers). Same verification path — no bypass.

# /platform/model/info
- Must answer <1 s: use the cached storage availability flag, never `storage.is_available` (lazy ping can block ~11 s when pdim is cold).
