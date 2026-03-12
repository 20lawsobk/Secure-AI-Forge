"""
patch_v2.py — One-shot fix for download_datasets.py on D: drive.
Run from: D:\ai_server>  python patch_v2.py
"""
import json, re, sys
from pathlib import Path

SCRIPT = Path(__file__).parent / 'download_datasets.py'
STATUS = Path(__file__).parent / 'knowledge' / 'dataset_download_status.json'

src = SCRIPT.read_text(encoding='utf-8')
orig = src

# ── 1. medley_solos → HuggingFace lostanlen/medley-solos-db ──────────────────
src = re.sub(
    r"Dataset\('medley_solos',\s+'http[^']*',\s+'[^']*'",
    "Dataset('medley_solos',   'hf',   'lostanlen/medley-solos-db'",
    src
)

# ── 2. NSynth filenames: add .jsonwav before .tar.gz ─────────────────────────
src = src.replace("'nsynth-train.tar.gz'", "'nsynth-train.jsonwav.tar.gz'")
src = src.replace("'nsynth-valid.tar.gz'", "'nsynth-valid.jsonwav.tar.gz'")
src = src.replace("'nsynth-test.tar.gz'",  "'nsynth-test.jsonwav.tar.gz'")
# Fix URL filenames too
src = src.replace('/nsynth-train.jsonwav.tar.gz',  '/nsynth-train.jsonwav.tar.gz')  # already correct if present
for old, new in [
    ("nsynth/nsynth-train.tar.gz", "nsynth/nsynth-train.jsonwav.tar.gz"),
    ("nsynth/nsynth-valid.tar.gz", "nsynth/nsynth-valid.jsonwav.tar.gz"),
    ("nsynth/nsynth-test.tar.gz",  "nsynth/nsynth-test.jsonwav.tar.gz"),
]:
    src = src.replace(old, new)

# ── 3. mtg_jamendo → Zenodo http_multi (MTG HF repo is private/missing) ─────
old_mtg = re.search(
    r"Dataset\('mtg_jamendo'.*?\),",
    src, re.DOTALL
)
new_mtg = (
    "Dataset('mtg_jamendo',    'http_multi', '',\n"
    "            est_gb=60.0,  music=True,  priority=1,\n"
    "            note='55K CC-licensed music tracks, genre/mood/instrument tags',\n"
    "            extra={'files': [\n"
    "                ('https://zenodo.org/records/3826813/files/raw_30s_segments_mp3.zip?download=1', 'raw_30s_segments_mp3.zip'),\n"
    "                ('https://zenodo.org/records/3826813/files/autotagging.tsv?download=1',          'autotagging.tsv'),\n"
    "                ('https://zenodo.org/records/3826813/files/tracks.tsv?download=1',               'tracks.tsv'),\n"
    "            ]}),"
)
if old_mtg:
    src = src[:old_mtg.start()] + new_mtg + src[old_mtg.end():]

# ── 4. Fix _load_status / _save_status — handle drive disconnection (OSError) ─
old_load = (
    "def _load_status() -> dict:\n"
    "    if STATUS_FILE.exists():\n"
    "        try:\n"
    "            return json.loads(STATUS_FILE.read_text())\n"
    "        except Exception:\n"
    "            pass\n"
    "    return {}"
)
new_load = (
    "def _load_status() -> dict:\n"
    "    try:\n"
    "        if STATUS_FILE.exists():\n"
    "            try:\n"
    "                return json.loads(STATUS_FILE.read_text())\n"
    "            except Exception:\n"
    "                pass\n"
    "    except OSError as e:\n"
    "        print(f'  [WARN] Cannot read status file (drive unavailable?): {e}', flush=True)\n"
    "    return {}"
)
if old_load in src:
    src = src.replace(old_load, new_load)

old_save = (
    "def _save_status(status: dict):\n"
    "    STATUS_FILE.write_text(json.dumps(status, indent=2, default=str))"
)
new_save = (
    "def _save_status(status: dict):\n"
    "    try:\n"
    "        STATUS_FILE.write_text(json.dumps(status, indent=2, default=str))\n"
    "    except OSError as e:\n"
    "        print(f'  [WARN] Cannot save status file (drive unavailable?): {e}', flush=True)"
)
if old_save in src:
    src = src.replace(old_save, new_save)

# ── 5. Fix premature download completion (HTTP closes early, file marked done) ─
old_verify = "    tmp.rename(filepath)\n    return filepath\n\ndef _extract"
new_verify = (
    "    # Verify full file received before finalizing\n"
    "    actual = tmp.stat().st_size\n"
    "    if total > 0 and actual < total:\n"
    "        raise Exception(\n"
    "            f'Incomplete download: received {actual/1e6:.0f} MB of {total/1e6:.0f} MB '\n"
    "            f'({actual/total*100:.1f}%) — connection closed early, will resume next run'\n"
    "        )\n\n"
    "    tmp.rename(filepath)\n"
    "    return filepath\n\ndef _extract"
)
if old_verify in src and "Incomplete download" not in src:
    src = src.replace(old_verify, new_verify)

# ── 6. Add --reset flag if missing ───────────────────────────────────────────
if "--reset" not in src:
    src = src.replace(
        "    parser.add_argument('--hf-token',",
        "    parser.add_argument('--reset',      type=str, default=None, metavar='NAME',\n"
        "                        help='Clear a dataset status so it will be retried')\n"
        "    parser.add_argument('--hf-token',",
    )
    src = src.replace(
        "    if args.hf_token:\n        HF_TOKEN = args.hf_token\n\n    if args.list:",
        "    if args.hf_token:\n        HF_TOKEN = args.hf_token\n\n"
        "    if args.reset:\n"
        "        import sys\n"
        "        s = _load_status()\n"
        "        if args.reset in s:\n"
        "            del s[args.reset]\n"
        "            _save_status(s)\n"
        "            print(f\"  Reset '{args.reset}' — will retry on next run\")\n"
        "        else:\n"
        "            print(f\"  '{args.reset}' not in status (already clear)\")\n"
        "        sys.exit(0)\n\n"
        "    if args.list:",
    )

# ── Write patched script ──────────────────────────────────────────────────────
if src != orig:
    SCRIPT.write_text(src, encoding='utf-8')
    print("  download_datasets.py patched successfully")
else:
    print("  No changes needed — script may already be up to date")

# ── 7. Clear broken status entries ───────────────────────────────────────────
TO_RESET = ['medley_solos', 'nsynth', 'mtg_jamendo', 'fma_large', 'fma_medium']
if STATUS.exists():
    try:
        s = json.loads(STATUS.read_text())
        removed = [k for k in TO_RESET if k in s]
        for k in removed:
            del s[k]
        STATUS.write_text(json.dumps(s, indent=2))
        if removed:
            print(f"  Status cleared for: {', '.join(removed)}")
        else:
            print("  Status entries already clear")
    except Exception as e:
        print(f"  [WARN] Could not update status file: {e}")
else:
    print("  Status file not found — nothing to clear")

print("\nDone. Run:  python download_datasets.py")
