"""
patch_v3.py — Fix medley_solos, nsynth, mtg_jamendo URLs + add "already downloaded" size check.
Run from: D:\ai_server>  python patch_v3.py
"""
import json, re, sys
from pathlib import Path

SCRIPT = Path(__file__).parent / 'download_datasets.py'
STATUS = Path(__file__).parent / 'knowledge' / 'dataset_download_status.json'

src = SCRIPT.read_text(encoding='utf-8')
orig = src

# ── 1. medley_solos: Zenodo 3464278 (soundata library confirmed; 3464302 = 404) ──
src = re.sub(
    r"Dataset\('medley_solos',\s+'hf',\s+'[^']*'",
    "Dataset('medley_solos',   'http', 'https://zenodo.org/records/3464278/files/Medley-solos-DB.tar.gz?download=1'",
    src,
)
# Also catch if it still has the old HF form with lostanlen
src = re.sub(
    r"Dataset\('medley_solos',\s+'hf',\s+'lostanlen[^']*'",
    "Dataset('medley_solos',   'http', 'https://zenodo.org/records/3464278/files/Medley-solos-DB.tar.gz?download=1'",
    src,
)

# ── 2. NSynth: tensorflow.org CDN (GCS magentadata bucket returns 404) ──────
src = src.replace(
    "https://storage.googleapis.com/magentadata/datasets/nsynth/nsynth-train.jsonwav.tar.gz",
    "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz",
)
src = src.replace(
    "https://storage.googleapis.com/magentadata/datasets/nsynth/nsynth-valid.jsonwav.tar.gz",
    "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz",
)
src = src.replace(
    "https://storage.googleapis.com/magentadata/datasets/nsynth/nsynth-test.jsonwav.tar.gz",
    "http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz",
)

# ── 3. mtg_jamendo: Zenodo requires account login — skip with clear note ────
old_mtg = re.search(r"Dataset\('mtg_jamendo'.*?\),", src, re.DOTALL)
new_mtg = (
    "# MTG-Jamendo: Zenodo (record 3826813) requires an account login to download.\n"
    "    # Download manually from https://mtg.upf.edu/download/datasets/mtg-jamendo\n"
    "    # then place files in D:\\ai_server\\datasets\\mtg_jamendo\\\n"
    "    Dataset('mtg_jamendo',    'skip', '',\n"
    "            est_gb=60.0,  music=True,  priority=1,\n"
    "            note='55K CC-licensed music tracks — MANUAL DOWNLOAD REQUIRED (Zenodo login)'),\n"
    "    #"  # closing comment to keep surrounding formatting
)
if old_mtg:
    # simpler: just replace the method and url, keep other fields
    src = re.sub(
        r"Dataset\('mtg_jamendo',\s+'[^']+',\s+'[^']*'",
        "Dataset('mtg_jamendo',    'skip', ''",
        src,
    )
    src = re.sub(
        r"(Dataset\('mtg_jamendo'.*?note=')[^']*(')",
        r"\g<1>55K CC-licensed music tracks — MANUAL DOWNLOAD REQUIRED (Zenodo needs login)\g<2>",
        src, flags=re.DOTALL,
    )
    # Remove the extra={'files': [...]} block if present
    src = re.sub(
        r"(Dataset\('mtg_jamendo'.*?priority=1,\s*\n\s*note=[^)]+),\s*\n\s*extra=\{[^}]+\}\)",
        r"\g<1>)",
        src, flags=re.DOTALL,
    )

# ── 4. Add 'skip' handler to the dispatcher if missing ───────────────────────
if "'skip'" not in src and "ds.method == 'skip'" not in src:
    # Find where the method dispatch happens (usually in a run() function)
    src = re.sub(
        r"(elif ds\.method == 'ytdlp')",
        "elif ds.method == 'skip':\n"
        "        print(f\"  [SKIP] {ds.name}: {ds.note}\", flush=True)\n"
        "        ok = False\n"
        "    \\1",
        src,
    )

# ── 5. Add size verification when file already exists (prevents stale partials) ─
old_exists = (
    "    if filepath.exists():\n"
    "        print(f\"  Already downloaded: {fname}\", flush=True)\n"
    "        return filepath"
)
new_exists = (
    "    if filepath.exists():\n"
    "        # Verify size matches server to catch incomplete files that were wrongly finalized\n"
    "        local_size = filepath.stat().st_size\n"
    "        try:\n"
    "            head_req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}, method='HEAD')\n"
    "            with urllib.request.urlopen(head_req, timeout=30) as r:\n"
    "                server_size = int(r.headers.get('Content-Length', 0))\n"
    "            if server_size > 0 and local_size < server_size * 0.99:\n"
    "                print(f\"  [WARN] {fname}: local {local_size/1e9:.2f} GB vs server {server_size/1e9:.2f} GB — incomplete, re-downloading\", flush=True)\n"
    "                filepath.unlink()\n"
    "            else:\n"
    "                print(f\"  Already downloaded: {fname} ({local_size/1e9:.2f} GB)\", flush=True)\n"
    "                return filepath\n"
    "        except Exception:\n"
    "            print(f\"  Already downloaded: {fname}\", flush=True)\n"
    "            return filepath"
)
if old_exists in src:
    src = src.replace(old_exists, new_exists)

# ── Write patched script ──────────────────────────────────────────────────────
if src != orig:
    SCRIPT.write_text(src, encoding='utf-8')
    print("  download_datasets.py patched successfully")
else:
    print("  No changes needed")

# ── 6. Clear broken status entries so they retry ─────────────────────────────
TO_RESET = ['medley_solos', 'nsynth', 'mtg_jamendo', 'fma_large', 'fma_medium']
if STATUS.exists():
    try:
        s = json.loads(STATUS.read_text())
        removed = [k for k in TO_RESET if k in s]
        for k in removed:
            del s[k]
        STATUS.write_text(json.dumps(s, indent=2))
        print(f"  Status cleared for: {', '.join(removed) if removed else '(already clear)'}")
    except Exception as e:
        print(f"  [WARN] Could not update status file: {e}")

print("""
  IMPORTANT — delete the incomplete FMA files before re-running:
    del D:\\ai_server\\datasets\\fma_medium\\fma_medium.zip
    del D:\\ai_server\\datasets\\fma_large\\fma_large.zip
  (These are leftover partial downloads wrongly marked as complete)

Done. Then run:  python download_datasets.py
""")
