"""
D: Drive Dataset Downloader
============================
Run this on your local machine to download all large training datasets
directly to D:\\ai_server\\datasets\\

Usage:
    cd D:\\ai_server
    python download_datasets.py

Options:
    python download_datasets.py --list             # Show all datasets + status
    python download_datasets.py --only music       # Music datasets only
    python download_datasets.py --only video       # Video datasets only
    python download_datasets.py --name nsynth      # One specific dataset
    python download_datasets.py --skip-large       # Skip anything over 100 GB
    python download_datasets.py --hf-token TOKEN   # HF token for gated repos
    python download_datasets.py --reset NAME       # Clear a dataset so it retries
    python download_datasets.py --threads 16       # Parallel streams per file (default 8)
    python download_datasets.py --parallel 3       # Concurrent datasets at once (default 1)

Requirements:
    pip install huggingface_hub datasets requests tqdm

All downloads resume if interrupted. Already-complete datasets are skipped.
"""

import argparse
import json
import shutil
import subprocess
import sys
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone

from pathlib import Path
from typing import List, Optional

# ── Speed tuning (override via CLI: --threads N  --parallel N) ───────────────
CHUNK_THREADS    = 8   # parallel byte-range streams per file (set 1 to disable)
PARALLEL_DATASETS = 1  # concurrent datasets downloading at once

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).resolve().parent
DATASETS    = ROOT / "datasets"
STATUS_FILE = ROOT / "knowledge" / "dataset_download_status.json"

DATASETS.mkdir(parents=True, exist_ok=True)
STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)

# ── Dataset definitions ───────────────────────────────────────────────────────

@dataclass
class Dataset:
    name:     str
    method:   str        # 'hf' | 'http' | 'http_multi' | 'ytdlp'
    source:   str
    est_gb:   float
    music:    bool
    priority: int        # 1=high 2=med 3=low
    note:     str = ""
    extra:    dict = field(default_factory=dict)


DATASETS_PLAN: List[Dataset] = [

    # ── Small/medium — verified public HF repos ────────────────────────────
    Dataset('gtzan',          'hf',   'marsyas/gtzan',
            est_gb=1.5,   music=True,  priority=1,
            note='Genre classification, 1K clips across 10 genres'),

    Dataset('musiccaps',      'hf',   'google/MusicCaps',
            est_gb=0.1,   music=True,  priority=1,
            note='5K high-quality music captions from Google'),

    # MagnaTagATune — direct HTTP from MIRLAB (more reliable than HF mirrors)
    Dataset('magnatagatune',  'http_multi',
            'https://mirg.city.ac.uk/datasets/magnatagatune/',
            est_gb=4.5,   music=True,  priority=1,
            note='25K clips, 188 mood/style/instrument tags',
            extra={'files': [
                ('https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.001', 'mp3.zip.001'),
                ('https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.002', 'mp3.zip.002'),
                ('https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip.003', 'mp3.zip.003'),
                ('https://mirg.city.ac.uk/datasets/magnatagatune/annotations_final.csv', 'annotations_final.csv'),
                ('https://mirg.city.ac.uk/datasets/magnatagatune/clip_info_final.csv', 'clip_info_final.csv'),
            ]}),

    # Medley-Solos-DB — Zenodo 3464278 (confirmed by soundata library; 3464302 = wrong record)
    Dataset('medley_solos',   'http', 'https://zenodo.org/records/3464278/files/Medley-solos-DB.tar.gz?download=1',
            est_gb=0.8,   music=True,  priority=1,
            note='21K clips across 8 instrument classes'),

    Dataset('emopia',         'hf',   'Nyanko7/emopia',
            est_gb=1.5,   music=True,  priority=2,
            note='Piano MIDI with emotion quadrants (valence/arousal)'),

    # NSynth — tensorflow.org CDN mirror (GCS magentadata bucket returns 404)
    Dataset('nsynth',         'http_multi', '',
            est_gb=22.0,  music=True,  priority=1,
            note='300K annotated musical notes — instrument + pitch conditioning',
            extra={'files': [
                ('http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz', 'nsynth-train.jsonwav.tar.gz'),
                ('http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz', 'nsynth-valid.jsonwav.tar.gz'),
                ('http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz',  'nsynth-test.jsonwav.tar.gz'),
            ]}),

    Dataset('maestro_v3',     'http', 'https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip',
            est_gb=120.0, music=True,  priority=1,
            note='200hrs piano audio + MIDI annotation pairs'),

    Dataset('fma_large',      'http', 'https://os.unil.cloud.switch.ch/fma/fma_large.zip',
            est_gb=93.0,  music=True,  priority=1,
            note='30K tracks across 161 genres — largest free music collection'),

    Dataset('fma_medium',     'http', 'https://os.unil.cloud.switch.ch/fma/fma_medium.zip',
            est_gb=22.0,  music=True,  priority=1,
            note='25K tracks, balanced across genres'),

    # ── Large video/visual ──────────────────────────────────────────────────
    Dataset('diffusiondb',    'hf',   'poloclub/diffusiondb',
            est_gb=88.0,  music=False, priority=2,
            note='1M Stable Diffusion prompts + images — visual quality training',
            extra={'config': 'large_random_1k'}),

    # Kinetics-700 — correct HF repo ID (was missing username prefix)
    Dataset('kinetics700',    'hf',   'HuggingFaceM4/kinetics700-2020',
            est_gb=450.0, music=False, priority=1,
            note='700 action classes, 650K clips — primary motion teacher'),

    Dataset('vggsound',       'hf',   'Loie/VGGSound',
            est_gb=450.0, music=True,  priority=2,
            note='200K clips with audio-visual labels — strong AV correspondence'),

    Dataset('audioset',       'hf',   'agkphysics/AudioSet',
            est_gb=2000.0,music=True,  priority=2,
            note='2M clips, 527 sound classes — audio conditioning backbone'),

    Dataset('webvid10m',      'hf',   'TempoFunk/webvid-10M',
            est_gb=2500.0,music=True,  priority=3,
            note='10M video+caption pairs — text-to-video alignment',
            extra={'streaming': True}),

    Dataset('laion_aesthetics','hf',  'laion/laion2B-en-aesthetic',
            est_gb=240.0, music=False, priority=2,
            note='2.3B image-text pairs at aesthetic score ≥5 — visual quality',
            extra={'streaming': True}),

    # MTG-Jamendo — Zenodo record 3826813 requires account login; HF repo is private.
    # Download manually: https://mtg.upf.edu/download/datasets/mtg-jamendo
    # Then place files in D:\ai_server\datasets\mtg_jamendo\
    Dataset('mtg_jamendo',    'skip', '',
            est_gb=60.0,  music=True,  priority=1,
            note='55K CC-licensed music tracks — MANUAL DOWNLOAD REQUIRED (Zenodo login needed)'),

    # ── yt-dlp music video sets ─────────────────────────────────────────────
    Dataset('musicvideo_hq',  'ytdlp', 'official music video HD 4K 2023 2024',
            est_gb=500.0, music=True,  priority=2,
            note='10K high-quality official music videos',
            extra={'max_clips': 10000, 'duration': 30}),

    Dataset('concert_live',   'ytdlp', 'live concert performance stadium arena HD',
            est_gb=200.0, music=True,  priority=3,
            note='Stage lighting, crowd dynamics, live performance',
            extra={'max_clips': 4000, 'duration': 30}),

    Dataset('albumart_lofi',  'ytdlp', 'album art aesthetic lofi chill music visualizer 4k',
            est_gb=50.0,  music=True,  priority=2,
            note='Lofi aesthetic + album art visualiser videos',
            extra={'max_clips': 2000, 'duration': 15}),
]

# ── Status tracking ───────────────────────────────────────────────────────────

def _load_status() -> dict:
    try:
        if STATUS_FILE.exists():
            try:
                return json.loads(STATUS_FILE.read_text())
            except Exception:
                pass
    except OSError as e:
        print(f"  [WARN] Cannot read status file (drive unavailable?): {e}", flush=True)
    return {}

def _save_status(status: dict):
    try:
        STATUS_FILE.write_text(json.dumps(status, indent=2, default=str))
    except OSError as e:
        print(f"  [WARN] Cannot save status file (drive unavailable?): {e}", flush=True)

def _mark(name: str, state: str, note: str = ""):
    status = _load_status()
    status[name] = {'state': state, 'ts': datetime.now(timezone.utc).isoformat(), 'note': note}
    _save_status(status)

def _is_done(name: str) -> bool:
    s = _load_status().get(name, {})
    return s.get('state') == 'done'

# ── Helpers ───────────────────────────────────────────────────────────────────

def _run(cmd: List[str], cwd: Optional[Path] = None) -> int:
    print(f"\n  $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=cwd)
    return result.returncode

def _ensure_pkg(pkg: str, import_name: Optional[str] = None):
    mod = import_name or pkg.replace('-', '_')
    try:
        _ = __import__(mod)
        # Verify the package is functional (datasets sometimes installs broken)
        if mod == 'datasets':
            from datasets import load_dataset  # noqa: F401
    except (ImportError, Exception):
        print(f"  Installing/repairing {pkg}...", flush=True)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', pkg])

_PRINT_LOCK = threading.Lock()

def _print(msg: str, end: str = '\n'):
    with _PRINT_LOCK:
        print(msg, end=end, flush=True)

def _http_head(url: str) -> tuple[int, bool]:
    """Return (content_length, accepts_ranges) via HEAD request."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}, method='HEAD')
        with urllib.request.urlopen(req, timeout=30) as r:
            total = int(r.headers.get('Content-Length', 0))
            ranges = r.headers.get('Accept-Ranges', '').strip().lower() == 'bytes'
            return total, ranges
    except Exception:
        return 0, False

def _http_download_chunk(url: str, start: int, end: int, part_path: Path,
                          counter: list, lock: threading.Lock, total: int):
    """Download a single byte-range chunk to part_path. Updates counter[0] with bytes done."""
    existing = part_path.stat().st_size if part_path.exists() else 0
    expected = end - start + 1
    if existing == expected:
        with lock:
            counter[0] += existing
        return  # already complete

    resume = existing
    headers = {
        'User-Agent': 'Mozilla/5.0',
        'Range': f'bytes={start + resume}-{end}',
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=180) as resp:
        mode = 'ab' if resume else 'wb'
        with open(part_path, mode) as f:
            buf_size = 8 * 1024 * 1024  # 8 MB read buffer
            while True:
                buf = resp.read(buf_size)
                if not buf:
                    break
                f.write(buf)
                with lock:
                    counter[0] += len(buf)
                    done = counter[0]
                    pct  = done / total * 100
                _print(f"\r  {done/1e6:.0f} MB / {total/1e6:.0f} MB  ({pct:.1f}%)  [{CHUNK_THREADS} streams]", end='')

def _http_download_parallel(url: str, dest: Path, fname: str, total: int) -> Path:
    """Split file into CHUNK_THREADS equal byte-range parts, download in parallel, then concatenate."""
    filepath = dest / fname
    tmp      = filepath.with_suffix(filepath.suffix + '.tmp')
    part_dir = dest / f'.{fname}.parts'
    part_dir.mkdir(parents=True, exist_ok=True)

    chunk_size = (total + CHUNK_THREADS - 1) // CHUNK_THREADS
    ranges = [(i * chunk_size, min((i + 1) * chunk_size - 1, total - 1)) for i in range(CHUNK_THREADS)]
    parts  = [part_dir / f'part{i:04d}' for i in range(len(ranges))]

    counter: list = [sum(p.stat().st_size for p in parts if p.exists())]
    lock = threading.Lock()

    _print(f"  Parallel download: {CHUNK_THREADS} streams × {chunk_size/1e6:.0f} MB", end='')
    _print('')

    with ThreadPoolExecutor(max_workers=CHUNK_THREADS) as ex:
        futures = {
            ex.submit(_http_download_chunk, url, s, e, p, counter, lock, total): i
            for i, ((s, e), p) in enumerate(zip(ranges, parts))
        }
        for f in as_completed(futures):
            exc = f.exception()
            if exc:
                raise exc
    _print('')

    # Concatenate parts → tmp → filepath
    _print(f"  Assembling {fname}...", end='')
    with open(tmp, 'wb') as out:
        for p in parts:
            out.write(p.read_bytes())
    _print(' done')

    actual = tmp.stat().st_size
    if actual < total * 0.99:
        raise Exception(f"Assembly incomplete: {actual/1e9:.2f} GB of {total/1e9:.2f} GB")

    tmp.rename(filepath)
    shutil.rmtree(part_dir, ignore_errors=True)
    return filepath

def _http_download_sequential(url: str, dest: Path, fname: str,
                               total: int, resume_pos: int = 0) -> Path:
    """Single-stream download with resume and progress bar."""
    filepath = dest / fname
    tmp      = filepath.with_suffix(filepath.suffix + '.tmp')

    headers = {'User-Agent': 'Mozilla/5.0'}
    if resume_pos > 0:
        headers['Range'] = f'bytes={resume_pos}-'
        _print(f"  Resuming {fname} from {resume_pos/1e6:.0f} MB...")

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            total_raw = int(resp.headers.get('Content-Length', 0))
            total     = total_raw + resume_pos if resume_pos else (total or total_raw)
            done      = resume_pos
            mode      = 'ab' if resume_pos else 'wb'
            buf_size  = 8 * 1024 * 1024  # 8 MB read buffer
            with open(tmp, mode) as f:
                while True:
                    buf = resp.read(buf_size)
                    if not buf:
                        break
                    f.write(buf)
                    done += len(buf)
                    if total:
                        _print(f"\r  {done/1e6:.0f} MB / {total/1e6:.0f} MB  ({done/total*100:.1f}%)", end='')
        _print('')
    except urllib.error.HTTPError as e:
        if e.code == 416 and tmp.exists():
            _print('\n  Already complete (416), finalizing...')
            tmp.rename(filepath)
            return filepath
        _print(f'\n  [WARN] Download interrupted: {e}')
        raise
    except Exception as e:
        _print(f'\n  [WARN] Download interrupted: {e}')
        raise

    actual = tmp.stat().st_size
    if total > 0 and actual < total * 0.99:
        raise Exception(
            f"Incomplete download: received {actual/1e6:.0f} MB of {total/1e6:.0f} MB "
            f"({actual/total*100:.1f}%) — connection closed early, will resume next run"
        )
    tmp.rename(filepath)
    return filepath

def _http_download(url: str, dest: Path, filename: Optional[str] = None) -> Path:
    """Download a file. Uses parallel chunked streams for large files; sequential otherwise."""
    dest.mkdir(parents=True, exist_ok=True)
    fname    = filename or url.split('/')[-1].split('?')[0]
    filepath = dest / fname
    tmp      = filepath.with_suffix(filepath.suffix + '.tmp')

    # ── Existing file: verify size against server ─────────────────────────────
    if filepath.exists():
        local_size = filepath.stat().st_size
        try:
            server_size, _ = _http_head(url)
            if server_size > 0 and local_size < server_size * 0.99:
                _print(f"  [WARN] {fname}: local {local_size/1e9:.2f} GB vs server {server_size/1e9:.2f} GB — resuming")
                filepath.rename(tmp)  # demote to .tmp so it resumes
            else:
                _print(f"  Already downloaded: {fname} ({local_size/1e9:.2f} GB)")
                return filepath
        except Exception:
            _print(f"  Already downloaded: {fname}")
            return filepath

    # ── Decide: parallel chunked vs sequential ────────────────────────────────
    PARALLEL_MIN = 50 * 1024 * 1024  # 50 MB threshold
    total, accepts_ranges = _http_head(url)

    if total > PARALLEL_MIN and accepts_ranges and CHUNK_THREADS > 1 and not tmp.exists():
        # Fresh large file → parallel chunked download
        return _http_download_parallel(url, dest, fname, total)

    # Sequential (small file, no Range support, or resuming an existing .tmp)
    resume_pos = tmp.stat().st_size if tmp.exists() else 0
    return _http_download_sequential(url, dest, fname, total, resume_pos)

def _extract(filepath: Path, dest: Path):
    """Extract zip or tar archives."""
    name = filepath.name
    if name.endswith('.zip') or name.endswith('.zip.001'):
        print(f"  Extracting {name}...", flush=True)
        import zipfile
        try:
            with zipfile.ZipFile(filepath) as z:
                z.extractall(dest)
        except zipfile.BadZipFile:
            print(f"  [WARN] {name} may be a multi-part zip — skipping extraction (join parts manually)", flush=True)
    elif name.endswith(('.tar.gz', '.tgz')):
        print(f"  Extracting {name}...", flush=True)
        import tarfile
        with tarfile.open(filepath) as t:
            t.extractall(dest)
    print(f"  Extracted to {dest}", flush=True)

# ── Downloaders ───────────────────────────────────────────────────────────────

HF_TOKEN: Optional[str] = None  # set via --hf-token arg


def _download_hf(ds: Dataset, dest: Path):
    """Download from HuggingFace Hub using huggingface_hub."""
    _ensure_pkg('huggingface_hub')
    from huggingface_hub import snapshot_download

    config   = ds.extra.get('config')
    streaming = ds.extra.get('streaming', False)

    if streaming:
        print(f"  [{ds.name}] Large streaming dataset — downloading metadata + first shard only.", flush=True)

    kwargs = dict(
        repo_id=ds.source,
        repo_type='dataset',
        local_dir=str(dest),
        ignore_patterns=['*.bin', '*.pt', '*.pth', '__pycache__/*'],
    )
    if HF_TOKEN:
        kwargs['token'] = HF_TOKEN

    try:
        snapshot_download(**kwargs)
        return True
    except Exception as e:
        print(f"  [WARN] snapshot_download failed: {e}", flush=True)
        # Fallback: try datasets library
        try:
            _ensure_pkg('datasets')
            from datasets import load_dataset
            kw: dict = {'split': 'train', 'streaming': True, 'trust_remote_code': True}
            if config:
                kw['name'] = config
            if HF_TOKEN:
                kw['token'] = HF_TOKEN
            dataset = load_dataset(ds.source, **kw)
            dest.mkdir(parents=True, exist_ok=True)
            manifest = dest / 'manifest.jsonl'
            n = 0
            with open(manifest, 'w') as f:
                for row in dataset:
                    f.write(json.dumps(row) + '\n')
                    n += 1
                    if n >= 1000:
                        break
            print(f"  Saved {n} rows manifest to {manifest}", flush=True)
            return True
        except Exception as e2:
            print(f"  [WARN] datasets fallback also failed: {e2}", flush=True)
            return False


def _download_http(ds: Dataset, dest: Path):
    """Download a single file via direct HTTP URL."""
    dest.mkdir(parents=True, exist_ok=True)
    url      = ds.source
    filename = url.split('/')[-1].split('?')[0] or f'{ds.name}.zip'

    print(f"  Downloading {filename} ({ds.est_gb:.0f} GB)...", flush=True)
    try:
        filepath = _http_download(url, dest, filename)
    except Exception as e:
        print(f"  [WARN] Download failed: {e}", flush=True)
        return False

    try:
        _extract(filepath, dest)
    except Exception as e:
        print(f"  [WARN] Extraction failed (file still saved): {e}", flush=True)
    return True


def _download_http_multi(ds: Dataset, dest: Path):
    """Download multiple files in parallel, then join any multi-part archives."""
    dest.mkdir(parents=True, exist_ok=True)
    files = ds.extra.get('files', [])
    if not files:
        _print(f"  [WARN] No files defined for {ds.name}")
        return False

    _print(f"  Downloading {len(files)} files in parallel...")
    results: dict[str, bool] = {}

    def _dl_one(url_fname):
        url, fname = url_fname
        _print(f"  → {fname}")
        try:
            _http_download(url, dest, fname)
            return fname, True
        except Exception as e:
            _print(f"  [WARN] Failed to download {fname}: {e}")
            return fname, False

    with ThreadPoolExecutor(max_workers=min(len(files), CHUNK_THREADS)) as ex:
        for fname, ok in ex.map(_dl_one, files):
            results[fname] = ok

    all_ok = all(results.values())

    # Extract non-split archives
    for _, fname in files:
        if results.get(fname) and not fname.endswith(('.001', '.002', '.003')):
            fp = dest / fname
            if fp.exists():
                try:
                    _extract(fp, dest)
                except Exception:
                    pass

    # Join and extract multi-part zips (zip.001 + zip.002 + ...)
    for _, fname in files:
        if not fname.endswith('.001'):
            continue
        base     = fname[:-4]
        out_zip  = dest / base
        part_files = sorted(dest.glob(base + '.*'))
        if len(part_files) > 1 and not out_zip.exists():
            _print(f"  Joining multi-part archive: {out_zip.name}...")
            try:
                with open(out_zip, 'wb') as out_f:
                    for pf in part_files:
                        out_f.write(pf.read_bytes())
                _extract(out_zip, dest)
            except Exception as e:
                _print(f"  [WARN] Multi-part join failed: {e}")

    return all_ok


def _download_ytdlp(ds: Dataset, dest: Path):
    """Download via yt-dlp."""
    try:
        import yt_dlp as _  # noqa: F401
    except ImportError:
        rc = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'yt-dlp'])
        if rc.returncode != 0:
            print("  [WARN] Could not install yt-dlp. Skipping.", flush=True)
            return False

    dest.mkdir(parents=True, exist_ok=True)
    max_clips = ds.extra.get('max_clips', 500)
    duration  = ds.extra.get('duration', 20)

    cmd = [
        sys.executable, '-m', 'yt_dlp',
        f'ytsearch{max_clips}:{ds.source}',
        '--output', str(dest / '%(id)s.%(ext)s'),
        '--format', 'bestvideo[height<=720]+bestaudio/best[height<=720]',
        '--merge-output-format', 'mp4',
        '--max-downloads', str(max_clips),
        '--match-filter', f'duration <= {duration * 3}',
        '--no-playlist',
        '--ignore-errors',
        '--no-warnings',
        '--quiet',
        '--progress',
    ]
    rc = _run(cmd)
    return rc == 0


# ── Main download loop ────────────────────────────────────────────────────────

def run(
    filter_type: Optional[str] = None,
    only_name:   Optional[str] = None,
    skip_large:  bool = False,
    dry_run:     bool = False,
):
    targets = DATASETS_PLAN

    if only_name:
        targets = [d for d in targets if d.name == only_name]
        if not targets:
            print(f"Dataset '{only_name}' not found. Run --list to see all.", flush=True)
            return

    if filter_type == 'music':
        targets = [d for d in targets if d.music]
    elif filter_type == 'video':
        targets = [d for d in targets if not d.music]

    if skip_large:
        targets = [d for d in targets if d.est_gb <= 100]

    # Sort by priority then size
    targets.sort(key=lambda d: (d.priority, d.est_gb))

    total_gb = sum(d.est_gb for d in targets)
    present  = sum(1 for d in targets if _is_done(d.name))

    print()
    print("=" * 65)
    print("  MaxCore D: Drive Dataset Downloader")
    print("=" * 65)
    print(f"  Target dir : {DATASETS}")
    print(f"  Datasets   : {len(targets)} ({present} already done)")
    print(f"  Est. total : {total_gb/1024:.1f} TB  ({total_gb:.0f} GB)")
    if HF_TOKEN:
        print("  HF Token   : set (authenticated)")
    else:
        print("  HF Token   : not set (gated repos may fail — use --hf-token)")
    print()

    if dry_run:
        print("  DRY RUN — no downloads will be made\n")

    def _process_one(ds: Dataset):
        dest = DATASETS / ds.name

        if _is_done(ds.name):
            _print(f"  ✓ {ds.name:<25} already complete — skipping")
            return

        size_str = f"{ds.est_gb:.0f} GB" if ds.est_gb < 1000 else f"{ds.est_gb/1024:.1f} TB"
        _print(f"\n{'='*65}")
        _print(f"  [{ds.name}]  ~{size_str}  (priority {ds.priority})")
        _print(f"  {ds.note}")

        if dry_run:
            _print(f"  DRY RUN: would download via {ds.method}")
            return

        _mark(ds.name, 'downloading')
        ok = False
        try:
            if ds.method == 'skip':
                _print(f"  [SKIP] Requires manual download — {ds.note}")
                ok = False
            elif ds.method == 'hf':
                ok = _download_hf(ds, dest)
            elif ds.method == 'http':
                ok = _download_http(ds, dest)
            elif ds.method == 'http_multi':
                ok = _download_http_multi(ds, dest)
            elif ds.method == 'ytdlp':
                ok = _download_ytdlp(ds, dest)
        except KeyboardInterrupt:
            _print(f"\n  Interrupted during {ds.name}. Progress saved.")
            _mark(ds.name, 'interrupted')
            sys.exit(0)
        except Exception as e:
            _print(f"  [ERROR] {ds.name}: {e}")
            ok = False

        if ok:
            _mark(ds.name, 'done')
            _print(f"  ✓ {ds.name} complete")
        else:
            _mark(ds.name, 'failed', 'download error')
            _print(f"  ✗ {ds.name} failed — check connection and retry")

    if PARALLEL_DATASETS > 1:
        _print(f"  Parallel mode: up to {PARALLEL_DATASETS} datasets simultaneously\n")
        with ThreadPoolExecutor(max_workers=PARALLEL_DATASETS) as ex:
            futures = [ex.submit(_process_one, ds) for ds in targets]
            for f in as_completed(futures):
                exc = f.exception()
                if exc and not isinstance(exc, SystemExit):
                    _print(f"  [ERROR] {exc}")
    else:
        for ds in targets:
            _process_one(ds)

    _print(f"\n{'='*65}")
    _print("  All downloads complete.")
    _print(f"  Datasets saved to: {DATASETS}")


def list_datasets():
    status = _load_status()
    print()
    print(f"{'Name':<25} {'Size':>8}  {'Priority'}  {'Status':<15}  Notes")
    print("-" * 80)
    total = 0.0
    for ds in sorted(DATASETS_PLAN, key=lambda d: (d.priority, d.est_gb)):
        st    = status.get(ds.name, {}).get('state', 'pending')
        icon  = '✓' if st == 'done' else ('↻' if st == 'downloading' else '·')
        size  = f"{ds.est_gb:.0f}GB" if ds.est_gb < 1000 else f"{ds.est_gb/1024:.1f}TB"
        total += ds.est_gb
        print(f"  {icon} {ds.name:<23} {size:>8}  P{ds.priority}        {st:<15}  {ds.note[:40]}")
    print("-" * 80)
    print(f"  Total: {total/1024:.1f} TB")
    print()


# ── Entry ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download large training datasets to D: drive')
    parser.add_argument('--list',       action='store_true',  help='List all datasets and status')
    parser.add_argument('--only',       type=str, default=None, metavar='TYPE',
                        help='Filter: music | video')
    parser.add_argument('--name',       type=str, default=None, metavar='NAME',
                        help='Download a single dataset by name')
    parser.add_argument('--skip-large', action='store_true',  help='Skip datasets over 100 GB')
    parser.add_argument('--dry-run',    action='store_true',  help='Show plan without downloading')
    parser.add_argument('--hf-token',   type=str, default=None, metavar='TOKEN',
                        help='HuggingFace access token for gated/private repos')
    parser.add_argument('--reset',      type=str, default=None, metavar='NAME',
                        help='Clear a dataset from the done-status so it will be retried (e.g. --reset fma_large)')
    parser.add_argument('--threads',    type=int, default=None, metavar='N',
                        help=f'Parallel byte-range streams per file (default: {CHUNK_THREADS})')
    parser.add_argument('--parallel',   type=int, default=None, metavar='N',
                        help=f'Concurrent datasets downloading at once (default: {PARALLEL_DATASETS})')
    args = parser.parse_args()

    if args.hf_token:
        HF_TOKEN = args.hf_token
    if args.threads is not None:
        CHUNK_THREADS = max(1, args.threads)
    if args.parallel is not None:
        PARALLEL_DATASETS = max(1, args.parallel)

    if args.reset:
        status = _load_status()
        name = args.reset
        if name in status:
            del status[name]
            _save_status(status)
            print(f"  ✓ Reset '{name}' — it will be re-downloaded on next run")
        else:
            print(f"  '{name}' was not in the status file (already cleared or never ran)")
        sys.exit(0)

    if args.list:
        list_datasets()
    else:
        run(
            filter_type=args.only,
            only_name=args.name,
            skip_large=args.skip_large,
            dry_run=args.dry_run,
        )
