import re, sys
from pathlib import Path

f = Path(__file__).parent / 'download_datasets.py'
src = f.read_text(encoding='utf-8')
original = src

fixes = {
    "rdiehl/medley-solos-db": "https://zenodo.org/records/3464302/files/Medley-solos-DB.tar.gz",
    "zenodo.org/record/3464302": "zenodo.org/records/3464302",
    "rdiehl/magnatagatune":   "ccmusic-database/magnatagatune",
    "Ivan-ZNN/NSynth":        "https://storage.googleapis.com/magentadata/datasets/nsynth/nsynth-train.jsonwav.tar.gz",
    "kinetics700-2020":       "HuggingFaceM4/kinetics700-2020",
}

method_fixes = {
    "Dataset('medley_solos',   'hf',": "Dataset('medley_solos',   'http',",
    "Dataset('nsynth',         'hf',": "Dataset('nsynth',         'http',",
}

for old, new in fixes.items():
    src = src.replace(old, new)

for old, new in method_fixes.items():
    src = src.replace(old, new)

src = src.replace('from datetime import datetime\n', 'from datetime import datetime, timezone\n')
src = src.replace('datetime.utcnow().isoformat()', 'datetime.now(timezone.utc).isoformat()')

if src == original:
    print('Nothing changed — file may already be patched or content differs.')
else:
    f.write_text(src, encoding='utf-8')
    print('Patched successfully. Changes:')
    for old, new in {**fixes, **method_fixes}.items():
        print(f'  {old[:50]} -> {new[:50]}')
