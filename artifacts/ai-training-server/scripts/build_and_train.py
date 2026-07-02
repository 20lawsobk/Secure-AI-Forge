"""
Standalone pipeline: build a combined training corpus, train a fresh BPE
tokenizer on it, train the TransformerLM from scratch, and save a checkpoint
compatible with server.py's loader (vocab/inv_vocab/merges/config).

Run from the ai-training-server directory:
    python3 scripts/build_and_train.py
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
WORKSPACE_ROOT = ROOT.parent.parent

REAL_DATA_FILES = [
    WORKSPACE_ROOT / "training" / "boostsheet_samples.json",
    ROOT.parent / "api-server" / "training" / "curriculum_phase_1_social.json",
]
SYNTHETIC_COUNT = 4000
COMBINED_DATA_PATH = ROOT / "training" / "combined_training_data.json"
VOCAB_SIZE = 4000
WEIGHTS_DIR = ROOT / "ai_model" / "weights"
WEIGHTS_PATH = WEIGHTS_DIR / "model.pt"


def load_json_items(path: Path) -> list:
    if not path.exists():
        print(f"[build_and_train] WARNING: missing data file {path}")
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        data = [data]
    return data


def main():
    import torch
    from ai_model.training.synthetic import generate_synthetic_samples
    from ai_model.training.dataset import CreativeDataset, extract_text_from_item
    from ai_model.training.trainer import train as run_train
    from ai_model.training.config import TrainConfig
    from ai_model.model.tokenizer import BPETokenizer
    from ai_model.model.transformer import TransformerLM

    t0 = time.time()

    # ── 1. Build combined corpus (real + freshly generated synthetic) ────────
    items: list = []
    for f in REAL_DATA_FILES:
        loaded = load_json_items(f)
        print(f"[build_and_train] Loaded {len(loaded)} items from {f}")
        items.extend(loaded)

    synth_path = ROOT / "training" / "synthetic_generated.json"
    synth_items = generate_synthetic_samples(str(synth_path), n=SYNTHETIC_COUNT)
    items.extend(synth_items)

    print(f"[build_and_train] Combined corpus: {len(items)} items")
    COMBINED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(COMBINED_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f)
    print(f"[build_and_train] Wrote combined dataset -> {COMBINED_DATA_PATH}")

    # ── 2. Train BPE tokenizer on the extracted text corpus ───────────────────
    texts = [extract_text_from_item(it) for it in items]
    texts = [t for t in texts if t and t.strip()]
    print(f"[build_and_train] Training BPE tokenizer on {len(texts)} texts "
          f"(target vocab_size={VOCAB_SIZE})...")

    tokenizer = BPETokenizer()
    tokenizer.train(texts, vocab_size=VOCAB_SIZE, min_freq=2)
    print(f"[build_and_train] Tokenizer trained: {tokenizer.vocab_size} tokens, "
          f"{len(tokenizer.merges)} merges")

    # ── 3. Build the training dataset with the new tokenizer ──────────────────
    max_len = int(os.environ.get("AI_MODEL_MAX_LEN", "1024"))
    dataset = CreativeDataset(str(COMBINED_DATA_PATH), tokenizer, max_len=max_len)
    print(f"[build_and_train] Dataset ready: {len(dataset)} encoded samples")

    if len(dataset) == 0:
        print("[build_and_train] ERROR: empty dataset, aborting.")
        sys.exit(1)

    # ── 4. Train the model from scratch on the new tokenizer/vocab ────────────
    dim = int(os.environ.get("AI_MODEL_DIM", "512"))
    n_layers = int(os.environ.get("AI_MODEL_LAYERS", "8"))
    n_heads = int(os.environ.get("AI_MODEL_HEADS", "8"))

    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        dim=dim, n_layers=n_layers, n_heads=n_heads, max_len=max_len,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[build_and_train] Model: dim={dim} layers={n_layers} heads={n_heads} "
          f"params={n_params:,}")

    cfg = TrainConfig({
        "model": {"dim": dim, "layers": n_layers, "heads": n_heads, "max_len": max_len},
        "train": {"data_path": str(COMBINED_DATA_PATH)},
    })
    print(f"[build_and_train] {cfg}")

    result = run_train(model, dataset, tokenizer, cfg, device="cpu")
    print(f"[build_and_train] Training result: {result}")

    # ── 5. Save checkpoint (back up the previous one first) ───────────────────
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    if WEIGHTS_PATH.exists():
        backup_path = WEIGHTS_DIR / "model.pre_bpe_backup.pt"
        if not backup_path.exists():
            os.replace(WEIGHTS_PATH, backup_path)
            print(f"[build_and_train] Backed up previous checkpoint -> {backup_path}")

    tokenizer.freeze()
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab": tokenizer.vocab,
        "inv_vocab": tokenizer.inv_vocab,
        "merges": tokenizer.merges,
        "config": {"dim": dim, "layers": n_layers, "heads": n_heads, "max_len": max_len},
    }, str(WEIGHTS_PATH))
    print(f"[build_and_train] Saved checkpoint -> {WEIGHTS_PATH}")

    elapsed = time.time() - t0
    print(f"[build_and_train] DONE in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
