from __future__ import annotations
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from ai_model.gpu.torch_backend import DigitalGPUBackend
from ai_model.gpu.accelerated_transformer import AcceleratedTransformerLM
from ai_model.training.dataset import CreativeDataset
from ai_model.model.tokenizer import SimpleTokenizer
from ai_model.training.config import TrainConfig

SAVE_PATH = "ai_model/weights/model_gpu.pt"


def _cosine_lr(step, warmup, total, base_lr, min_lr):
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def train_on_digital_gpu(
    data_path: str = "training/boostsheet_samples_v2.json",
    config: TrainConfig | None = None,
    lanes: int = 32,
    resume: bool = True,
):
    if config is None:
        config = TrainConfig()

    gpu_backend = DigitalGPUBackend(lanes=lanes)
    print(f"Digital GPU initialized: {gpu_backend.status()}")

    tokenizer = SimpleTokenizer()

    total_epochs_done = 0
    best_val_loss = float("inf")
    checkpoint = None

    if resume:
        try:
            checkpoint = torch.load(SAVE_PATH, map_location="cpu", weights_only=False)
            tokenizer.vocab = checkpoint["vocab"]
            tokenizer.inv_vocab = checkpoint["inv_vocab"]
            tokenizer.next_id = checkpoint["next_id"]
            total_epochs_done = checkpoint.get("total_epochs", 0)
            best_val_loss = checkpoint.get("best_val_loss", float("inf"))
            print(f"Resumed from GPU checkpoint (epoch {total_epochs_done}, best_val={best_val_loss:.4f})")
        except FileNotFoundError:
            try:
                cpu_ckpt = torch.load("ai_model/weights/model.pt", map_location="cpu", weights_only=False)
                tokenizer.vocab = cpu_ckpt["vocab"]
                tokenizer.inv_vocab = cpu_ckpt["inv_vocab"]
                tokenizer.next_id = cpu_ckpt["next_id"]
                print(f"Loaded vocab from CPU checkpoint (vocab={tokenizer.vocab_size})")
            except FileNotFoundError:
                pass

    dataset = CreativeDataset(data_path, tokenizer, max_len=config.max_len)
    print(f"Dataset: {len(dataset)} samples, vocab: {tokenizer.vocab_size}")

    model = AcceleratedTransformerLM(
        vocab_size=tokenizer.vocab_size,
        dim=config.dim,
        n_layers=config.layers,
        n_heads=config.heads,
        max_len=config.max_len,
        dropout=0.1,
        gpu_backend=gpu_backend,
    )

    if checkpoint and "model_state_dict" in checkpoint:
        old_cfg = checkpoint.get("config", {})
        if old_cfg.get("dim") == config.dim and old_cfg.get("layers") == config.layers:
            try:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                print("Loaded GPU model weights from checkpoint")
            except Exception as e:
                print(f"Could not load weights (architecture changed): {e}")

    params = sum(p.numel() for p in model.parameters())
    print(f"Accelerated model: {params:,} parameters (running on Digital GPU)")

    total = len(dataset)
    val_size = max(1, int(total * 0.1))
    train_size = total - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)

    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.98), eps=1e-9)

    pad_id = tokenizer.token_to_id("<PAD>")
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=config.label_smoothing)

    grad_accum = config.gradient_accumulation_steps
    steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum))
    total_steps = config.epochs * steps_per_epoch
    warmup_steps = int(total_steps * config.warmup_ratio)
    min_lr = config.lr * config.min_lr_ratio

    best_state = None
    global_step = 0
    start_time = time.time()
    profile_data = []

    print(f"\n{'='*60}")
    print(f"Training on Digital GPU ({lanes}-lane SIMD)")
    print(f"Epochs: {config.epochs} | Batch: {config.batch_size} x {grad_accum} accum")
    print(f"LR: {config.lr} | Steps: {total_steps} | Warmup: {warmup_steps}")
    print(f"{'='*60}\n")

    model.train()
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        count = 0
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            lr_now = _cosine_lr(global_step, warmup_steps, total_steps, config.lr, min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            x = batch["input_ids"]
            y = batch["labels"]

            logits = model(x)
            loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
            loss = loss / grad_accum

            loss.backward()
            epoch_loss += loss.item() * grad_accum
            count += 1

            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            gpu_backend.flush_vram()

        avg_train = epoch_loss / max(count, 1)

        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["input_ids"]
                y = batch["labels"]
                logits = model(x)
                vloss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=pad_id)
                val_loss += vloss.item()
                val_count += 1
                gpu_backend.flush_vram()
        val_loss = val_loss / max(val_count, 1)
        model.train()

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        ppl = math.exp(min(val_loss, 20))
        elapsed = time.time() - start_time
        epoch_num = total_epochs_done + epoch + 1

        gpu_stats = gpu_backend.status()

        print(
            f"  Epoch {epoch_num} | Train: {avg_train:.4f} | Val: {val_loss:.4f} | "
            f"PPL: {ppl:.2f} | VRAM: {gpu_stats['vram_mb']}MB | {elapsed:.0f}s"
            f"{' *best' if is_best else ''}",
            flush=True,
        )

        profile_data.append({
            "epoch": epoch_num,
            "train_loss": avg_train,
            "val_loss": val_loss,
            "ppl": ppl,
            "vram_mb": gpu_stats["vram_mb"],
            "elapsed_s": elapsed,
        })

    if best_state is not None:
        model.load_state_dict(best_state)

    total_epochs_done += config.epochs

    torch.save({
        "model_state_dict": best_state or model.state_dict(),
        "vocab": tokenizer.vocab,
        "inv_vocab": tokenizer.inv_vocab,
        "next_id": tokenizer.next_id,
        "config": {
            "dim": config.dim,
            "layers": config.layers,
            "heads": config.heads,
            "max_len": config.max_len,
        },
        "total_epochs": total_epochs_done,
        "best_val_loss": best_val_loss,
        "backend": "digital_gpu",
        "gpu_lanes": lanes,
        "profile": profile_data,
    }, SAVE_PATH)

    final_ppl = math.exp(min(best_val_loss, 20))
    total_time = time.time() - start_time

    print(f"\nDigital GPU training complete in {total_time:.0f}s")
    print(f"Best val loss: {best_val_loss:.4f} | PPL: {final_ppl:.2f}")
    print(f"Saved to {SAVE_PATH}")
    print(f"GPU status: {gpu_backend.status()}")

    return model, best_val_loss, profile_data


if __name__ == "__main__":
    cfg = TrainConfig()
    cfg.epochs = 3
    cfg.lr = 5e-4
    cfg.batch_size = 4
    train_on_digital_gpu(config=cfg)
