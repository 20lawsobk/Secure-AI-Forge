from __future__ import annotations
import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW

from ai_model.gpu.hyper_backend import HyperGPUBackend
from ai_model.gpu.hyper_core import PrecisionMode
from ai_model.gpu.hyper_transformer import HyperTransformerLM
from ai_model.training.dataset import CreativeDataset
from ai_model.model.tokenizer import SimpleTokenizer
from ai_model.training.config import TrainConfig

SAVE_PATH = "ai_model/weights/model_hyper.pt"


def _cosine_lr(step, warmup, total, base_lr, min_lr):
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def train_on_hyper_gpu(
    data_path: str = "training/boostsheet_samples_v2.json",
    config: TrainConfig | None = None,
    lanes: int = 512,
    tensor_cores: int = 8,
    resume: bool = True,
):
    if config is None:
        config = TrainConfig()

    torch.set_num_threads(max(4, os.cpu_count() or 4))
    torch.set_grad_enabled(True)

    backend = HyperGPUBackend(
        lanes=lanes,
        tensor_cores=tensor_cores,
        precision=PrecisionMode.MIXED,
        training_mode=True,
    )

    print(f"\n{'='*70}")
    print(f"HYPER GPU ACCELERATED TRAINING")
    print(f"{'='*70}")
    gpu_s = backend.status()
    print(f"Engine: {gpu_s['engine']}")
    print(f"SIMD Lanes: {gpu_s['lanes']}")
    print(f"Tensor Cores: {gpu_s['tensor_cores']}")
    print(f"Precision: {gpu_s['precision']}")
    print(f"CPU threads: {torch.get_num_threads()}")
    print(f"{'='*70}\n")

    tokenizer = SimpleTokenizer()

    total_epochs_done = 0
    best_val_loss = float("inf")
    checkpoint = None

    if resume:
        for path in [SAVE_PATH, "ai_model/weights/model_gpu.pt", "ai_model/weights/model.pt"]:
            try:
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                tokenizer.vocab = checkpoint["vocab"]
                tokenizer.inv_vocab = checkpoint["inv_vocab"]
                tokenizer.next_id = checkpoint["next_id"]
                total_epochs_done = checkpoint.get("total_epochs", 0)
                best_val_loss = checkpoint.get("best_val_loss", float("inf"))
                print(f"Resumed from {path} (epoch {total_epochs_done}, best_val={best_val_loss:.4f})")
                break
            except FileNotFoundError:
                continue

    dataset = CreativeDataset(data_path, tokenizer, max_len=config.max_len)
    print(f"Dataset: {len(dataset)} samples, vocab: {tokenizer.vocab_size}")

    model = HyperTransformerLM(
        vocab_size=tokenizer.vocab_size,
        dim=config.dim,
        n_layers=config.layers,
        n_heads=config.heads,
        max_len=config.max_len,
        dropout=0.1,
        backend=backend,
    )

    if checkpoint and "model_state_dict" in checkpoint:
        old_cfg = checkpoint.get("config", {})
        if old_cfg.get("dim") == config.dim and old_cfg.get("layers") == config.layers:
            try:
                model.load_state_dict(checkpoint["model_state_dict"], strict=False)
                print("Loaded model weights from checkpoint")
            except Exception as e:
                print(f"Could not load weights: {e}")

    params = sum(p.numel() for p in model.parameters())
    print(f"HyperGPU model: {params:,} parameters")

    total = len(dataset)
    val_size = max(1, int(total * 0.1))
    train_size = total - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)

    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.98),
        eps=1e-9,
        fused=False,
    )

    pad_id = tokenizer.token_to_id("<PAD>")
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=config.label_smoothing)

    grad_accum = config.gradient_accumulation_steps
    steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum))
    total_steps = config.epochs * steps_per_epoch
    warmup_steps = int(total_steps * config.warmup_ratio)
    min_lr = config.lr * config.min_lr_ratio

    print(f"Samples: {train_size} train, {val_size} val | Batch: {config.batch_size}")
    print(f"Steps/epoch: {steps_per_epoch} | Total: {total_steps} | Warmup: {warmup_steps}")

    use_autocast = hasattr(torch, 'autocast')
    if use_autocast:
        print(f"Mixed precision: BFloat16 autocast ENABLED")
    else:
        print(f"Mixed precision: Not available")

    best_state = None
    global_step = 0
    start_time = time.time()
    profile_data = []

    model.train()
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        count = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader):
            lr_now = _cosine_lr(global_step, warmup_steps, total_steps, config.lr, min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            x = batch["input_ids"]
            y = batch["labels"]

            if use_autocast:
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    logits = model(x)
                    loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                    loss = loss / grad_accum
            else:
                logits = model(x)
                loss = loss_fn(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / grad_accum

            loss.backward()
            epoch_loss += loss.item() * grad_accum
            count += 1

            if (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        avg_train = epoch_loss / max(count, 1)

        model.eval()
        val_loss = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["input_ids"]
                y = batch["labels"]
                if use_autocast:
                    with torch.autocast('cpu', dtype=torch.bfloat16):
                        logits = model(x)
                        vloss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=pad_id
                        )
                else:
                    logits = model(x)
                    vloss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=pad_id
                    )
                val_loss += vloss.item()
                val_count += 1
        val_loss = val_loss / max(val_count, 1)
        model.train()

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        ppl = math.exp(min(val_loss, 20))
        elapsed = time.time() - start_time
        epoch_num = total_epochs_done + epoch + 1

        gpu_stats = backend.status()

        print(
            f"  Epoch {epoch_num} | Train: {avg_train:.4f} | Val: {val_loss:.4f} | "
            f"PPL: {ppl:.2f} | GPU ops: {gpu_stats['total_ops']} | "
            f"TC TFLOPs: {gpu_stats['total_tensor_core_tflops']} | {elapsed:.0f}s"
            f"{' *best' if is_best else ''}",
            flush=True,
        )

        profile_data.append({
            "epoch": epoch_num,
            "train_loss": avg_train,
            "val_loss": val_loss,
            "ppl": ppl,
            "gpu_ops": gpu_stats["total_ops"],
            "tc_tflops": gpu_stats["total_tensor_core_tflops"],
            "elapsed_s": elapsed,
        })

    if best_state is not None:
        model.load_state_dict(best_state)

    total_epochs_done += config.epochs

    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
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
        "backend": "hyper_gpu",
        "gpu_lanes": lanes,
        "tensor_cores": tensor_cores,
        "profile": profile_data,
    }, SAVE_PATH)

    final_ppl = math.exp(min(best_val_loss, 20))
    total_time = time.time() - start_time
    gpu_final = backend.status()

    print(f"\n{'='*70}")
    print(f"HYPER GPU TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Best val loss: {best_val_loss:.4f} | PPL: {final_ppl:.2f}")
    print(f"GPU: {gpu_final['total_ops']} ops | {gpu_final['total_compute_ms']:.0f}ms compute")
    print(f"TC TFLOPs: {gpu_final['total_tensor_core_tflops']}")
    print(f"Saved to {SAVE_PATH}")
    print(f"{'='*70}\n")

    return model, best_val_loss, profile_data


if __name__ == "__main__":
    cfg = TrainConfig()
    cfg.epochs = 10
    cfg.lr = 1e-3
    cfg.batch_size = 64
    train_on_hyper_gpu(config=cfg)
