from __future__ import annotations
import math
import time
from typing import Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW

from .config import TrainConfig
from .dataset import CreativeDataset


def _cosine_lr(step: int, warmup: int, total: int, base_lr: float, min_lr: float) -> float:
    if step < warmup:
        return base_lr * step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def train(
    model: nn.Module,
    dataset_or_tokenizer,
    tokenizer_or_data_path=None,
    config: TrainConfig = None,
    device: str = "cpu",
) -> dict:
    if config is None:
        config = TrainConfig()

    if isinstance(dataset_or_tokenizer, CreativeDataset):
        dataset = dataset_or_tokenizer
        tokenizer = tokenizer_or_data_path
    else:
        tokenizer = dataset_or_tokenizer
        data_path = tokenizer_or_data_path if isinstance(tokenizer_or_data_path, str) else config.data_path
        max_len = getattr(model, "max_len", config.max_len)
        if hasattr(model, "pos_emb"):
            max_len = model.pos_emb.num_embeddings
        dataset = CreativeDataset(data_path, tokenizer, max_len=max_len)

    if len(dataset) == 0:
        return {"final_loss": 0.0, "loss": 0.0, "epochs": 0, "samples": 0}

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
    )
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, num_workers=0)

    model = model.to(device)
    model.train()

    optimizer = AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    pad_id = tokenizer.token_to_id("<PAD>")
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_id, label_smoothing=config.label_smoothing)

    grad_accum = config.gradient_accumulation_steps
    steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum))
    total_steps = config.epochs * steps_per_epoch
    warmup_steps = int(total_steps * config.warmup_ratio)
    min_lr = config.lr * config.min_lr_ratio

    use_autocast = hasattr(torch, "autocast")
    global_step = 0
    best_val_loss = float("inf")
    best_state = None
    start_time = time.time()

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        count = 0
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader):
            lr_now = _cosine_lr(global_step, warmup_steps, total_steps, config.lr, min_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)

            if use_autocast and device == "cpu":
                with torch.autocast("cpu", dtype=torch.bfloat16):
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
        val_loss_total = 0.0
        val_count = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["input_ids"].to(device)
                y = batch["labels"].to(device)
                logits = model(x)
                vloss = nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=pad_id,
                )
                val_loss_total += vloss.item()
                val_count += 1

        val_loss = val_loss_total / max(val_count, 1)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        ppl = math.exp(min(val_loss, 20))
        elapsed = time.time() - start_time
        print(
            f"  Epoch {epoch + 1}/{config.epochs} | Train: {avg_train:.4f} | "
            f"Val: {val_loss:.4f} | PPL: {ppl:.2f} | {elapsed:.0f}s",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    final_ppl = math.exp(min(best_val_loss, 20))
    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.0f}s | Best val loss: {best_val_loss:.4f} | PPL: {final_ppl:.2f}")

    return {
        "final_loss": best_val_loss,
        "loss": best_val_loss,
        "perplexity": final_ppl,
        "epochs": config.epochs,
        "samples": len(dataset),
        "elapsed_seconds": total_time,
    }


def evaluate(
    model: nn.Module,
    dataset_or_tokenizer,
    tokenizer=None,
    device: str = "cpu",
) -> float:
    if isinstance(dataset_or_tokenizer, CreativeDataset):
        dataset = dataset_or_tokenizer
        tok = tokenizer
    else:
        return 1.0

    if len(dataset) == 0:
        return 1.0

    model.eval()
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
    pad_id = tok.token_to_id("<PAD>")

    total_loss = 0.0
    count = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=pad_id,
            )
            total_loss += loss.item()
            count += 1

    avg_loss = total_loss / max(count, 1)
    ppl = math.exp(min(avg_loss, 20))
    return ppl
