"""cpt/train.py

Continued Pre-Training (CPT) of MambaByte on Javanese/Sundanese text.

Trains from the English base (JunxiongWang/MambaByte_PG19_972M) using LoRA
via PEFT. Reuses the text corpus from GFD_BLT/CPT/data/text_corpus/.

Usage:
    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python -u cpt/train.py configs/cpt_jv.yaml
    CUDA_VISIBLE_DEVICES=4 PYTHONPATH=. python -u cpt/train.py configs/cpt_su.yaml

Config options (train section):
  total_steps:        optimizer steps (default 10000)
  lr:                 peak learning rate (default 2e-4)
  min_lr:             cosine decay floor (default 0 → constant LR)
  warmup_steps:       linear warmup length (default 0)
  grad_accum_steps:   micro-steps per optimizer step (default 1)
  weight_decay:       AdamW weight decay (default 0.01)
  batch_size:         micro-batch size (default 4)
  seq_len:            bytes per sample (default 1024)
  save_every:         checkpoint interval in optimizer steps (default 500)
  log_every:          log interval in optimizer steps (default 20)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Dataset


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class ByteTextDataset(Dataset):
    """Streams raw bytes from a text file, returns fixed-length windows."""

    def __init__(self, text_file: str, seq_len: int = 1024):
        self.seq_len = seq_len
        with open(text_file, "rb") as f:
            data = f.read()
        # Convert to token IDs (byte value = token ID for MambaByte_PG19)
        self.data = torch.frombuffer(data, dtype=torch.uint8).long()
        print(f"[dataset] {text_file}: {len(self.data):,} bytes, "
              f"{len(self):,} windows of {seq_len}")

    def __len__(self) -> int:
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx: int) -> dict:
        chunk = self.data[idx: idx + self.seq_len + 1]
        return {
            "input_ids": chunk[:-1].clone(),
            "labels":    chunk[1:].clone(),
        }


def get_lr(step: int, warmup_steps: int, total_steps: int, lr: float, min_lr: float) -> float:
    """Linear warmup then cosine decay. Returns constant lr when min_lr==0 and warmup==0."""
    if warmup_steps > 0 and step < warmup_steps:
        return lr * step / warmup_steps
    if min_lr == 0.0 and warmup_steps == 0:
        return lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def build_model_and_optimizer(cfg: dict):
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    from peft import LoraConfig, inject_adapter_in_model

    lm_cfg = cfg["lm"]
    device = lm_cfg.get("device", "cuda")
    # float16 causes NaN in Mamba SSM slow path (exp(A*dt) overflows on V100/sm_70).
    # float32 is numerically stable and fits in 32GB V100 (~7GB total incl. AdamW state).
    dtype = torch.float32

    print(f"[train] Loading base MambaByte from {lm_cfg['model_name_or_path']} ...")
    model = MambaLMHeadModel.from_pretrained(
        lm_cfg["model_name_or_path"],
        device=device,
        dtype=dtype,
    )

    lora_cfg = lm_cfg.get("lora_config", {})
    rank = lora_cfg.get("rank", 16)
    alpha = lora_cfg.get("alpha", 32.0)
    target_modules = lora_cfg.get("target_modules", ["in_proj", "out_proj", "x_proj", "dt_proj"])

    # inject_adapter_in_model directly modifies the nn.Modules in-place, preserving
    # the gradient graph — get_peft_model wraps in PeftModel which breaks .backward()
    # for non-HF models like MambaLMHeadModel.
    peft_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=target_modules,
        bias="none",
    )
    # The Mamba fast path passes raw .weight tensors directly to a fused CUDA kernel,
    # bypassing nn.Linear.forward() entirely — so LoRA adapters are never called.
    # Disable fast path so the slow path uses self.in_proj(u) / self.out_proj(y),
    # which go through the LoRA-wrapped linear forward.
    for module in model.modules():
        if hasattr(module, "use_fast_path"):
            module.use_fast_path = False

    model = inject_adapter_in_model(peft_config, model)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {trainable/total*100:.4f}")

    train_cfg = cfg["train"]
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(train_cfg.get("lr", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
    )
    return model, optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfg = load_config(args.config)

    train_cfg = cfg["train"]
    out_dir = train_cfg["checkpoint_dir"]
    os.makedirs(out_dir, exist_ok=True)

    dataset = ByteTextDataset(
        text_file=train_cfg["text_file"],
        seq_len=int(train_cfg.get("seq_len", 1024)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 4)),
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    model, optimizer = build_model_and_optimizer(cfg)
    device = cfg["lm"].get("device", "cuda")
    total_steps      = int(train_cfg.get("total_steps", 10000))
    save_every       = int(train_cfg.get("save_every", 500))
    log_every        = int(train_cfg.get("log_every", 20))
    warmup_steps     = int(train_cfg.get("warmup_steps", 0))
    min_lr           = float(train_cfg.get("min_lr", 0.0))
    peak_lr          = float(train_cfg.get("lr", 2e-4))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))

    log_path = os.path.join(out_dir, "train.log")
    log_f = open(log_path, "w")

    step = 0          # optimizer steps
    model.train()
    data_iter = iter(loader)

    while step < total_steps:
        # ── cosine LR update ──────────────────────────────────────────────
        cur_lr = get_lr(step, warmup_steps, total_steps, peak_lr, min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        # ── gradient accumulation ─────────────────────────────────────────
        accum_loss = 0.0
        optimizer.zero_grad()
        for micro in range(grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)

            out = model(input_ids=input_ids)
            logits = out.logits if hasattr(out, "logits") else out[0]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            ) / grad_accum_steps

            loss.backward()
            accum_loss += loss.item()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1

        if step % log_every == 0:
            bpb = accum_loss / math.log(2)
            msg = (f"step {step:>7}/{total_steps}  loss {accum_loss:.4f}"
                   f"  bpb {bpb:.4f}  grad_norm {grad_norm:.3f}  lr {cur_lr:.2e}")
            print(msg)
            log_f.write(msg + "\n")
            log_f.flush()

        if step % save_every == 0 or step == total_steps:
            ckpt_path = os.path.join(out_dir, f"step_{step:07d}.pt")
            lora_sd = {k: v for k, v in model.state_dict().items() if "lora_" in k}
            torch.save({"step": step, "lora_state_dict": lora_sd,
                        "lora_config": cfg["lm"].get("lora_config", {})}, ckpt_path)
            print(f"[checkpoint] Saved step {step} -> {ckpt_path}")

    # Save final
    final_path = os.path.join(out_dir, "final.pt")
    lora_sd = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    torch.save({"step": step, "lora_state_dict": lora_sd,
                "lora_config": cfg["lm"].get("lora_config", {})}, final_path)
    print(f"[train] Done. Final checkpoint -> {final_path}")
    log_f.close()


if __name__ == "__main__":
    main()
