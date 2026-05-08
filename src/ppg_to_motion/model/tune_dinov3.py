#!/usr/bin/env python3
"""DINOv3-to-Accel: fine-tune DINOv3 ViT-B/16 (LoRA) for STFT image → accelerometer reconstruction.

Architecture overview
---------------------
  [B, 3, F_in, T_in]                   (stft_image, pre-normalised to ImageNet stats)
    → pad to multiples of patch_size   (65→80, 188→192)
    → DINOv3 backbone                  last_hidden_state [B, 1+4+N, C]
    → drop CLS + 4 register tokens     [B, N, C]  (N = F_patches × T_patches = 60)
    → reshape to 2-D patch grid        [B, C, F_patches, T_patches]
    → mean-pool over F_patches          [B, C, T_patches]  (frequency collapse)
    → per-token MLP decoder            [B, T_patches, samples_per_patch]
    → flatten                          [B, T_patches × samples_per_patch]  (= 3000)

With default stft_image shape [3, 65, 188] and patch_size=16:
  F_patches = 80 // 16 = 5
  T_patches = 192 // 16 = 12
  samples_per_patch = 3000 // 12 = 250
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel

try:
    from peft import LoraConfig, TaskType, get_peft_model
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# ── Fixed dataset constants (must match builder.py) ──────────────────────────
ACC_LEN       = 3000                                      # 30 s × 100 Hz
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────
class AccelReconDataset(Dataset):
    """
    Wraps a single HF Arrow split produced by make_splits.py.
    Returns (stft_image [3, 65, 188], acc [3000]) float32 tensors.
    ImageNet normalisation is applied here so the DataLoader workers stay cheap.
    """

    def __init__(
        self,
        split_dir: str | Path,
        train: bool = False,
        spec_aug: bool = False,
        freq_mask: int = 4,
        time_mask: int = 4,
    ):
        self.ds        = load_from_disk(str(split_dir))
        self.train     = train
        self.spec_aug  = spec_aug
        self.freq_mask = freq_mask
        self.time_mask = time_mask

    def __len__(self) -> int:
        return len(self.ds)

    @staticmethod
    def _mask_axis(x: torch.Tensor, axis: int, max_w: int) -> torch.Tensor:
        """Zero-out a random contiguous band along `axis` (SpecAugment)."""
        if max_w <= 0:
            return x
        w = int(np.random.randint(0, max_w + 1))
        if w == 0:
            return x
        n     = x.size(axis)
        start = int(np.random.randint(0, max(1, n - w)))
        sl    = [slice(None)] * x.ndim
        sl[axis] = slice(start, start + w)
        x = x.clone()
        x[tuple(sl)] = 0.0
        return x

    def __getitem__(self, idx: int):
        row = self.ds[idx]
        img = torch.from_numpy(np.asarray(row["stft_image"], dtype=np.float32))  # [3, 65, 188]
        acc = torch.from_numpy(np.asarray(row["acc"],        dtype=np.float32))  # [3000]

        img = (img - IMAGENET_MEAN) / IMAGENET_STD

        # Z-score normalise acc per segment so the model learns a standardised signal
        acc = (acc - acc.mean()) / (acc.std() + 1e-6)

        if self.train and self.spec_aug:
            img = self._mask_axis(img, axis=1, max_w=self.freq_mask)
            img = self._mask_axis(img, axis=2, max_w=self.time_mask)

        return img, acc


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────
class DinoAccelReconstructor(nn.Module):
    """
    DINOv3 ViT-B/16 backbone with LoRA adapters, frequency-collapse, and an
    MLP decoder that reconstructs a 1-D accelerometer magnitude signal.

    Parameters
    ----------
    lora_r, lora_alpha : LoRA rank and scaling — primary hyperparameter knobs.
    samples_per_patch  : accelerometer samples decoded per temporal patch token.
                         Must satisfy T_patches × samples_per_patch == ACC_LEN.
    """

    def __init__(
        self,
        model_id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        samples_per_patch: int = 250,
        decoder_hidden: int = 512,
        local_files_only: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        if not PEFT_AVAILABLE:
            raise ImportError("peft is required. Install with: pip install peft")

        # ── Backbone ─────────────────────────────────────────────────────────
        self.backbone = AutoModel.from_pretrained(model_id, local_files_only=local_files_only)
        if hasattr(self.backbone, "head"):
            self.backbone.head = nn.Identity()
        if gradient_checkpointing and hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()

        self.patch_size        = int(getattr(self.backbone.config, "patch_size", 16))
        # 1 CLS token + N register tokens (DINOv3-reg has 4)
        self.num_prefix_tokens = 1 + int(getattr(self.backbone.config, "num_register_tokens", 4))
        self.hidden            = int(self.backbone.config.hidden_size)
        self.samples_per_patch = samples_per_patch

        # ── LoRA ─────────────────────────────────────────────────────────────
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=["query", "value", "fc1", "fc2"],
            task_type=TaskType.FEATURE_EXTRACTION,
        )
        self.backbone = get_peft_model(self.backbone, lora_cfg)
        for n, p in self.backbone.named_parameters():
            if "lora_" not in n:
                p.requires_grad = False

        # ── Decoder ──────────────────────────────────────────────────────────
        # Maps each temporal token [C] → [samples_per_patch] acc values.
        self.decoder = nn.Sequential(
            nn.LayerNorm(self.hidden),
            nn.Linear(self.hidden, decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, samples_per_patch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, 3, F_in, T_in]  ImageNet-normalised stft_image

        Returns
        -------
        [B, T_patches × samples_per_patch]  reconstructed acc signal
        """
        ps = self.patch_size
        B, _, F_in, T_in = x.shape

        # Pad height and width to multiples of patch_size
        Ht = max(ps, math.ceil(F_in / ps) * ps)
        Wt = max(ps, math.ceil(T_in / ps) * ps)
        if F_in != Ht or T_in != Wt:
            x = F.interpolate(x, size=(Ht, Wt), mode="bilinear", align_corners=False)

        # Backbone → all tokens
        tokens       = self.backbone(pixel_values=x).last_hidden_state  # [B, 1+num_reg+N, C]
        patch_tokens = tokens[:, self.num_prefix_tokens:, :]             # [B, N, C]

        # Reshape flat patch sequence → 2-D spatial grid
        F_p = Ht // ps
        T_p = Wt // ps
        C   = patch_tokens.size(-1)
        grid    = patch_tokens.view(B, F_p, T_p, C).permute(0, 3, 1, 2)  # [B, C, F_p, T_p]

        # Frequency collapse: mean over frequency patches → temporal sequence
        temporal = grid.mean(dim=2).permute(0, 2, 1)  # [B, T_p, C]

        # Decode each temporal token to a chunk of acc samples
        out = self.decoder(temporal)   # [B, T_p, samples_per_patch]
        return out.reshape(B, -1)      # [B, T_p × samples_per_patch]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def human_time(s: float) -> str:
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}"


def build_optimizer(
    model: nn.Module,
    lr_lora: float,
    lr_decoder: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """Split-LR AdamW: lower rate for LoRA adapters, higher for decoder."""
    lora_params    = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    decoder_params = list(model.decoder.parameters())
    return torch.optim.AdamW(
        [
            {"params": lora_params,    "lr": lr_lora},
            {"params": decoder_params, "lr": lr_decoder},
        ],
        weight_decay=weight_decay,
    )


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    sum_mse = sum_mae = n_samples = 0
    for x, y in loader:
        x, y  = x.to(device), y.to(device)
        pred  = model(x)
        L     = min(pred.size(1), y.size(1))          # guard against shape mismatch
        pred, y = pred[:, :L], y[:, :L]
        sum_mse    += F.mse_loss(pred, y, reduction="sum").item()
        sum_mae    += (pred - y).abs().sum().item()
        n_samples  += y.numel()
    return {"MSE": sum_mse / n_samples, "MAE": sum_mae / n_samples}


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Device: {device}")

    # ── Data ────────────────────────────────────────────────────────────────
    splits = Path(args.splits_dir)
    train_ds = AccelReconDataset(splits / "train", train=True,
                                  spec_aug=args.spec_aug,
                                  freq_mask=args.freq_mask, time_mask=args.time_mask)
    val_ds   = AccelReconDataset(splits / "val")
    test_ds  = AccelReconDataset(splits / "test")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.eval_batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.eval_batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True)

    # ── Model ───────────────────────────────────────────────────────────────
    model = DinoAccelReconstructor(
        model_id=args.model_name,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        samples_per_patch=args.samples_per_patch,
        decoder_hidden=args.decoder_hidden,
        local_files_only=args.local_files_only,
        gradient_checkpointing=args.grad_checkpoint,
    ).to(device)

    # Verify reconstruction length matches ACC_LEN
    with torch.no_grad():
        dummy   = torch.zeros(1, 3, 65, 188, device=device)
        out_len = model(dummy).size(1)
    if out_len != ACC_LEN:
        print(
            f"WARNING: model output length {out_len} != ACC_LEN {ACC_LEN}. "
            f"Adjust --samples-per-patch so that T_patches × samples_per_patch = {ACC_LEN}."
        )

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.2f}%)")

    # ── Optimizer + scheduler ───────────────────────────────────────────────
    optimizer = build_optimizer(model, args.lr_lora, args.lr_decoder, args.weight_decay)
    criterion = nn.MSELoss()

    total_steps  = max(1, (len(train_loader) // max(1, args.grad_accum)) * args.epochs)
    warmup_steps = int(args.warmup * total_steps)

    def cosine_warmup(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_warmup)
    scaler    = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available() and not args.no_amp)

    os.makedirs(args.out_dir, exist_ok=True)
    best_val_mse = float("inf")
    t0           = time.time()

    # ── Epoch loop ──────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = n_seen = 0
        optimizer.zero_grad(set_to_none=True)

        for step, (x, y) in enumerate(train_loader, 1):
            x, y = x.to(device), y.to(device)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available() and not args.no_amp):
                pred = model(x)
                L    = min(pred.size(1), y.size(1))
                loss = criterion(pred[:, :L], y[:, :L]) / args.grad_accum

            scaler.scale(loss).backward()

            if step % args.grad_accum == 0:
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        args.grad_clip,
                    )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            total_loss += loss.item() * y.size(0) * args.grad_accum
            n_seen     += y.size(0)

            if step % args.log_every == 0:
                print(
                    f"Epoch {epoch:03d}/{args.epochs} | Step {step:05d} | "
                    f"Loss {loss.item()*args.grad_accum:.6f} | "
                    f"LR(lora) {optimizer.param_groups[0]['lr']:.2e} | "
                    f"T {human_time(time.time()-t0)}",
                    flush=True,
                )

        val_metrics = evaluate(model, val_loader, device)
        tr_loss     = total_loss / max(1, n_seen)
        print(
            f"[Epoch {epoch:03d}] train_loss={tr_loss:.6f} | "
            f"val_MSE={val_metrics['MSE']:.6f} | val_MAE={val_metrics['MAE']:.6f} | "
            f"T={human_time(time.time()-t0)}"
        )

        if val_metrics["MSE"] < best_val_mse:
            best_val_mse = val_metrics["MSE"]
            save_path    = os.path.join(args.out_dir, "best.pt")
            torch.save({"model": model.state_dict(), "args": vars(args), "val": val_metrics},
                        save_path)
            print(f"  -> Saved best (val_MSE={best_val_mse:.6f}) to {save_path}")

    # ── Final test eval ─────────────────────────────────────────────────────
    ckpt = torch.load(os.path.join(args.out_dir, "best.pt"), map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    test_metrics = evaluate(model, test_loader, device)
    print(json.dumps({"best_val_MSE": best_val_mse, "test": test_metrics}, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DINOv3 (LoRA) STFT-image → accelerometer signal reconstruction"
    )
    # Data
    p.add_argument("--splits-dir", type=str, required=True,
                   help="Directory with train/ val/ test/ HF splits (output of make_splits.py)")

    # Model
    p.add_argument("--model-name",       type=str,   default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--decoder-hidden",   type=int,   default=512,
                   help="Hidden dim of the MLP decoder")
    p.add_argument("--samples-per-patch", type=int,  default=250,
                   help="Acc samples decoded per temporal patch token (T_patches × this = ACC_LEN)")
    p.add_argument("--grad-checkpoint",  action="store_true")

    # LoRA — primary hyperparameter knobs
    p.add_argument("--lora-r",       type=int,   default=8,    help="LoRA rank r")
    p.add_argument("--lora-alpha",   type=int,   default=16,   help="LoRA scaling alpha")
    p.add_argument("--lora-dropout", type=float, default=0.05)

    # SpecAug
    p.add_argument("--spec-aug",   action="store_true")
    p.add_argument("--freq-mask",  type=int, default=4)
    p.add_argument("--time-mask",  type=int, default=4)

    # Training
    p.add_argument("--batch-size",       type=int,   default=32)
    p.add_argument("--eval-batch-size",  type=int,   default=64)
    p.add_argument("--epochs",           type=int,   default=50)
    p.add_argument("--lr-lora",          type=float, default=1e-5,
                   help="Learning rate for LoRA adapter parameters")
    p.add_argument("--lr-decoder",       type=float, default=1e-4,
                   help="Learning rate for reconstruction decoder")
    p.add_argument("--weight-decay",     type=float, default=1e-4)
    p.add_argument("--warmup",           type=float, default=0.1,
                   help="Fraction of total steps used for linear warmup")
    p.add_argument("--grad-clip",        type=float, default=1.0)
    p.add_argument("--grad-accum",       type=int,   default=1)
    p.add_argument("--num-workers",      type=int,   default=4)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--cpu",              action="store_true")
    p.add_argument("--no-amp",           action="store_true")
    p.add_argument("--log-every",        type=int,   default=100)
    p.add_argument("--out-dir",          type=str,   default="./dinov3_accel_out")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    train(args)