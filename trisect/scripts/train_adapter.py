"""Train the adapter (`adapter_model.py`) by regression against the true VD context.

Inputs:  data/targets/targets.h5 (z_sem, z_struct) and data/targets/a_psi_true_context.h5
         (`extract_vd_context.py`), joined on the 73k id, non-shared1000 only
Outputs: data/targets/a_psi_checkpoint.pt (overwritten every epoch: protection against a kill)

No brain data. The loss is MSE between predicted and true context (text and image halves), reported
with the mean per-token cosine. Regression is a proxy for the denoising loss through the frozen
generator (see `adapter_model.py`). The struct width is read from the z_struct shape of --targets; a
sweep file is self-describing. It runs on CPU or GPU and never loads diffusers.

Usage (from the trisect/ folder):
    python scripts/train_adapter.py --epochs 30 --batch-size 128 --lr 3e-4
    python scripts/train_adapter.py --targets data/targets/targets_structd64.h5 --out data/targets/a_psi_checkpoint_structd64.pt
"""
import argparse
import os
import time
from datetime import datetime

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from adapter_model import APsiMinimal

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGETS_H5 = "data/targets/targets.h5"
CONTEXT_H5 = "data/targets/a_psi_true_context.h5"
CKPT_PATH = "data/targets/a_psi_checkpoint.pt"


class ZToContextDataset(Dataset):
    """Pairs (z_sem, z_struct) from targets.h5 with the true VD context from a_psi_true_context.h5,
    both indexed by the same 73k image id, for one split ("train" or "val")."""

    def __init__(self, targets_path, context_path, split):
        with h5py.File(context_path, "r") as f:
            self.indices_73k = f[split]["image_indices_73k"][:]
        self.targets_path = targets_path
        self.context_path = context_path
        self.split = split
        self._targets_f = None
        self._context_f = None

    def _files(self):
        if self._targets_f is None:
            self._targets_f = h5py.File(self.targets_path, "r")
            self._context_f = h5py.File(self.context_path, "r")
        return self._targets_f, self._context_f

    def __len__(self):
        return len(self.indices_73k)

    def __getitem__(self, i):
        targets_f, context_f = self._files()
        idx73k = int(self.indices_73k[i])
        z_sem = targets_f["z_sem"][idx73k].astype(np.float32)  # (2,128)
        z_struct = targets_f["z_struct"][idx73k].astype(np.float32)  # (8,8,32)
        prompt_embeds = context_f[self.split]["prompt_embeds"][i].astype(np.float32)  # (77,768)
        image_embeddings = context_f[self.split]["image_embeddings"][i].astype(np.float32)  # (257,768)
        return (
            torch.from_numpy(z_sem),
            torch.from_numpy(z_struct),
            torch.from_numpy(prompt_embeds),
            torch.from_numpy(image_embeddings),
        )


def token_cosine(pred, true):
    return F.cosine_similarity(pred, true, dim=-1).mean().item()


def run_epoch(model, loader, optimizer=None, heartbeat_label=None):
    training = optimizer is not None
    model.train(training)
    total_loss, total_text_cos, total_image_cos, n_batches = 0.0, 0.0, 0.0, 0
    n_total_batches = len(loader)
    t0 = time.monotonic()
    for z_sem, z_struct, prompt_true, image_true in loader:
        z_sem, z_struct = z_sem.to(DEVICE), z_struct.to(DEVICE)
        prompt_true, image_true = prompt_true.to(DEVICE), image_true.to(DEVICE)

        with torch.set_grad_enabled(training):
            prompt_pred, image_pred = model(z_sem, z_struct)
            loss = F.mse_loss(prompt_pred, prompt_true) + F.mse_loss(image_pred, image_true)

        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_text_cos += token_cosine(prompt_pred.detach(), prompt_true)
        total_image_cos += token_cosine(image_pred.detach(), image_true)
        n_batches += 1

        # Per-batch heartbeat, timestamped: a stall shows up here within one batch's time instead
        # of only at the next epoch print (a past run hung for hours with only epoch 1's line in
        # the log, and nothing distinguished "still working" from "stuck").
        if heartbeat_label is not None:
            now = datetime.now().strftime("%H:%M:%S")
            elapsed = time.monotonic() - t0
            print(
                f"  [{now}] {heartbeat_label} batch {n_batches}/{n_total_batches} "
                f"loss={loss.item():.4f} elapsed={elapsed:.1f}s",
                flush=True,
            )

    return total_loss / n_batches, total_text_cos / n_batches, total_image_cos / n_batches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", type=str, default=CKPT_PATH)
    ap.add_argument("--targets", type=str, default=TARGETS_H5,
                     help="assembled targets.h5 variant (assemble_targets.py --struct-d/--sem-d)")
    args = ap.parse_args()

    with h5py.File(args.targets, "r") as f:
        struct_channels = f["z_struct"].shape[-1]
        sem_dim = f["z_sem"].shape[-1]
    print(f"targets: {args.targets} (struct_channels={struct_channels}, sem_dim={sem_dim})")

    train_ds = ZToContextDataset(args.targets, CONTEXT_H5, "train")
    val_ds = ZToContextDataset(args.targets, CONTEXT_H5, "val")
    print(f"train: {len(train_ds)} images, val: {len(val_ds)} images")

    # num_workers=0 deliberately: DataLoader workers fork() the process, and both CUDA (already
    # initialized below) and h5py (opened lazily per-worker) are not fork-safe. Forking a new
    # worker pool every epoch (default persistent_workers=False) risks a hang (process stuck at
    # ~100% single-core CPU, 0% GPU util, no exception). The dataset is tiny (6000/300 items via
    # simple h5py indexing); single-process loading is not a real bottleneck.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = APsiMinimal(struct_channels=struct_channels, sem_dim=sem_dim).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"A_psi minimal: {n_params / 1e6:.1f}M parameters, device={DEVICE}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        train_loss, train_text_cos, train_image_cos = run_epoch(
            model, train_loader, optimizer, heartbeat_label=f"epoch {epoch:3d} train"
        )
        val_loss, val_text_cos, val_image_cos = run_epoch(
            model, val_loader, optimizer=None, heartbeat_label=f"epoch {epoch:3d} val"
        )
        print(
            f"epoch {epoch:3d}  train loss {train_loss:.4f} (cos txt {train_text_cos:.3f} img {train_image_cos:.3f})"
            f"  val loss {val_loss:.4f} (cos txt {val_text_cos:.3f} img {val_image_cos:.3f})",
            flush=True,
        )
        # Overwritten every epoch (not versioned): cheap insurance against a hang/kill losing
        # all progress, not a resumable-training checkpoint history.
        torch.save(
            {
                "state_dict": model.state_dict(),
                "epoch": epoch,
                "struct_channels": struct_channels,
                "sem_dim": sem_dim,
            },
            args.out,
        )

    print(f"Done. Saved {args.out}")


if __name__ == "__main__":
    main()
