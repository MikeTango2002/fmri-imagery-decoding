"""Train the MLPEncoder (mlp_encoder.py) on the four subjects at once (Section 8.1.1).

Inputs:  targets.h5, pca_zlow/z_low.npy, NSD-Core betas and split (decoder_training_utils.py)
Outputs: --out-dir: model.pt, model_best.pt, fit_target_moments.npz, history.csv, resource_log.jsonl,
         run_summary.json (peak VRAM, wall-clock time, disk, configuration)

One row per distinct image, one random repetition per access: a batch needs 64 distinct stimuli or
InfoNCE disables itself. Voxel counts differ by subject; voxels travel as a list and are regrouped by
subject in the forward pass. Cosine LR with 5% warm-up. --delta-ctr (InfoNCE weight) is not chosen
here: it was swept on the NSD-core development split only. --smoke-test runs the same code on synthetic data.

Usage:
    python train_mlp_encoder.py --epochs 60 --batch-size 256 --delta-ctr <value> --gpus 0 1 --out-dir DIR
    python train_mlp_encoder.py --smoke-test --epochs 3 --batch-size 96 --out-dir DIR
"""
import argparse
import json
import math
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent  # the folder that holds icnn/
sys.path.insert(0, str(_REPO_ROOT / "icnn" / "src"))
from prepare_nsd_imagery_icnn import NUM_VOXELS  # noqa: E402  (needs the sys.path insert above)

from mlp_encoder import FACTORS, MLPEncoder, encoder_loss  # noqa: E402
from decoder_training_utils import (  # noqa: E402  (reused verbatim, not re-derived)
    MIN_BATCH_FOR_CTR,
    load_subject_split,
    load_targets,
)
from trisect.nsd import zscore_fmri  # noqa: E402
from trisect.ridge import restandardize  # noqa: E402

SUBJECTS = (1, 2, 5, 7)

INDEX_DIR = "data/nsdcore_index"
BETAS_DIR = "data/mindeyev2_dataset"
TARGETS_H5 = "data/targets/targets.h5"
ZLOW_PATH = "data/targets/pca_zlow/z_low.npy"
R_DIR = "data/betas_trisect/nsdcore_r"
DEV_MASK_DIR = "data/targets"
OUT_DIR = "data/mlp_encoder_checkpoints"


# --------------------------------------------------------------------------------------------
# Data: multisubject dataset, collate, and the subject-grouped forward pass
# --------------------------------------------------------------------------------------------

class MultiSubjectFitDataset(Dataset):
    """One row per DISTINCT (subj, image) pair. A pair is unique across subjects by construction:
    the non-shared trials of NSD-Core are subject-specific, and shared1000 is excluded from every
    fit. __getitem__ picks one of that image's trial reps at random; the batch membership is
    fixed and only the repetition is stochastic."""

    def __init__(self, per_subj: dict, sem: np.ndarray, struct: np.ndarray, low: np.ndarray, rng):
        self.entries = []  # list of (subj, rows[])
        for subj, d in per_subj.items():
            fit_rows = np.flatnonzero(d["fit_mask"])
            fit_img_ids = d["img73k"][fit_rows]
            uniq_ids, inverse = np.unique(fit_img_ids, return_inverse=True)
            for i in range(len(uniq_ids)):
                self.entries.append((subj, fit_rows[inverse == i]))
        self.per_subj = per_subj
        self.sem, self.struct, self.low = sem, struct, low
        self.rng = rng

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        subj, rows = self.entries[i]
        d = self.per_subj[subj]
        row = int(rows[self.rng.integers(len(rows))])
        img_id = int(d["img73k"][row])
        return (
            subj,
            torch.from_numpy(d["fmri_z"][row]),
            torch.from_numpy(self.sem[img_id]),
            torch.from_numpy(self.struct[img_id]),
            torch.from_numpy(self.low[img_id]),
        )


def collate_multisubject(batch):
    """Voxels cannot be stacked across subjects (V differs per subject): they are kept as a
    per-item list and regrouped by forward_multisubject() instead."""
    subjs = [b[0] for b in batch]
    voxels = [b[1] for b in batch]
    sem = torch.stack([b[2] for b in batch])
    struct = torch.stack([b[3] for b in batch])
    low = torch.stack([b[4] for b in batch])
    return subjs, voxels, sem, struct, low


def forward_multisubject(model, subjs, voxels, device) -> dict:
    """Regroup a mixed-subject batch by subject, call MLPEncoder.forward() once per subject
    present, and scatter the per-factor outputs back to the original row order of the batch.

    It uses the out-of-place index_copy (not index_copy_); no tensor needed for the backward
    pass is modified in place: each subject writes a disjoint set of rows exactly once, and out[f]
    is rebound to the new tensor at each call."""
    subjs_arr = np.asarray(subjs)
    batch_size = len(subjs)
    out = None
    for subj in np.unique(subjs_arr):
        idx = np.flatnonzero(subjs_arr == subj)
        v = torch.stack([voxels[i] for i in idx]).to(device, non_blocking=True)
        pred = model(v, subj=int(subj))
        if out is None:
            out = {f: torch.zeros(batch_size, pred[f].shape[-1], device=device,
                                   dtype=pred[f].dtype) for f in FACTORS}
        idx_t = torch.as_tensor(idx, device=device)
        for f in FACTORS:
            out[f] = out[f].index_copy(0, idx_t, pred[f])
    return out


def dev_tensors_multisubject(per_subj: dict, sem, struct, low, device):
    """Concatenate the development trials of every subject into one pool: lists of per-trial
    subject ids (python ints, for forward_multisubject) plus stacked target tensors, aligned by
    row. The number of trials is not equalized across subjects; a subject with more
    development trials weighs more in the pooled loss, as in any sum over a mixed-subject batch."""
    # Voxel rows cannot be concatenated across subjects (different V per subject, the same
    # reason collate_multisubject keeps them as a per-item list); they are built row by row.
    subjs, voxels, img_ids = [], [], []
    for subj, d in per_subj.items():
        rows = np.flatnonzero(d["dev_mask"])
        for row in rows:
            subjs.append(subj)
            voxels.append(torch.from_numpy(d["fmri_z"][row]))
            img_ids.append(d["img73k"][row])
    img_ids = np.array(img_ids)
    targets = {
        "sem": torch.from_numpy(sem[img_ids]).to(device),
        "struct": torch.from_numpy(struct[img_ids]).to(device),
        "low": torch.from_numpy(low[img_ids]).to(device),
    }
    return subjs, voxels, targets, img_ids


def dev_retrieval_gallery_multisubject(per_subj: dict, sem, device):
    """Union of the distinct development images of every subject (safe to union directly, same
    uniqueness argument as for the fit set): a retrieval gallery and, for each trial, the label
    of its image in the gallery."""
    all_img_ids = []
    for d in per_subj.values():
        rows = np.flatnonzero(d["dev_mask"])
        all_img_ids.append(d["img73k"][rows])
    all_img_ids = np.concatenate(all_img_ids)
    uniq_ids, labels = np.unique(all_img_ids, return_inverse=True)
    gallery = torch.from_numpy(sem[uniq_ids]).to(device)
    return gallery, torch.from_numpy(labels).to(device)


# --------------------------------------------------------------------------------------------
# Synthetic data path (--smoke-test): no files touched, real per-subject voxel dims
# --------------------------------------------------------------------------------------------

def build_synthetic_subject(subj: int, id_offset: int, n_fit: int, n_dev: int,
                             rng: np.random.Generator):
    """Return the same 5-tuple as load_subject_split (betas, img73k, is_shared, r, dev_trial), so
    that the fit_mask / dev_mask derivation (~is_shared & ~dev_trial and ~is_shared & dev_trial)
    works unchanged on synthetic data. id_offset keeps the synthetic image ids of every subject in
    a disjoint block: the real data satisfy the invariant "non-shared trials are subject-specific",
    and here it is reproduced on purpose."""
    n = n_fit + n_dev
    v = NUM_VOXELS[subj]
    betas = rng.normal(size=(n, v)).astype(np.float32) * 300.0  # zscore_fmri divides by /300
    img73k = np.arange(id_offset, id_offset + n, dtype=np.int64)
    is_shared = np.zeros(n, dtype=bool)
    r = rng.uniform(0.3, 1.0, size=n).astype(np.float32)
    dev_trial = np.zeros(n, dtype=bool)
    dev_trial[n_fit:] = True
    return betas, img73k, is_shared, r, dev_trial


def build_synthetic_data(subjects, n_fit_per_subj: int, n_dev_per_subj: int, dim_low: int,
                          seed: int):
    """One synthetic sem / struct / low target pool that covers the synthetic image ids of every
    subject (disjoint blocks, see build_synthetic_subject), plus a per_subj dict of raw data in
    the shape that load_subject_split returns. The caller applies zscore_fmri as it would to real
    data; only the files are replaced, not the processing code."""
    rng = np.random.default_rng(seed)
    n_total_images = len(subjects) * (n_fit_per_subj + n_dev_per_subj)
    sem = rng.normal(size=(n_total_images, 256)).astype(np.float32)
    struct = rng.normal(size=(n_total_images, 2048)).astype(np.float32)
    low = rng.normal(size=(n_total_images, dim_low)).astype(np.float32)

    raw = {}
    offset = 0
    for subj in subjects:
        raw[subj] = build_synthetic_subject(subj, offset, n_fit_per_subj, n_dev_per_subj, rng)
        offset += n_fit_per_subj + n_dev_per_subj
    return raw, sem, struct, low


# --------------------------------------------------------------------------------------------
# Resource / duration logging (RAM, VRAM, disk, wall-clock; for the thesis' cost tables)
# --------------------------------------------------------------------------------------------

class ResourceMonitor:
    """psutil for RAM and process RSS, torch.cuda for per-GPU VRAM, shutil.disk_usage for the
    output filesystem. All three are cheap (<10 ms); one call per epoch has no measurable
    effect on training throughput."""

    def __init__(self, gpu_ids, out_dir: Path):
        self.gpu_ids = gpu_ids
        self.out_dir = out_dir
        try:
            import psutil
            self._proc = psutil.Process()
            self._psutil = psutil
        except ImportError:
            self._proc, self._psutil = None, None
        self.disk_used_start_gb = self._disk_usage_gb(out_dir)
        if torch.cuda.is_available():
            for g in gpu_ids:
                torch.cuda.reset_peak_memory_stats(g)

    @staticmethod
    def _disk_usage_gb(path: Path) -> float:
        """Size of `path` itself (checkpoints and so on), not of the whole filesystem: it answers
        "how much disk does this run cost"."""
        total = 0
        if path.exists():
            for f in path.rglob("*"):
                if f.is_file():
                    total += f.stat().st_size
        return total / 2**30

    def snapshot(self) -> dict:
        out = {"timestamp": datetime.now().isoformat(timespec="seconds")}
        if self._proc is not None:
            out["ram_rss_gb"] = self._proc.memory_info().rss / 2**30
            vm = self._psutil.virtual_memory()
            out["ram_system_used_gb"] = vm.used / 2**30
            out["ram_system_total_gb"] = vm.total / 2**30
        du = shutil.disk_usage(self.out_dir if self.out_dir.exists() else self.out_dir.parent)
        out["disk_free_gb"] = du.free / 2**30
        out["disk_used_by_out_dir_gb"] = self._disk_usage_gb(self.out_dir)
        out["vram"] = {}
        for g in self.gpu_ids:
            out["vram"][g] = {
                "allocated_gb": torch.cuda.memory_allocated(g) / 2**30,
                "reserved_gb": torch.cuda.memory_reserved(g) / 2**30,
                "max_allocated_gb": torch.cuda.max_memory_allocated(g) / 2**30,
            }
        return out

    def peak_vram_gb(self) -> dict:
        return {g: torch.cuda.max_memory_allocated(g) / 2**30 for g in self.gpu_ids}


# --------------------------------------------------------------------------------------------
# LR schedule: cosine, with a linear warm-up
# --------------------------------------------------------------------------------------------

def make_lr_lambda(total_steps: int, warmup_frac: float):
    warmup_steps = max(1, int(warmup_frac * total_steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return lr_lambda, warmup_steps


# --------------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------------

def train(args):
    device_str = "cuda" if (torch.cuda.is_available() and args.gpus != []) else "cpu"
    if device_str == "cuda":
        gpu_ids = args.gpus if args.gpus is not None else list(range(torch.cuda.device_count()))
    else:
        gpu_ids = []
    device = torch.device(f"cuda:{gpu_ids[0]}") if gpu_ids else torch.device("cpu")
    print(f"device={device}  gpus={gpu_ids or 'cpu'}"
          + (f"  ({', '.join(torch.cuda.get_device_name(g) for g in gpu_ids)})" if gpu_ids else ""))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    monitor = ResourceMonitor(gpu_ids, out_dir)

    rng = np.random.default_rng(args.seed)
    if args.smoke_test:
        print(f"--smoke-test: synthetic data, {args.smoke_n_fit} fit / {args.smoke_n_dev} dev "
              f"images per subject, dim_low={args.smoke_dim_low} (no files touched)")
        raw, sem, struct, low = build_synthetic_data(
            args.subj, args.smoke_n_fit, args.smoke_n_dev, args.smoke_dim_low, args.seed)
        dim_low = args.smoke_dim_low
    else:
        sem, struct, low = load_targets(args.targets, args.zlow)
        dim_low = low.shape[1]
        raw = {}
        for subj in args.subj:
            raw[subj] = load_subject_split(subj, Path(args.index_dir), Path(args.betas_dir),
                                            Path(args.r_dir), Path(args.dev_mask_dir))

    per_subj = {}
    for subj in args.subj:
        betas, img73k, is_shared, r, dev_trial = raw[subj]
        fit_mask = ~is_shared & ~dev_trial
        dev_mask = ~is_shared & dev_trial
        fmri_z = zscore_fmri(betas, fit_mask)
        per_subj[subj] = {"fmri_z": fmri_z, "r": r, "img73k": img73k,
                           "fit_mask": fit_mask, "dev_mask": dev_mask}
        print(f"  subj{subj:02d}: fit={fit_mask.sum()}  dev={dev_mask.sum()}  "
              f"shared1000(excluded)={is_shared.sum()}")

    # Target moments of the fit split (the same moments zscore_fmri() uses for the voxels, here for
    # the three factor targets). Training does not need them, because the loss regresses onto
    # target-scale values; predict_mlp.py does. Computed once over the union of the distinct fit
    # images of every subject and saved, instead of being derived again by the prediction scripts.
    fit_target_moments = compute_fit_target_moments(per_subj, sem, struct, low)
    np.savez(out_dir / "fit_target_moments.npz", **fit_target_moments)

    fit_ds = MultiSubjectFitDataset(per_subj, sem, struct, low, rng)
    print(f"combined fit set: {len(fit_ds)} distinct (subj, image) pairs across "
          f"{len(args.subj)} subjects")

    loader = DataLoader(fit_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
                         drop_last=True, collate_fn=collate_multisubject)
    steps_per_epoch = len(loader)
    if steps_per_epoch == 0:
        sys.exit(f"--batch-size {args.batch_size} > fit set {len(fit_ds)}, nothing to train on")

    use_ctr = args.batch_size >= MIN_BATCH_FOR_CTR
    if not use_ctr and args.delta_ctr > 0:
        print(f"  WARNING: --batch-size {args.batch_size} < {MIN_BATCH_FOR_CTR}, L_ctr will be "
              f"disabled every step (the InfoNCE term needs at least MIN_BATCH_FOR_CTR distinct stimuli)")

    dev_subjs, dev_voxels, dev_targets, dev_img_ids = dev_tensors_multisubject(
        per_subj, sem, struct, low, device)
    gallery, gallery_labels = dev_retrieval_gallery_multisubject(per_subj, sem, device)

    model = MLPEncoder(subjects=args.subj, dim_low=dim_low, hidden=args.hidden,
                        n_trunk_layers=args.trunk_layers, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params / 1e6:.2f}M parameters, dims={model.dims}")

    train_model = nn.DataParallel(model, device_ids=gpu_ids) if len(gpu_ids) > 1 else model

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.epochs * steps_per_epoch
    lr_lambda, warmup_steps = make_lr_lambda(total_steps, args.warmup_frac)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"{args.epochs} epochs x {steps_per_epoch} steps/epoch = {total_steps} steps "
          f"({warmup_steps} warm-up, {args.warmup_frac:.0%})")

    amp_dtype = torch.bfloat16 if device.type == "cuda" else None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"resumed from {args.resume} at epoch {start_epoch}")
    else:
        start_epoch = 1

    history_path = out_dir / "history.csv"
    history_fields = ["epoch", "global_step", "lr", "train_loss", "train_sem", "train_struct",
                       "train_low", "train_ctr", "dev_loss", "dev_sem", "dev_struct", "dev_low",
                       "dev_sem_retrieval_acc", "elapsed_s", "ram_rss_gb", "disk_free_gb",
                       "vram_max_allocated_gb"]
    write_header = not (history_path.exists() and args.resume)
    history_file = open(history_path, "a" if args.resume else "w", newline="")
    history_writer = csv.DictWriter(history_file, fieldnames=history_fields)
    if write_header:
        history_writer.writeheader()

    resource_log_path = out_dir / "resource_log.jsonl"
    resource_log_file = open(resource_log_path, "a" if args.resume else "w")

    run_t0 = time.monotonic()
    global_step = (start_epoch - 1) * steps_per_epoch
    best_epoch, best_dev_loss, best_retrieval_acc = None, float("inf"), None
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.monotonic()
        epoch_loss = epoch_sem = epoch_struct = epoch_low = epoch_ctr = 0.0
        model.train()
        for subjs, voxels, sem_t, struct_t, low_t in loader:
            target = {"sem": sem_t.to(device), "struct": struct_t.to(device),
                      "low": low_t.to(device)}
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                pred = forward_multisubject(train_model, subjs, voxels, device)
                losses = encoder_loss(pred, target, delta_ctr=args.delta_ctr,
                                       ctr_temp=args.ctr_temp)
            loss = losses["total"]
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_sem += losses["sem"].item()
            epoch_struct += losses["struct"].item()
            epoch_low += losses["low"].item()
            if use_ctr:
                epoch_ctr += losses["ctr"].item()
            global_step += 1

        dev_loss, dev_sem, dev_struct, dev_low, retrieval_acc = evaluate_dev(
            model, dev_subjs, dev_voxels, dev_targets, gallery, gallery_labels, device)
        elapsed = time.monotonic() - t0
        snap = monitor.snapshot()
        now = datetime.now().strftime("%H:%M:%S")
        train_ctr = epoch_ctr / steps_per_epoch if use_ctr else float("nan")
        vram_max = max((v["max_allocated_gb"] for v in snap["vram"].values()), default=0.0)
        print(f"[{now}] epoch {epoch:3d}/{args.epochs}  lr={scheduler.get_last_lr()[0]:.2e}  "
              f"train_loss={epoch_loss/steps_per_epoch:.4f}  train_ctr={train_ctr:.4f}  "
              f"dev_loss={dev_loss:.4f} (sem={dev_sem:.4f} struct={dev_struct:.4f} "
              f"low={dev_low:.4f})  dev_retrieval@1={retrieval_acc:.3f}  ({elapsed:.1f}s)  "
              f"ram={snap.get('ram_rss_gb', float('nan')):.2f}GB  vram_peak={vram_max:.2f}GB",
              flush=True)

        history_writer.writerow({
            "epoch": epoch, "global_step": global_step, "lr": scheduler.get_last_lr()[0],
            "train_loss": epoch_loss / steps_per_epoch, "train_sem": epoch_sem / steps_per_epoch,
            "train_struct": epoch_struct / steps_per_epoch, "train_low": epoch_low / steps_per_epoch,
            "train_ctr": train_ctr, "dev_loss": dev_loss, "dev_sem": dev_sem,
            "dev_struct": dev_struct, "dev_low": dev_low,
            "dev_sem_retrieval_acc": retrieval_acc, "elapsed_s": elapsed,
            "ram_rss_gb": snap.get("ram_rss_gb", float("nan")),
            "disk_free_gb": snap["disk_free_gb"], "vram_max_allocated_gb": vram_max,
        })
        history_file.flush()
        resource_log_file.write(json.dumps({"epoch": epoch, "elapsed_s": elapsed, **snap}) + "\n")
        resource_log_file.flush()

        # The adapters alone have ~300M parameters (4 subjects x NUM_VOXELS[s] x hidden), and the two
        # momentum buffers of AdamW roughly triple that on disk. The optimizer and scheduler state
        # is therefore kept only in model.pt, the one checkpoint that --resume reads. model_best.pt
        # and every --ckpt-every snapshot are for evaluation, are never resumed from, and hold
        # only the weights (3.66 GB become ~1.2 GB per file at this model size).
        light_ckpt = {"state_dict": model.state_dict(), "epoch": epoch,
                      "dims": model.dims, "subjects": model.subjects}
        if dev_loss < best_dev_loss:
            best_epoch, best_dev_loss, best_retrieval_acc = epoch, dev_loss, retrieval_acc
            torch.save(light_ckpt, out_dir / "model_best.pt")
        if args.ckpt_every and epoch % args.ckpt_every == 0:
            torch.save(light_ckpt, out_dir / f"model_epoch{epoch:04d}.pt")
        torch.save({**light_ckpt, "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()},
                   out_dir / "model.pt")  # cheap insurance, overwritten every epoch, resumable

    history_file.close()
    resource_log_file.close()
    total_elapsed = time.monotonic() - run_t0

    summary = {
        "best_epoch": best_epoch, "best_dev_loss": best_dev_loss,
        "best_dev_sem_retrieval_acc": best_retrieval_acc,
        "epochs": args.epochs, "batch_size": args.batch_size, "delta_ctr": args.delta_ctr,
        "lr": args.lr, "hidden": args.hidden, "trunk_layers": args.trunk_layers,
        "n_params": n_params, "subjects": list(args.subj),
        "total_wall_clock_s": total_elapsed,
        "peak_vram_gb": monitor.peak_vram_gb(),
        "disk_used_by_out_dir_gb": monitor._disk_usage_gb(out_dir),
        "disk_used_delta_gb": monitor._disk_usage_gb(out_dir) - monitor.disk_used_start_gb,
        "gpu_names": [torch.cuda.get_device_name(g) for g in gpu_ids],
        "n_gpus": len(gpu_ids) or 0,
        "smoke_test": args.smoke_test,
    }
    with open(out_dir / "run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out_dir}  (best epoch {best_epoch}, dev_loss={best_dev_loss:.4f}, "
          f"dev_retrieval@1={best_retrieval_acc:.3f}, wall-clock={total_elapsed:.1f}s)")


def compute_fit_target_moments(per_subj: dict, sem, struct, low) -> dict:
    """Per-factor mean and std of the targets of the fit split, over the union of the distinct fit
    images of every subject (each counted once, not once per trial). Only this script has them;
    restandardize_output() below is the matching consumer."""
    fit_img_ids = []
    for d in per_subj.values():
        rows = np.flatnonzero(d["fit_mask"])
        fit_img_ids.append(np.unique(d["img73k"][rows]))
    fit_img_ids = np.unique(np.concatenate(fit_img_ids))
    return {
        "sem_mean": sem[fit_img_ids].mean(axis=0), "sem_std": sem[fit_img_ids].std(axis=0),
        "struct_mean": struct[fit_img_ids].mean(axis=0), "struct_std": struct[fit_img_ids].std(axis=0),
        "low_mean": low[fit_img_ids].mean(axis=0), "low_std": low[fit_img_ids].std(axis=0),
    }


def restandardize_output(pred_np: np.ndarray, factor: str, fit_target_moments: dict) -> np.ndarray:
    """Helper for the prediction scripts (the training loop does not use it): rescale a batch of
    predictions of the trained model from their own pool scale onto
    fit_target_moments[f'{factor}_{mean,std}'], with the per-dimension recipe of
    trisect.ridge.restandardize(), the same one that the output of fit_ridge.py goes through."""
    return restandardize(pred_np, pred_np.mean(axis=0), pred_np.std(axis=0),
                          fit_target_moments[f"{factor}_mean"], fit_target_moments[f"{factor}_std"])


@torch.no_grad()
def evaluate_dev(model, dev_subjs, dev_voxels, dev_targets, gallery, gallery_labels, device):
    model.eval()
    pred = forward_multisubject(model, dev_subjs, dev_voxels, device)
    losses = encoder_loss(pred, dev_targets, delta_ctr=0.0)
    pred_sem_n = torch.nn.functional.normalize(pred["sem"], dim=-1)
    gallery_n = torch.nn.functional.normalize(gallery, dim=-1)
    nn_idx = (pred_sem_n @ gallery_n.T).argmax(dim=-1)
    acc = (nn_idx == gallery_labels).float().mean().item()
    model.train()
    return (losses["total"].item(), losses["sem"].item(), losses["struct"].item(),
            losses["low"].item(), acc)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subj", type=int, nargs="+", default=list(SUBJECTS), choices=list(SUBJECTS))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=256,
                     help="64-256; the contrastive term needs the upper end (more InfoNCE negatives), "
                          "so 256 is the default. The floor MIN_BATCH_FOR_CTR=64 still applies "
                          "below it")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--warmup-frac", type=float, default=0.05,
                     help="linear LR warm-up fraction of total steps (cosine schedule, 5%% warm-up)")
    ap.add_argument("--delta-ctr", type=float, default=0.0,
                     help="weight of the InfoNCE term on z_sem. This script does not decide it: the "
                          "value comes from a sweep with a criterion fixed in advance on the "
                          "NSD-core development split. The default 0.0 (disabled) is on purpose, so "
                          "that a bare invocation cannot count as the real run.")
    ap.add_argument("--ctr-temp", type=float, default=0.07)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--trunk-layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--gpus", type=int, nargs="*", default=None,
                     help="GPU ids to use, e.g. --gpus 0 1 (DataParallel across them, batch "
                          "split by torch's default scatter). Omit for all visible GPUs, pass "
                          "--gpus with no values or on a CPU-only machine for CPU.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=0,
                     help="also save model_epoch{N}.pt every N epochs, on top of model.pt "
                          "(last) and model_best.pt (lowest dev_loss); 0 = off")
    ap.add_argument("--resume", type=str, default=None,
                     help="path to a checkpoint (model.pt/model_best.pt/model_epochNNNN.pt) to "
                          "resume model+optimizer+scheduler state from")
    ap.add_argument("--index-dir", type=str, default=INDEX_DIR)
    ap.add_argument("--betas-dir", type=str, default=BETAS_DIR)
    ap.add_argument("--targets", type=str, default=TARGETS_H5)
    ap.add_argument("--zlow", type=str, default=ZLOW_PATH)
    ap.add_argument("--r-dir", type=str, default=R_DIR)
    ap.add_argument("--dev-mask-dir", type=str, default=DEV_MASK_DIR)
    ap.add_argument("--out-dir", type=str, default=OUT_DIR,
                     help="pass a distinct dir per experiment arm (e.g. a delta-ctr sweep) -- "
                          "this script never namespaces runs for you")
    ap.add_argument("--smoke-test", action="store_true",
                     help="synthetic in-memory data at the real per-subject voxel counts, no files "
                          "needed. A check that the loop runs, not a real training.")
    ap.add_argument("--smoke-n-fit", type=int, default=300)
    ap.add_argument("--smoke-n-dev", type=int, default=80)
    ap.add_argument("--smoke-dim-low", type=int, default=64)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train(args)


if __name__ == "__main__":
    main()
