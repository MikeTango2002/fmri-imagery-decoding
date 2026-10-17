"""Helpers shared by the MLP encoder scripts: target loading, the NSD-core split, InfoNCE.

The three functions and two constants below are the ones that the MLP scripts import.
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import torch.nn.functional as F

# nsdgeneral voxel counts per subject, as in prepare_nsd_imagery_icnn.py (the reproduction of iCNN)
NUM_VOXELS = {1: 15724, 2: 14278, 5: 13039, 7: 12682}

MIN_BATCH_FOR_CTR = 64  # InfoNCE needs at least this many distinct stimuli per batch


def load_targets(targets_h5, zlow_path):
    """(73000, 256/2048/d_low) targets, same 73k id used everywhere. sem is (img_128 || txt_128),
    in the same order as the ridge weights; the predicted sem lines up with these targets
    column for column."""
    print(f"Loading targets: {targets_h5}")
    with h5py.File(targets_h5, "r") as f:
        z_sem = f["z_sem"][:].astype(np.float32)        # (73000, 2, 128)
        z_struct = f["z_struct"][:].astype(np.float32)  # (73000, 8, 8, 32)
    print(f"Loading z_low: {zlow_path}")
    z_low = np.load(zlow_path).astype(np.float32)         # (73000, d_low)

    sem = np.concatenate([z_sem[:, 0, :], z_sem[:, 1, :]], axis=1)  # (73000, 256)
    struct = z_struct.reshape(z_struct.shape[0], -1)                # (73000, 2048)
    return sem, struct, z_low


def load_subject_split(subj, index_dir: Path, betas_dir: Path, r_dir: Path, dev_mask_dir: Path):
    index_file = index_dir / f"subj{subj:02d}_trial73k_index.npz"
    if not index_file.exists():
        sys.exit(f"missing {index_file}: run prepare_nsdcore_index.py first")
    idx = np.load(index_file)
    img73k, is_shared = idx["img73k"], idx["is_shared"]

    betas_file = betas_dir / f"betas_all_subj{subj:02d}_fp32_renorm.hdf5"
    if not betas_file.exists():
        sys.exit(f"missing {betas_file}: run download_data_from_internet.sh --betas first")
    with h5py.File(betas_file, "r") as h:
        betas = np.asarray(h["betas"], dtype=np.float32)  # (30000, V)
    assert betas.shape[1] == NUM_VOXELS[subj], \
        f"{betas.shape[1]} voxels, expected {NUM_VOXELS[subj]} (nsdgeneral mask mismatch)"

    r_file = r_dir / f"subj{subj:02d}_r.npy"
    if not r_file.exists():
        sys.exit(f"missing {r_file}: run prepare_betas.py --stage core first")
    r = np.load(r_file).astype(np.float32)  # (30000,)

    dev_mask_file = dev_mask_dir / f"nsdcore_dev_mask_subj{subj:02d}.npy"
    if not dev_mask_file.exists():
        sys.exit(f"missing {dev_mask_file}: run derive_nsdcore_dev_mask.py first")
    dev_mask_73k = np.load(dev_mask_file)  # (73000,) bool, per IMAGE id
    dev_trial = dev_mask_73k[img73k]       # (30000,) bool, per TRIAL

    return betas, img73k, is_shared, r, dev_trial


def info_nce_loss(pred, target, temp):
    """Symmetric InfoNCE, adapted from the soft_clip_loss of the MindEye1 code (it normalizes
    internally here instead of requiring the caller to, otherwise the same formula): the predicted
    sem embedding of row i should be the closest match to the true sem target i among the other
    (distinct) stimuli of the batch, in both directions."""
    pred_n = F.normalize(pred, dim=-1)
    target_n = F.normalize(target, dim=-1)
    target_target = (target_n @ target_n.T) / temp
    pred_target = (pred_n @ target_n.T) / temp
    loss1 = -(pred_target.log_softmax(-1) * target_target.softmax(-1)).sum(-1).mean()
    loss2 = -(pred_target.T.log_softmax(-1) * target_target.T.softmax(-1)).sum(-1).mean()
    return (loss1 + loss2) / 2
