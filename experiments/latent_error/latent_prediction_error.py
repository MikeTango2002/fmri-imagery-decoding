"""Predicted against true latents, per branch and subject, on shared1000 and NSD-Imagery (Section 8.3).

Inputs:  nsdcore_predictions/ and imagery_predictions/ (predict_ridge_*.py), train_target_stats.npz,
         targets.h5, targets_imagery18.npz, zlow_target_noise_ceiling.json (read if present)
Outputs: data/results/latent_error/latent_prediction_error.tsv

The four branches share one ridge kernel and differ only in the space they are rendered into;
they are reported side by side: if all collapse out of distribution, the failure is in the kernel.
Everything is centered on the TRAINING target mean, the ridge's own baseline output; an uncentered
cosine is inflated by the shared mean. Read the z_low ceiling (zlow_target_noise_ceiling.py) first.

Usage (pure numpy, takes seconds):
    python latent_prediction_error.py [--out PATH]
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_ridge import two_way  # noqa: E402

SUBJECTS = [1, 2, 5, 7]
BRANCHES = ["z_sem_img", "z_sem_txt", "z_struct", "z_low"]
S1K_PRED_DIR = "data/betas_trisect/nsdcore_predictions"
IMG_PRED_DIR = "data/betas_trisect/imagery_predictions"
WEIGHTS_DIR = "data/betas_trisect/ridge_weights"
TARGETS_H5 = "data/targets/targets.h5"
ZLOW_PATH = "data/targets/pca_zlow/z_low.npy"
IMG18_TARGETS = "data/targets/targets_imagery18.npz"
OUT_DEFAULT = "data/results/latent_error/latent_prediction_error.tsv"
CEILING_JSON = "data/results/latent_error/zlow_target_noise_ceiling.json"

# NSD-Imagery pools. Vision and imagery hold the same 12 benchmark stimuli; their difference is
# the regime alone; the concepts (6 stimuli) are outside the benchmark and reported for completeness.
# z_sem_txt has no imagery target (no caption source was fixed for the 18 stimuli): s1k rows only.
IMAGERY_POOLS = {
    "img_vision": ["visionsimple", "visioncomplex"],
    "img_imagery": ["imagerysimple", "imagerycomplex"],
    "img_concepts": ["imageryconcepts"],
}


def flat(a):
    return a.reshape(a.shape[0], -1).astype(np.float64)


def cosines(pred, gt):
    """Full (n,n) cosine matrix between centered predictions and centered targets."""
    p = pred / np.maximum(np.linalg.norm(pred, axis=1, keepdims=True), 1e-12)
    g = gt / np.maximum(np.linalg.norm(gt, axis=1, keepdims=True), 1e-12)
    return p @ g.T


def stats_for(pred, gt, mean):
    """pred/gt: (n, d) raw; mean: (d,) training-target mean. -> dict of scalars.

    cos_diag     mean_i cos(pred_i - m, gt_i - m): does the prediction point at the right target
    cos_offdiag  mean_{i!=j} cos(pred_i - m, gt_j - m): the generic direction shared by all stimuli;
                 cos_diag - cos_offdiag (cos_gap) is the part that identifies
    r2_vs_mean   1 - mean_i ||pred_i - gt_i||^2 / mean_i ||gt_i - m||^2; negative means worse than
                 emitting the training mean, which is normal after restandardize() re-inflates
    norm_ratio   mean_i ||pred_i - m|| / mean_i ||gt_i - m||: shrinkage left after restandardize()
    two_way      fit_ridge.two_way(gt, pred); 0.50 is chance, normalized by pool size, and shared1000
                 (n=1000) and NSD-Imagery (n=12) are comparable
    top1         rank-1 identification in the same pool
    """
    p, g = flat(pred) - mean, flat(gt) - mean
    n = p.shape[0]
    c = cosines(p, g)
    off = c[~np.eye(n, dtype=bool)]
    err = np.mean(np.sum((flat(pred) - flat(gt)) ** 2, axis=1))
    var = np.mean(np.sum(g ** 2, axis=1))
    rank1 = np.mean(np.argmax(c, axis=1) == np.arange(n))
    return {
        "n": n,
        "cos_diag": float(np.mean(np.diag(c))),
        "cos_offdiag": float(np.mean(off)),
        "cos_gap": float(np.mean(np.diag(c)) - np.mean(off)),
        "r2_vs_mean": float(1.0 - err / var),
        "norm_ratio": float(np.mean(np.linalg.norm(p, axis=1)) / np.mean(np.linalg.norm(g, axis=1))),
        "two_way": float(two_way(flat(gt), flat(pred))),
        "top1": float(rank1),
    }


def load_s1k_targets(ids):
    with h5py.File(TARGETS_H5, "r") as f:
        z_sem = f["z_sem"][:].astype(np.float32)
        z_struct = f["z_struct"][:].astype(np.float32)
    z_low = np.load(ZLOW_PATH, mmap_mode="r")
    return {
        "z_sem_img": z_sem[ids, 0], "z_sem_txt": z_sem[ids, 1],
        "z_struct": z_struct[ids], "z_low": np.asarray(z_low[ids], dtype=np.float32),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args()

    img18 = np.load(IMG18_TARGETS)
    img_gt = {"z_sem_img": img18["z_sem_img"], "z_struct": img18["z_struct"],
              "z_low": img18["z_low"][0]}

    rows = []
    for subj in SUBJECTS:
        sd = f"subj{subj:02d}"
        stats = np.load(os.path.join(WEIGHTS_DIR, sd, "train_target_stats.npz"))
        means = {b: stats[f"{b}_mean"].astype(np.float64) for b in BRANCHES}

        ids = np.load(os.path.join(S1K_PRED_DIR, sd, "img73k_ids.npy"))
        s1k_gt = load_s1k_targets(ids)
        for b in BRANCHES:
            pred = np.load(os.path.join(S1K_PRED_DIR, sd, f"{b}_pred.npy"))
            rows.append(dict(subject=sd, domain="s1k", branch=b,
                             **stats_for(pred, s1k_gt[b], means[b])))

        labels = np.load(os.path.join(IMG_PRED_DIR, sd, "labels.npy"), allow_pickle=True)
        conds = np.array([str(l).rsplit("-", 1)[0] for l in labels])
        gidx = np.array([int(str(l).rsplit("-", 1)[1]) for l in labels])
        for pool, keep in IMAGERY_POOLS.items():
            sel = np.isin(conds, keep)
            order = np.argsort(gidx[sel])
            for b in BRANCHES:
                if b not in img_gt:
                    continue
                pred = np.load(os.path.join(IMG_PRED_DIR, sd, f"{b}_pred.npy"))[sel][order]
                gt = img_gt[b][gidx[sel][order]]
                rows.append(dict(subject=sd, domain=pool, branch=b,
                                 **stats_for(pred, gt, means[b])))

    ceiling = {}
    if os.path.isfile(CEILING_JSON):
        with open(CEILING_JSON) as f:
            ceiling = json.load(f)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cols = ["subject", "domain", "branch", "n", "cos_diag", "cos_offdiag", "cos_gap",
            "r2_vs_mean", "norm_ratio", "two_way", "top1"]
    with open(args.out, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])
                              for c in cols) + "\n")
    # console summary: mean over subjects
    if ceiling:
        print(f"\nz_low TARGET noise ceiling ({CEILING_JSON}) -- VDVAE posterior draws:")
        for pool in ["s1k", "imagery12"]:
            c = ceiling.get(pool)
            if c:
                print(f"  {pool:<10} cos_diag ceiling={c['cos_ceiling_perfect_decoder']:.4f}   "
                      f"two_way ceiling={c['two_way']:.4f}")
    print()
    hdr = f"{'domain':<13}{'branch':<12}" + "".join(f"{c:>12}" for c in cols[4:])
    print(hdr); print("-" * len(hdr))
    for domain in ["s1k", "img_vision", "img_imagery", "img_concepts"]:
        for b in BRANCHES:
            sel = [r for r in rows if r["domain"] == domain and r["branch"] == b]
            if not sel:
                continue
            line = f"{domain:<13}{b:<12}"
            line += "".join(f"{np.mean([r[c] for r in sel]):>12.4f}" for c in cols[4:])
            print(line)
        print()
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
