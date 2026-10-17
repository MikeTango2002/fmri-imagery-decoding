"""Write the MLPEncoder predictions in the layout of the ridge scripts (Section 8.1.1).

Inputs:  --ckpt-dir (model, fit_target_moments.npz) and the NSD-Imagery or shared1000 betas
Outputs: --out-dir/subj0N/{z_sem_img,z_sem_txt,z_struct,z_low}_pred.npy; reconstruct.py only needs
         another --pred-dir

The raw output is restandardized onto the fit-split target moments (not the ridge's), with the pool
convention of the matching ridge script. --trial-level should be ON for imagery: the MLP was trained on
single trials (std ~1) and repetition-averaged betas (std ~0.30) are a ~3.4x shrink it does not absorb,
unlike the ridge (see README). It predicts per trial and averages the predictions instead.

Usage:
    python predict_mlp.py --dataset imagery --trial-level --ckpt-dir CKPT_DIR
    python predict_mlp.py --dataset shared1000 --ckpt-dir CKPT_DIR
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mlp_encoder import MLPEncoder  # noqa: E402
from fit_ridge import average_by_image  # noqa: E402
from predict_ridge_imagery import (  # noqa: E402  (reused verbatim, not re-derived)
    average_by_label, group_rows, load_trial_level,
)
from decoder_training_utils import load_subject_split  # noqa: E402
from trisect.nsd import zscore_fmri  # noqa: E402
from trisect.ridge import restandardize  # noqa: E402

SUBJECTS = (1, 2, 5, 7)

CKPT_DIR_DEFAULT = "data/mlp_encoder_checkpoints/delta_ctr_0.2"
IMAGERY_DIR = "data/betas_trisect/nsd_imagery"
INDEX_DIR = "data/nsdcore_index"
BETAS_DIR = "data/mindeyev2_dataset"
DEV_MASK_DIR = "data/targets"
R_DIR = "data/betas_trisect/nsdcore_r"
OUT_DIR_IMAGERY = "data/betas_trisect/mlp_predictions_imagery"
OUT_DIR_SHARED1000 = "data/betas_trisect/mlp_predictions_shared1000"


def load_model(ckpt_dir: Path):
    ckpt_path = ckpt_dir / "model_best.pt"
    if not ckpt_path.exists():
        sys.exit(f"missing {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = MLPEncoder(subjects=ckpt["subjects"], dim_low=ckpt["dims"]["low"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    moments_path = ckpt_dir / "fit_target_moments.npz"
    if not moments_path.exists():
        sys.exit(f"missing {moments_path}")
    moments = np.load(moments_path)
    return model, moments


@torch.no_grad()
def predict_raw(model: MLPEncoder, voxels: np.ndarray, subj: int) -> dict:
    """(N, V_subj) -> {factor: (N, D) numpy}, MODEL'S OWN SCALE (not yet restandardized)."""
    voxels_t = torch.from_numpy(voxels.astype(np.float32))
    out = model(voxels_t, subj=subj)
    return {f: v.numpy() for f, v in out.items()}


def split_branches(raw: dict) -> dict:
    """{sem, struct, low} model output -> the four ridge-branch-shaped arrays (native shape,
    flattened: callers reshape z_struct after restandardizing)."""
    sem = raw["sem"]
    d_sem = sem.shape[-1] // 2
    return {
        "z_sem_img": sem[:, :d_sem],
        "z_sem_txt": sem[:, d_sem:],
        "z_struct": raw["struct"],          # stays flat (N, 2048) until reshaped at save time
        "z_low": raw["low"],
    }


BRANCH_FACTOR = {"z_sem_img": "sem", "z_sem_txt": "sem", "z_struct": "struct", "z_low": "low"}
BRANCH_RESHAPE = {"z_struct": (8, 8, 32)}


def target_moments_for_branch(branch: str, moments) -> tuple:
    """fit_target_moments.npz stores one mean/std per FACTOR (sem 256-d, unsplit), split sem's
    moments the same way split_branches() splits sem's predictions."""
    factor = BRANCH_FACTOR[branch]
    mean, std = moments[f"{factor}_mean"], moments[f"{factor}_std"]
    if branch == "z_sem_img":
        return mean[:128], std[:128]
    if branch == "z_sem_txt":
        return mean[128:], std[128:]
    return mean, std


def write_predictions(out_dir: Path, branches: dict, labels=None, img73k_ids=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    if labels is not None:
        np.save(out_dir / "labels.npy", labels)
    if img73k_ids is not None:
        np.save(out_dir / "img73k_ids.npy", img73k_ids)
    for branch, arr in branches.items():
        np.save(out_dir / f"{branch}_pred.npy", arr)
        print(f"  [{branch}] -> {out_dir / f'{branch}_pred.npy'} {arr.shape}  "
              f"(mean {arr.mean():.3f}, std {arr.std():.3f})", flush=True)


def process_imagery(ckpt_dir: Path, imagery_dir: Path, out_dir: Path, pool_mode: str,
                    trial_level: bool = False):
    print(f"\n=== MLPEncoder (imagery{', trial-level' if trial_level else ''}) ===")
    model, moments = load_model(ckpt_dir)

    for subj in model.subjects:
        subj_imagery_dir = imagery_dir / f"subj{subj:02d}"
        labels = np.load(subj_imagery_dir / "labels_averaged.npy")
        if trial_level:
            betas, labels_flat = load_trial_level(subj_imagery_dir)
            src_name = "betas_flat.npy"
        else:
            betas = np.load(subj_imagery_dir / "betas_averaged.npy").astype(np.float32)  # (30, V)
            labels_flat = None
            src_name = "betas_averaged.npy"
        print(f" -- subj0{subj}: input {src_name} {betas.shape}, std {betas.std():.4f} "
              f"(training distribution is ~1.0)", flush=True)

        raw = split_branches(predict_raw(model, betas, subj))
        if trial_level:
            raw = {b: average_by_label(a, labels_flat, labels) for b, a in raw.items()}
        pred = {}
        for branch, arr in raw.items():
            tgt_mean, tgt_std = target_moments_for_branch(branch, moments)
            out = np.empty_like(arr)
            for group_name, rows in group_rows(labels, pool_mode):
                pool = arr[rows]
                pool_mean, pool_std = pool.mean(axis=0), pool.std(axis=0)
                out[rows] = restandardize(pool, pool_mean, pool_std, tgt_mean, tgt_std)
            pred[branch] = out.reshape((out.shape[0],) + BRANCH_RESHAPE.get(branch, tgt_mean.shape))

        subj_out = out_dir / f"subj{subj:02d}"
        write_predictions(subj_out, pred, labels=labels)
    print(f"wrote {out_dir}")


def process_shared1000(ckpt_dir: Path, index_dir: Path, betas_dir: Path, r_dir: Path,
                        dev_mask_dir: Path, out_dir: Path):
    print(f"\n=== MLPEncoder (shared1000) ===")
    model, moments = load_model(ckpt_dir)

    for subj in model.subjects:
        print(f" -- subj0{subj} --")
        betas, img73k, is_shared, r, dev_trial = load_subject_split(
            subj, index_dir, betas_dir, r_dir, dev_mask_dir)
        fit_mask = ~is_shared & ~dev_trial  # matches train_mlp_encoder.py's per_subj fit split
        fmri_z = zscore_fmri(betas, fit_mask)

        test_mask = is_shared
        test_img73k = img73k[test_mask]
        raw = split_branches(predict_raw(model, fmri_z[test_mask], subj))

        pred = {}
        uniq_ids_ref = None
        for branch, arr in raw.items():
            tgt_mean, tgt_std = target_moments_for_branch(branch, moments)
            pool_mean, pool_std = arr.mean(axis=0), arr.std(axis=0)  # predict_ridge_shared1000.py's convention: one pool, before averaging
            restd = restandardize(arr, pool_mean, pool_std, tgt_mean, tgt_std)
            avg, uniq_ids = average_by_image(restd, test_img73k)
            if uniq_ids_ref is None:
                uniq_ids_ref = uniq_ids
            assert np.array_equal(uniq_ids, uniq_ids_ref)
            pred[branch] = avg.reshape((avg.shape[0],) + BRANCH_RESHAPE.get(branch, tgt_mean.shape))

        subj_out = out_dir / f"subj{subj:02d}"
        write_predictions(subj_out, pred, img73k_ids=uniq_ids_ref)
    print(f"wrote {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["imagery", "shared1000"])
    ap.add_argument("--ckpt-dir", type=str, default=CKPT_DIR_DEFAULT)
    ap.add_argument("--imagery-dir", type=str, default=IMAGERY_DIR)
    ap.add_argument("--pool-mode", type=str, default="modality12",
                     choices=["modality12", "pertype"],
                     help="--dataset imagery only, mirrors predict_ridge_imagery.py's default")
    ap.add_argument("--trial-level", action="store_true",
                    help="--dataset imagery only: predict from betas_flat.npy (384 single "
                         "trials, std ~1, the training distribution) and average the "
                         "predictions, instead of predicting from betas_averaged.npy "
                         "(std ~0.30, out of distribution for a non-affine decoder). See the "
                         "module docstring -- this should be ON.")
    ap.add_argument("--index-dir", type=str, default=INDEX_DIR)
    ap.add_argument("--betas-dir", type=str, default=BETAS_DIR)
    ap.add_argument("--r-dir", type=str, default=R_DIR)
    ap.add_argument("--dev-mask-dir", type=str, default=DEV_MASK_DIR)
    ap.add_argument("--out-dir", type=str, default=None,
                     help="defaults to mlp_predictions_{imagery,shared1000} under "
                          "data/betas_trisect/")
    args = ap.parse_args()

    ckpt_dir = Path(args.ckpt_dir)
    if args.dataset == "imagery":
        out_dir = Path(args.out_dir) if args.out_dir else Path(OUT_DIR_IMAGERY)
        process_imagery(ckpt_dir, Path(args.imagery_dir), out_dir, args.pool_mode,
                        trial_level=args.trial_level)
    else:
        if args.trial_level:
            sys.exit("--trial-level applies to --dataset imagery only; the shared1000 arm "
                     "already predicts per trial and averages afterwards")
        out_dir = Path(args.out_dir) if args.out_dir else Path(OUT_DIR_SHARED1000)
        process_shared1000(ckpt_dir, Path(args.index_dir), Path(args.betas_dir), Path(args.r_dir),
                            Path(args.dev_mask_dir), out_dir)

    print("\nall done")


if __name__ == "__main__":
    main()
