"""Measure how much of the z_low target is VDVAE posterior sampling noise (Section 8.3).

Inputs:  COCO images (shared1000 ids), the 18 NSD-Imagery stimuli, the frozen data/targets/pca_zlow basis
Outputs: data/results/latent_error/zlow_target_noise_ceiling.json

The stored target is one SAMPLE of the VDVAE posterior, not its mean. Each pool is encoded twice with
different seeds and projected through the PCA. cos_draws is the signal fraction: a perfect decoder that
emits the posterior mean scores its square root against one draw, the ceiling of cos_diag in
latent_prediction_error.py. two_way(draw0, draw1) is the ceiling of its two_way column. Needs a GPU.

Usage (about 2 minutes):
    python zlow_target_noise_ceiling.py [--out PATH] [--batch-size N]
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "vdvae_lib"))

from extract_imagery_targets import load_stimuli  # noqa: E402
from fit_ridge import two_way  # noqa: E402
from trisect.pca import load_pca_basis  # noqa: E402

IDS_PATH = "data/betas_trisect/nsdcore_predictions/subj01/img73k_ids.npy"
IMAGES_H5 = "data/mindeyev2_dataset/coco_images_224_float16.hdf5"
STIMULI_PT = "data/nsddata_stimuli/stimuli/imagery_stimuli_18.pt"
ZLOW_PATH = "data/targets/pca_zlow/z_low.npy"
TRAIN_STATS = "data/betas_trisect/ridge_weights/subj01/train_target_stats.npz"
OUT_DEFAULT = "data/results/latent_error/zlow_target_noise_ceiling.json"


def cos_rows(a, b):
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return float(np.mean(np.sum(a * b, axis=1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--batch-size", type=int, default=50)
    args = ap.parse_args()

    from model_utils import load_vaes, set_up_data  # noqa: E402
    from extract_vdvae_latents import DEVICE, H_DICT, NUM_LATENTS, VDVAE_RES, dotdict  # noqa: E402

    basis = load_pca_basis(ZLOW_PATH.rsplit("/", 1)[0], dtype=np.float64, layout=True)
    mu, comps = basis["mean"], basis["components"]
    whit = np.sqrt(np.clip(basis["explained_variance"], 1e-12, None))
    dscale = basis["dim_scale"]
    tmean = np.load(TRAIN_STATS)["z_low_mean"].astype(np.float64)

    H = dotdict(dict(H_DICT))
    H, preprocess_fn = set_up_data(H)
    vae = load_vaes(H)

    def project(batch_255):
        with torch.no_grad():
            im = F.interpolate(batch_255, size=(VDVAE_RES, VDVAE_RES),
                               mode="bilinear", align_corners=False)
            data_input, _ = preprocess_fn(im.permute(0, 2, 3, 1).contiguous())
            _, stats = vae.decoder.forward(vae.encoder.forward(data_input), get_latents=True)
            raw = np.hstack([stats[i]["z"].cpu().numpy().reshape(data_input.shape[0], -1)
                             for i in range(NUM_LATENTS)]).astype(np.float64)
        return (((raw / dscale) - mu) @ comps.T) / whit

    def encode_h5(ids, seed):
        torch.manual_seed(seed)
        out = []
        with h5py.File(IMAGES_H5, "r") as f:
            for s in range(0, len(ids), args.batch_size):
                im = torch.from_numpy(f["images"][ids[s:s + args.batch_size]].astype(np.float32))
                out.append(project(im.to(DEVICE) * 255.0))
        return np.vstack(out)

    def encode_pt(imgs01, seed):
        torch.manual_seed(seed)
        return project(imgs01.to(DEVICE) * 255.0)

    res = {}
    ids = np.load(IDS_PATH)
    d0, d1 = encode_h5(ids, 0), encode_h5(ids, 1)
    stored = np.asarray(np.load(ZLOW_PATH, mmap_mode="r")[ids], dtype=np.float64)
    # The stored target is one more draw; cos_reencoded_vs_stored should land on cos_draws; a
    # matching std shows that the dim_scale step is applied.
    res["s1k"] = {
        "n": int(len(ids)),
        "cos_draws": cos_rows(d0 - tmean, d1 - tmean),
        "two_way": two_way(d0, d1),
        "cos_reencoded_vs_stored": cos_rows(d0 - tmean, stored - tmean),
        "std_stored": float(stored.std()), "std_reencoded": float(d0.std()),
    }

    stim = load_stimuli(STIMULI_PT)
    s0, s1 = encode_pt(stim, 0), encode_pt(stim, 1)
    for name, sl in [("imagery12", slice(0, 12)), ("imagery18", slice(0, 18))]:
        res[name] = {"n": int(sl.stop - sl.start),
                     "cos_draws": cos_rows(s0[sl] - tmean, s1[sl] - tmean),
                     "two_way": two_way(s0[sl], s1[sl])}

    for k, v in res.items():
        v["cos_ceiling_perfect_decoder"] = float(np.sqrt(max(v["cos_draws"], 0.0)))
        print(f"{k:<11} n={v['n']:<5} cos_draws={v['cos_draws']:.4f}  "
              f"cos_ceiling={v['cos_ceiling_perfect_decoder']:.4f}  2WC={v['two_way']:.4f}")
    if abs(res["s1k"]["std_stored"] - res["s1k"]["std_reencoded"]) > 0.02:
        print("WARNING: re-encoded std does not match the stored target -- dim_scale step wrong")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
