#!/usr/bin/env python
"""The shared1000 bridge for MindEye2, scored with the same harness as the NSD-Imagery benchmark.

Inputs:  `evals/all_images.pt` and `evals/{model}/{model}_all_enhancedrecons.pt` (same order), the shared1000
         betas, GNet weights and masks
Outputs: --out_tsv with image metrics, brain correlation and two-way on pools of 12

The harness is validated on the paper each model reproduces before it is trusted on NSD-Imagery
(Section 4.4); `eval_bridge_icnn.py` does it for iCNN. Brain correlation is recomputed because Kneeland's
Early is the correlation over the union V1|V2|V3|V4, not derivable from the six separate ones of
`final_evaluations.py`. It reuses `load_masks()` / `brain_correlation()` of `gnet_brain_correlation.py`.
PixCorr, SSIM, EffNet-B and SwAV run on all ~1000 images; two-way on random pools of 12 (the published
row is blank for 982 distractors), compared with the NSD-Imagery vision row.

Usage (MindEye2 environment; GPU needed):
    OPENBLAS_CORETYPE=SkylakeX python eval_bridge_mindeye2.py --subj 1 --n_pools 20   # smoke test
"""
import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import webdataset as wds
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from final_evaluations_imagery import Nets, TARGET_DIM, _blas_sanity, get_pix_corr, get_ssim
from gnet_brain_correlation import (GNET_SUBJECT_SIZES, _pil_from_tensor, brain_correlation,
                                    load_masks)
from models import GNet8_Encoder

# torch >= 2.6 defaults torch.load to weights_only=True, which rejects gnet_multisubject.pt
# (numpy scalars stored alongside the weights, models.py:764 loads it with no kwargs). Same
# patch as gnet_brain_correlation.py; these are on-disk artifacts of this work, not downloads.
if "weights_only" in __import__("inspect").signature(torch.load).parameters:
    _torch_load = torch.load

    def _torch_load_compat(*a, **kw):
        kw.setdefault("weights_only", False)
        return _torch_load(*a, **kw)

    torch.load = _torch_load_compat

# Table 1, "NSD Shared1000 Test Set" row, MindEye2 (Kneeland et al. 2025).
# The four two-way cells are "-" in the paper (982 distractors, not comparable to 11); only
# these seven can be compared against a published number.
PUBLISHED = {"PixCorr": 0.322, "SSIM": 0.431, "EffNet-B": 0.619, "SwAV": 0.344,
            "Early": 0.360, "Higher": 0.368, "VisCtx": 0.373}

# The NSD-Imagery vision row of this work, with the authors' off-by-one included (Section 4.5), for
# the pools-of-12 comparison, the same convention as eval_bridge_icnn.py.
OURS_VISION = {"AlexNet(2)": 0.641, "AlexNet(5)": 0.739, "InceptionV3": 0.618, "CLIP": 0.655}

# subj -> new_test trial count, final_evaluations.py:513-522 (only 1/2/5/7 matter here: the
# four benchmark subjects have the full 40 sessions and therefore 3000, not the reduced count
# of the four partial-session subjects).
NUM_TEST_TRIALS = {1: 3000, 2: 3000, 3: 2371, 4: 2188, 5: 3000, 6: 2371, 7: 3000, 8: 2188}


def load_test_betas_averaged(data_path: Path, subj: int):
    """(n_unique_test_images, V) betas, averaged over repetitions, in ascending-COCO-id order.

    Exact transcription of final_evaluations.py:500-544 (load) and 565-576 (average): the same
    hdf5, the same single webdataset shard, the same behav columns. The ascending-COCO-id
    order is what all_images.pt / all_recons.pt are implicitly assumed to be in by the
    authors' own harness (final_evaluations.py never re-sorts them), already load-bearing
    for the six brain-correlation columns of the original MindEye2 evaluation.
    """
    with h5py.File(data_path / f"betas_all_subj0{subj}_fp32_renorm.hdf5", "r") as f:
        betas = torch.Tensor(f["betas"][:])
    num_voxels = betas.shape[-1]

    def my_split_by_node(urls):
        return urls

    num_test = NUM_TEST_TRIALS[subj]
    test_url = str(data_path / f"wds/subj0{subj}/new_test/0.tar")
    test_data = (wds.WebDataset(test_url, resampled=False, nodesplitter=my_split_by_node)
                .decode("torch")
                .rename(behav="behav.npy", past_behav="past_behav.npy",
                        future_behav="future_behav.npy", olds_behav="olds_behav.npy")
                .to_tuple("behav", "past_behav", "future_behav", "olds_behav"))
    test_dl = torch.utils.data.DataLoader(test_data, batch_size=num_test, shuffle=False,
                                          drop_last=True, pin_memory=True)

    test_images_idx = np.empty(0, dtype=int)
    test_voxels = None
    n_batches = 0
    for behav, *_ in test_dl:
        test_voxels = betas[behav[:, 0, 5].cpu().long()]
        test_images_idx = np.append(test_images_idx, behav[:, 0, 0].cpu().numpy())
        n_batches += 1
    test_images_idx = test_images_idx.astype(int)
    assert n_batches * num_test == len(test_voxels) == len(test_images_idx), \
        f"subj0{subj}: {n_batches} batches x {num_test} != {len(test_voxels)}/{len(test_images_idx)}"

    uniq = np.unique(test_images_idx)
    out = torch.zeros((len(uniq), num_voxels))
    for i, img in enumerate(uniq):
        locs = np.where(test_images_idx == img)[0]
        out[i] = test_voxels[locs].mean(dim=0)
    return out, num_voxels


def compute_brain_corr(args, subj: int, recons: torch.Tensor):
    """Early (V1|V2|V3|V4 union) / Higher (higher_vis) / VisCtx (nsd_general), per stimulus."""
    betas_avg, num_voxels = load_test_betas_averaged(Path(args.data_path), subj)
    assert num_voxels == GNET_SUBJECT_SIZES[subj], (
        f"subj0{subj}: betas have {num_voxels} voxels but GNet predicts "
        f"{GNET_SUBJECT_SIZES[subj]} -- voxel space mismatch, stop.")
    assert betas_avg.shape[0] == recons.shape[0], (
        f"subj0{subj}: {betas_avg.shape[0]} unique test images but {recons.shape[0]} "
        "reconstructions -- all_images.pt/all_recons.pt and the betas disagree on N")

    masks = load_masks(Path(args.masks), subj, num_voxels)
    print(f"  masks  EVC {int(masks['EVC'].sum())} | higher {int(masks['higher_vis'].sum())} "
          f"| nsd_general {int(masks['nsd_general'].sum())} of {num_voxels}")

    gnet = GNet8_Encoder(device=args.device, subject=subj, model_path=args.gnet)
    recon_list = [_pil_from_tensor(recons[i]) for i in range(recons.shape[0])]
    beta_primes = gnet.predict(recon_list)
    assert beta_primes.shape == betas_avg.shape, \
        f"GNet returned {tuple(beta_primes.shape)}, betas are {tuple(betas_avg.shape)}"
    scores = brain_correlation(betas_avg, beta_primes, masks)
    del gnet
    torch.cuda.empty_cache()
    # region keys are EVC/higher_vis/nsd_general (gnet_brain_correlation.py convention) ->
    # rename to the Early/Higher/VisCtx labels this table uses everywhere else.
    return {"Early": scores["EVC"], "Higher": scores["higher_vis"], "VisCtx": scores["nsd_general"]}


def image_metrics(nets: Nets, images: torch.Tensor, recons: torch.Tensor, rng, n_pools: int):
    """PixCorr/SSIM/EffNet-B/SwAV on every stimulus; the four two-way metrics on n_pools
    random pools of 12 (comparable to the NSD-Imagery vision row, not to Table 1)."""
    recons_512 = transforms.Resize((TARGET_DIM, TARGET_DIM))(recons)  # cell 5, verbatim

    per_stim = {
        "PixCorr": np.asarray(get_pix_corr(images, recons_512)),
        "SSIM": np.asarray(get_ssim(images, recons_512)),
        "EffNet-B": np.asarray(nets.get_efficientnet(images, recons_512)),
        "SwAV": np.asarray(nets.get_swav(images, recons_512)),
    }

    n = images.shape[0]
    pools = [rng.choice(n, 12, replace=False) for _ in range(n_pools)]
    twoway = {k: [] for k in ("AlexNet(2)", "AlexNet(5)", "InceptionV3", "CLIP")}
    for k, sel in enumerate(pools):
        i, r = images[sel], recons_512[sel]
        a2, a5 = nets.get_alexnet(i, r)
        twoway["AlexNet(2)"].append(np.mean(a2))
        twoway["AlexNet(5)"].append(np.mean(a5))
        twoway["InceptionV3"].append(np.mean(nets.get_inceptionv3(i, r)))
        twoway["CLIP"].append(np.mean(nets.get_clip(i, r)))
        if (k + 1) % 50 == 0:
            print(f"    {k + 1}/{len(pools)} pools", flush=True)
    twoway = {k: np.asarray(v) for k, v in twoway.items()}
    return per_stim, twoway


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subj", type=int, nargs="+", default=[1, 2, 5, 7])
    p.add_argument("--data_path", default="mindeyev2_dataset",
                   help="dir with betas_all_subj0N_fp32_renorm.hdf5 and wds/subj0N/new_test/")
    p.add_argument("--evals_root", default="evals", help="dir with all_images.pt and "
                                                          "{model}/{model}_all_enhancedrecons.pt")
    p.add_argument("--model_template", default="finetuned_subj{subj02}_40sess_4096hid")
    p.add_argument("--gnet", default="mindeyev2_dataset/gnet_multisubject.pt")
    p.add_argument("--masks", default="mindeyev2_dataset/brain_region_masks.hdf5")
    p.add_argument("--n_pools", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--skip_brain_corr", action="store_true",
                   help="only the 4 image-quality metrics, no GNet")
    p.add_argument("--out_tsv", default="")
    return p.parse_args()


def main():
    args = parse_args()
    _blas_sanity()
    rng = np.random.default_rng(args.seed)

    images = torch.load(Path(args.evals_root) / "all_images.pt", map_location="cpu").float()
    print(f"ground truth: {tuple(images.shape)} from {args.evals_root}/all_images.pt")

    nets = Nets(args.device)

    per_subj_image, per_subj_brain = {}, {}
    for subj in args.subj:
        print(f"\n=== subj0{subj} ===")
        model = args.model_template.format(subj02=f"{subj:02d}")
        recon_path = Path(args.evals_root) / model / f"{model}_all_enhancedrecons.pt"
        recons = torch.load(recon_path, map_location="cpu").float()
        assert recons.shape[0] == images.shape[0], (
            f"{recon_path}: {recons.shape[0]} reconstructions but {images.shape[0]} "
            "ground-truth images")
        print(f"  recons {tuple(recons.shape)} <- {recon_path}")

        per_stim, twoway = image_metrics(nets, images, recons, rng, args.n_pools)
        per_subj_image[subj] = (per_stim, twoway)

        if not args.skip_brain_corr:
            per_subj_brain[subj] = compute_brain_corr(args, subj, recons)
            vals = "  ".join(f"{r}={float(v.mean()):+.3f}" for r, v in per_subj_brain[subj].items())
            print(f"  brain corr  {vals}")

    # ---- report: image-quality metrics against Table 1 --------------------------------
    print("\nAgainst the published Shared1000 row (Table 1, MindEye2):")
    print(f"{'metric':>10} {'measured':>10} {'SE':>7} {'published':>10} {'diff':>8}")
    image_metric_names = ("PixCorr", "SSIM", "EffNet-B", "SwAV")
    means_by_metric = {m: [] for m in image_metric_names}
    for m in image_metric_names:
        all_v = np.concatenate([per_subj_image[s][0][m] for s in args.subj])
        se = all_v.std(ddof=1) / np.sqrt(len(all_v))
        for s in args.subj:
            means_by_metric[m].append(per_subj_image[s][0][m].mean())
        mean_of_means = np.mean(means_by_metric[m])
        print(f"{m:>10} {mean_of_means:>10.4f} {se:>7.4f} {PUBLISHED[m]:>10.3f} "
              f"{mean_of_means - PUBLISHED[m]:>+8.4f}")

    # ---- report: brain correlation against Table 1 -------------------------------------
    if not args.skip_brain_corr:
        print("\nAgainst the published Shared1000 row (Table 1, MindEye2) -- brain correlation:")
        print(f"{'region':>10} {'measured':>10} {'SE':>7} {'published':>10} {'diff':>8}")
        for r in ("Early", "Higher", "VisCtx"):
            all_v = torch.cat([per_subj_brain[s][r] for s in args.subj]).numpy()
            se = all_v.std(ddof=1) / np.sqrt(len(all_v))
            mean_of_means = np.mean([float(per_subj_brain[s][r].mean()) for s in args.subj])
            print(f"{r:>10} {mean_of_means:>10.4f} {se:>7.4f} {PUBLISHED[r]:>10.3f} "
                  f"{mean_of_means - PUBLISHED[r]:>+8.4f}")

    # ---- report: two-way on pools of 12, against the NSD-Imagery vision row --------
    print("\nTwo-way on pools of 12 (chance 40.91%, off-by-one included), against our own")
    print("NSD-Imagery vision row of Section 4.5:")
    print(f"{'metric':>12} {'bridge':>8} {'SE':>7} {'nsdi vision':>12} {'diff':>8}")
    for m in ("AlexNet(2)", "AlexNet(5)", "InceptionV3", "CLIP"):
        all_v = np.concatenate([per_subj_image[s][1][m] for s in args.subj])
        se = all_v.std(ddof=1) / np.sqrt(len(all_v))
        mean_of_means = np.mean([per_subj_image[s][1][m].mean() for s in args.subj])
        print(f"{m:>12} {mean_of_means:>8.4f} {se:>7.4f} {OURS_VISION[m]:>12.3f} "
              f"{mean_of_means - OURS_VISION[m]:>+8.4f}")

    if args.out_tsv:
        import csv
        Path(args.out_tsv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_tsv, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["subj", "metric", "scope", "mean", "se", "n"])
            for s in args.subj:
                per_stim, twoway = per_subj_image[s]
                for m, v in per_stim.items():
                    w.writerow([s, m, "all", f"{v.mean():.6f}",
                                f"{v.std(ddof=1) / np.sqrt(len(v)):.6f}", len(v)])
                for m, v in twoway.items():
                    w.writerow([s, m, "pools12", f"{v.mean():.6f}",
                                f"{v.std(ddof=1) / np.sqrt(len(v)):.6f}", len(v)])
                if not args.skip_brain_corr:
                    for r, v in per_subj_brain[s].items():
                        v = v.numpy()
                        w.writerow([s, r, "all", f"{v.mean():.6f}",
                                    f"{v.std(ddof=1) / np.sqrt(len(v)):.6f}", len(v)])
        print(f"\nwrote {args.out_tsv}")


if __name__ == "__main__":
    sys.exit(main())
