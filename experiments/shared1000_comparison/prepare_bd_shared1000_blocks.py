#!/usr/bin/env python3
"""Put Brain-Diffuser's shared1000 reconstructions on the block set of TRISECT's shared1000 arm.

Inputs:  the Brain-Diffuser tree (--bd-root: prepare_nsddata.py, reconstructions, nsd_test_stim), the block set
Outputs: bd_block_index.json (kept blocks, row order), a symlink farm in the layout of evaluate_block_set.py,
         optionally data/shared1000_blockset_gt425 (ground truth at 425 pixels, for the published comparison)

1. Brain-Diffuser's test row order is reconstructed (first appearance of each shared id in the 37 x 750 trials)
   and checked on pixels: rank-1 identification inside its own 982-image test set (chance 1/982).
2. A block is kept only if its 12 ids are in the test set of all four subjects; the pool stays at 12 (chance
   40.91%) and the cells already measured for TRISECT are untouched.

Usage (from the project root, with the Brain-Diffuser repository at ../brain_diffuser):
    python prepare_bd_shared1000_blocks.py              # index + check + symlinks + 425 ground truth
    python prepare_bd_shared1000_blocks.py --no-gt425   # skip the 425 ground truth
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import scipy.io as spio
import torch

SUBJECTS = (1, 2, 5, 7)
N_TEST = 12
N_STIM = 6
STIM_SIZE = 425
NUM_TRIALS = 37 * 750          # prepare_nsddata.py's own constant
BD_ROOT_DEFAULT = "../brain_diffuser"
BLOCKSET_DEFAULT = "data/shared1000_blockset"
LINK_ROOT_DEFAULT = "data/bd_s1k_blockset"
GT425_DEFAULT = "data/shared1000_blockset_gt425"
OUT_JSON_DEFAULT = "data/results/shared1000_comparison/bd_block_index.json"
CONDS = ("imagerysimple", "imagerycomplex")


def _todict(matobj):
    return {k: getattr(matobj, k) for k in matobj._fieldnames}


def bd_test_order(expdesign_path: Path, sub: int) -> np.ndarray:
    """prepare_nsddata.py's `test_im_idx`, for one subject, byte-for-byte the same loop."""
    stim_order = spio.loadmat(str(expdesign_path), struct_as_record=False, squeeze_me=True)
    subjectim = np.asarray(stim_order["subjectim"])
    masterordering = np.asarray(stim_order["masterordering"])
    sig_test = {}
    for idx in range(NUM_TRIALS):
        if masterordering[idx] > 1000:
            continue
        nsd_id = int(subjectim[sub - 1, masterordering[idx] - 1]) - 1
        sig_test.setdefault(nsd_id, []).append(idx)
    return np.array(list(sig_test.keys()), dtype=np.int64)


def load_blockset(blockset: Path):
    with open(blockset / "manifest.json") as f:
        man = json.load(f)
    assert man["n_stim_per_block"] == N_STIM and man["stim_size"] == STIM_SIZE
    return man


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    a -= a.mean()
    b -= b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d > 0 else 0.0


def _feat(img_chw: np.ndarray, stride: int) -> np.ndarray:
    """Strided pixels, centred and unit-normed, a correlation reduces to a dot product."""
    v = img_chw[:, ::stride, ::stride].astype(np.float64).ravel()
    v -= v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def verify_mapping(bd_root: Path, blockset: Path, man, ids_by_subj, n_check, seed, stride=5):
    """Rank-1 identification of each checked blockset image inside BD's whole 982-image test set.

    A comparison of rho(same image, 224->425 vs 425-native) against rho(mismatched pair) is
    MISCALIBRATED: the same photograph rendered at two resolutions
    can fall to 0.89 on heavily textured content, while two different photographs sharing a
    global layout reach 0.52; the two distributions overlap, and no threshold on that statistic
    could decide. Identification against the full pool has a clean null (chance 1/982) and is
    strictly harder to pass. Both statistics are still reported, the verdict rests on rank-1.
    """
    rng = np.random.default_rng(seed)
    id_to_block_pos = {i: (blk["block"], j)
                       for blk in man["blocks"] for j, i in enumerate(blk["img73k_ids"])}
    sub = SUBJECTS[0]
    stim = np.load(bd_root / "data" / "processed_data" / f"subj{sub:02d}" /
                   f"nsd_test_stim_sub{sub}.npy", mmap_mode="r")
    ids = ids_by_subj[sub]
    assert len(ids) == stim.shape[0], f"{len(ids)} ids vs {stim.shape[0]} stimulus rows"
    row_of = {int(v): k for k, v in enumerate(ids)}
    common = [i for i in ids if i in id_to_block_pos]
    picks = rng.choice(len(common), size=min(n_check, len(common)), replace=False)

    print(f"  building the {stim.shape[0]}-image identification pool (stride {stride})...")
    pool = np.stack([_feat(np.asarray(stim[r]).transpose(2, 0, 1), stride)
                     for r in range(stim.shape[0])])                  # (982, d)

    gt_cache, rhos, controls, ranks = {}, [], [], []
    for p in picks:
        img_id = int(common[p])
        b, j = id_to_block_pos[img_id]
        if b not in gt_cache:
            gt_cache[b] = torch.load(blockset / "blocks" / f"block{b:02d}_gt.pt",
                                     map_location="cpu").float().numpy()
        q = _feat(gt_cache[b][j], stride)
        sims = pool @ q
        target = row_of[img_id]
        ranks.append(int((sims > sims[target]).sum()) + 1)
        rhos.append(float(sims[target]))
        controls.append(float(np.max(np.delete(sims, target))))
    return np.array(rhos), np.array(controls), np.array(ranks), int(stim.shape[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bd-root", default=BD_ROOT_DEFAULT)
    ap.add_argument("--blockset", default=BLOCKSET_DEFAULT)
    ap.add_argument("--link-root", default=LINK_ROOT_DEFAULT)
    ap.add_argument("--gt425-dir", default=GT425_DEFAULT)
    ap.add_argument("--no-gt425", action="store_true")
    ap.add_argument("--out", default=OUT_JSON_DEFAULT)
    ap.add_argument("--n-check", type=int, default=60)
    ap.add_argument("--stride", type=int, default=5,
                    help="pixel stride of the identification features (425 -> 85 per side)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bd_root = Path(args.bd_root)
    blockset = Path(args.blockset)
    man = load_blockset(blockset)
    print(f"blockset: {man['n_blocks']} blocks, {man['n_images_used']} images")

    # --- 1. BD's own row order, per subject ---
    expdesign = bd_root / "data" / "nsddata" / "experiments" / "nsd" / "nsd_expdesign.mat"
    ids_by_subj = {}
    for sub in SUBJECTS:
        ids = bd_test_order(expdesign, sub)
        n_png = len(list((bd_root / "results" / "versatile_diffusion" /
                          f"subj{sub:02d}").glob("*.png")))
        assert len(ids) == n_png, (f"subj{sub:02d}: reconstructed order has {len(ids)} ids but "
                                   f"{n_png} PNGs exist -- the loop does not match this run")
        ids_by_subj[sub] = ids
        print(f"  subj{sub:02d}: {len(ids)} test ids, first 5 = {ids[:5].tolist()}")

    # --- 1b. the pixel check that has to pass before anything is measured ---
    rhos, controls, ranks, pool_n = verify_mapping(bd_root, blockset, man, ids_by_subj,
                                                    args.n_check, args.seed, args.stride)
    n_top1 = int((ranks == 1).sum())
    print(f"\nmapping check on {len(ranks)} ids, identification in a pool of {pool_n} "
          f"(chance 1/{pool_n}): rank-1 {n_top1}/{len(ranks)}, worst rank {int(ranks.max())}")
    print(f"  for the record: rho(target) min={rhos.min():.4f} mean={rhos.mean():.4f} | "
          f"best distractor max={controls.max():.4f}")
    if n_top1 != len(ranks):
        bad = [int(r) for r in ranks if r != 1]
        raise SystemExit("MAPPING CHECK FAILED -- refusing to build the blockset. "
                         f"{len(bad)} id(s) not identified at rank 1: {bad}")
    print("mapping check PASSED")

    # --- 2. which blocks survive ---
    # Whole or dropped whole: Brain-Diffuser has 982 images per subject (the shared ones of its 37 sessions), the
    # block set 996 in 83 blocks. Keeping the pool at exactly 12 keeps the chance of the two-way metrics at 40.91%.
    sets = {sub: set(map(int, ids)) for sub, ids in ids_by_subj.items()}
    kept, dropped = [], []
    for blk in man["blocks"]:
        ok = all(all(i in sets[sub] for i in blk["img73k_ids"]) for sub in SUBJECTS)
        (kept if ok else dropped).append(blk["block"])
    print(f"\nblocks kept: {len(kept)}/{man['n_blocks']}  (dropped {len(dropped)}: {dropped})")
    print(f"CI widening vs the full 83 blocks: x{np.sqrt(man['n_blocks'] / len(kept)):.3f}")

    # --- 3a. symlink farm ---
    link_root = Path(args.link_root)
    n_links = 0
    for sub in SUBJECTS:
        row_of = {int(v): k for k, v in enumerate(ids_by_subj[sub])}
        src_dir = (bd_root / "results" / "versatile_diffusion" / f"subj{sub:02d}").resolve()
        for b in kept:
            blk = man["blocks"][b]
            for cond, half in zip(CONDS, (blk["img73k_ids"][:N_STIM], blk["img73k_ids"][N_STIM:])):
                d = link_root / f"subj{sub:02d}" / f"subj{b:02d}" / cond / "sample000"
                d.mkdir(parents=True, exist_ok=True)
                for j, img_id in enumerate(half):
                    dst = d / f"{j:05d}.png"
                    if dst.is_symlink() or dst.exists():
                        dst.unlink()
                    os.symlink(src_dir / f"{row_of[int(img_id)]}.png", dst)
                    n_links += 1
    print(f"wrote {n_links} symlinks under {link_root}")

    # --- 3b. 425-native ground truth, for the comparison with the published Brain-Diffuser values ---
    # Needed only there (Section 4.4.3): the block set's own ground truth is a 224-native copy upsampled to 425
    # (its manifest says so), and the absolute PixCorr and SSIM must be comparable with the published ones.
    if not args.no_gt425:
        gt_dir = Path(args.gt425_dir) / "blocks"
        gt_dir.mkdir(parents=True, exist_ok=True)
        sub = SUBJECTS[0]
        stim = np.load(bd_root / "data" / "processed_data" / f"subj{sub:02d}" /
                       f"nsd_test_stim_sub{sub}.npy", mmap_mode="r")
        row_of = {int(v): k for k, v in enumerate(ids_by_subj[sub])}
        for b in kept:
            blk = man["blocks"][b]
            imgs = np.stack([np.asarray(stim[row_of[int(i)]]).transpose(2, 0, 1)
                             for i in blk["img73k_ids"]]).astype(np.float32) / 255.0
            t = torch.from_numpy(imgs)                                   # (12,3,425,425)
            gt = torch.cat([t, t[N_STIM:]], dim=0)                       # filler, as make_shared1000_blockset.py
            assert tuple(gt.shape) == (18, 3, STIM_SIZE, STIM_SIZE)
            assert -1e-4 <= float(gt.min()) and float(gt.max()) <= 1.0 + 1e-4
            torch.save(gt, gt_dir / f"block{b:02d}_gt.pt")
        with open(Path(args.gt425_dir) / "manifest.json", "w") as f:
            json.dump({**{k: v for k, v in man.items() if k != "blocks"},
                       "gt_resolution_note": "425-native, from brain_diffuser's own "
                                             "nsd_test_stim_sub1.npy (uint8) -- built for the "
                                             "comparison with the published Brain-Diffuser values "
                                             "(Section 4.4.3)",
                       "n_blocks": man["n_blocks"], "kept_blocks": kept,
                       "blocks": [b for b in man["blocks"] if b["block"] in set(kept)]}, f, indent=2)
        print(f"wrote {len(kept)} 425-native GT blocks to {gt_dir}")

    # --- index ---
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump({
            "source": "brain_diffuser benchmark-A reconstructions, results/versatile_diffusion",
            "row_order": "reproduced from nsd_expdesign.mat exactly as prepare_nsddata.py does "
                         "(first appearance over 37*750 trials); subject-specific, not sorted",
            "gen_rep": 1,
            "trisect_arm": "oracle_budget/S1K_T3_subj0N.tsv (s=0.85 m=0.6, 4 replicas)",
            "n_test_ids": {f"subj{s:02d}": int(len(ids_by_subj[s])) for s in SUBJECTS},
            "blocks_kept": kept,
            "blocks_dropped": dropped,
            "n_blocks_kept": len(kept),
            "ci_widening_vs_83": float(np.sqrt(man["n_blocks"] / len(kept))),
            "mapping_check": {"test": "rank-1 identification of the blockset image inside "
                                      f"BD's own {int(pool_n)}-image test set",
                              "n": int(len(ranks)), "pool": int(pool_n),
                              "rank1": int(n_top1), "worst_rank": int(ranks.max()),
                              "rho_min": float(rhos.min()), "rho_mean": float(rhos.mean()),
                              "best_distractor_max": float(controls.max()),
                              "note": "the first, miscalibrated criterion (min rho >= 0.95, "
                                      "control <= 0.5) failed at 0.892/0.519 on a mapping this "
                                      "test then identified 100% at rank 1"},
            "test_ids": {f"subj{s:02d}": ids_by_subj[s].tolist() for s in SUBJECTS},
        }, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
