"""Prepare NSD-Imagery betas for the Takagi & Nishimoto decoders.

Inputs:  the 1.8 mm volumetric NSD-Imagery betas (version 3), the `streams` atlas, the ridge weights (--weights_dir)
Outputs: --out_dir/<subject>_<roi>_<cond>_betas_ave.npy, one file per ROI, plus <subject>_<cond>_stim_ids.npy

Masks with the `streams` atlas (one file per ROI, in the voxel order the ridges were fitted on), divides by 300,
z-scores each run separately and averages the repetitions of each stimulus. Nothing is refitted. Seven sanity checks
("[check N]") run inline and abort at the first failure: the 720 trials; the voxel count of every ROI against the
ridge weights; mean 0 and std 1 in every run; no NaN; 8 / 16 repetitions; the stimulus indices; the scale of the
NSD-Core betas. They matter: every mistake here gives chance-level metrics, exactly like success.

Usage (from codes/utils/):
    python prepare_nsd_imagery_takagi.py --subject subj01 subj02 subj05 subj07 \
      --data_root ../../nsd --out_dir ../../mrifeat_imagery --weights_dir ../../decoded
"""
import argparse
import os
import pickle
import sys

import nibabel as nb
import numpy as np
import pandas as pd
from scipy.stats import zscore

ROIS = ["early", "ventral", "midventral", "midlateral", "lateral", "parietal"]

# One key per run type: 3 vision, 6 imagery (3 stimulus sets x 2 repeats), 3 attention.
RUN_TYPES = ["visA", "visB", "visC",
             "imgA_1", "imgA_2", "imgB_1", "imgB_2", "imgC_1", "imgC_2",
             "attA", "attB", "attC"]

# Conditions exported to disk. `visionconcepts` and attA/B/C are normalized together
# with the rest (they are part of their own runs) but excluded from the benchmark.
CONDITIONS = ["visionsimple", "visioncomplex",
              "imagerysimple", "imagerycomplex", "imageryconcepts"]

# Expected stimulus indices into imagery_stimuli_18.pt, per condition (check 7).
STIM_RANGE = {"simple": (0, 6), "complex": (6, 12), "concepts": (12, 18)}

# Weight files whose `roi_sizes` are the only published reference for V_roi (check 2).
WEIGHT_FILES = {
    "init_latent": "{s}/{s}_early_weights_init_latent.npz",
    "blip": "{s}/{s}_early_ventral_midventral_midlateral_lateral_parietal_weights_blip.npz",
}


def fail(msg):
    print(f"CHECK FAILED: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def streams_mapping(data_root):
    """{roi_name: int_label}, read exactly like nsd_access/nsda.py:230-234 does."""
    ctab = os.path.join(data_root, "nsddata/freesurfer/fsaverage/label/streams.mgz.ctab")
    mapp = pd.read_csv(ctab, delimiter=" ", header=None, index_col=0).to_dict()[1]
    return {name: value for value, name in mapp.items()}


def extract_imagery_betas_per_roi(subject, data_root, rois):
    """Return {roi: (720, V_roi) float32}, unnormalized, in fit-time voxel order."""
    beta_file = os.path.join(data_root, "nsddata_betas/ppdata", subject,
                             "func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR",
                             "betas_nsdimagery.nii.gz")
    atlas_file = os.path.join(data_root, "nsddata/ppdata", subject,
                              "func1pt8mm/roi/streams.nii.gz")

    betas = nb.load(beta_file).get_fdata()          # (X, Y, Z, 720)
    # (720, Z, Y, X): the SAME axis order as the NSD-core hdf5 betas ridge.py was fit on
    # (make_subjmri.py:64 masks with atlas.transpose([2,1,0])). Using (3, 0, 1, 2) here
    # would keep the same voxels but scramble their order, and the resulting noise is
    # indistinguishable from this model's expected chance-level result.
    betas = betas.transpose((3, 2, 1, 0))
    if betas.shape[0] != 720:                                                  # check 1
        fail(f"{subject}: expected 720 trials, found {betas.shape[0]}")

    atlas = nb.load(atlas_file).get_fdata()         # (X, Y, Z)
    atlas = atlas.transpose([2, 1, 0])              # (Z, Y, X), as in make_subjmri.py:64
    if atlas.shape != betas.shape[1:]:
        fail(f"{subject}: atlas {atlas.shape} vs betas {betas.shape[1:]}")

    mapping = streams_mapping(data_root)
    out = {}
    for roi in rois:
        if roi not in mapping:
            fail(f"'{roi}' absent from the streams ctab; found {sorted(mapping)}")
        out[roi] = betas[:, atlas == mapping[roi]].astype(np.float32)          # (720, V)
    return out, mapping


def normalize_imagery_betas(x, exps, nan_to_zero):
    """(720, V) raw betas -> (720, V) float32, /300 then z-scored within each run."""
    x = x / 300.0
    x_norm = np.zeros_like(x, dtype=np.float32)
    for c in RUN_TYPES:
        idx = np.arange(len(exps))[exps == c]
        x_norm[idx] = zscore(x[idx], axis=0).astype(np.float32)

        mean, std = float(x_norm[idx].mean()), float(x_norm[idx].std())        # check 4
        if not np.isnan(mean) and (abs(mean) > 1e-3 or abs(std - 1) > 1e-2):
            fail(f"run {c}: mean {mean:.3e}, std {std:.3e} (expected ~0 and ~1)")

    n_nan = int(np.isnan(x_norm).sum())                                        # check 5
    if n_nan:
        if not nan_to_zero:
            fail(f"{n_nan} NaN after the per-run z-score "
                 f"({n_nan / x_norm.shape[0]:.1f} per trial on average). Typical cause: a "
                 f"voxel with zero variance inside one run. Rerun with --nan_to_zero to "
                 f"zero them, and report it.")
        print(f"  --nan_to_zero: zeroing {n_nan} NaN values", flush=True)
        x_norm = np.nan_to_num(x_norm, nan=0.0)
    return x_norm


def build_cond_idx(exps):
    a = np.arange(len(exps))
    return {
        "visionsimple":    a[exps == "visA"],
        "visioncomplex":   a[exps == "visB"],
        "visionconcepts":  a[exps == "visC"],   # excluded from the protocol (paper, sec. 3)
        "imagerysimple":   a[np.isin(exps, ["imgA_1", "imgA_2"])],
        "imagerycomplex":  a[np.isin(exps, ["imgB_1", "imgB_2"])],
        "imageryconcepts": a[np.isin(exps, ["imgC_1", "imgC_2"])],
    }


def group_by_stimulus(x_norm, idx, cues, image_map):
    """(720, V) -> (n_stim, V), averaged over repetitions.

    Takagi never averages repetitions at inference time; make_subjmri.py:69-73 does it
    upstream, which is why ridge.py:191 reads `_betas_ave_te.npy` and why the diffusion
    script indexes one row per stimulus.
    """
    conditionals = np.array([image_map[c] for c in cues[idx]])
    x_cond = x_norm[idx]
    stim_ids = np.unique(conditionals)
    reps = [x_cond[conditionals == s] for s in stim_ids]
    n_reps = min(r.shape[0] for r in reps)
    x_stack = np.stack([r[:n_reps] for r in reps])          # (n_stim, n_reps, V)
    return x_stack.mean(axis=1), stim_ids, n_reps


def check_voxel_counts(subject, x_roi, weights_dir):
    """check 2: V_roi against the block sizes recorded at fit time (ridge.py:194)."""
    seen = False
    for target, pattern in WEIGHT_FILES.items():
        path = os.path.join(weights_dir, pattern.format(s=subject))
        if not os.path.exists(path):
            print(f"  [check 2] {target}: {path} missing, skipped", flush=True)
            continue
        seen = True
        with np.load(path) as w:
            rois = [str(r) for r in w["roi"]]
            sizes = [int(n) for n in w["roi_sizes"]]
        for roi, n in zip(rois, sizes):
            if roi not in x_roi:
                fail(f"{subject}: weights for {target} need ROI '{roi}', not extracted")
            if x_roi[roi].shape[1] != n:
                fail(f"{subject}/{roi}: {x_roi[roi].shape[1]} voxels, weights expect {n}. "
                     f"Usual causes: atlas of another subject, func1mm instead of "
                     f"func1pt8mm, or the ctab read with a different delimiter.")
        print(f"  [check 2] {target}: {list(zip(rois, sizes))} OK", flush=True)
    if not seen:
        fail(f"{subject}: no weight file found under {weights_dir} -- check 2 is the only "
             f"existing reference for the voxel counts, do not skip it")


def check_training_beta_scale(subject, weights_dir, mrifeat_dir):
    """check 8: using the per-run z-score in place of the StandardScaler of the fit holds only
    if the fit betas are raw integers."""
    path = os.path.join(weights_dir, WEIGHT_FILES["init_latent"].format(s=subject))
    if not os.path.exists(path):
        return
    with np.load(path) as w:
        med = float(np.median(w["scaler_scale"]))
    tr = os.path.join(mrifeat_dir, subject, f"{subject}_early_betas_tr.npy")
    dtype = "n/a"
    if os.path.exists(tr):
        dtype = str(np.load(tr, mmap_mode="r").dtype)
    print(f"  [check 8] betas_tr dtype {dtype}, median scaler_scale {med:.1f}", flush=True)
    if med < 10:
        fail(f"{subject}: median scaler_scale is {med:.3f}, expected 1e2-1e3. The training "
             f"betas would then already be in PSC units, which invalidates the choice of "
             f"the per-run z-score in place of the StandardScaler. Stop and "
             f"re-derive the normalization before going further.")


def process_subject(subject, opt, imagery_dict):
    print(f"\n=== {subject}", flush=True)
    exps = np.asarray(imagery_dict["exps"])
    cues = np.asarray(imagery_dict["cues"])
    image_map = imagery_dict["image_map"]
    if len(exps) != 720:                                                       # check 1
        fail(f"exps has {len(exps)} entries, expected 720")

    outdir = os.path.join(opt.out_dir, subject)
    os.makedirs(outdir, exist_ok=True)

    x_roi, mapping = extract_imagery_betas_per_roi(subject, opt.data_root, opt.rois)
    print("  streams map:", {k: mapping[k] for k in sorted(mapping)}, flush=True)
    for roi in opt.rois:
        print(f"  {roi}: {x_roi[roi].shape}", flush=True)
    check_voxel_counts(subject, x_roi, opt.weights_dir)
    check_training_beta_scale(subject, opt.weights_dir, opt.mrifeat_dir)

    uniq, cnt = np.unique(exps, return_counts=True)                            # check 6
    print("  trials per run:", dict(zip(uniq.tolist(), cnt.tolist())), flush=True)

    cond_idx = build_cond_idx(exps)
    for roi in opt.rois:
        np.save(os.path.join(outdir, f"{subject}_{roi}_unnormalized.npy"), x_roi[roi])
        x_norm = normalize_imagery_betas(x_roi[roi], exps, opt.nan_to_zero)
        np.save(os.path.join(outdir, f"{subject}_{roi}_norm.npy"), x_norm)

        for cond in CONDITIONS:
            ave, stim_ids, n_reps = group_by_stimulus(
                x_norm, cond_idx[cond], cues, image_map)

            kind = cond.replace("vision", "").replace("imagery", "")           # check 7
            lo, hi = STIM_RANGE[kind]
            if not np.array_equal(stim_ids, np.arange(lo, hi)):
                fail(f"{subject}/{cond}: stim_ids {stim_ids.tolist()}, expected "
                     f"{list(range(lo, hi))}. The rows would not line up with "
                     f"imagery_stimuli_18.pt[{lo}:{hi}] and nothing downstream would notice.")
            expected_reps = 8 if cond.startswith("vision") else 16             # check 6
            if n_reps != expected_reps:
                fail(f"{subject}/{cond}: {n_reps} repetitions, expected {expected_reps}. "
                     f"On an imagery condition, 8 means one of the two repeated runs "
                     f"(imgX_2) was lost.")

            np.save(os.path.join(outdir, f"{subject}_{roi}_{cond}_betas_ave.npy"), ave)
            np.save(os.path.join(outdir, f"{subject}_{cond}_stim_ids.npy"), stim_ids)
            if roi == opt.rois[0]:
                print(f"  {cond}: {ave.shape} from {len(cond_idx[cond])} trials, "
                      f"{n_reps} reps, stim {stim_ids.tolist()}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subject", type=str, nargs="+", required=True,
                        help="subj01 subj02 subj05 subj07")
    parser.add_argument("--data_root", type=str, default="../../nsd",
                        help="NSD root, holding nsddata/ nsddata_betas/ nsddata_stimuli/")
    parser.add_argument("--out_dir", type=str, default="../../mrifeat_imagery",
                        help="never point this at mrifeat/: those are the NSD-core test set")
    parser.add_argument("--weights_dir", type=str, default="../../decoded",
                        help="the .npz written by ridge.py --save_weights, for check 2")
    parser.add_argument("--mrifeat_dir", type=str, default="../../mrifeat",
                        help="NSD-core betas, read only to print their dtype (check 8)")
    parser.add_argument("--rois", type=str, nargs="+", default=ROIS)
    parser.add_argument("--nan_to_zero", action="store_true",
                        help="zero the NaN left by a zero-variance voxel inside a run "
                             "instead of aborting (check 5)")
    opt = parser.parse_args()

    if os.path.abspath(opt.out_dir).rstrip("/").endswith("/mrifeat"):
        fail("--out_dir points at mrifeat/: those files are the NSD-core test set and the "
             "only non-chance anchor this model has. Use mrifeat_imagery/.")

    pkl = os.path.join(opt.data_root, "nsddata_stimuli/stimuli/nsdimagery_stimuli.pkl3")
    with open(pkl, "rb") as f:
        imagery_dict = pickle.load(f)
    print("image_map:", imagery_dict["image_map"], flush=True)

    for subject in opt.subject:
        process_subject(subject, opt, imagery_dict)

    print("\nall checks passed", flush=True)


if __name__ == "__main__":
    main()
