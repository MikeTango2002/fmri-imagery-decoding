#!/usr/bin/env python3
"""Map the `streams` atlas onto the voxel order of the nsdgeneral mask used by the beta tensors.

Inputs:  nsdgeneral.nii.gz and streams.nii.gz (1.8 mm) of subjects 1, 2, 5, 7, and streams.mgz.ctab
Outputs: data/betas_trisect/streams_roi/subj0N/{streams_labels.npy (V,), roi_counts.json}
         the label of each beta column; -1 = outside the atlas

The betas keep only the nsdgeneral voxels, in (X, Y, Z) C-order. Only that mask decides which voxels exist
and in what order; streams is read as a per-voxel label in the same order. Takagi's hdf5 is (Z, Y, X):
mixing the two permutes the columns, and no voxel-count assert catches it.

Usage (CPU only):
    python src/map_streams_rois.py
"""
import json
import os
import sys
from pathlib import Path

import nibabel as nb
import numpy as np

NUM_VOXELS = {1: 15724, 2: 14278, 3: 15226, 4: 13153,
              5: 13039, 6: 17907, 7: 12682, 8: 14386}
SUBJECTS = [1, 2, 5, 7]
ROIS = ["early", "ventral", "midventral", "midlateral", "lateral", "parietal"]
# "Higher ventral" is used by the restricted-ROI arm: the non-early streams ROIs on the ventral
# side, ventral and midventral, against "early" (V1-V3, informative for vision but with little
# top-down signal for imagery). lateral, midlateral and parietal are dorsal/parietal streams;
# they are fitted separately and not folded into either bucket. The arm uses exactly
# {ventral, midventral}, not the full non-early set; the choice is written here so that it can
# be audited.
HIGHER_VENTRAL_ROIS = ["ventral", "midventral"]


def streams_value_map(ctab_path: Path) -> dict:
    """{roi_name: int_value}, parsed as nsd_access.read_atlas_results parses streams.mgz.ctab:
    whitespace-delimited, column 0 = int value, column 1 = name. The targets environment has no
    pandas; the file is parsed directly instead of with pd.read_csv."""
    mapp = {}
    for line in ctab_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            value = int(parts[0])
        except ValueError:
            continue
        mapp[parts[1]] = value
    return mapp


def run(data_root: Path, out_root: Path) -> None:
    ctab_path = data_root / "takagi/nsd/nsddata/freesurfer/fsaverage/label/streams.mgz.ctab"
    value_of = streams_value_map(ctab_path)
    missing_rois = [r for r in ROIS if r not in value_of]
    if missing_rois:
        sys.exit(f"ctab is missing expected ROI names: {missing_rois} (found: {sorted(value_of)})")
    print("streams ctab values:", {r: value_of[r] for r in ROIS})

    summary = {"rois": ROIS, "higher_ventral_rois": HIGHER_VENTRAL_ROIS,
               "roi_values": {r: value_of[r] for r in ROIS}, "subjects": {}}

    for subj in SUBJECTS:
        tag = f"subj{subj:02d}"
        ng_file = data_root / f"mindeye_I/dataset/nsddata/ppdata/{tag}/func1pt8mm/roi/nsdgeneral.nii.gz"
        st_file = data_root / f"takagi/nsd/nsddata/ppdata/{tag}/func1pt8mm/roi/streams.nii.gz"
        for f in (ng_file, st_file):
            if not f.exists():
                sys.exit(f"missing file: {f}")

        ng_img = nb.load(ng_file)
        st_img = nb.load(st_file)
        nsd_general = ng_img.get_fdata()          # (X, Y, Z)
        streams = st_img.get_fdata()               # (X, Y, Z)

        if nsd_general.shape != streams.shape:
            sys.exit(f"{tag}: shape mismatch nsdgeneral {nsd_general.shape} vs streams {streams.shape} "
                      "-- the two atlases are not in the same grid, cannot align by flatten order")

        # Affine equality: both files must describe the SAME physical grid (voxel i in one is
        # voxel i in the other), otherwise a shared flatten-order index means nothing.
        affine_max_diff = float(np.max(np.abs(ng_img.affine - st_img.affine)))
        if affine_max_diff > 1e-4:
            sys.exit(f"{tag}: affine mismatch (max|diff|={affine_max_diff:.6g}) between nsdgeneral "
                      "and streams volumes -- not the same grid, do not trust this mapping")

        # The recipe of MindEye2's prepare_nsd_imagery.py, which built the betas the ridge is applied to:
        # betas.transpose((3, 0, 1, 2)), mask = (nsdgeneral == 1).flatten(), x = betas.reshape(720, -1)[:, mask].
        # Here streams replaces the betas as the values and nsdgeneral stays the mask; voxel i of the
        # labels is the atlas at the 3D location of beta column i: aligned by construction, not by assumption.
        mask = np.nan_to_num(np.where(nsd_general == 1.0, True, False)).flatten()  # C-order, (X,Y,Z)
        v = int(mask.sum())
        expected_v = NUM_VOXELS[subj]
        if v != expected_v:
            sys.exit(f"{tag}: nsdgeneral mask has {v} voxels, expected {expected_v} "
                      "(this IS the check prepare_nsd_imagery.py / fit_ridge.py rely on -- "
                      "if this fails here, every downstream ridge fit for this subject is already broken)")

        streams_flat = streams.flatten()            # same C-order, (X,Y,Z), SAME convention as mask
        streams_labels = streams_flat[mask].astype(np.int64)   # (V,), aligned to beta columns
        streams_labels[streams_labels == 0] = -1     # 0 = outside streams atlas (not a real ROI)

        roi_counts = {r: int((streams_labels == value_of[r]).sum()) for r in ROIS}
        higher_ventral_count = int(sum(roi_counts[r] for r in HIGHER_VENTRAL_ROIS))
        outside_count = int((streams_labels == -1).sum())
        covered = sum(roi_counts.values())

        print(f"{tag}: V_nsdgeneral={v}  roi_counts={roi_counts}  "
              f"higher_ventral={higher_ventral_count}  outside_streams_atlas={outside_count}  "
              f"(covered={covered}/{v} = {100*covered/v:.1f}%)  affine_max_diff={affine_max_diff:.2e}")

        out_dir = out_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "streams_labels.npy", streams_labels)
        with open(out_dir / "roi_counts.json", "w") as f:
            json.dump({
                "V_nsdgeneral": v,
                "roi_counts": roi_counts,
                "higher_ventral_count": higher_ventral_count,
                "outside_streams_atlas": outside_count,
                "affine_max_diff": affine_max_diff,
                "roi_value_map": value_of,
                "convention": "(X,Y,Z) C-order flatten, masked by nsdgeneral==1.0 -- "
                               "matches mindeye_II/src/prepare_nsd_imagery.py, NOT takagi's (Z,Y,X)",
            }, f, indent=2)

        summary["subjects"][tag] = {
            "V_nsdgeneral": v, "roi_counts": roi_counts,
            "higher_ventral_count": higher_ventral_count,
            "outside_streams_atlas": outside_count, "affine_max_diff": affine_max_diff,
        }

    with open(out_root / "preflight_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out_root}/preflight_summary.json + per-subject streams_labels.npy/roi_counts.json")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path,
                    default=Path(os.environ["PROJECT_ROOT"]),
                    help="parent dir containing mindeye_I/ and takagi/ (server default)")
    p.add_argument("--out-root", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data/betas_trisect/streams_roi")
    args = p.parse_args()
    run(args.data_root, args.out_root)
