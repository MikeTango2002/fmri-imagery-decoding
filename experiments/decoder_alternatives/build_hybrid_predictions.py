"""Recombine two prediction folders into a hybrid one: sem and struct from one, z_low from the other.

Inputs:  --sem-source (MLP, e.g. mlp_predictions_imagery_trial) and --low-source (ridge, e.g.
         imagery_predictions_trial), per subject
Outputs: --out-dir/subj0N/{labels.npy | img73k_ids.npy, z_sem_img, z_sem_txt, z_struct, z_low}_pred.npy

Nothing is retrained: this tests whether the MLP helps on the semantic side while the ridge keeps
the low-level side (Section 8.1.1). Row alignment is asserted, not assumed: the key array (labels.npy
for imagery, img73k_ids.npy for nsdcore) must be identical in both sources, or the script aborts
before writing anything.

Usage:
    python build_hybrid_predictions.py --dataset imagery --sem-source DIR --low-source DIR --out-dir DIR
"""
import argparse
from pathlib import Path

import numpy as np

SUBJECTS = (1, 2, 5, 7)
SEM_BRANCHES = ("z_sem_img_pred.npy", "z_sem_txt_pred.npy", "z_struct_pred.npy")
LOW_BRANCH = "z_low_pred.npy"


def key_file(dataset: str) -> str:
    return "labels.npy" if dataset == "imagery" else "img73k_ids.npy"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=["imagery", "nsdcore"], required=True)
    ap.add_argument("--sem-source", required=True,
                     help="pred-dir supplying z_sem_img/z_sem_txt/z_struct")
    ap.add_argument("--low-source", required=True, help="pred-dir supplying z_low")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--subj", type=int, nargs="+", default=list(SUBJECTS))
    args = ap.parse_args()

    key = key_file(args.dataset)
    sem_root, low_root, out_root = Path(args.sem_source), Path(args.low_source), Path(args.out_dir)

    for subj in args.subj:
        tag = f"subj{subj:02d}"
        sem_dir, low_dir, out_dir = sem_root / tag, low_root / tag, out_root / tag
        sem_key = np.load(sem_dir / key, allow_pickle=True)
        low_key = np.load(low_dir / key, allow_pickle=True)
        if sem_key.shape != low_key.shape or not np.array_equal(sem_key, low_key):
            raise ValueError(
                f"{tag}: {key} mismatch between {sem_dir} ({sem_key.shape}) and {low_dir} "
                f"({low_key.shape}) -- refusing to combine, row order is not guaranteed aligned")

        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / key, sem_key)
        for branch_file in SEM_BRANCHES:
            np.save(out_dir / branch_file, np.load(sem_dir / branch_file))
        np.save(out_dir / LOW_BRANCH, np.load(low_dir / LOW_BRANCH))
        print(f"  {tag}: {len(sem_key)} rows, sem/struct <- {sem_root.name}, "
              f"z_low <- {low_root.name} -> {out_dir}")

    print(f"Done. Hybrid pred-dir written to {out_root}")


if __name__ == "__main__":
    main()
