"""Score a set packaged as blocks with Kneeland's own image-metric harness (`final_evaluations_imagery.py`).

Inputs:  --recon-root from `reconstruct.py --dataset proxy` (or a shared1000 run), and --proxy-dir with the
         per-block ground truth (data/proxy_set/blocks/blockNN_gt.pt; `make_shared1000_blockset.py` writes
         the same format)
Outputs: one TSV with the columns of the harness's own --out; "subj" holds the block index, "mode" is "imagery"

The harness takes ONE --gt file for every --subj, but each block has its own ground truth; the
wrapper imports the harness and reuses `Nets`, `build_final_recons`, `run_one` and `load_ground_truth`
unchanged (it stays bit-identical to what the five baselines went through). Loading `Nets` once avoids a
model reload per block. A small argparse.Namespace stands in for the harness `args` (model="template",
path_template, recon_root, gen_rep).

Usage (MindEye2 environment, CPU: EffNet/SwAV stay on CPU as in the harness):
    python evaluate_block_set.py --recon-root RECON_ROOT/T0_s0.75_m0.4 --label T0_s0.75_m0.4 --gen-rep 4 \
        --out results/oracle_budget/setting_sweep/T0_s0.75_m0.4.tsv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

MINDEYE_II_SRC = Path(__file__).resolve().parents[2] / "mindeye_II" / "src"
sys.path.insert(0, str(MINDEYE_II_SRC))
import final_evaluations_imagery as fe  # noqa: E402  (needs the sys.path insert above)

PATH_TEMPLATE = "subj{subj02}/{cond}/sample{sample03}/{stim05}.png"
PROXY_DIR_DEFAULT = "data/proxy_set"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recon-root", required=True,
                     help="reconstruct.py --dataset proxy's --out-dir for this arm")
    ap.add_argument("--proxy-dir", default=PROXY_DIR_DEFAULT)
    ap.add_argument("--label", required=True, help="filed under 'model' in the output TSV")
    ap.add_argument("--gen-rep", type=int, default=4,
                     help="must match reconstruct.py's --gen-rep for this arm")
    ap.add_argument("--blocks", type=int, nargs="+", default=None,
                     help="default: every block in manifest.json")
    ap.add_argument("--device", default="cuda", help="cuda or cpu for the GPU-side nets "
                     "(AlexNet/InceptionV3/CLIP) -- EffNet/SwAV always run on CPU, matching "
                     "final_evaluations_imagery.py's Nets class")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    fe._blas_sanity()

    with open(Path(args.proxy_dir) / "manifest.json") as f:
        manifest = json.load(f)
    blocks = args.blocks if args.blocks is not None else list(range(manifest["n_blocks"]))
    print(f"{len(blocks)} block(s), recon_root={args.recon_root}, label={args.label}")

    print("loading the six feature extractors...")
    nets = fe.Nets(args.device)

    fake_args = argparse.Namespace(model="template", path_template=PATH_TEMPLATE,
                                    recon_root=args.recon_root, recons_pt="",
                                    gen_rep=args.gen_rep)

    rows = []
    for b in blocks:
        gt_path = Path(args.proxy_dir) / "blocks" / f"block{b:02d}_gt.pt"
        gt_true = fe.load_ground_truth(gt_path)[:fe.N_TEST]
        print(f"\n=== {args.label}  block{b:02d}  imagery ===")
        rows += fe.run_one(fake_args, nets, gt_true, gt_true, b, "imagery", args.label)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["model", "subj", "mode", "stim_type", "sample", "repetition"] + fe.METRICS
    with open(out, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(f"{r[c]:.6f}" if c in fe.METRICS else str(r[c])
                              for c in cols) + "\n")
    print(f"\nwrote {len(rows)} rows to {out}")

    print(f"\n--- mean over {len(blocks)} block(s) x 12 stimuli x {args.gen_rep} rep(s) ---")
    for m in fe.METRICS:
        print(f"  {m}: {np.mean([r[m] for r in rows]):.4f}")
    for stim_type in ("simple", "complex"):
        sel = [r for r in rows if r["stim_type"] == stim_type]
        print(f"  --- {stim_type} only ---")
        for m in fe.METRICS:
            print(f"    {m}: {np.mean([r[m] for r in sel]):.4f}")


if __name__ == "__main__":
    main()
