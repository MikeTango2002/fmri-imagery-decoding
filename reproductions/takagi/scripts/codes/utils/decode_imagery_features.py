"""Apply the ridge decoders fitted on NSD-Core to the prepared NSD-Imagery betas.

Inputs:  the .npz of `ridge.py --save_weights` (--weights_dir) and the prepared betas (--imagery_dir)
Outputs: --out_dir/<subject>_<rois>_scores_<target>_<cond>.npy

The inference half of ridge.py: no fit, only the forward pass. RAM: the `blip` .npz is ~17 GB and cannot be
memory-mapped; run one subject at a time and never two `blip` processes on a host without checking `free -h`.
`init_latent` is ~0.1 GB.

Usage (from codes/utils/):
    python decode_imagery_features.py --subject subj01 --target init_latent --imagery_dir ../../mrifeat_imagery \
      --weights_dir ../../decoded --out_dir ../../decoded_imagery
"""
import argparse
import os
import time

import numpy as np

CONDITIONS = ["visionsimple", "visioncomplex",
              "imagerysimple", "imagerycomplex", "imageryconcepts"]

WEIGHT_FILES = {
    "init_latent": "{s}/{s}_early_weights_init_latent.npz",
    "blip": "{s}/{s}_early_ventral_midventral_midlateral_lateral_parietal_weights_blip.npz",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subject", type=str, required=True,
                        help="subj01 or subj02 or subj05 or subj07")
    parser.add_argument("--target", type=str, required=True,
                        choices=sorted(WEIGHT_FILES), help="init_latent or blip")
    parser.add_argument("--imagery_dir", type=str, default="../../mrifeat_imagery",
                        help="--out_dir of prepare_nsd_imagery_takagi.py")
    parser.add_argument("--weights_dir", type=str, default="../../decoded")
    parser.add_argument("--out_dir", type=str, default="../../decoded_imagery")
    parser.add_argument("--conditions", type=str, nargs="+", default=CONDITIONS)
    opt = parser.parse_args()

    subject = opt.subject
    wpath = os.path.join(opt.weights_dir, WEIGHT_FILES[opt.target].format(s=subject))
    t = time.time()
    print(f"loading {wpath} ({os.path.getsize(wpath) / 2 ** 30:.2f} GB)", flush=True)
    w = np.load(wpath)
    coef = w["coef"]
    intercept = w["intercept"]
    rois = [str(r) for r in w["roi"]]           # the hstack order recorded at fit time
    sizes = [int(n) for n in w["roi_sizes"]]
    print(f"  loaded in {time.time() - t:.1f}s: roi {rois}, sizes {sizes}, "
          f"coef {coef.shape}", flush=True)

    savedir = os.path.join(opt.out_dir, subject)
    os.makedirs(savedir, exist_ok=True)

    for cond in opt.conditions:
        blocks = []
        for roi, n in zip(rois, sizes):
            path = os.path.join(opt.imagery_dir, subject,
                                f"{subject}_{roi}_{cond}_betas_ave.npy")
            b = np.load(path)
            assert b.shape[1] == n, f"{roi}: {b.shape[1]} voxels, {n} expected"
            blocks.append(b)
        X = np.hstack(blocks).astype(np.float32)                    # (6, V_tot)
        assert X.shape[1] == coef.shape[0], f"X {X.shape} vs coef {coef.shape}"

        # No (X - scaler_mean) / scaler_scale here: the per-run z-score already puts every
        # voxel at mean 0 / std 1, which is exactly what the training StandardScaler
        # produced. Applying both would shrink the signal by two or three orders of
        # magnitude and collapse the predictions towards zero.
        pred = X @ coef + intercept                                 # (6, 6400) | (6, 173568)

        out = os.path.join(savedir,
                           f'{subject}_{"_".join(rois)}_scores_{opt.target}_{cond}.npy')
        np.save(out, pred)
        print(f"{cond}: X {X.shape} -> {pred.shape}, "
              f"|pred| mean {np.abs(pred).mean():.4f}, std {pred.std():.4f} -> {out}",
              flush=True)

    print(f"done in {time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
