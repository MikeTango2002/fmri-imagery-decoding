"""Rebuild Brain-Diffuser's shared1000 test predictions (VDVAE, CLIP text, CLIP vision) without its features.

Inputs:  <bd-root>/data/{regression_weights/subj0N/*.pkl, train_target_stats.npz, processed_data/}
Outputs: <bd-root>/data/predicted_features/subj0N/nsd_{vdvae,cliptext,clipvision}_predtest_nsdgeneral.npy

They feed the swap experiment (Section 8.4). The cleanup after benchmark A removed the extracted features;
the *_regression.py scripts cannot run again (they refit the ridge). The frozen weights and training moments are
all the predict-only half needs: reg.predict(test_fmri), then a per-dimension re-standardization onto the
training moments. The output is bit-identical to what *_regression.py wrote.

Usage (numpy only; from the project root):
    export OPENBLAS_CORETYPE=SkylakeX   # large matmuls (982 x ~14k) x (~14k x 91168)
    python bd_predict_shared1000.py --bd-root ../brain_diffuser --sub 1 2 5 7
"""
import argparse
import os
import pickle

import numpy as np

SUBJECTS = (1, 2, 5, 7)
NUM_VOXELS = {1: 15724, 2: 14278, 5: 13039, 7: 12682}

# name -> (weights .pkl, stats key prefix, output filename template)
BRANCHES = {
    "vdvae": ("vdvae_regression_weights.pkl", "vdvae",
              "nsd_vdvae_nsdgeneral_pred_sub{sub}_31l_alpha50k.npy"),
    "cliptext": ("cliptext_regression_weights.pkl", "cliptext",
                 "nsd_cliptext_predtest_nsdgeneral.npy"),
    "clipvision": ("clipvision_regression_weights.pkl", "clipvision",
                   "nsd_clipvision_predtest_nsdgeneral.npy"),
}


def load_reg(path):
    with open(path, "rb") as f:
        d = pickle.load(f)
    return d["weight"], d["bias"]


def predict_flat(X, W, b):
    # vdvae_regression.py: reg.fit(train_fmri, train_latents) -> coef_ is (D, V).
    return X @ W.T + b


def predict_tokenwise(X, W, b):
    # cliptext/clipvision_regression.py: one Ridge per token, coef_ stacked as (T, dim, V).
    N, T = X.shape[0], W.shape[0]
    out = np.zeros((N, T, W.shape[1]), dtype=np.float32)
    for t in range(T):
        out[:, t] = X @ W[t].T + b[t]
    return out


def restandardize(pred, tgt_mean, tgt_std):
    """*_regression.py's own re-inflation, onto the TRAINING targets' moments directly (no
    pool indirection: imagery_regression_predict.py's --restandardize_pool only exists
    because NSD-Imagery has no equivalent of "the test set's own predictions", which
    shared1000 does: this is exactly what the original scripts compute)."""
    std_norm = (pred - pred.mean(axis=0)) / pred.std(axis=0)
    return std_norm * tgt_std + tgt_mean


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bd-root", default="../brain_diffuser")
    ap.add_argument("--sub", nargs="+", type=int, default=list(SUBJECTS))
    ap.add_argument("--branches", nargs="+", default=list(BRANCHES), choices=list(BRANCHES))
    args = ap.parse_args()

    if os.environ.get("OPENBLAS_CORETYPE") != "SkylakeX":
        raise SystemExit("OPENBLAS_CORETYPE is not SkylakeX -- export it before running "
                          "(the CPU of the server miscomputes matrix products without it)")

    data_root = os.path.join(args.bd_root, "data")
    for sub in args.sub:
        if sub not in SUBJECTS:
            raise SystemExit(f"subject {sub} is not one of {SUBJECTS}")
        print(f"=== subj{sub:02d} ===", flush=True)
        stats_path = os.path.join(data_root, "regression_weights", f"subj{sub:02d}",
                                   "train_target_stats.npz")
        stats = np.load(stats_path)

        test_path = os.path.join(data_root, "processed_data", f"subj{sub:02d}",
                                  f"nsd_test_fmriavg_nsdgeneral_sub{sub}.npy")
        # The input z-score with the TRAINING moments is applied, as *_regression.py does for this test set.
        # imagery_regression_predict.py skips it, because the NSD-Imagery betas are normalized differently.
        test_fmri = np.load(test_path).astype(np.float64) / 300.0
        test_fmri = (test_fmri - stats["fmri_mean"]) / stats["fmri_std"]
        print(f"  test_fmri: {test_fmri.shape} mean={test_fmri.mean():.4f} "
              f"std={test_fmri.std():.4f}")

        out_dir = os.path.join(data_root, "predicted_features", f"subj{sub:02d}")
        os.makedirs(out_dir, exist_ok=True)

        for branch in args.branches:
            wname, skey, out_tmpl = BRANCHES[branch]
            wpath = os.path.join(data_root, "regression_weights", f"subj{sub:02d}", wname)
            print(f"  loading {wname} ({os.path.getsize(wpath) / 1e9:.1f} GB)", flush=True)
            W, b = load_reg(wpath)
            if W.shape[-1] != NUM_VOXELS[sub]:
                raise SystemExit(f"{wname}: weights have {W.shape[-1]} voxels, "
                                  f"subject {sub} has {NUM_VOXELS[sub]}")

            raw = (predict_flat(test_fmri, W, b) if branch == "vdvae"
                   else predict_tokenwise(test_fmri, W, b)).astype(np.float32)
            del W, b

            tgt_mean, tgt_std = stats[skey + "_mean"], stats[skey + "_std"]
            if raw.shape[1:] != tgt_mean.shape:
                raise SystemExit(f"{branch}: prediction shape {raw.shape[1:]} vs target "
                                  f"moments {tgt_mean.shape}")
            pred = restandardize(raw, tgt_mean, tgt_std).astype(np.float32)

            out_path = os.path.join(out_dir, out_tmpl.format(sub=sub))
            np.save(out_path, pred)
            print(f"    {branch} -> {os.path.basename(out_path)} {pred.shape}", flush=True)

    print("done", flush=True)


if __name__ == "__main__":
    main()
