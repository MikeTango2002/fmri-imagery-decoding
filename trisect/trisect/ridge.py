"""Re-standardization of ridge predictions onto the training targets' moments.

One formula, several entry points. A copy of it once gave a mu whose std was 5-25x too small and
cost 13-17 pp of identification accuracy; there is a single implementation here.
Upstream reference: Brain-Diffuser `scripts/imagery_regression_predict.py`, lines 83-90.
"""
import numpy as np


def restandardize(pred, pool_mean, pool_std, tgt_mean, tgt_std, eps=1e-8):
    """Per-dimension affine map: pool moments -> train-target moments.

    The canonical body. `pool_mean`/`pool_std` are the moments of the pool the predictions are
    normalized against, computed by the caller. Which pool to use (the test trials themselves, the
    vision trials, the training set) is a choice of the calling script
    (`predict_ridge_imagery.py --restandardize-pool`).
    """
    return (pred - pool_mean) / np.maximum(pool_std, eps) * tgt_std + tgt_mean


def restandardize_from_pred(pred, train_target, eps=1e-8):
    """restandardize() with the pool taken from `pred` itself, as the upstream recipe does.

    Identical arithmetic to
    restandardize(pred, pred.mean(0), pred.std(0), train_target.mean(0), train_target.std(0)).
    """
    return restandardize(pred, pred.mean(axis=0), pred.std(axis=0),
                         train_target.mean(axis=0), train_target.std(axis=0), eps=eps)


def restandardize_scalar(pred, tgt_mean, tgt_std, eps=1e-8):
    """Global-scalar variant of restandardize(): match the target distribution's OVERALL mean and
    std instead of per-dimension ones.

    restandardize() does two things at once, and on z_low they pull in opposite directions.
    (1) It buys back the SCALE the ridge shrank away: without it z_low leaves the ridge at std
    ~0.024 against a target scale of ~1, and the VDVAE decoder would get a near-zero latent, i.e.
    a flat init image. That part is needed. (2) It also EQUALIZES variance dimension by dimension,
    and that costs 6.0 pp of branch two-way identification on z_low: PCA components are ordered by
    variance by construction; flattening them destroys the very ordering the representation
    encodes. This variant keeps (1) and drops (2): one scalar affine map for the whole branch.

    The target's global moments are recovered exactly from the per-dimension ones by the law of
    total variance: Var_global = E[sigma_d^2] + Var(mu_d).
    """
    tgt_mean_g = float(tgt_mean.mean())
    tgt_std_g = float(np.sqrt((tgt_std ** 2).mean() + tgt_mean.var()))
    pool_mean_g, pool_std_g = float(pred.mean()), float(pred.std())
    return (pred - pool_mean_g) / max(pool_std_g, eps) * tgt_std_g + tgt_mean_g


def dim_gain(r, gamma, floor=0.0):
    """Per-dimension gain a_k: a_k proportional to |r_k|^gamma, normalized to mean 1 so it
    reallocates amplitude across dimensions instead of rescaling the branch as a whole (that
    global rescale is restandardize_scalar()'s job; the two must not be confounded).

    gamma=0 -> gain is exactly all-ones (|r_k|**0 == 1 elementwise, mean 1 already), and
    restandardize_gain(pred, train_target, dim_gain(r, 0)) is bit-identical to
    restandardize_from_pred(pred, train_target). That identity is the sanity check of the family.

    `floor` implements the "branch off" extreme: pass floor=0 with an all-zero `r` (or call
    restandardize_gain directly with a zero gain array) to silence a branch.
    """
    r = np.abs(np.asarray(r, dtype=np.float64))
    if gamma == 0:
        return np.ones_like(r)
    g = np.maximum(r, floor) ** gamma
    m = g.mean()
    return g / m if m > 0 else np.ones_like(r)


def sample_residual(pred_mmse, resid_var, rng):
    """The `z ~ q` alternative to restandardize() for a branch whose ridge output IS the MMSE
    estimate (std_k ~ r_k * sigma_k): add per-dimension isotropic Gaussian noise up to the train
    target's own variance instead of rescaling every dimension to that variance deterministically.

    `restandardize()` matches std_k exactly by dividing out the POOL's std and multiplying back
    by tgt_std_k. That rescales the same ~56 effective directions the ridge actually predicts;
    the ~970 near-zero-r_k dimensions of z_low come out at FULL target amplitude even though they
    are (by construction) linear combinations of those same few voxel directions: correlated
    noise, not isotropic. This function adds fresh N(0, resid_var_k) noise instead; the
    non-predicted variance is decorrelated across dimensions the way VDVAE's own posterior samples
    are (its training distribution), while the total per-dimension variance still lands at the
    train target's sigma_k^2 (Var(pred_mmse_k) + resid_var_k = sigma_k^2 by construction of
    resid_var_k, see estimate_resid_var() below): same amplitude as restandardize(), different
    covariance.

    `pred_mmse` is the branch's raw ridge prediction (the output of predict_flat, averaged over
    repetitions, NOT yet restandardized). `resid_var` is (D,), one value per flattened
    dimension. `rng` is a numpy Generator, seeded by the caller (the `diffusion_seed` scheme of
    `reconstruct.py`; replicates are reproducible).
    """
    resid_var = np.maximum(np.asarray(resid_var, dtype=np.float64), 0.0)
    noise = rng.normal(0.0, np.sqrt(resid_var), size=pred_mmse.shape)
    return pred_mmse + noise.astype(pred_mmse.dtype)


def estimate_resid_var(pred_mmse_dev, tgt_std):
    """Per-dimension residual variance for sample_residual(): sigma_k^2 (train target, already on
    disk in train_target_stats.npz) minus the DEV-split prediction's own variance, floored at 0.
    Estimated only on the dev split, because selection and calibration never touch shared1000 or
    NSD-Imagery. `pred_mmse_dev` is the raw output of predict_flat on the dev split, averaged
    over repetitions, shape (n_dev, D).
    """
    tgt_var = np.asarray(tgt_std, dtype=np.float64) ** 2
    pred_var = pred_mmse_dev.astype(np.float64).var(axis=0)
    return np.maximum(tgt_var - pred_var, 0.0)


def restandardize_gain(pred, train_target, gain, eps=1e-8):
    """restandardize_from_pred() with an extra per-dimension amplitude gain on the target scale.

    Generalizes restandardize_from_pred(): the uniform equalization (every PCA dimension inflated
    to the FULL train-target std regardless of how predictable it is, i.e. gain==1 everywhere) is
    the gamma=0 member of the family dim_gain(r, gamma). `gain` must broadcast against the
    flattened per-dimension axis of `pred`/`train_target` (their trailing dims, reshaped by the
    caller to (n, D) first: this function does not reshape).
    """
    pred_mean, pred_std = pred.mean(axis=0), pred.std(axis=0)
    pred_std = np.maximum(pred_std, eps)
    tgt_mean, tgt_std = train_target.mean(axis=0), train_target.std(axis=0)
    gain = np.asarray(gain, dtype=tgt_std.dtype)
    return (pred - pred_mean) / pred_std * (tgt_std * gain) + tgt_mean
