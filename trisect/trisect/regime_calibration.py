"""Regime calibration: three corrections of the ridge input, fitted on the NSD-Imagery calibration set
(attention + vision-conceptual trials, label-blind).

All three share one projection, z = x @ V_k (top-K eigenvectors of X^T X, from the eigendecomposition
that `ridge_posterior_sigma2.py` caches). The corrections realign the calibration set's amplitude or
covariance in z-space onto the training distribution's, before `predict_flat()`:
    gain (1 scalar) -> diag_gain (K scalars) -> coral (K x K, full covariance)
No correction is the identity member of the same family. `leverage_u()` is the leverage baseline.

Why it is exact on the training side: train_fmri is voxelwise zero-mean; Z_train is zero-mean and
C_train = diag(eigval_k / (n_train - 1)). Only C_calib is estimated, from M=336 rows. coral shrinks it
(Ledoit-Wolf, toward a scaled identity) because M cannot support a plain K x K estimate when K nears
M. Shrinkage and K are fixed a priori, never tuned on a reconstruction metric.
"""
import numpy as np


def project_topk(x, eigvecs, eigvals, k):
    """x: (N, V). eigvecs: (V, V) ascending by eigval. -> z (N, k), eigvecs_topk (V, k) [V_k,
    columns = the k LARGEST-eigenvalue directions, kept in ascending order within the slice],
    eigvals_topk (k,)."""
    V_k = eigvecs[:, -k:]
    eigvals_topk = eigvals[-k:]
    z = x.astype(np.float64) @ V_k
    return z, V_k, eigvals_topk


def train_var_topk(eigvals_topk, n_train):
    """Var_train(z_k) for the top-k directions; exact, not estimated (see module docstring)."""
    return eigvals_topk / (n_train - 1)


def fit_gain_scalar(Z_calib, eigvals_topk, n_train):
    """Single scalar g: sqrt(total train variance in the K-subspace / total calib variance in
    the same subspace). g > 1 means the calibration set's signal in the directions the ridge
    uses is WEAKER than training's, the coarsest member of the correction ladder."""
    train_var = train_var_topk(eigvals_topk, n_train)
    calib_var = Z_calib.var(axis=0, ddof=1)
    g = float(np.sqrt(train_var.sum() / max(calib_var.sum(), 1e-12)))
    return g


def fit_diag_gain(Z_calib, eigvals_topk, n_train):
    """Per-direction gain, no off-diagonal realignment, the diagonal fallback when the full
    covariance estimate is unstable."""
    train_var = train_var_topk(eigvals_topk, n_train)
    calib_var = np.maximum(Z_calib.var(axis=0, ddof=1), 1e-12)
    return np.sqrt(train_var / calib_var)


def fit_coral(Z_calib, eigvals_topk, n_train, shrinkage=0.3, cond_threshold=1e4):
    """Full K x K covariance realignment: A = C_train^(1/2) @ C_calib_shrunk^(-1/2).

    C_train is exactly diag(eigvals_topk/(n_train-1)) (see module docstring). C_calib is
    Ledoit-Wolf shrunk toward its own scaled identity (mean diagonal * I) BEFORE inversion.
    `shrinkage` and `cond_threshold` are fixed constants, never tuned against a reconstruction
    metric; `cond_threshold` only decides whether this K is numerically usable at all, returning
    `stable=False` if not (the caller retries at a smaller K or uses fit_diag_gain instead).

    Returns (A, mu_calib, diagnostics). A broadcasts as z_aligned = (z - mu_calib) @ A.T.
    """
    mu_calib = Z_calib.mean(axis=0)
    Zc = Z_calib - mu_calib
    n_calib = Zc.shape[0]
    C_calib = (Zc.T @ Zc) / (n_calib - 1)
    mean_var = float(np.trace(C_calib)) / C_calib.shape[0]
    eps = shrinkage * mean_var
    C_shrunk = (1.0 - shrinkage) * C_calib + eps * np.eye(C_calib.shape[0])

    w, U = np.linalg.eigh(C_shrunk)
    w = np.maximum(w, 1e-12)
    cond_number = float(w.max() / w.min())
    stable = cond_number <= cond_threshold

    C_calib_inv_sqrt = (U * (1.0 / np.sqrt(w))) @ U.T
    train_var = train_var_topk(eigvals_topk, n_train)
    C_train_sqrt = np.diag(np.sqrt(train_var))
    A = C_train_sqrt @ C_calib_inv_sqrt

    diagnostics = {
        "n_calib": int(n_calib), "k": int(C_calib.shape[0]), "shrinkage": float(shrinkage),
        "cond_number_shrunk": cond_number, "cond_threshold": float(cond_threshold),
        "stable": bool(stable), "raw_calib_min_eig": float(np.linalg.eigvalsh(C_calib).min()),
        "raw_calib_max_eig": float(np.linalg.eigvalsh(C_calib).max()),
    }
    return A, mu_calib, diagnostics


def apply_correction(x, V_k, mu_calib, transform):
    """x: (N, V) raw input (same per-run-zscored voxel space the calibration set and the
    NSD-Imagery eval trials both live in). `transform`: callable Zc (N,k) -> Zc_aligned (N,k),
    applied to the MEAN-CENTERED (on mu_calib) top-k coefficients. Only the top-K subspace is
    touched; x's component orthogonal to V_k passes through unchanged.

    z' - z is added back in voxel space rather than replacing x outright; a correction that
    is a no-op in Z-space (transform = identity) is an EXACT no-op on x (bitwise, up to the
    float64 projection round-trip). Callers must run this identity check before trusting
    anything downstream.
    """
    x64 = x.astype(np.float64)
    z = x64 @ V_k
    zc = z - mu_calib
    zc_aligned = transform(zc)
    delta = (zc_aligned - zc) @ V_k.T
    return (x64 + delta).astype(x.dtype)


def gain_transform(g):
    return lambda zc: g * zc


def diag_gain_transform(diag_gain):
    return lambda zc: zc * diag_gain[None, :]


def coral_transform(A):
    return lambda zc: zc @ A.T


def leverage_u(x, eigvecs, eigvals, alpha):
    """u(x) = x^T (X^T X + alpha I)^-1 x via the FULL eigenbasis (not top-K, alpha regularizes
    every direction, small eigenvalues included; truncating would overstate u). x: (N, V).
    Returns (N,): the per-trial leverage baseline, computed once here for the calibration set
    and reusable verbatim for any other pool (train/vision/imagery)."""
    z = x.astype(np.float64) @ eigvecs
    return np.sum(z ** 2 / (eigvals + alpha)[None, :], axis=1)
