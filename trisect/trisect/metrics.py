"""Pixel-space correlation, the one metric TRISECT computes outside the Kneeland harness.

This is a quick in-loop check (used by `decode_zlow_to_image.py` and `smoke_vd_init_image.py`), not the
benchmark's PixCorr: the reported numbers come from `evaluation/`.
`pixcorr()` works on batches; `pixcorr_single()` returns nan, not 0, for a constant image.
"""
import numpy as np


def pixcorr(pred, true):
    """Pearson correlation of flattened RGB pixels, per image. Both (n, ...), same shape."""
    p = pred.reshape(pred.shape[0], -1).astype(np.float64)
    t = true.reshape(true.shape[0], -1).astype(np.float64)
    p = p - p.mean(axis=1, keepdims=True)
    t = t - t.mean(axis=1, keepdims=True)
    num = (p * t).sum(axis=1)
    den = np.sqrt((p ** 2).sum(axis=1) * (t ** 2).sum(axis=1))
    return num / np.maximum(den, 1e-8)


def pixcorr_single(pred, true):
    """Scalar pixcorr between two same-size images, nan on a degenerate (constant) input.

    It differs from pixcorr() only in returning nan instead of 0 when an image is constant:
    a smoke test wants to see that, a batch loop does not.
    """
    p = np.asarray(pred, dtype=np.float64).ravel()
    t = np.asarray(true, dtype=np.float64).ravel()
    p = p - p.mean()
    t = t - t.mean()
    denom = np.sqrt((p ** 2).sum() * (t ** 2).sum())
    return float((p * t).sum() / denom) if denom > 0 else float("nan")
