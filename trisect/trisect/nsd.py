"""NSD-core betas normalization: one recipe, three call shapes.

All three do /300 and then a voxelwise z-score with the moments of the FIT split; the variants cannot
drift apart without an error:
    zscore_fmri()             all rows, normalized with the moments of `fit_mask` (ridge, MLP)
    zscore_fmri_train_test()  (train, test) view, Brain-Diffuser's ridge recipe
    zscore_fmri_arrays()      the same for arrays already split and repetition-averaged
"""
import numpy as np

BETAS_SCALE = 300.0  # NSD betas are stored as int16 * 300 (NSD Data Manual)


def zscore_fmri(betas, fit_mask, eps=1e-8):
    """/300 then voxelwise z-score with the FIT split's own moments, applied to ALL rows.

    `fit_mask` selects the rows whose mean/std define the normalization: the train split for the
    ridge, the smaller train-minus-dev split for the MLP; dev never leaks into the
    normalization either. Returns the whole array, normalized.
    """
    fmri = betas / BETAS_SCALE
    fit_fmri = fmri[fit_mask]
    mean = fit_fmri.mean(axis=0)
    scale = fit_fmri.std(axis=0, ddof=1)
    scale = np.maximum(scale, eps)  # guard a degenerate zero-variance voxel
    return (fmri - mean) / scale


def zscore_fmri_train_test(betas, train_mask, test_mask, eps=1e-8):
    """(train, test) split view of zscore_fmri(), with the TRAIN split's moments.

    This is the recipe of the upstream Brain-Diffuser ridge. It is a separate function because it
    never materializes the full normalized array for the rows nobody asked for.
    """
    fmri = betas / BETAS_SCALE
    train_fmri = fmri[train_mask]
    mean = train_fmri.mean(axis=0)
    scale = train_fmri.std(axis=0, ddof=1)
    scale = np.maximum(scale, eps)
    return (fmri[train_mask] - mean) / scale, (fmri[test_mask] - mean) / scale


def zscore_fmri_arrays(train_betas, test_betas, eps=1e-8):
    """Same formula as zscore_fmri_train_test(), but for train/test arrays that are already
    split and already REPETITION-AVERAGED (averaged repetitions raise the ridge's input SNR).
    The averaging has to happen on raw betas, before this call: the moments are computed from the
    averaged train rows, never from the single-trial ones. Average first, z-score after.
    """
    train_fmri = train_betas / BETAS_SCALE
    test_fmri = test_betas / BETAS_SCALE
    mean = train_fmri.mean(axis=0)
    scale = train_fmri.std(axis=0, ddof=1)
    scale = np.maximum(scale, eps)
    return (train_fmri - mean) / scale, (test_fmri - mean) / scale
