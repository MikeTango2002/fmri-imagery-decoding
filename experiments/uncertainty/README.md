# experiments/uncertainty/

The two ways of using per-trial uncertainty that Section 8.1.2 tests and rejects: gating the generator
by how much the ridge is trusted on a trial, and weighting the repetitions of an image by their
reliability.

## Files

| File | What it does | Thesis section |
|---|---|---|
| `gating_oracle_ceiling.py` | Upper bound of any per-trial choice of strength and mixing: for every image, the best of the nine settings of the 3×3 sweep, scored on the same replica (selection on generation noise) and cross-fitted across replicas. Reads only TSVs already on disk. | 8.1.2 (Table 8.3) |
| `ridge_posterior_sigma2.py` | Closed-form posterior variance of the ridge per branch, calibrated on NSD-core training trials only, and its Spearman correlation with the leave-one-out squared error. | 8.1.2 |
| `sigma2_vs_error_diagnostics.py` | Diagnostics of that criterion: a power analysis of the test, a screen of other per-trial signals, and a comparison of leverage on training and on out-of-design trials. | 8.1.2 |
| `weighted_repetition_fusion.py` | Precision-weighted fusion of the three repetitions of an image on the development split, against the uniform mean and against averaging the voxels before the ridge. | 8.1.2 |

## Notes

- `sigma2_vs_error_diagnostics.py` needs the cache written by `ridge_posterior_sigma2.py` (the
  eigendecomposition of X^T X, about 2 GB per subject), which is not included in this repository.
- The metric file of the weighted fusion is `results/uncertainty/weighted_repetition_fusion_dev.tsv`.
  `f1_diagnostics.json`, written next to the cache of `ridge_posterior_sigma2.py`, keeps its original name.
- Some names (`D1_ALPHAS`, `d1_power`, `d2_feature_screen`, `d3_shift`, `f1_diagnostics.json`) keep the label
  of the work. The main README, "Working labels in some names", says what each stands for.
