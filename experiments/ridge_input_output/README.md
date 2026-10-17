# experiments/ridge_input_output/

Changes to the input and to the output of the ridge decoder of Section 5.3, tested in Section 8.1.5.
None was adopted. The ridge fitted on image means was worse, the restricted ROI and the regime
calibration did not show a useful pattern in latent space, the reliability weighting did not transfer
to NSD-Imagery, and the per-dimension gain broke the generator.

## Files

| Alternative | File | What it does | Thesis section |
|---|---|---|---|
| Image means or single trials | `compare_averaged_vs_single_trials.py` | Ridge fitted on image means against ridge fitted on single trials, on the low-level factor, with the criterion fixed before the run. | 8.1.5, 5.3 |
| | `run_averaged_vs_single_trials.sh` | Fits the control (single trials, grouped cross-validation) and the treatment (image means), then runs the comparison. CPU only. | 8.1.5 |
| Voxel reliability and ROI | `map_streams_rois.py` | Maps the `streams` atlas onto the voxel order of the `nsdgeneral` mask. | 8.1.5 |
| | `voxel_reliability_and_roi.py` | Ridge fits with reliability-weighted voxels and with one `streams` ROI at a time, scored in latent space on shared1000. | 8.1.5 |
| | `run_voxel_reliability_and_roi.sh` | Runs the two voxel-side arms (reliability weighting, `streams` ROI) for all subjects and branches. CPU only. | 8.1.5 |
| | `compare_voxel_reliability_weighting.py` | End-to-end paired comparison of the weighted and the unweighted ridge, 83 blocks × 4 subjects. | 8.1.5 |
| Regime calibration | `regime_calibration_extract.py` | Extracts the calibration set per subject: the NSD-Imagery attention trials and vision-conceptual trials, which the benchmark does not score. | 8.1.5 |
| | `regime_calibration_fit.py` | Fits the three corrections (scalar gain, per-direction gain, CORAL) in the eigenbasis of the ridge, without using any label. | 8.1.5 |
| | `regime_calibration_precheck.py` | Applies each correction before the ridge and compares in latent space on the vision and imagery pools. | 8.1.5 |
| Per-dimension gain | `dimension_gain_precheck.py` | Gain proportional to the predictability of each output dimension, checked in latent space on the development split. | 8.1.5 |
| | `compare_dimension_gain.py` | End-to-end paired comparison on shared1000 (−11.76 pp in the thesis). | 8.1.5 |
| | `run_gain_and_reliability_reconstructions.sh` | Full-pipeline generation and scoring of the gain arm and of the reliability-weighted arm on the 83 blocks of shared1000, against the final model as control. | 8.1.5 |

The regime-calibration module itself (`regime_calibration.py`: the three corrections and the projection
on the eigenbasis) is in `trisect/trisect/`.

## Notes

- The cheap latent-space steps run on the CPU; the end-to-end steps need the GPU and use
  `trisect/scripts/reconstruct.py` and `evaluation/evaluate_block_set.py`.
- `regime_calibration_fit.py` and `dimension_gain_precheck.py` need the eigendecomposition of X^T X
  cached by `experiments/uncertainty/ridge_posterior_sigma2.py`, which therefore runs first.
- `compare_averaged_vs_single_trials.py` imports `stats_for()` from `experiments/latent_error/latent_prediction_error.py`.
- Neither the restricted ROI nor the regime calibration was run end to end (Section 8.1.5).
- The metric files are in `results/ridge_input_output/`. Two outputs outside `results/` keep their
  original names: `h4_all_results.json` (latent-space scores of `voxel_reliability_and_roi.py`) and
  the folder `data/betas_trisect/h3_correction`.
- The shell scripts expect the Python files in `src/` under the project root, as in the layout of the
  top-level README.
- Some names (`H0_FLOOR_JSON`, `h2_e2e`, `h4_e2e`, `h3_correction`, `betas_trisect/h4`) keep the label of the
  work. The main README, "Working labels in some names", says what each stands for.
