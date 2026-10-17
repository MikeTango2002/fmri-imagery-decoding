# experiments/low_level_factor/

Alternatives for the low-level factor `z_low`, the weakest part of the model (Sections 5.5, 8.1.3 and
8.4). Each one was tested before it was built, and none replaced the PCA of the VDVAE latent.

## Files

| Alternative | File | What it does | Thesis section |
|---|---|---|---|
| Pixel target | `extract_pixel_low_target.py` | The 16 × 16 RGB thumbnail of the image as low-level target, no PCA and no VDVAE. | 8.1.3 |
| | `run_pixel_low_target_imagery.sh` | Predictions, reconstruction and scoring on NSD-Imagery with the pixel target, four subjects. | 8.1.3 |
| Uncompressed target | `fit_ridge_uncompressed_zlow.py` | Ridge on the raw 91,168-dimensional latent, next to a refit of the compressed one. | 8.1.3 |
| | `run_uncompressed_zlow.sh` | Generation and scoring of both arms on NSD-Imagery. | 8.1.3 |
| | `compare_uncompressed_zlow.py` | Paired comparison with the criterion fixed before the run. | 8.1.3 |
| Decodable basis | `decodable_basis_precheck.py` | SVD of the ridge predictions instead of the image PCA, checked on the development split for k = 64, 128 and 256. | 8.1.3 |
| | `decodable_basis_predict_shared1000.py`, `decodable_basis_predict_imagery.py` | Write `z_low_pred.npy` from the decodable basis, for shared1000 and for NSD-Imagery. | 8.1.3 |
| | `run_decodable_basis_eval.sh` | Scores the reconstructions of the four subjects with the harness. | 8.1.3 |
| | `compare_decodable_basis_vs_pca.py` | Paired reading against the ridge with the PCA, on 65 blocks, with the noise floor. | 8.1.3 |
| Residual sampling | `residual_sampling_precheck.py` | Decoder-side check of `sample_residual()`, which fills the part of `z_low` the ridge does not predict with noise, instead of re-standardizing. | 8.1.3 |
| Scalar re-standardization | `run_scalar_restandardization.sh` | Three arms on NSD-Imagery: one mean and scale for the whole `z_low` branch, the production setting, and no re-standardization. | 8.1.3 |

## Steps that are not scripts

- Ridge for the pixel target. `run_pixel_low_target_imagery.sh` expects the weights in
  `data/betas_trisect/ridge_weights_check_pixel16`, which were made with, once per subject,
  `python src/fit_ridge.py --subj N --branches z_low --zlow data/targets/pixel_low_res16/z_low.npy --out-dir data/betas_trisect/ridge_weights_check_pixel16`
  (the file written by `extract_pixel_low_target.py` is `data/targets/pixel_low_res16/z_low.npy`).
- Reconstructions of the decodable basis. Before `run_decodable_basis_eval.sh`, the four
  subjects were reconstructed with `reconstruct.py` on the 65 blocks, with `--pred-dir` pointing at
  the output of `decodable_basis_predict_shared1000.py`. The command is not saved as a script. The evaluation script shows its settings: one replica (`--gen-rep 1`), strength 0.85
  and mixing 0.6, and the output folders `data/nsdcore_recons_b3/S1K_I3K64_subj0N_s0.85_m0.6`.

## Notes

- The metric files are in `results/low_level_factor/`, one folder per alternative. Arm labels
  (`S1K_I3K64_*`, `v0repro`, `zlowraw`, `zlowscalar`, `bfull`, `d1024refit`, `pixel16`) and the `H0` keys of
  the stored summary stay, because they are also written inside the files or name folders outside
  `results/`.
- The generator settings of the NSD-Imagery comparison of the decodable basis were not logged and were
  most likely strength 0.85 and mixing 0.6 (Section 8.1.5 states the caveat).
- The shell scripts expect the Python files in `src/` under the project root, as in the layout of the
  top-level README.
- Some variable names (`G6_DIR`, `I3_DIR`) keep the label of the work. The main README, "Working labels in
  some names", says what each stands for.
