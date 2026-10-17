# results/

The per-image metric files from which the tables of Chapters 4 to 8 are computed. They are metrics computed on
the reconstructions of the models, not data of the Natural Scenes Dataset or of NSD-Imagery. Most files hold one
row per model, subject, condition, stimulus and sample, with the values of the metrics. No brain response, no
stimulus image and no reconstructed image is in this folder.

## Layout and names

The folders follow `experiments/`: the script that wrote a file can be read from its path. The scripts of
`evaluation/`, `experiments/` and `trisect/` read and write these files under `data/results/`, so
`cp -r results data/` puts every file where the scripts expect it (see the main `README.md`). The two
reproduction pipelines that write metric files (`reproductions/mindeye1/` and `reproductions/mindeye2/`) write
them in `results/baselines/` and `results/brain_correlation/` next to the clones of the models instead.

Some file names keep the label of the run they hold, because the same label is written inside the file (the
`model` column of the TSV) or names a folder of reconstructions outside `results/`. The labels are not words of
the thesis:

| Label | Meaning |
|---|---|
| `S1K` | the `shared1000` block set (blocks of 12 images), with the subject in `subj0N` |
| `T0`, `T1`, `T2`, `T3` | the four oracle arms of Section 8.2 (Table 8.5); `T1nol2` is T1 without the final L2 term on the semantic code; `devT3` is T3 on the development split |
| `h0` | the final scheme of diffusion seeds, one per replica, subject and stimulus (Section 5.5) |
| `p3`, `p4b` | TRISECT (`p3`) and the T2 arm (`p4b`) at strength 0.75 and mixing 0.4, the final settings. The other arms use the setting that won on their selection set: the `*_setting_choice.json` file and the file names in `setting_sweep/` give it |
| `H2G1`, `H4REL` | the per-dimension gain (gamma = 1) and the reliability-weighted ridge of Section 8.1.5 |
| `I3K64` | the decodable basis with k = 64 of Section 8.1.3 |
| `BB`, `BT`, `TB` | the swap arms of Table 8.9 (T is TRISECT, B is Brain-Diffuser; first letter the context, second the low-level code) |
| `v0repro`, `zlowraw`, `zlowscalar` | the three arms of the scalar re-standardization test (Section 8.1.3): the production setting, no re-standardization of `z_low`, one mean and scale for the whole `z_low` branch |
| `bfull`, `d1024refit` | the ridge on the uncompressed `z_low` and the refit of the compressed one (Section 8.1.3) |
| `pixel16` | the 16 × 16 pixel target for the low-level factor (Section 8.1.3) |
| `trisect_b3_h0` | the `model` column of the TRISECT rows on NSD-Imagery |

## Files

| Folder or files | Written by | Thesis |
|---|---|---|
| `baselines/{brain_diffuser,icnn,mindeye_I,mindeye_II,takagi}.tsv` and `..._chance.tsv` | `evaluation/final_evaluations_imagery.py`; the `_chance` files come from the same script with `--roll_gt` | Tables 4.10 to 4.14, 4.16, 7.1, 7.2, 7.4, 7.6 to 7.8 |
| `baselines/shared1000_bridge/` | `evaluation/eval_bridge_mindeye2.py`, `evaluation/eval_bridge_icnn.py` | The `shared1000` bridges of Section 4.4 and Table 4.15 |
| `brain_correlation/*_brain_corr.tsv` | `evaluation/gnet_brain_correlation.py` | The brain columns of Tables 4.10 to 4.16 and 7.1 to 7.9 |
| `brain_correlation/icnn_brain_corr_upsampled_512px.tsv` | the same script, iCNN images upsampled to 512 pixels before GNet, as the benchmark code does | The check of Section 4.5.5. The tables use the native 224 pixels (`icnn_brain_corr.tsv`) |
| `trisect/trisect.tsv`, `trisect/trisect_chance.tsv` | `evaluation/final_evaluations_imagery.py`, run by `trisect/scripts/run_evaluate_imagery.sh` | Chapter 7 |
| `trisect/ci_margin_trisect_vs_*.tsv` | `evaluation/bootstrap_ci_imagery.py` | Table 7.5 |
| `trisect/shared1000/` | `trisect/scripts/run_reconstruct_shared1000.sh` (TRISECT on `shared1000` at strength 0.75 and mixing 0.4) | Table 7.3 |
| `oracle_budget/` | `experiments/oracle_budget/` | Tables 8.4 to 8.7 |
| `structural_factor/` | `experiments/structural_factor/` (the baseline arm is `oracle_budget/T2.tsv`) | Section 8.1.4 |
| `decoder_alternatives/mlp/`, `decoder_alternatives/hybrid/` | `experiments/decoder_alternatives/` | Section 8.1.1, Table 8.2 |
| `uncertainty/` | `experiments/uncertainty/weighted_repetition_fusion.py` | Section 8.1.2 |
| `low_level_factor/` | `experiments/low_level_factor/` | Section 8.1.3 |
| `ridge_input_output/` | `experiments/ridge_input_output/` (`imagery_transfer/` holds the NSD-Imagery runs of the reliability-weighted ridge and of the decodable basis, with their paired intervals in `ci/`) | Section 8.1.5 |
| `shared1000_comparison/` | `experiments/shared1000_comparison/` (comparison with Brain-Diffuser on `shared1000`, generator noise, swap of inputs) | Sections 6.4, 7.1, 8.4, Tables 7.3 and 8.9 |
| `latent_error/` | `experiments/latent_error/` | Section 8.3, Table 8.8 |

`latent_error/latent_prediction_error_ceiling.json` has no script in this repository that writes it under this
name.

## What is not here

Logs, arrays (`.npy`, `.npz`) and smoke tests. Only the metric files that a chapter or a script of this
repository refers to are included.
