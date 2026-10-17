# TRISECT: zero-shot reconstruction of mental images from fMRI

This repository holds the implementation of TRISECT and the rest of the code of the master's thesis
*Zero-Shot Reconstruction of Mental Images from fMRI Using Compact Latent Representations*, by Marco Turchetti
(Sapienza University of Rome, 2026).

The thesis retrains five fMRI-to-image decoding models from scratch and scores them on NSD-Imagery (Kneeland et
al., 2025), a benchmark in which a decoder trained only on seen images is tested on mental images. It then
compares them with TRISECT (Chapter 5), a compact model that predicts three latent factors of the image with
ridge regression and turns them into a picture with a frozen generator.

Every table and figure of Chapters 4 to 8 comes from the scripts of this repository, and the README of each
folder gives the commands in the order in which they run. The data, the model weights and the reconstructed
images are not included: the section "Data" below says where each data set comes from, and the scripts rebuild
the rest. The numbers of the thesis were produced on a machine with two NVIDIA RTX 6000 Ada GPUs of 48 GB each,
which the READMEs and the comments of the code call "the server", and the training and the generation of the six
models took weeks of GPU time.

## Where to start

- From a number or a figure of the thesis: go to "From the thesis to the code" below. Each row names the
  folder and the scripts, and the metric files that hold the numbers are in `results/`.
- From the method: read `trisect/README.md` (Chapter 5), then `evaluation/README.md`, which describes the
  scoring that every model goes through (Chapter 6).
- From one baseline: read `reproductions/<model>/README.md`. It gives the upstream commit, the deviations
  from the original procedure, and the scripts in the order in which they ran.
- From one ablation: read the `README.md` of its folder in `experiments/`. The files are listed in the
  order in which they run.
- From one file: its first lines say what it does, what it reads, what it writes and how to call it.

## What is here

| Folder | Content | Thesis |
|---|---|---|
| `trisect/` | The method of the thesis: from the NSD betas to the reconstructions. | Chapter 5, 7 |
| `evaluation/` | The evaluation code of the benchmark authors, the brain-correlation metric and the confidence intervals. One harness scores all six models. | Chapter 4, 6 |
| `experiments/` | Only the experiments that Chapter 8 reports, one folder per group of alternatives. | Chapter 5, 8 |
| `reproductions/` | The five reproduced models. For each one: the changes to the original code as patches, the files that were added, the frozen environment. | Chapter 4 |
| `figures/` | The three scripts that draw figures of the thesis. | Chapter 4, 7 |
| `results/` | The per-image metric files (TSV, JSON) from which the tables are computed. | Chapter 4 to 8 |

The original code of the five models is not copied. Each `reproductions/<model>/README.md` gives the upstream
repository and the commit, and says how to rebuild the working copy: clone the commit, apply the patches, copy
the added files. The one exception is `trisect/vdvae_lib/`, seven files of VDVAE copied from its repository
(its licence is next to them). The licence of every upstream project is in `THIRD_PARTY_NOTICES.md`.

```
trisect/             README.md  scripts/ (data, targets, adapter, ridge, generation)  trisect/ (shared package)
                     vdvae_lib/  tests/  setup_venv.sh  requirements-lock.txt  download_*.sh
evaluation/          README.md and the evaluation scripts
experiments/         code_width/  decoder_alternatives/  latent_error/  low_level_factor/  oracle_budget/
                     ridge_input_output/  shared1000_comparison/  structural_factor/  uncertainty/
reproductions/       mindeye1/  mindeye2/  brain_diffuser/  takagi/  icnn/
                     each with README.md  patches/  scripts/  LICENSE.upstream  requirements-lock.txt
figures/             make_margin_figures.py  make_shared1000_figure.py  make_imagery_figure.py
results/             baselines/  brain_correlation/  trisect/  and one folder per folder of experiments/
```

## From the thesis to the code

Every table and figure of Chapters 4 to 8 has a row below. "Metric files" are the TSV files in `results/`. The
entry "none" in the Folder column means that no script computes the item (a diagram, a table copied from a paper, or a list of settings).

### Chapter 4: the five reproductions

Benchmark A is the check of each model on the test set of its own paper. Benchmark B is NSD-Imagery, scored with
the common code of `evaluation/`.

| Thesis | What | Folder | Scripts |
|---|---|---|---|
| Table 4.1 | Upstream commit and software stack of the five models | `reproductions/<model>/` | `README.md` of each model; `requirements-lock.txt` |
| Table 4.2 | Declared deviations | `reproductions/<model>/` | The "Deviations" table of each `README.md` |
| Table 4.3 | Compute cost | `reproductions/<model>/` | Read from the logs of the training and generation scripts; `mindeye2/scripts/profile_memory.sh` for the memory sweep of MindEye2 |
| Table 4.4 | A, MindEye1 | `reproductions/mindeye1/` | `patches/`; `scripts/Retrievals_topk.py` (retrieval columns). The metric script is the unchanged upstream one |
| Table 4.5 | A, MindEye2 | `reproductions/mindeye2/` | `scripts/run_training.sh`, `run_inference_eval.sh`; `evaluation/final_evaluations.py` |
| Table 4.6 | A, Brain-Diffuser | `reproductions/brain_diffuser/` | `patches/eval_extract_features.patch` and the unchanged upstream metric script |
| Table 4.7 | A, Takagi et al. | `reproductions/takagi/` | `scripts/run_full_shared1000.sh`; `patches/identification.patch` |
| Table 4.8 | A, iCNN | `reproductions/icnn/` | `scripts/feature-decoding/run_A_layers.sh`, `scripts/recon_icnn_A.py`, `scripts/diag_profile_distribution.py`; the decoding scripts are the unchanged upstream ones, with `patches/` |
| Table 4.9 | A, summary | none | Built from Tables 4.4 to 4.8 |
| Section 4.4 | `shared1000` bridges (the common code on `shared1000`) | `evaluation/` | `eval_bridge_mindeye2.py`, `eval_bridge_icnn.py`; metric files in `results/baselines/shared1000_bridge/` |
| Figure 4.1 | `shared1000` reconstructions of six models | `figures/` | `make_shared1000_figure.py` |
| Table 4.10 | B, MindEye1 | `reproductions/mindeye1/` | `scripts/run_benchmark_B.sh` (and the scripts it calls); `evaluation/` |
| Table 4.11 | B, MindEye2 | `reproductions/mindeye2/` | `scripts/run_imagery_pipeline.sh`; `evaluation/` |
| Table 4.12 | B, Brain-Diffuser | `reproductions/brain_diffuser/` | `scripts/prepare_nsd_imagery.py`, `dump_train_target_stats.py`, `imagery_regression_predict.py`, `vdvae_reconstruct_images_imagery.py`, `versatilediffusion_reconstruct_images_imagery.py`; `evaluation/` |
| Table 4.13 | B, Takagi et al. | `reproductions/takagi/` | `scripts/run_full_imagery.sh`; `evaluation/` |
| Table 4.14 | B, iCNN | `reproductions/icnn/` | `scripts/src/prepare_nsd_imagery_icnn.py`, `decode_features_chunked.py`, `recon_icnn_imagery.py`, `export_icnn_recons.py`; the diagnoses of Section 4.5.5 in the same folder; `evaluation/` |
| Table 4.15 | iCNN, `shared1000` bridge under the stage-2 variants | `reproductions/icnn/` | `scripts/src/decode_features_chunked.py --test shared1000`, `scripts/recon_icnn_A.py`, `scripts/src/decode_features_fastl2lir.py`; `evaluation/eval_bridge_icnn.py`; metric files in `results/baselines/shared1000_bridge/`. The Takagi bridge row: `reproductions/takagi/scripts/codes/utils/eval_extract_features_imagery.py`, `evaluate_reconstruction_imagery.py`, `aggregate_imagery_metrics.py` |
| Table 4.16 | B, summary | `evaluation/` | `aggregate_imagery_metrics.py` on the metric files of the five models |

The five Benchmark B rows (Tables 4.10 to 4.14) share one scoring step. `final_evaluations_imagery.py` writes the
eight image metrics, `gnet_brain_correlation.py` the three brain correlations, and `aggregate_imagery_metrics.py`
merges them. The metric files are `results/baselines/<model>.tsv` and `<model>_chance.tsv` (the
measured chance, from `--roll_gt`) and `results/brain_correlation/<model>_brain_corr.tsv`.

### Chapter 5: TRISECT

| Thesis | What | Folder | Scripts |
|---|---|---|---|
| Figure 5.1 | Overview of the method | none | A diagram |
| Section 5.2 | The three latent factors (targets, PCA bases) | `trisect/scripts/` | `extract_clip_image_raw.py`, `extract_clip_text_raw.py`, `extract_blip2_captions.py`, `extract_dinov2_pooled.py`, `extract_vdvae_latents.py`, `assemble_targets.py`, `fit_pca_zstruct.py`, `fit_pca_zlow.py`; `experiments/code_width/pca_width_zsem.py` for the semantic basis |
| Section 5.3 | The ridge decoder | `trisect/scripts/` | `prepare_betas.py`, `prepare_nsdcore_index.py`, `fit_ridge.py`, `predict_ridge_shared1000.py`, `predict_ridge_imagery.py` |
| Figure 5.2 | The adapter | none | A diagram; the model is `trisect/scripts/adapter_model.py` |
| Section 5.4 | Training the adapter | `trisect/scripts/` | `train_adapter.py`, `extract_vd_context.py` |
| Section 5.5 | Generation | `trisect/scripts/` | `reconstruct.py`, `vd_context.py`, `decode_zlow_to_image.py`, `run_reconstruct_imagery.sh`, `run_reconstruct_shared1000.sh` |
| Table 5.1 | Specifications and hyperparameters | none | Read from the constants of the scripts above; the counts of the baselines are copied from their papers |

### Chapter 6: experimental setup

| Thesis | What | Folder | Scripts |
|---|---|---|---|
| Figure 6.1 | Preparation of the NSD-Imagery betas | none | A diagram; the code is `prepare_nsd_imagery*.py` of each reproduction and `trisect/scripts/prepare_betas.py` |
| Table 6.1 | The data sets and their roles | none | The splits are made by `trisect/scripts/derive_nsdcore_dev_mask.py`, `evaluation/make_shared1000_blockset.py` and `experiments/oracle_budget/make_proxy_set.py` |
| Table 6.2 | The eleven metrics | `evaluation/` | `final_evaluations_imagery.py`, `gnet_brain_correlation.py` |
| Section 6.4 | Confidence intervals and the paired criterion | `evaluation/` | `bootstrap_ci_imagery.py` |

### Chapter 7: results

TRISECT is generated by `trisect/scripts/run_reconstruct_imagery.sh` and scored by `run_evaluate_imagery.sh`, which
call the scripts of `evaluation/`. Its metric files are `results/trisect/trisect.tsv` and
`trisect_chance.tsv`, and `results/brain_correlation/trisect_brain_corr.tsv`.

| Thesis | What | Folder | Scripts |
|---|---|---|---|
| Tables 7.1, 7.4 | Vision and imagery, all stimuli | `evaluation/` | `aggregate_imagery_metrics.py` |
| Table 7.2 | Margins above measured chance | `figures/` | `make_margin_figures.py` recomputes the margins from the metric files and checks them against the table (0.1 pp) |
| Table 7.3 | TRISECT minus Brain-Diffuser, vision | `evaluation/`, `experiments/shared1000_comparison/` | Row 1: `bootstrap_ci_imagery.py` (paired). Row 2: `compare_trisect_vs_bd_shared1000.py`, 65 blocks of `shared1000` |
| Table 7.5 | TRISECT minus each baseline, imagery | `evaluation/` | `bootstrap_ci_imagery.py` (margin mode and paired); metric files `results/trisect/ci_margin_trisect_vs_*.tsv` |
| Tables 7.6, 7.7 | Simple and complex stimuli | `evaluation/` | `aggregate_imagery_metrics.py` (scopes `simple` and `complex`) |
| Table 7.8 | Margin per stimulus type | `evaluation/` | The per-type means of `aggregate_imagery_metrics.py`, run on the score files and on the `_chance` files; the subtraction is not scripted |
| Table 7.9 | Conceptual stimuli, brain correlation | `evaluation/` | `gnet_brain_correlation.py`; metric files `results/brain_correlation/*_brain_corr.tsv`, `trisect_conceptual_brain_corr.tsv` |
| Figure 7.1 | Reconstructions of the imagery trials | `figures/` | `make_imagery_figure.py`, with the options given in `figures/README.md` |

### Chapter 8: ablations

| Thesis | What | Folder | Scripts |
|---|---|---|---|
| Table 8.1 | Every alternative that was run | see the next rows | One row per group of alternatives |
| Table 8.1: MLP decoder, hybrid MLP / ridge | Section 8.1.1 | `experiments/decoder_alternatives/` | `train_mlp_encoder.py`, `predict_mlp.py`, `run_mlp_vs_ridge.sh`, `build_hybrid_predictions.py`, `run_hybrid_vs_ridge.sh` |
| Table 8.2 | MLP minus ridge, paired | `experiments/decoder_alternatives/` | `compare_mlp_vs_ridge.py` |
| Table 8.1: uncertainty gating, weighted fusion | Section 8.1.2 | `experiments/uncertainty/` | `ridge_posterior_sigma2.py`, `sigma2_vs_error_diagnostics.py`, `weighted_repetition_fusion.py` |
| Table 8.3 | Ceiling of a per-trial choice of strength and mixing | `experiments/uncertainty/` | `gating_oracle_ceiling.py` |
| Table 8.1: pixel, uncompressed, scalar, residual, decodable basis | Section 8.1.3 | `experiments/low_level_factor/` | See the table of that folder's `README.md` |
| Table 8.1: structural factor removed, wider codes | Section 8.1.4 | `experiments/structural_factor/`, `experiments/code_width/` | `run_structural_ablation.sh`; `run_semantic_width_sweep.sh`, `run_structural_width_sweep.sh` |
| Table 8.4 | Structural factor removed or shuffled | `experiments/structural_factor/` | `analyze_structural_ablation.py` |
| Table 8.1: averaged repetitions, voxel reliability, restricted ROI, regime calibration, per-dimension gain | Section 8.1.5 | `experiments/ridge_input_output/` | See the table of that folder's `README.md` |
| Table 8.5 | The four oracle arms | `experiments/oracle_budget/` | `make_proxy_set.py`, `run_oracle_t0.sh`, `run_oracle_t1_t2.sh`, `run_oracle_shared1000.sh`, `pick_oracle_constants.py` |
| Tables 8.6, 8.7 | The budget, on `shared1000` and on the proxy set | `experiments/oracle_budget/` | `oracle_budget.py`, `oracle_budget_fixed_constants.py` |
| Table 8.8 | Ridge predictions against the true factors | `experiments/latent_error/` | `latent_prediction_error.py` |
| Table 8.9 | Swap of predicted inputs with Brain-Diffuser | `experiments/shared1000_comparison/` | `bd_predict_shared1000.py`, `run_bd_swap.sh`, `run_bd_swap_eval.sh`, `compare_bd_swap.py` |

The `README.md` of each experiment folder lists its files in the order in which they run, and says what is not
a script (steps that were typed by hand and are written there as commands).

## Data

No data set is stored in this repository. All the data are public: NSD and NSD-Imagery require accepting the
terms of use of the Natural Scenes Dataset first (<https://naturalscenesdataset.org>), and two of the Hugging Face
datasets below are gated, with access given on request on the dataset page. A Hugging Face access token is then
needed by the download scripts.

| Data | Source | Download | Used by |
|---|---|---|---|
| NSD-core betas of the 40 sessions, masked to `nsdgeneral` (`betas_all_subj0N_fp32_renorm.hdf5`), the 73,000 NSD images at 224 px and the trial index | Hugging Face dataset `pscotti/mindeyev2` (gated) | `trisect/download_data_from_internet.sh --betas`, or `reproductions/mindeye2/scripts/download_mindeyev2.py` | TRISECT, `experiments/`, MindEye2, iCNN |
| NSD design file (`nsd_expdesign.mat`) | NSD bucket `s3://natural-scenes-dataset` (public) | `trisect/download_data_from_internet.sh` | TRISECT, iCNN |
| Human COCO captions of the 73,000 images (`COCO_73k_annots_curated.npy`) | Hugging Face dataset `pscotti/naturalscenesdataset` (gated) | `trisect/download_data_from_internet.sh` | TRISECT, Brain-Diffuser |
| NSD-Imagery betas (1.8 mm, betas version 3) and the `nsdgeneral` masks | NSD bucket (public) | `trisect/download_nsd_imagery.sh`, or `reproductions/mindeye1/scripts/download_nsd_imagery.sh` | all six models |
| NSD-Imagery trial table, the 18 stimuli (`imagery_stimuli_18.pt`) and the preprocessed tensors of Kneeland et al. (2025) | Hugging Face dataset `reesekneeland/umn-imagery` (public) | the same two scripts | all six models, `evaluation/` |
| GNet weights (`gnet_multisubject.pt`) and `brain_region_masks.hdf5` | Hugging Face dataset `pscotti/mindeyev2` | `reproductions/mindeye2/scripts/download_mindeyev2.py` | `evaluation/gnet_brain_correlation.py` |
| VDVAE weights (`imagenet64-iter-1600000-model-ema.th`) | <https://openaipublic.blob.core.windows.net/very-deep-vaes-assets/vdvae-assets-2/imagenet64-iter-1600000-model-ema.th> | by hand, into `data/vdvae_weights/` | TRISECT, Brain-Diffuser |
| 200 COCO val2017 images that NSD never shows (the proxy set) | COCO (<https://cocodataset.org>) | by hand, with the procedure of `experiments/oracle_budget/README.md` | `experiments/oracle_budget/`, `experiments/structural_factor/` |
| The data of benchmark A of each reproduction (for example the webdataset shards of MindEye1 and the Deeprecon data of iCNN) | see the README of the model | see the README of the model | the five reproductions |

The pretrained networks (Versatile Diffusion, CLIP, BLIP-2, DINOv2 and the feature networks of the metrics) are
downloaded from Hugging Face or from `torchvision` on first use.

The two download scripts of `trisect/` write into `trisect/data/` by default, and the variables `DEST` and
`DATA_ROOT` change the target folder. They read the Hugging Face token from `HF_thesis_token.txt` at the root of
the repository (`TOKEN_FILE` changes the path); `.gitignore` keeps this file out of any commit. The scripts of
`trisect/`, `evaluation/` and `experiments/` read and write `data/` relative to the folder from which they are
started, which is `work/` in the layout of the next section:

```
DEST=$PWD/work/data bash trisect/download_data_from_internet.sh --betas
DATA_ROOT=$PWD/work/data bash trisect/download_nsd_imagery.sh
```

## Layout of the working tree and `PYTHONPATH`

The scripts keep the imports and the paths of the project in which they were run. Two things follow.

- Imports. Scripts import each other by bare module name (`from vd_context import ...`,
  `from trisect.ridge import ...`). With the folders of this repository, the imports resolve with

  ```
  export PYTHONPATH=trisect:trisect/scripts:trisect/vdvae_lib:trisect/tests:evaluation:\
  experiments/oracle_budget:experiments/latent_error:experiments/uncertainty:experiments/shared1000_comparison:\
  reproductions/icnn/scripts/src:reproductions/mindeye1/scripts
  ```

  Every local import of the repository resolves with this path; the third-party packages come from the
  environments of the section "Environments".
- Shell scripts. The `.sh` files of `trisect/` and `experiments/` do `cd "$(dirname "$0")/.."` and call
  `src/<name>.py`. They were written for a project folder with one flat `src/` and a `data/` next to it. No two
  files of `trisect/scripts/`, `trisect/tests/`, `evaluation/` and `experiments/*/` share a name, and that folder
  can therefore be rebuilt with

  ```
  mkdir -p work/src work/data
  cp trisect/scripts/*.py trisect/scripts/*.sh trisect/tests/*.py evaluation/*.py experiments/*/*.py experiments/*/*.sh work/src/
  cp -r trisect/trisect trisect/vdvae_lib work/src/
  cp -r results work/data/
  ```

  The scripts read and write their metric files under `data/results/`, and the last line puts the metric files
  of this repository in that place.

  The copy does not replace the `PYTHONPATH`. Some scripts import modules that stay in `reproductions/`
  (`fit_ridge.py` and `prepare_betas.py` import `prepare_nsd_imagery_icnn.py` of iCNN), and the `.sh` files
  change directory before they call Python. In `work/` the eleven folders are given as absolute paths,
  set once from the root of the repository:

  ```
  T=$PWD
  export PYTHONPATH=$T/trisect:$T/trisect/scripts:$T/trisect/vdvae_lib:$T/trisect/tests:$T/evaluation:\
  $T/experiments/oracle_budget:$T/experiments/latent_error:$T/experiments/uncertainty:\
  $T/experiments/shared1000_comparison:$T/reproductions/icnn/scripts/src:$T/reproductions/mindeye1/scripts
  ```

The scripts also expect the working copies of the reproductions next to the project folder (for example
`mindeye_II/src/` for the evaluation code that imports `models.py`). Each `reproductions/<model>/README.md`
says how to build its working copy. The few scripts that need the folder of these working copies read it from
the variable `PROJECT_ROOT`, the folder that holds the working copies and the `uv` binary.

## Environments

The five reproductions and TRISECT need incompatible software stacks and therefore each have their own environment.
`reproductions/<model>/requirements-lock.txt` is the list of packages of the environment that produced the
numbers (`uv pip freeze`), and `setup_venv.sh` builds it. The first line of each lock file says what to know
about it (for example, which packages were left over from an earlier installation).
`trisect/` has its own lock, `trisect/requirements-lock.txt`, taken from `.venv_targets`, the environment that
`trisect/setup_venv.sh` builds.

## Working labels in some names

Some variable names and some names of data files and folders carry the internal label of a run (`G6_DIR`,
`h4_all_results.json`, `d2_3_recons`). These labels do not appear in the thesis. The table below says what each
one stands for, and `results/README.md` explains the labels in the names of the metric files.

| Name | Where | What it stands for |
|---|---|---|
| `G6`, `G6_DIR`, `--g6-dir`, `G6_DEVICE` | `shared1000_comparison/` (`run_bd_shared1000_blocks.sh`, `compare_bd_swap.py`, `compare_trisect_vs_bd_shared1000.py`), `low_level_factor/compare_decodable_basis_vs_pca.py` | The folder of the comparison with Brain-Diffuser on the `shared1000` blocks (Section 8.4), `data/results/shared1000_comparison`. In `compare_decodable_basis_vs_pca.py` it is the root of all the results (`data/results`), because that script reads the block index of the comparison. `G6_DEVICE` is the device of the run (`cuda` by default) |
| `I2`, `I2_DIR`, `--i2-dir` | `shared1000_comparison/compare_bd_swap.py` | The folder `swap/` of the swap with Brain-Diffuser: the arms BT, TB and BB (Table 8.9) |
| `I3`, `I3_DIR`, `--i3-dir` | `low_level_factor/compare_decodable_basis_vs_pca.py` | The folder `low_level_factor/decodable_basis/`: the decodable low-level basis (Section 8.1.3) |
| `D2_3`, `D2_3_DIR`, `--d2-3-dir`, `d2_3_recons` | `decoder_alternatives/` | The MLP decoder trained on the four subjects together, its metric folder (`results/decoder_alternatives/mlp/`) and its folder of reconstructions (Section 8.1.1) |
| `G4_0`, `G4_0_DIR`, `--g4-0-dir`, `g4_0_recons`, `g4_0_hybrid_*`, `g4_0_ctrl_*` | `decoder_alternatives/` | The hybrid decoder, with its control that swaps the roles (Section 8.1.1): its metric folder (`results/decoder_alternatives/hybrid/`), reconstructions and predictions |
| `F6_DELTA` | `shared1000_comparison/compare_trisect_vs_bd_shared1000.py` | TRISECT minus Brain-Diffuser on NSD-Imagery in AlexNet(2) and AlexNet(5), in percentage points, as in Tables 7.3 (vision) and 7.5 (imagery). It is the reference of the ordering test of Section 8.4 |
| `H0_FLOOR_JSON`, `h0_floor` | `ridge_input_output/compare_dimension_gain.py`, `compare_voxel_reliability_weighting.py` | `generation_noise_floor.json`, written by `shared1000_comparison/generation_noise.py`: the generator noise of Section 6.4, measured as the difference between two generations of the same TRISECT arm. The two comparisons require the difference of their arm to exceed it, and write it under the key `h0_floor` |
| `h2_e2e` | `ridge_input_output/compare_dimension_gain.py` | Name of the end-to-end comparison of the per-dimension gain with the uniform ridge (Section 8.1.5) |
| `h4_e2e`, `betas_trisect/h4`, `h4_all_results.json` | `ridge_input_output/` (`voxel_reliability_and_roi.py`, `compare_voxel_reliability_weighting.py`, `run_voxel_reliability_and_roi.sh`), `trisect/scripts/predict_ridge_*.py` | The reliability-weighted ridge and the ridge restricted to one `streams` ROI (Section 8.1.5): the end-to-end comparison, the folder of weights and predictions, and the file of latent-space scores |
| `h3_correction` | `ridge_input_output/` (`regime_calibration_fit.py`, `regime_calibration_precheck.py`) | The folder `data/betas_trisect/h3_correction/` with the three regime-calibration corrections (Section 8.1.5) |
| `D1_ALPHAS`, `d1_power`, `d2_feature_screen`, `d3_shift`, `f1_diagnostics.json`, `f1_d3_imagery.json` | `uncertainty/sigma2_vs_error_diagnostics.py` | The diagnostics of the correlation check between the ridge variance and the error (Section 8.1.2): the power analysis against a null of well-specified data (`D1_ALPHAS` is its grid of ridge penalties), the screen of candidate per-trial signals, the shift of leverage between training and `shared1000`, and the two JSON files that hold them |
| `G0`, `G1`, `G0_SUBJ`, `G1_SUBJ` | `reproductions/mindeye2/scripts/run_imagery_pipeline.sh`, `run_inference_eval.sh`, `reproductions/takagi/scripts/run_full_shared1000.sh` | Not a task label: the lists of jobs sent to GPU 0 and to GPU 1 |
| `B1` | `evaluation/final_evaluations_imagery.py` | Not a task label: EfficientNet-B1 |

In the scripts of `reproductions/`, "Benchmark A" and "Benchmark B" in a message mean the two evaluations that
each reproduction README defines: A is the metric of the original paper on `shared1000`, B is NSD-Imagery.

## Citation

```bibtex
@mastersthesis{turchetti2026trisect,
  author = {Marco Turchetti},
  title  = {Zero-Shot Reconstruction of Mental Images from fMRI Using Compact Latent Representations},
  school = {Sapienza University of Rome},
  year   = {2026}
}
```

## Licence

The code of this repository is released under the MIT licence (`LICENSE`). The code of the five models and of
the other projects keeps its own licence, listed in `THIRD_PARTY_NOTICES.md`. `results/` holds metrics computed
on reconstructions: it holds no data of the Natural Scenes Dataset (see `results/README.md`).
