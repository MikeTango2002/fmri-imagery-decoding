# reproductions/mindeye2/

Retraining of MindEye2 (Scotti et al., ICML 2024) from scratch and its evaluation on NSD-Imagery.
This folder holds only what was added or changed. The upstream code is not copied.

| | |
|---|---|
| Upstream | <https://github.com/MedARC-AI/MindEyeV2> (MIT, © 2023 MedARC; `LICENSE.upstream`) |
| Commit | `c184718` ("Fix: open-clip-torch version (#41)", the head of `main`) |
| Thesis | Section 4.3.2 (procedure), 4.4.2 (benchmark A), 4.5.2 (benchmark B), Appendix A |

## Layout

```
patches/                 two patches: models.py, and Train.py as converted from the notebook
scripts/                 the files that were added (tests/ and one accelerate config included)
LICENSE.upstream         the licence of the upstream repository (the patches carry lines of its code)
requirements-lock.txt    the frozen environment that produced the thesis numbers
```

## How to rebuild the working copy

The upstream repository ships only Jupyter notebooks. The training and the inference run Python
scripts that were converted from them with `jupyter nbconvert` (version 7.17.1), which copies the code
cells as they are. The converted `.py` files are therefore not stored here: they are rebuilt.
Run `nbconvert` in an environment without IPython (`pip install nbconvert==7.17.1` alone). With IPython
installed, `nbconvert` rewrites the `%` commands and adds blank lines between cells, and `Train.patch` does not apply.

```bash
git clone https://github.com/MedARC-AI/MindEyeV2.git mindeye_II
cd mindeye_II && git checkout c184718
(cd src && for nb in Train recon_inference enhanced_recon_inference final_evaluations; do
    jupyter nbconvert --to script "$nb.ipynb"; done)                  # 1. notebooks -> scripts
git apply <this folder>/patches/models.patch <this folder>/patches/Train.patch   # 2. from the repo root
cp -r <this folder>/scripts/* src/                                    # 3. the added files
mv src/accelerate_2gpu_zero2_gas4.yaml .                              #    the config sits in the repo root
cp <repository>/evaluation/{final_evaluations_imagery,gnet_brain_correlation,aggregate_imagery_metrics}.py src/
(cd src && bash apply_upstream_patches.sh)                            # 4. fixes of the converted scripts
```

Both patches apply to a fresh clone of `c184718` after step 1 with `git apply --check`. After step 4,
`Train.py`, `recon_inference.py`, `enhanced_recon_inference.py` and `final_evaluations.py` hold the code that
produced the numbers of the thesis.

Without the data, step 4 ends with `SOME PATCHES MISSING` and exit code 1. The only `MISS` lines are fixes
(c) and (d): they link `brain_region_masks.hdf5` and the `evals/*.pt` files from the dataset folder
(`mindeyev2_dataset/`), and a clone without data does not have it. The other fixes and the six marked
features report `ok`.

Why the patch of `Train.py` is made against the converted file and not against `Train.ipynb`: of the six
changes to the training script, only one was ever copied into the notebook, and the notebook was never
run. The five others exist only in the `.py`; a patch of the notebook would not give the code that ran.

All scripts run from `src/`. They read and write `../train_logs/`, `../evals_imagery_smoke.png` and, for
benchmark B, `../../results/baselines/` and `../../results/brain_correlation/` (in the folder that holds the
repositories).
`run_imagery_pipeline.sh` also reads the ground-truth stimuli and the preprocessed betas from the
working copy of MindEye1, which must sit next to this one (`../../mindeye_I/dataset`).

## Stack

- Python 3.11.15, one virtual environment for this model only (`scripts/setup_venv.sh`, built with `uv`).
- `torch==2.1.0` (the default PyPI wheel, built for CUDA 12.1, which already supports the Ada GPUs),
  `torchvision==0.16.0`, `diffusers==0.23.0`, `xformers==0.0.22.post7`, `deepspeed==0.13.1`,
  `accelerate==0.24.1`, `transformers==4.37.2`: the pins of the authors. No pin had to change.
- The frozen package list is `requirements-lock.txt`. It also lists CUDA 13 packages: they are left over
  from a transient upgrade to torch 2.13 (installing `dalle2-pytorch` alone upgrades torch), which was
  undone by reinstalling torch 2.1.0 before the first training run. The torch that ran is 2.1.0.
- GPUs: two NVIDIA RTX 6000 Ada (sm_89, 48 GB), used together by each training run (DeepSpeed ZeRO stage
  2, optimizer state on the GPUs). The paper used eight A100 GPUs.

## Patches

| Patch | What it changes | Why |
|---|---|---|
| `models.patch` | Optional gradient checkpointing of the mixer blocks, of the low-level head and of the six transformer layers of the diffusion prior. Off by default: with the flag off the code path is the upstream one. | The prior keeps three copies of its attention matrices per layer, and at batch 63 the training does not fit in 48 GB without it. The gradients are the same (`scripts/tests/test_grad_checkpointing.py`). |
| `Train.patch` | Against the output of `nbconvert`. Six optional features, each marked in the code and each a no-op at its default: `--frozen_cache` (reads the outputs of the three frozen encoders from disk), `--grad_accum_steps` (with the epoch length and the learning-rate schedule counted in updates), `--ckpt_slim` (checkpoint without optimizer state, written to a temporary file and renamed), `--pixcorr_stride`, `--grad_checkpointing` (wires the flag into `models.py`), `--profile_memory`. Also closes the figure of the low-level reconstructions after saving it. | Fit the full configuration of the paper on two GPUs with the same batch, number of updates and schedule. None of them touches the model, the losses or the hyperparameters. |

`scripts/apply_upstream_patches.sh` makes the converted scripts run outside a notebook and on the server,
and checks that the marked sites of the patches are present (a new `nbconvert` run removes them):

| Fix | What it changes |
|---|---|
| (0) | Comments out the IPython commands that `nbconvert` copies (`%load_ext autoreload`). |
| (a) | Defines `plotting` in non-interactive runs. |
| (b) | Points the weights of the SDXL refiner to the local dataset instead of the cluster of the authors. |
| (c), (d) | Link the mask file and the three ground-truth `.pt` files so that they are found from `src/`. |
| (f) | `Accelerator(dispatch_batches=False)` in `Train.py`. Without it the two processes block each other at the first backward pass; with it each GPU reads its own data stream, which is what the data loader was built for. |

## Scripts

| File | What it does | Benchmark |
|---|---|---|
| `setup_venv.sh` | Builds the virtual environment with `uv` (no conda on the server). | A, B |
| `apply_upstream_patches.sh` | The fixes above, plus a check that the six marked features are in `Train.py` and `models.py`. `--check` only verifies. | A, B |
| `download_mindeyev2.py` | Downloads the HDF5 data and webdataset shards of MindEye2 from HuggingFace (all 40 sessions per subject, already masked to `nsdgeneral`). | A |
| `precompute_frozen.py`, `frozen_cache.py` | Compute once, and serve, the outputs of the three frozen encoders (ViT-bigG, VAE, ConvNeXt) for the 73,000 stimuli. With no image augmentation they are the same at every epoch. | A |
| `check_cache_fidelity.py` | Compares the cache with a fresh recomputation, against the noise floor of two recomputations at different batch sizes. | A |
| `tests/test_grad_checkpointing.py` | Gradient equivalence with and without checkpointing, on the real modules at reduced size: relative difference under 1e-4, and exactly 0 for the control row. | A |
| `accelerate_2gpu_zero2_gas4.yaml` | DeepSpeed ZeRO-2 on 2 GPUs with four accumulation steps: 2 × 63 × 4 = 504, the optimizer batch of the paper's eight GPUs. | A |
| `run_training.sh` | Stage 1 (pretraining on the other seven subjects) and stage 2 (fine-tuning), subject by subject. Has a preflight and skips a stage that is already done. | A |
| `profile_memory.sh` | Runs `Train.py --profile_memory` at several batch sizes to measure memory and time per sample. See note 1. | A |
| `run_inference_eval.sh` | Reconstructions (`recon_inference.py`), SDXL refinement (`enhanced_recon_inference.py`) and the metrics of the authors (`final_evaluations.py`), two subjects at a time on the two GPUs. | A |
| `download_nsd_imagery.sh` | Fetches the NSD-Imagery betas, masks, trial metadata and the 18 stimuli, reusing the files of the MindEye1 working copy when they are there. | B |
| `prepare_nsd_imagery.py` | Mask → divide by 300 → z-score per run → nest over repetitions. Writes the two combined conditions (`x_visionall`, `x_imageryall`) and the ground truth, and stops if the result differs from the tensors of Kneeland et al. or if the voxel count differs from the training data. | B (Section 4.2) |
| `recon_inference_imagery.py` | `recon_inference.py` ported to NSD-Imagery with five confined changes (arguments, data loader, number of repetitions, seeding, output names). | B |
| `enhanced_recon_inference_imagery.py` | The same for the SDXL refiner. Saves the blend of 0.75 refined and 0.25 low-level image. | B |
| `run_imagery_pipeline.sh` | The whole chain of benchmark B in six steps, with an automated PixCorr gate after a smoke test. | B |

## Commands

All commands run from `src/` of the working copy, with the virtual environment active.

Benchmark A (the paper's own metric on `shared1000`; it only checks that the training worked):

```bash
python download_mindeyev2.py --data_path=mindeyev2_dataset
python precompute_frozen.py --data_path=mindeyev2_dataset --cache_dir=mindeyev2_dataset \
  --out=mindeyev2_dataset/frozen_cache.hdf5 --parts=clip,vae,cnx
python check_cache_fidelity.py --data_path=mindeyev2_dataset --cache_dir=mindeyev2_dataset --parts=clip,vae,cnx --n=128
python tests/test_grad_checkpointing.py
setsid nohup bash run_training.sh --prune-pretrain > ../train_logs/logs/run_training.log 2>&1 < /dev/null &
nohup bash run_inference_eval.sh > ../train_logs/logs/run_inference_eval.log 2>&1 &
```

`run_training.sh` launches, for each subject, the two commands

```bash
accelerate launch --config_file=../accelerate_2gpu_zero2_gas4.yaml Train.py \
  --model_name=multisubject_subj0N_4096hid --multi_subject --subj=N --batch_size=63 --grad_checkpointing=prior <common>
accelerate launch --config_file=../accelerate_2gpu_zero2_gas4.yaml Train.py \
  --model_name=finetuned_subj0N_40sess_4096hid --no-multi_subject --subj=N --batch_size=24 --grad_checkpointing=prior \
  --multisubject_ckpt=../train_logs/multisubject_subj0N_4096hid <common>
```

with `<common>` = `--max_lr=3e-4 --mixup_pct=.33 --num_epochs=150 --use_prior --prior_scale=30 --clip_scale=1
--blurry_recon --blur_scale=.5 --no-use_image_aug --n_blocks=4 --hidden_dim=4096 --num_sessions=40
--frozen_cache=<cache> --grad_accum_steps=4 --pixcorr_stride=10 --ckpt_slim`. The flags of the paper are
those of the upstream README FAQ; the last three are additions of this reproduction and do not change the recipe. The set of the four
subjects is 1, 2, 5, 7, and a training run takes both GPUs.

Benchmark B (NSD-Imagery, the thesis deliverable):

```bash
bash download_nsd_imagery.sh
nohup bash run_imagery_pipeline.sh > ../train_logs/logs/imagery_pipeline.log 2>&1 &
```

`run_imagery_pipeline.sh` runs `prepare_nsd_imagery.py`, a smoke test on the first subject (vision, one
sample) with the PixCorr gate, the full matrix (subjects × vision/imagery × 10 samples, split in
(subject, mode) pairs across the two GPUs), the image metrics (`evaluation/final_evaluations_imagery.py`),
the GNet brain correlation (`evaluation/gnet_brain_correlation.py`) and the aggregation
(`evaluation/aggregate_imagery_metrics.py`).

## Deviations from the original procedure

| Deviation | Where it matters |
|---|---|
| Two GPUs instead of eight. The per-GPU batch is the one of the paper (63 in stage 1, 24 in stage 2) and four steps are accumulated before each update; the optimizer batch (504, 192) and the number of updates (8,850, 23,400) are therefore the paper's. The model has no batch normalization, and only the order of floating-point sums changes. | A, B |
| Stage-1 batch of 63, as in the paper, not the 42 of the repository command for the released checkpoints. | A, B |
| Gradient checkpointing of the prior, outputs of the frozen encoders read from disk, diagnostic computed every tenth step, checkpoints without optimizer state. None changes the model or the gradients. | A, B |
| No CPU offload of the optimizer, which the authors used: it needs a CUDA compiler that the server does not have, and the optimizer state stays on the GPUs, partitioned by ZeRO-2. | A, B |
| `dispatch_batches=False` (fix (f) above). | A, B |
| On NSD-Imagery the repetitions are averaged with their real number, 8 for vision and 16 for imagery, where the upstream code fixes three; the ten samples are ten complete runs of the generation chain with new seeds and no selection. | B |
| The refined and the low-level image are blended (0.75 / 0.25) before saving, because the upstream evaluation script blends before every metric and the common harness does not. | B |

The 40 sessions of the benchmark specification are used, as in the paper: MindEye2 reads the HDF5 files
of its authors and has no counterpart of the 37-session split of the other models.

## Notes

1. What is not here. Other configuration and helper files of the working copy are not included. One
   script of this folder reaches two of them: `profile_memory.sh` calls `fit_memory_scaling.py` for the
   final regression (the sweep itself does not need it) and takes `accelerate_2gpu_zero2.yaml` as its default
   configuration; it must therefore be called with `--acc-cfg ../accelerate_2gpu_zero2_gas4.yaml
   --grad-accum 4`.
2. `recon_inference.py` and `enhanced_recon_inference.py` have no file here. Converted from the
   notebooks and fixed by `apply_upstream_patches.sh`, they are the scripts of the upstream repository, and only
   the `_imagery` ports were written for this reproduction. `final_evaluations.py` is produced the same way; a copy of it is also in
   `evaluation/`.
