# reproductions/brain_diffuser/

Retraining of Brain-Diffuser (Ozcelik and VanRullen, 2023) from scratch and its evaluation on NSD-Imagery.
This folder holds only what was added or changed. The upstream code is not copied.

| | |
|---|---|
| Upstream | <https://github.com/ozcelikfu/brain-diffuser> (MIT, © 2023 Furkan Ozcelik; `LICENSE.upstream`) |
| Commit | `31d52c0` ("Merge pull request #4 from ozcelikfu/evaluation", the head of `main`) |
| Thesis | Section 4.3.3 (procedure), 4.4.3 (benchmark A), 4.5.3 (benchmark B), Appendix A |

## Layout

```
patches/                 the four upstream files that were changed, as git patches
scripts/                 the files that were added (diagnostics/ included)
LICENSE.upstream         the licence of the upstream repository (the patches carry lines of its code)
requirements-lock.txt    the frozen environment that produced the thesis numbers
```

## How to rebuild the working copy

```bash
git clone https://github.com/ozcelikfu/brain-diffuser.git brain_diffuser
cd brain_diffuser && git checkout 31d52c0
for p in <this folder>/patches/*.patch; do git apply "$p"; done   # the patches use the paths of the upstream repository
cp <this folder>/scripts/setup_venv.sh .                           # the environment script sits in the repository root
cp <this folder>/scripts/*.py scripts/                             # the scripts run from the repository root
cp -r <this folder>/scripts/diagnostics .
```

Every patch applies to a fresh clone of `31d52c0` with `git apply --check`. All scripts run from the root of the working copy:
they read and write `data/` and `results/` and several of them stop if the `scripts/` folder is not
in the current directory.

## Stack

- Python 3.10.11 (the authors used 3.8.13), one virtual environment for this model only
  (`scripts/setup_venv.sh`, built with `uv`; there is no conda on the server).
- PyTorch 2.0.1 built for CUDA 11.8, not 1.12.1. The build of the authors has no kernels for the
  GPUs of the server (RTX 6000 Ada, sm_89). It is the only change that the hardware forces. NumPy
  1.23.4, SciPy 1.9.3 and scikit-learn 1.1.3, which run the ridge regressions, keep the versions of
  the original `environment.yml`. Three other pins had to be raised, because they have no wheel for
  Python 3.10 (`opencv-python`, `matplotlib`) or no longer build (`pyyaml`); the script explains each
  one. The frozen package list is `requirements-lock.txt`.
- `OPENBLAS_CORETYPE=SkylakeX` must be exported before every run that fits or applies a ridge.
  The processor of the server (Xeon Gold 6448Y, Sapphire Rapids) is newer than the OpenBLAS inside NumPy
  and SciPy, which does not recognise it and computes `A @ A.T` wrongly, without any warning. With
  the wrong kernel benchmark A came out at chance (two-way scores of 50.6–51.4%); with the variable it
  reproduces the paper. `scripts/diagnostics/blas_selfcheck.py` checks the matrix products in a few
  seconds (exit code 1 if they are wrong) and runs before every regression. The variable is not
  needed on other machines.
- GPUs: one 48 GB GPU generates all the images (the authors used two 12 GB GPUs); two jobs run in
  parallel, one per GPU.

## Patches

| Patch | What it changes | Why |
|---|---|---|
| `vdvae_regression.patch` | Adds `import pickle`. | The last line of the script calls `pickle.dump` and the import is missing: the fit finished, the predictions were written, then the script stopped with a `NameError` and the fitted decoder was lost. Benchmark B applies exactly these weights. |
| `versatilediffusion_reconstruct_images.patch` | `cuda(1)` → `cuda(0)` in the places that put tensors and the diffusion model on a device. | The code assumes two 12 GB GPUs; the server has one 48 GB GPU per job. |
| `eval_extract_features.patch` | `device = 1` → `device = 0`. | Same reason. |
| `download_nsddata.patch` | Adds `--no-sign-request` to every `aws s3 cp`; copies `nsdgeneral.nii.gz` by name instead of a wildcard. | The NSD bucket is public and needs no credentials. `aws s3 cp` does not expand a wildcard, and the original line therefore failed and left the `roi/` folder empty. Only `nsdgeneral` is read. |

None of the patches touches the regressions (except the import) or the generative models: the recipe is the
one of the authors.

## Scripts

| File | What it does | Benchmark |
|---|---|---|
| `setup_venv.sh` | Builds the virtual environment with `uv`. | A, B |
| `diagnostics/blas_selfcheck.py` | Refuses to continue if this interpreter computes matrix products or ridge solutions wrongly. | A, B |
| `prepare_nsd_imagery.py` | NSD-Imagery betas → `nsdgeneral` mask → divide by 300 → z-score per run → average over repetitions; writes the ground-truth PNGs. Compares the result with the preprocessed tensors of Kneeland et al. and stops if they differ. | B (Section 4.2) |
| `dump_train_target_stats.py` | Writes `data/regression_weights/subj0N/train_target_stats.npz` for the four subjects: the mean and standard deviation of the training betas and of the three training targets (note 2). Run once, after benchmark A and before any intermediate file is deleted. | B |
| `imagery_regression_predict.py` | Applies the three NSD-core ridge decoders to the NSD-Imagery rows and re-standardizes the predictions. | B (Section 4.3.3) |
| `vdvae_reconstruct_images_imagery.py` | First stage: VDVAE images of one condition and one seed, with a fixed seed. | B |
| `versatilediffusion_reconstruct_images_imagery.py` | Second stage: Versatile Diffusion image-to-image from the first stage and the CLIP predictions, with the same seed. | B |

The two reconstruction scripts are copies of the upstream scripts with a few confined changes: the input is
the prediction of one NSD-Imagery condition, the number of images comes from the input, the output path
carries condition and seed, and the random generator is seeded (see Deviations). The VDVAE script builds
its reference latent from a dummy batch, because it only uses the shapes of the layers.

## Commands

All commands run from the root of the working copy, with the virtual environment active and
`OPENBLAS_CORETYPE=SkylakeX` exported.

Benchmark A (the paper's own metric on `shared1000`; it only checks that the training worked). The
steps are the ones of the upstream README; the scripts that are not named here are used unchanged:

```bash
python diagnostics/blas_selfcheck.py && echo ok
(cd data && python download_nsddata.py)             # patched; accept the NSD terms on the NSD site first
# data/annots/COCO_73k_annots_curated.npy: download it from the Hugging Face dataset pscotti/naturalscenesdataset
(cd data && python prepare_nsddata.py -sub 1)       # and 2, 5, 7; these two scripts use paths relative to data/
python scripts/vdvae_extract_features.py -sub 1     # first stage
python scripts/vdvae_regression.py -sub 1           # patched (import pickle)
python scripts/vdvae_reconstruct_images.py -sub 1
python scripts/cliptext_extract_features.py -sub 1  # second stage
python scripts/clipvision_extract_features.py -sub 1
python scripts/cliptext_regression.py -sub 1
python scripts/clipvision_regression.py -sub 1
python scripts/versatilediffusion_reconstruct_images.py -sub 1   # patched (one GPU)
python scripts/save_test_images.py                  # metrics
python scripts/eval_extract_features.py -sub 0      # patched (one GPU)
python scripts/eval_extract_features.py -sub 1
python scripts/evaluate_reconstruction.py -sub 1
```

The steps repeat for subjects 2, 5 and 7. Before the second stage, the pretrained VDVAE and Versatile
Diffusion files go where the upstream README says. Training uses the 37 sessions that the released
`download_nsddata.py` fetches (Section 4.2).

Benchmark B (NSD-Imagery, the thesis deliverable). The decoders are not retrained: B uses the twelve
weight files of benchmark A (three per subject) and the moments of the training targets (see note 2).

```bash
# 0. once, after benchmark A and before anything is deleted: the moments of the training targets (note 2)
python scripts/dump_train_target_stats.py
# 1. data: the files of NSD-Imagery (see note 3), then
python scripts/prepare_nsd_imagery.py --sub 1 2 5 7 --data_root data --reference_dir data/reference
# 2. predictions of the three decoders, one subject at a time (each weight file is about 11 GB)
for S in 1 2 5 7; do
  python scripts/imagery_regression_predict.py --sub $S \
    --conditions visionsimple visioncomplex imagerysimple imagerycomplex imageryconcepts \
    --restandardize_pool modality12
done
# 3. reconstructions: 5 conditions x 10 seeds x 4 subjects, both stages for each seed.
#    Subjects 1 and 2 on GPU 0; the same loop with S in "5 7" and CUDA_VISIBLE_DEVICES=1 on GPU 1.
for S in 1 2; do
  for COND in visioncomplex visionsimple imagerycomplex imagerysimple imageryconcepts; do
    for SEED in 0 1 2 3 4 5 6 7 8 9; do
      CUDA_VISIBLE_DEVICES=0 python scripts/vdvae_reconstruct_images_imagery.py \
        --sub $S --condition $COND --seed $SEED -bs 6
      CUDA_VISIBLE_DEVICES=0 python scripts/versatilediffusion_reconstruct_images_imagery.py \
        --sub $S --condition $COND --seed $SEED -diff_str 0.75 -mix_str 0.4
    done
  done
done
```

The reconstructions land in `results/versatile_diffusion_imagery/subj0N/<condition>_seed<S>/<i>.png` of the
working copy (not the `results/` of this repository, which holds only metric files; six images per condition). They are scored by the common harness of `evaluation/`:
`final_evaluations_imagery.py --model brain_diffuser --recon_root results/versatile_diffusion_imagery`, and the same
`--model brain_diffuser` for `gnet_brain_correlation.py`.

## Deviations from the original procedure

| Deviation | Where it matters |
|---|---|
| PyTorch 2.0.1 instead of 1.12.1, Python 3.10.11 instead of 3.8.13, three raised pins (see Stack). | A, B |
| The 37-session split of the released download script, not the 40 sessions of the benchmark specification. | A, B (Section 4.2) |
| One 48 GB GPU for the generation, instead of two 12 GB GPUs. | A, B |
| The `import pickle` fix (without it the weights that B applies are never saved). | A, B |
| The input z-score with the training statistics is not applied to NSD-Imagery. The per-run z-score already puts every voxel on the unit scale on which the ridges were fitted; a second z-score would rescale each voxel by a different factor, with no error. | B (Section 4.3.3) |
| The predictions are re-standardized, as in the original code, but with the moments of a pool of 12 stimuli (the two image-based conditions of one task; the six conceptual stimuli are a pool of their own), not of the 982 `shared1000` predictions. Without it the strongly regularized predictions are too small for the generator. | B (Section 4.3.3) |
| 10 samples per stimulus come from 10 seeds, each one fixing both noise sources (the VDVAE decoder and the initial noise of the image-to-image step). The upstream scripts never seed them. | B |
| Defaults of the code for the generation: 50 DDIM steps, strength 0.75, mixing 0.4, one sample per call (`n_samples = 1`). | A, B |

Neither paper documents the two normalizations, and the public code of the benchmark authors has no
Brain-Diffuser part; they are choices of this reproduction, and no alternative was tested.

## Notes

1. What is not here. The metric scripts that score NSD-Imagery with Brain-Diffuser's own feature
   extractor (`eval_extract_features_imagery.py`, `evaluate_reconstruction_imagery.py`; the numbers of the
   thesis come from the common harness) and the upstream scripts for the ROI analysis (`roi_*.py`), which
   are not run. Of the diagnostics only `blas_selfcheck.py` is shipped.
2. The training-target moments. `imagery_regression_predict.py` reads
   `data/regression_weights/subj0N/train_target_stats.npz`, a file of about 5 MB per subject written by
   `scripts/dump_train_target_stats.py`, which runs once, after benchmark A and before any intermediate file
   is deleted. It holds the
   mean and standard deviation of the training arrays that the ridges predict or are fitted on:
   `fmri_mean`/`fmri_std` (the training betas after `/300`, `ddof=1`; informative only, B does not use
   them), `vdvae_mean`/`vdvae_std` (the `train_latents` of `nsd_vdvae_features_31l.npz`),
   `cliptext_mean`/`cliptext_std` and `clipvision_mean`/`clipvision_std` (the `nsd_cliptext_train.npy` and
   `nsd_clipvision_train.npy` features). They come from the 37-session training arrays; they can only be
   rebuilt while `data/processed_data/` and `data/extracted_features/` still exist. Without this file the
   weights predict on a wrong scale and the reconstructions fall to chance without any error.
3. No download script for NSD-Imagery. The files come from two sources: `aws s3 cp --no-sign-request` for
   `betas_nsdimagery.nii.gz` of each subject (the `nsdimagerybetas_fithrf_GLMdenoise_RR` folder, 1.8 mm, betas
   version 3), and `huggingface_hub.hf_hub_download` from the dataset `reesekneeland/umn-imagery` for
   `nsdimagery_stimuli.pkl3`, `imagery_stimuli_18.pt` and, as the reference of `prepare_nsd_imagery.py`,
   `preprocessed_data/subject{N}/nsd_imagery{,_unnormalized}.pt`. `reproductions/mindeye1/scripts/download_nsd_imagery.sh`
   downloads the same files in the layout that `prepare_nsd_imagery.py` expects, once its `DATA_ROOT`
   is changed from `../dataset` to `data`. `nsdgeneral.nii.gz` of each subject must also be in place
   (`data/download_nsddata.py` fetches it).
4. The upstream modules. The two reconstruction scripts import `hps`, `utils`, `data`, `vae`, `train_helpers`,
   `image_utils`, `model_utils` (from `vdvae/`) and `lib.*` (from `versatile_diffusion/`). They arrive with the
   clone of `31d52c0`; `sys.path` points to those folders.
