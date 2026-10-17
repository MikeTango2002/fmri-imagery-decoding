# trisect/

The method of the thesis, from the NSD betas to the reconstructions (Chapter 5, with the data
preparation of Chapter 6). The folder holds everything that produces TRISECT's numbers. The scripts
that test alternatives to it are in `experiments/`, and the scoring code is in `evaluation/`.

How the folders are put on the Python path, and how the `.sh` files find `src/`, is described in the
top-level README. Every script reads and writes under a `data/` folder of the working directory. The data are not
in this repository (see "What is not here").

## Files

### Scripts: data and targets (`scripts/`)

| File | What it does | Thesis section |
|---|---|---|
| `extract_clip_image_raw.py` | CLIP ViT-L/14 image embeddings of the 73,000 NSD images, before the PCA. | 5.2 |
| `extract_clip_text_raw.py` | CLIP ViT-L/14 text embeddings of one caption per image (`--source human` or `blip2`; the final model uses `blip2`). | 5.2 |
| `extract_blip2_captions.py` | BLIP-2 captions of the 73,000 images, the source of the text half of `z_sem`. | 5.2 |
| `extract_dinov2_pooled.py` | DINOv2 ViT-L/14 patch tokens, pooled to an 8 x 8 grid, before the structural PCA. | 5.2 |
| `extract_vdvae_latents.py` | VDVAE latents of the 73,000 images, the raw material of `z_low`. | 5.2 |
| `fit_pca_zstruct.py` | Fits the structural PCA (1024 channels to 32 per position). | 5.2 |
| `fit_pca_zlow.py` | Fits the PCA of `z_low`, after a standardization of each VDVAE layer. | 5.2 |
| `assemble_targets.py` | Puts the finished targets (`z_sem`, `z_struct`) into one HDF5 file. | 5.2 |
| `derive_shared1000_mask.py` | Writes the 73k-space mask of `shared1000`, so that every PCA fit leaves these images out. | 5.2, 6.1 (Table 6.1) |
| `derive_nsdcore_dev_mask.py` | Writes the development split: 1,000 of each subject's 9,000 training images. | 6.1 (Table 6.1) |
| `derive_nsd73k_coco_ids.py` | Writes the COCO id of each of the 73,000 NSD images (`nsd73k_coco_ids.npy`). It is used to check that the 150 COCO photographs of the proxy set were never in NSD. | 6.1 (the proxy set) |
| `extract_imagery_targets.py` | The frozen targets (`z_sem`, `z_struct`, `z_low`) of the 18 NSD-Imagery stimuli, with `--n-draws` draws of the VDVAE posterior. They are only for the comparison of predictions and targets. | 8.3 |

### Scripts: ridge decoder (`scripts/`)

| File | What it does | Thesis section |
|---|---|---|
| `prepare_nsdcore_index.py` | Trial-to-image index of NSD-core, rebuilt from the NSD design file and checked against the one of MindEye2. | 5.3, 6.1 |
| `prepare_betas.py` | Prepares the NSD-core and NSD-Imagery betas and the reliability of each trial. | 5.3, 6.1 (Figure 6.1) |
| `fit_ridge.py` | Closed-form ridge, one per branch and per subject, trained on NSD-core only. | 5.3 |
| `predict_ridge_shared1000.py` | Applies the weights to the `shared1000` trials and saves the predictions. | 5.3 |
| `predict_ridge_imagery.py` | Applies the weights to the NSD-Imagery betas and re-standardizes the predictions. | 5.3 |

### Scripts: adapter and generation (`scripts/`)

| File | What it does | Thesis section |
|---|---|---|
| `adapter_model.py` | The adapter: from (`z_sem`, `z_struct`) to the context of Versatile Diffusion. | 5.4 (Figure 5.2) |
| `extract_vd_context.py` | The true Versatile Diffusion context of a subset of images, the target of the adapter. | 5.4 |
| `train_adapter.py` | Trains the adapter by regression on that context. No brain data. | 5.4 |
| `vd_context.py` | Runs the dual-guided pipeline of Versatile Diffusion from context tensors and an initial image. | 5.5 |
| `decode_zlow_to_image.py` | Decodes `z_low` to a 64 x 64 image (inverse PCA, VDVAE decoder). | 5.5 |
| `reconstruct.py` | The reconstruction step: `--gen-rep` images per stimulus from the predicted factors (ten in the full run). | 5.5 |
| `run_reconstruct_imagery.sh` | Reconstructs the NSD-Imagery arm of the final model (Chapter 7). | 5.5, 7 |
| `run_reconstruct_shared1000.sh` | Reconstructs `shared1000` at the generator setting of Brain-Diffuser, for the paired comparison. | 5.5, 6.4 |
| `run_evaluate_imagery.sh` | Scores the NSD-Imagery reconstructions with the scripts of `evaluation/`. | 7 |

A new run of `reconstruct.py` does not give the same images as the run behind the thesis. The diffusion
noise is fixed by `diffusion_seed()`, but the VDVAE decoder samples its output image with the global random
state of PyTorch, which the script does not seed. On NSD-Imagery a rerun moved the two-way metrics by at most
1.6 pp, below the noise floor of Section 6.4.

### Package (`trisect/`)

Functions that were copied into several scripts. They live here so that a normalization recipe
cannot change in one copy only. The evaluation code does not use this package.

| File | What it does | Thesis section |
|---|---|---|
| `__init__.py` | Declares the package and says what belongs in it. | 5 |
| `nsd.py` | The normalization of the NSD-core betas: division by 300, then voxelwise z-score with the moments of the fitting split (three call shapes of one recipe). | 5.3 |
| `pca.py` | Loads the frozen PCA bases and gives the layer layout of the VDVAE latent. | 5.2 |
| `targets.py` | Helpers for the target scripts (`extract_*` and the comparison scripts of `experiments/`). | 5.2 |
| `ridge.py` | The re-standardization of the ridge predictions on the moments of the training targets. | 5.3 |
| `metrics.py` | The quick pixel correlation used inside the scripts. The numbers of the thesis come from `evaluation/`. | 5.5 |
| `regime_calibration.py` | The three corrections of the ridge input of the regime calibration (gain, gain per direction, full covariance). Used by `experiments/ridge_input_output/regime_calibration_*.py`. | 8.1.5 |

### Tests (`tests/`)

Three smoke tests of the Versatile Diffusion code of `scripts/vd_context.py`. They need a GPU and
write PNG files for a visual check. No table of the thesis reports their output; they are the
checks behind the generation code of Section 5.5. Run them in this order, with the same
`--image-index` and `--seed`.

| File | What it checks | Thesis section |
|---|---|---|
| `smoke_vd_pipeline.py` | Test 1 of 3. The unmodified dual-guided pipeline reconstructs a real NSD image from one image and one caption. | 5.5 |
| `smoke_vd_context_roundtrip.py` | Test 2 of 3. The manual denoising loop of `vd_context.py`, given the true context, gives the same image as test 1. | 5.5 |
| `smoke_vd_init_image.py` | Test 3 of 3. The `init_image` and `strength` path behaves like img2img: the correlation with the input image grows as `strength` goes down. | 5.5 |

### Download and setup (root of this folder)

| File | What it does | Thesis section |
|---|---|---|
| `download_data_from_internet.sh` | Downloads the NSD-core files that TRISECT needs (design file, images, trial index, captions; `--betas` adds the betas). Needs a Hugging Face token in `HF_thesis_token.txt` at the root of the repository. | 6.1 |
| `download_nsd_imagery.sh` | Downloads the NSD-Imagery betas, the stimuli and the reference tensors of the benchmark authors, for subjects 1, 2, 5 and 7. | 6.1 |
| `setup_venv.sh` | Builds the Python environment. `requirements-lock.txt` is the frozen list of the environment that ran. | 6.2 |

### Other folders

`vdvae_lib/` holds seven files of OpenAI's VDVAE, copied from its repository (see its own
README and licence).

## Data

The two download scripts of this folder fetch the raw files: `download_data_from_internet.sh` the
NSD-core files (with `--betas` also the betas of the 40 sessions) and `download_nsd_imagery.sh` the
NSD-Imagery files. The section "Data" of the top-level README gives the source of each file and the
access that the gated Hugging Face datasets need. The VDVAE weights are the one file that no script
downloads: `extract_vdvae_latents.py` reads them from
`data/vdvae_weights/imagenet64-iter-1600000-model-ema.th` and, when the file is missing, stops with the
`curl` command that fetches it.

## What is not here

- The data: NSD betas, NSD-Imagery betas, masks, targets, PCA bases, ridge weights, adapter
  weights and reconstructions. The two download scripts fetch the raw files, and the other
  scripts rebuild the rest.
- The Versatile Diffusion, BLIP-2, CLIP and DINOv2 weights. They are downloaded from their
  own hosts on first use.
- The numbers. The metric files of TRISECT are in `results/`.
