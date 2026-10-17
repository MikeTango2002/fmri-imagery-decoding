# reproductions/takagi/

Retraining of Takagi and Nishimoto (2023) from scratch and its evaluation on NSD-Imagery. The benchmark
evaluates the variant "+Decoded Text" of the technical report (the `text` method of the code), which is
trained together with the CVPR method that benchmark A checks.
This folder holds only what was added or changed. The upstream code is not copied.

| | |
|---|---|
| Upstream | <https://github.com/yu-takagi/StableDiffusionReconstruction> (MIT, © 2023 Yu Takagi; `LICENSE.upstream`) |
| Commit | `e187d4b` ("Update README.md", the head of `main`) |
| Thesis | Section 3.2.2 (the method), 4.2 (sessions), 4.3.4 (procedure), 4.4.4 (benchmark A), 4.5.4 (benchmark B), Appendix A |

## Layout

```
patches/                 the nine upstream files that were changed, as git patches, and the licence texts of the code they touch
scripts/                 the files that were added; the folders mirror the working copy (scripts/codes/utils/ goes to codes/utils/)
requirements-lock.txt    the frozen environment of the main venv, the one that produced the thesis numbers
LICENSE.upstream         the licence of the upstream repository
```

## How to rebuild the working copy

```bash
git clone https://github.com/yu-takagi/StableDiffusionReconstruction.git takagi
cd takagi && git checkout e187d4b
for p in <this folder>/patches/*.patch; do git apply "$p"; done   # the patches use the paths of the upstream repository
cp -r <this folder>/scripts/. .                                    # the added files go to the places they have in the working copy
```

Every patch applies to a fresh clone of `e187d4b` with `git apply --check`. All scripts run from the root of the working copy (the shell scripts) or
from the folder named in their first lines (`codes/utils/`, `codes/caption/BLIP/`, `codes/diffusion_sd1/`): the
upstream scripts read `../../nsd`, `../../mrifeat` and so on relative to that folder. The shell scripts find
the root from their own location, and the Python scripts from `__file__` or from the relative paths above.

## Licences of the patches

All the patched files are MIT, except one:

- `patches/ddpm_sd1.patch` changes one line of `ldm/models/diffusion/ddpm.py` of CompVis Stable Diffusion v1,
  which is released under CreativeML Open RAIL-M, not MIT. The patch contains no weights and no model, only the
  changed line and its context. The licence text is next to it (`patches/LICENSE.stable-diffusion-v1.RAIL-M`); the
  use restrictions of that licence apply to the code the patch touches.
- `patches/ddpm_sd2.patch`: Stable Diffusion v2, MIT, © 2022 Stability AI (`patches/LICENSE.stable-diffusion-v2.MIT`).
- `patches/blip.patch`, `decode_captions.patch`, `img2feat_blip.patch`: BLIP, BSD-3-Clause, © 2022 Salesforce
  (`patches/LICENSE.BLIP.BSD-3-Clause`).
- The other four patches (`ridge`, `identification`, `diffusion_decoding`, `img2feat_decoded`) are files of the
  upstream repository itself (MIT, `LICENSE.upstream`).

## Stack

- Main environment (`scripts/setup_venv.sh`, built with `uv`; there is no conda on the server): Python 3.10.13,
  PyTorch 2.0.1 built for CUDA 11.8, not 1.11.0. The version of the authors has no kernels for the GPUs of the
  server (RTX 6000 Ada, sm_89); it is the change the hardware forces. PyTorch Lightning goes from 1.4.2 to 1.9.5 (1.4.2
  does not run on PyTorch 2), and NumPy is pinned to 1.26.4, because PyTorch 2.0.1 is built against the NumPy 1.x
  interface. The repository itself pins no versions (its `requirements.txt` lists eight unpinned packages). The frozen
  list is `requirements-lock.txt`.
- BLIP environment (`scripts/setup_venv_blip.sh`): Python 3.9 with `transformers==4.15.0`, the version BLIP pins;
  the main environment uses 4.19.2; BLIP cannot share it. It extracts the BLIP features, fits their ridge and decodes
  the captions. There is no lock for this environment: its versions are the ones pinned in the script.
- `OPENBLAS_CORETYPE=SkylakeX`: the shell scripts export it. The processor of the server (Xeon Gold 6448Y, Sapphire
  Rapids) is newer than some OpenBLAS builds, which then compute wrong matrix products without any warning. The
  NumPy 1.26.4 of the main venv bundles OpenBLAS 0.3.23, which already recognizes the processor; the variable is a
  precaution here. It is not needed on other machines.
- Fixes of the environment. `setup_venv.sh` includes the fixes that the environment of the thesis needed after
  its first build: NumPy pinned, a `.pth` file that makes `ldm` importable, `taming-transformers-rom1504`, the
  CLIP weights fetched with `wget` and the cache variables in `activate`. The lock lists both taming packages, a
  trace of that first build. No script installs `scikit-image==0.19.3`, which has to be installed by hand, and
  `run_full_imagery.sh` stops if another version is found.
- GPUs: one job per GPU (`CUDA_VISIBLE_DEVICES=0` and `1`), the scripts split the four subjects over the two.

## Patches

| Patch | What it changes | Why |
|---|---|---|
| `ridge.patch` | Adds `--save_weights`: the scaler statistics and the ridge coefficients are written as an `.npz`, after a check that they reproduce the predictions of the fitted model (largest difference about $10^{-6}$; a warning, not an abort, if it fails). Adds progress logging (one line per step, a heartbeat while `himalaya` fits). | Upstream saves only the predictions for the 982 test images and loses the fitted model when the process ends; NSD-Imagery could therefore not be decoded without a refit. Alphas, pipeline, solver and data are untouched. |
| `diffusion_decoding.patch` | Adds `--imgidx_end`, `--skip_existing` and `--n_iter` (default 5, the upstream value). Many images are generated in one process, loading Stable Diffusion once, and the random state is restored before each image. | The restore gives the same noise as an upstream run with one image per process (the first call of each process reaches the same state). `--n_iter 10` is what benchmark B needs; the first five samples equal those of a run with 5. |
| `identification.patch` | Adds `--nimage` (default 982, as upstream) and `--out`: the accuracy is also written to a CSV, one row per subject, method, feature and number of images. | Upstream prints the accuracy of benchmark A and stores nothing. The numbers are unchanged. |
| `img2feat_decoded.patch` | Adds `import torchvision.models.feature_extraction`. | `import torchvision` alone does not load it. |
| `decode_captions.patch`, `img2feat_blip.patch`, `blip.patch` | The BLIP checkpoint and the `bert-base-uncased` tokenizer are cached inside the repository; the paths of the BLIP scripts no longer depend on the folder they are started from (upstream used `../../util/`, a folder that does not exist: it is called `utils`). | The caches stay inside the repository folder (nothing may be written outside it on the shared server), and a script started from another folder died with a misleading import error. |
| `ddpm_sd1.patch`, `ddpm_sd2.patch` | `pytorch_lightning.utilities.distributed` → `pytorch_lightning.utilities.rank_zero`. | The import moved in the Lightning that PyTorch 2 needs. The second copy belongs to the depth variant, which is not run; it is kept because the file was changed. |

None of the patches touches a regression, a feature or a generation setting: the recipe is the one of the authors.
Generation uses the defaults of the code in both benchmarks: 50 DDIM steps, strength 0.8, guidance scale 5.0, seed 42.

## Scripts

| File | What it does | Benchmark |
|---|---|---|
| `setup_venv.sh` | Builds the main virtual environment (Stable Diffusion v1.4, `ldm`, `himalaya`) and fetches the CLIP weights. | A, B |
| `setup_venv_blip.sh` | Builds the BLIP environment. | A, B |
| `run_full_shared1000.sh` | Benchmark A on the 982 test images and four subjects: generation (`cvpr` and `text`, 5 samples), identification accuracy with the seven features in parallel, and the shared1000 bridge to the common harness. `--n-iter` changes the number of samples. | A |
| `codes/utils/prepare_nsd_imagery_takagi.py` | NSD-Imagery betas → `streams` atlas, one file per region, in the voxel order of the fit → divide by 300 → z-score per run → average over repetitions. Eight inline checks stop the run on the first failure. | B (Section 4.2) |
| `codes/utils/decode_imagery_features.py` | Applies the ridge weights of benchmark A to the prepared betas: `X @ coef + intercept`, with no scaler (see Deviations). | B |
| `codes/caption/BLIP/decode_captions_imagery.py` | Brain captions for one NSD-Imagery condition (`decode_captions.py` with three changes: score source, six stimuli, output name). Runs in the BLIP environment. | B |
| `codes/diffusion_sd1/diffusion_decoding_imagery.py` | Reconstruction of one condition with the `text` method (`diffusion_decoding.py` with four changes). `--n_iter 10` gives the ten samples of the protocol. | B |
| `run_full_imagery.sh` | Benchmark B as one restartable pipeline, six stages: betas, features, captions, reconstruction, metrics, aggregation. | B |
| `regen_B_images.sh` | Regenerates the 1,200 reconstructions of benchmark B (betas, features, generation) with the same code, weights and seed. | B |
| `codes/utils/eval_extract_features_imagery.py` | The six feature extractors of the common harness (Inception, CLIP, AlexNet layers 2 and 5, EffNet-B, SwAV) on the 12-image pool of one modality, or `--gt` for the ground truth, or `--bridge` for the 100-image shared1000 pilot. A port of the Brain-Diffuser extractor to the paths of this model. | A (bridge), B |
| `codes/utils/evaluate_reconstruction_imagery.py` | Per-image metrics (PixCorr, SSIM, the two-way comparisons and the three distances) from those features; one TSV per subject, modality and sample. Refuses to run if the BLAS of the host returns wrong products. | A (bridge), B |
| `codes/utils/aggregate_imagery_metrics.py` | Per stimulus, then per stimulus type, then the mean over the ten samples and over the four subjects. Warns loudly if a cell has fewer than ten samples. `--bridge` aggregates the shared1000 pilot. | A (bridge), B |
| `codes/utils/diag_readonly_checks.py` | Five checks that read what is on disk and write nothing: whether the ground truth of generation and evaluation agree, the preparation checks, whether the twelve reconstructions differ from each other, whether the two modalities share files, the captions. | B |

## Data and checkpoints

No script downloads them. The commands below fetch them, from the root of the working copy and with the main
environment active (it has `awscli`). The NSD terms must be accepted first. Only the files that the code opens are
fetched; syncing all of `nsddata` and `nsddata_stimuli` would pull more than 100 GB that nothing reads.

```bash
# NSD-core: design, stimulus table, streams atlas and colour table, responses, the .hdf5 betas of 37 sessions, the images
mkdir -p nsd && cd nsd
B=s3://natural-scenes-dataset
aws s3 cp --no-sign-request $B/nsddata/experiments/nsd/nsd_expdesign.mat           ./nsddata/experiments/nsd/nsd_expdesign.mat
aws s3 cp --no-sign-request $B/nsddata/experiments/nsd/nsd_stim_info_merged.csv    ./nsddata/experiments/nsd/nsd_stim_info_merged.csv
aws s3 cp --no-sign-request $B/nsddata/freesurfer/fsaverage/label/streams.mgz.ctab ./nsddata/freesurfer/fsaverage/label/streams.mgz.ctab
for S in subj01 subj02 subj05 subj07; do
  aws s3 cp --no-sign-request $B/nsddata/ppdata/$S/behav/responses.tsv           ./nsddata/ppdata/$S/behav/responses.tsv
  aws s3 cp --no-sign-request $B/nsddata/ppdata/$S/func1pt8mm/roi/streams.nii.gz ./nsddata/ppdata/$S/func1pt8mm/roi/streams.nii.gz
  aws s3 sync --no-sign-request $B/nsddata_betas/ppdata/$S/func1pt8mm/betas_fithrf_GLMdenoise_RR \
      ./nsddata_betas/ppdata/$S/func1pt8mm/betas_fithrf_GLMdenoise_RR --exclude "*" --include "betas_session*.hdf5"
done
aws s3 cp --no-sign-request $B/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5 ./nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5
# NSD-Imagery: the betas (1.8 mm, betas version 3) of each subject
for S in subj01 subj02 subj05 subj07; do
  aws s3 cp --no-sign-request $B/nsddata_betas/ppdata/$S/func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR/betas_nsdimagery.nii.gz \
      ./nsddata_betas/ppdata/$S/func1pt8mm/nsdimagerybetas_fithrf_GLMdenoise_RR/betas_nsdimagery.nii.gz
done
cd ..
# NSD-Imagery: the stimulus table and the 18 stimulus images (Hugging Face dataset of Kneeland et al.)
uv pip install huggingface_hub        # not installed by setup_venv.sh
python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ("nsddata_stimuli/stimuli/nsdimagery_stimuli.pkl3", "nsddata_stimuli/stimuli/imagery_stimuli_18.pt"):
    print(hf_hub_download("reesekneeland/umn-imagery", f, repo_type="dataset", local_dir="nsd"))
PY
# Stable Diffusion v1.4 (needs a Hugging Face token)
mkdir -p codes/diffusion_sd1/stable-diffusion/models/ldm/stable-diffusion-v1
wget --header="Authorization: Bearer <HF_TOKEN>" -O codes/diffusion_sd1/stable-diffusion/models/ldm/stable-diffusion-v1/sd-v1-4.ckpt \
    https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt
```

The COCO annotations are downloaded by `nsd_access` on first use, the BLIP checkpoint by the patched BLIP scripts,
and the CLIP ViT-L/14 weights by `setup_venv.sh`.

## Commands

All commands run with the main environment active unless marked, and `OPENBLAS_CORETYPE=SkylakeX` exported.

Benchmark A (the paper's own metric on `shared1000`; it only checks that the training worked). Upstream scripts
that are not named in the patch table are used unchanged:

```bash
cd codes/utils
for S in subj01 subj02 subj05 subj07; do python make_subjmri.py --subject $S; done   # 37 sessions
python img2feat_sd.py --imgidx 0 73000 --gpu 0                    # writes nsdfeat/init_latent/ and nsdfeat/c/
for S in subj01 subj02 subj05 subj07; do
  for F in init_latent c; do for U in each ave; do python make_subjstim.py --featname $F --use_stim $U --subject $S; done; done
  python ridge.py --target init_latent --roi early   --subject $S --save_weights    # patched
  python ridge.py --target c           --roi ventral --subject $S
done
# BLIP environment (source .venv_blip/bin/activate):
(cd ../caption/BLIP && python img2feat_blip.py --gpu 0)                             # BLIP features of the 73,000 images
for S in subj01 subj02 subj05 subj07; do
  python make_subjstim.py --featname blip --use_stim ave  --subject $S
  python make_subjstim.py --featname blip --use_stim each --subject $S
  python ridge.py --target blip --roi early ventral midventral midlateral lateral parietal --subject $S --save_weights
  (cd ../caption/BLIP && python decode_captions.py --subject $S)
done
# main environment again, from the root of the working copy:
bash run_full_shared1000.sh        # generation, identification accuracy (benchmark A), shared1000 bridge
```

Benchmark B (NSD-Imagery, the thesis deliverable). The decoders are not retrained: B uses the weight files of
benchmark A (`decoded/<subj>/*_weights_*.npz`).

```bash
# main environment, from the root of the working copy; the pipeline runs six stages (the reconstructions are done at stage 4;
# stages 5 and 6 score them with the scripts of note 1, which are in this folder):
bash run_full_imagery.sh                 # betas, features, captions (this stage switches to .venv_blip by itself), reconstructions
# scoring: the common harness of evaluation/, which reads the layout of the reconstructions directly
python <evaluation>/final_evaluations_imagery.py --model takagi --recon_root decoded/image-text-imagery \
    --gt nsd/nsddata_stimuli/stimuli/imagery_stimuli_18.pt --out takagi_imagery.tsv
```

The reconstructions land in `decoded/image-text-imagery/<subj>/<condition>/samples/{stimulus:05}_{sample:03}.png`:
six stimuli and ten samples per condition. Stages 5 and 6 of `run_full_imagery.sh` call the metric scripts of note 1; the numbers of the thesis come from the
common harness, not from them.

## Deviations from the original procedure

| Deviation | Where it matters |
|---|---|
| PyTorch 2.0.1 instead of 1.11.0, PyTorch Lightning 1.9.5, Python 3.10, a second environment for BLIP (see Stack). | A, B |
| The 37-session split of the released `make_subjmri.py` (`range(1, 38)`), 24,980 single trials per subject, not the 40 sessions of the benchmark specification. | A, B (Section 4.2) |
| The regions follow the code, not the description in the benchmark paper: the latent is predicted from `early` (5,917 voxels for subject 1), the BLIP features from six regions together (`early`, `ventral`, `midventral`, `midlateral`, `lateral`, `parietal`; 26,688 voxels), the CLIP text embedding of the CVPR method from `ventral`. | A, B |
| The training standardization is not applied to NSD-Imagery. The training betas are in integer units (median scale 367 to 569 per subject); the prepared NSD-Imagery betas already have unit variance after the per-run z-score. Applying the scaler on top would divide them again by several hundred and collapse the predictions towards zero, with no error. The model has no rescaling of its predictions. | B (Section 4.3.4) |
| Ten samples per stimulus (`--n_iter 10`); the code of the authors draws 5. Each sample draws new noise in one loop that starts from the fixed seed; the first five equal those of a 5-sample run. Benchmark A keeps 5. | B |
| Many images per process, with the random state restored before each one (same noise as upstream's one image per process). | A, B |

## Notes

1. The three metric scripts of `codes/utils/` are here, and they are not the source of the NSD-Imagery numbers.
   `eval_extract_features_imagery.py`, `evaluate_reconstruction_imagery.py` and `aggregate_imagery_metrics.py` are called by phases 3 and 4 of
   `run_full_shared1000.sh` (the shared1000 bridge), by stages 5 and 6 of `run_full_imagery.sh`, by the last part of
   `regen_B_images.sh`, and imported by `diag_readonly_checks.py`. They are a port of the Brain-Diffuser metric code to the
   layout of this model: the earlier port of the common harness with which the bridge row of Takagi was scored (Table 4.15). The NSD-Imagery numbers of the thesis do not come
   from them (they come from `evaluation/`), but the Takagi row of the shared1000 bridge does. `regen_B_images.sh` also uses
   `eval_extract_features_imagery.py` for its identity check.
2. What else is not here. The environments of the GAN and depth variants of the upstream repository,
   which are not evaluated, and the scripts that fetch data (see Data and checkpoints).
3. Upstream modules. The scripts use files that arrive with the clone of `e187d4b`: `make_subjmri.py`, `make_subjstim.py`,
   `img2feat_sd.py`, `nsd_access/`, the `ldm` package of Stable Diffusion v1, `models/blip.py` and the three patched
   files `diffusion_decoding.py`, `img2feat_decoded.py` and `identification.py`. `diag_readonly_checks.py` also looks for
   `make_subjmri.py` in the current folder.
4. Benchmark B images were regenerated. The 1,200 images were deleted after the metrics were computed and rebuilt with
   `regen_B_images.sh` for the qualitative figure. The inputs are identical, the images are close copies and not bit
   identical (the median correlation with the original images is between 0.981 and 0.998). The metrics were not recomputed on them.
5. `ponte` in the names of some log and result files (`ponte_gpu0.log`, `ponte_shared1000_summary.md`) is Italian for "bridge".
   The names were kept because they are names of files that the runs wrote.
