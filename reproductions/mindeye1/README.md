# reproductions/mindeye1/

Retraining of MindEye1 (Scotti et al., NeurIPS 2023) from scratch and its evaluation on NSD-Imagery.
This folder holds only what was added or changed. The upstream code is not copied.

| | |
|---|---|
| Upstream | <https://github.com/MedARC-AI/fMRI-reconstruction-NSD> (MIT, © MedARC; `LICENSE.upstream`) |
| Commit | `6b30ea5` ("Update README.md citations", the head of `main`) |
| Thesis | Section 4.3.1 (procedure), 4.4.1 (benchmark A), 4.5.1 (benchmark B), Appendix A |

## Layout

```
patches/                 the four upstream files that were changed, as git patches
scripts/                 the files that were added, and an empty cache/ (download_nsd.py writes there and does not create it)
LICENSE.upstream         the licence of the upstream repository (the patches carry lines of its code)
requirements-lock.txt    the frozen environment that produced the thesis numbers
```

## How to rebuild the working copy

```bash
git clone https://github.com/MedARC-AI/fMRI-reconstruction-NSD.git mindeye_I
cd mindeye_I && git checkout 6b30ea5
for p in <this folder>/patches/*.patch; do git apply "$p"; done   # patches use the paths src/<file>.py
cp -r <this folder>/scripts/* src/                                  # all scripts run from src/; also creates src/cache/
mv src/run_benchmark_B.sh .                                         # except this one: it runs as `bash ../run_benchmark_B.sh`
```

The scripts keep the layout of the upstream README: they read and write `../train_logs/`, `../dataset/`
and `cache/` relative to `src/`. Every patch applies to a fresh clone of `6b30ea5` with `git apply --check`.

## Stack

- Python 3.10.8, one virtual environment for this model only (`scripts/setup_venv.sh`, built with `uv`).
- PyTorch 2.13.0 with CUDA 13 wheels, not 2.0.1. `setup_venv.sh` asks for `torch==2.0.1`, but the
  environment on the server holds 2.13.0 (installed before the first training checkpoint), and this build
  produced every MindEye1 number of the thesis. No run with 2.0.1 was made to measure the effect of the
  change. The frozen package list is `requirements-lock.txt`.
- Two packages of the lock install `clip/clip.py`: `clip-anytorch` and OpenAI CLIP (`clip @ git+...`). The first
  imports `pkg_resources`, which the locked `setuptools==83.0.0` no longer has. If `clip-anytorch` is installed last,
  `import clip` fails and six scripts stop. `setup_venv.sh` installs OpenAI CLIP after the other packages; an environment built from
  the lock must do the same.
- `diffusers==0.14.0` (the upstream README pins 0.13.0), `transformers==4.27.4`, `huggingface-hub==0.13.4`,
  `numpy==1.26.4`.
- GPUs: NVIDIA RTX 6000 Ada (sm_89). One process per GPU and per subject; DDP is not used because
  `Train_MindEye.py` multiplies the number of epochs by the number of processes.

## Patches

| Patch | What it changes | Why |
|---|---|---|
| `Reconstructions.patch` | Writes `all_images_subj0N.pt` instead of `all_images.pt`. | The four subjects would overwrite each other's ground truth. |
| `Retrievals.patch` | Removes the hard-coded `subj = 1` that overrides `--subj`; shards the test set (`LAION_START`, `LAION_STOP`); tags the output names with the shard. | Branch of the LAION-5B retrieval, not used for any number. |
| `train_autoencoder.patch` | Renames the keys of the attention weights of the pretrained autoencoder (`to_q` → `query`, ...); points the data paths to the local `cache/`; creates the cache folder; removes the block that resumes from a checkpoint. | The checkpoint was saved with a newer `diffusers`; paths were absolute paths of the authors' cluster. |
| `utils.patch` | The data loader also looks for shards already in the local cache; the LAION candidate download is parallel and has a timeout. | Local data; the second change is in the unused LAION branch. |

`Train_MindEye.py` and `models.py` are not changed: the training recipe is the one of the authors.

## Scripts

| File | What it does | Benchmark |
|---|---|---|
| `setup_venv.sh` | Builds the virtual environment with `uv` (no conda on the server). | A, B |
| `download_nsd.py` | Downloads the webdataset shards of NSD-core released by the authors (subjects 1, 2, 5, 7). | A |
| `Retrievals_topk.py` | Top-1 image and brain retrieval on `shared1000`: the header of `Retrievals.py` plus its retrieval block, without the LAION part. Upstream calls `sys.exit(0)` before that block when run from the command line; the notebook was the only way to reach it. | A (Table 4.4) |
| `download_nsd_imagery.sh` | Downloads the NSD-Imagery betas, the `nsdgeneral` masks, the trial metadata, the 18 stimuli and the authors' preprocessed tensors. | B |
| `prepare_nsd_imagery.py` | Mask → divide by 300 → z-score per run → nest over repetitions; writes the ground truth; compares with the tensors of Kneeland et al. and stops if they differ. | B (Section 4.2) |
| `Reconstructions_imagery.py` | `Reconstructions.py` with four confined changes: arguments, data loader, seeding after argument parsing, output names. Everything else is the configuration the benchmark authors evaluated. | B |
| `Reconstructions_imagery_batch.py` | The same computation for the whole matrix (subjects × conditions × seeds) in one process, loading the models once. | B |
| `run_benchmark_B.sh` | The whole chain of benchmark B in five stages (see below). | B |
| `export_recons.py` | Writes the reconstruction tensors as PNG grids next to the ground truth. | figures only |

## Commands

All commands run from `src/` of the working copy, with the virtual environment active.

Benchmark A (the paper's own metric on `shared1000`; it only checks that the training worked):

```bash
python download_nsd.py --data_path=cache --subj 1 2 5 7
python Train_MindEye.py --model_name=prior_257_final_subj01_bimixco_softclip_byol --data_path=cache/ \
  --subj=1 --hidden --norm_embs --clip_variant=ViT-L/14 --batch_size=32 --num_epochs=240 \
  --max_lr=3e-4 --mixup_pct=.33 --use_image_aug --n_samples_save=0 --ckpt_interval=5
python train_autoencoder.py --model_name=autoencoder_subj01_4x_locont_no_reconst --subj_id=01
python Reconstructions.py --model_name=prior_257_final_subj01_bimixco_softclip_byol \
  --autoencoder_name=models/autoencoder_subj01_4x_locont_no_reconst --data_path=cache/ \
  --vd_cache_dir=../train_logs/models/versatile-diffusion --subj=1 --img2img_strength=.85 --recons_per_sample=16
python Reconstruction_Metrics.py --recon_path=<recons>.pt --all_images_path=all_images_subj01.pt   # upstream, unchanged
python Retrievals_topk.py --model_name=prior_257_final_subj01_bimixco_softclip_byol --data_path=cache/ --subj=1
```

The training commands repeat for subjects 2, 5 and 7 (`--subj=2`, ...). Training uses the 37-session split of
the authors' shards (Section 4.2).

Benchmark B (NSD-Imagery, the thesis deliverable):

```bash
bash download_nsd_imagery.sh
python prepare_nsd_imagery.py --subj 1 2 5 7 --data_root ../dataset --out_dir ../dataset/preprocessed
export PROJECT_ROOT=<folder that holds the repositories and .uv/>
nohup bash ../run_benchmark_B.sh > ../train_logs/logs/rerun_B.log 2>&1 &
```

`run_benchmark_B.sh` has a preflight (checkpoints, betas, free memory on each GPU) and five stages:
(1) 200 reconstruction tensors, 5 conditions × 10 seeds × 4 subjects, with one process per GPU;
(2) an optional comparison with a run made on betas z-scored with `ddof=0`;
(3) per-image metrics; (4) aggregation; (5) brain correlation with GNet. Stages 2 to 5 start only if stage 1
wrote all 200 tensors. The `GPUS` and `RESUME` variables are explained at the top of the script.

## Deviations from the original procedure

| Deviation | Where it matters |
|---|---|
| PyTorch 2.13.0 instead of 2.0.1 (see Stack). | A, B |
| The 37-session split of the released shards, not the 40 sessions of the benchmark specification. | A, B (Section 4.2) |
| The retrieval branch on LAION-5B is not run: the search service has been offline since December 2023. | A only |
| Top-1 retrieval columns of benchmark A come from `Retrievals_topk.py`, the retrieval block of `Retrievals.py` as a script. | A (Table 4.4) |
| On NSD-Imagery each reconstruction is the best of 16 candidates, as in the paper; MindEye1 therefore generates 160 images per stimulus against 10 for the other models. With one candidate every metric moves less than the noise floor of the benchmark (Section 4.3.1). | B |
| Betas are z-scored with `ddof=1`, as in the tensors of the benchmark authors. | B |
| The four patched upstream files (above). | A |

## Notes

What is not here. The per-model metric scripts `Reconstruction_Metrics_imagery.py` and
`Reconstruction_Metrics_imagery_batch.py`, and the script that aggregates their per-image TSVs
(`aggregate_imagery_metrics.py` of the MindEye1 working copy). Stages 3 and 4 of `run_benchmark_B.sh`
call them by name; those stages cannot run from this folder. The NSD-Imagery numbers of the thesis do
not come from them: all models are scored by the common harness in `evaluation/`. Stage 5 runs
`gnet_brain_correlation.py` from the `src/` folder of the MindEye2 repository; in this repository the
file is `evaluation/gnet_brain_correlation.py`.
