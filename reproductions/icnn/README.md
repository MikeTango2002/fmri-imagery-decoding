# reproductions/icnn/

Retraining of iCNN (Shen et al., 2019; trained on NSD by Shirakawa et al.) from scratch and its evaluation on
NSD-Imagery. The benchmark authors did not release their iCNN code; the NSD pipeline of benchmark B is therefore new code
written for the thesis. It keeps the estimator of the Kamitani Lab scripts and applies the two changes that the
benchmark paper lists in its Appendix A.6.
This folder holds only what was added or changed. The upstream code is not copied.

| | |
|---|---|
| Upstream | <https://github.com/KamitaniLab/feature-decoding> (MIT, © 2024 Kamitani Lab; `LICENSE.upstream`) |
| Commit | `ef0f1cc` ("Fix file mode", the head of `main`) |
| Thesis | Section 3.2.1 (the method), 4.2 (sessions), 4.3.5 (procedure), 4.4.5 (benchmark A), 4.5.5 (benchmark B), Appendix A |

## Layout

```
patches/                 the two upstream configuration files that were changed, as git patches
scripts/                 the files that were added; the folders mirror the working copy (scripts/src/ goes to icnn/src/,
                         scripts/feature-decoding/ to icnn/feature-decoding/, next to the clone)
requirements-lock.txt    the frozen environment of the only venv of this model
LICENSE.upstream         the licence of the upstream repository
```

## How to rebuild the working copy

The working copy is a folder `icnn/` that holds the clone of `feature-decoding` and the added files side by side.

```bash
mkdir icnn && cd icnn
git clone https://github.com/KamitaniLab/feature-decoding.git feature-decoding
(cd feature-decoding && git checkout ef0f1cc && for p in <this folder>/patches/*.patch; do git apply "$p"; done)
cp -r <this folder>/scripts/. .      # setup_venv.sh, recon_icnn_A.py, diag_profile_distribution.py, src/ and feature-decoding/run_A_layers.sh
```

Both patches apply to a fresh clone of `ef0f1cc` with `git apply --check`. The scripts of `src/` are started from `icnn/feature-decoding/` (as `../src/x.py`) or
from `icnn/` (as `src/x.py`): the first lines of each file say which. The data paths are relative (`./data/...`, as in
the upstream scripts). The scripts that need the root of the other reproductions read it from `PROJECT_ROOT`
(`export PROJECT_ROOT=<folder that holds mindeye_I/, mindeye_II/, takagi/ and icnn/>`): `src/link_existing_nsd_data.sh`
and the default `--betas_dir` of `src/prepare_nsdcore_icnn.py`.
`src/link_existing_nsd_data.sh` needs the data of MindEye1 and Takagi to be downloaded first (see their READMEs): it links
those files and reports the ones that are missing.

## Licences

`feature-decoding` is MIT (`LICENSE.upstream`); the patches and the scripts adapted from its examples can be
distributed with that notice. The package `bdpy` (0.26) and `fastl2lir` (0.11) are installed from PyPI and not copied.
`recon_icnn_A.py` and `src/recon_icnn_imagery.py` are adapted from `codes/gan/recon_icnn_image_vgg19_dgn_relu7gen_gd.py`
of the Takagi repository (MIT, see `reproductions/takagi/`), which is its copy of the example script of the Kamitani Lab.
The repository `KamitaniLab/spurious_reconstruction` (Shirakawa et al.) has no licence: no file of it is here
and no code was taken from it (see the note on what is not here).

## Stack

- One environment (`scripts/setup_venv.sh`, built with `uv`; there is no conda on the server): Python 3.10.13
  (`bdpy` needs Python below 3.12), PyTorch 2.0.1 built for CUDA 11.8, NumPy held at 1.26.4 (the 1.x ABI that
  PyTorch 2.0.1 is built for). The GPUs of the server (RTX 6000 Ada, sm_89) need a CUDA 11.8 build or newer. All the
  caches stay inside the folder (`TORCH_HOME` and `XDG_CACHE_HOME` are written into the `activate` script). The frozen
  list is `requirements-lock.txt`.
- Python 3 port instead of Caffe. The original code (`KamitaniLab/DeepImageReconstruction`, Python 2.7 and a
  2016 fork of Caffe) cannot be built for sm_89. Stage 1 (fMRI to VGG19 features) uses the official Python 3 port
  `feature-decoding`. Stage 2 (image from features) uses `bdpy.recon.torch.icnn`, with the VGG19 and the generator of
  the lab converted to PyTorch.
- Stage 1 is linear algebra and runs on CPU and RAM for benchmark A (`fastl2lir`); the closed form of benchmark B runs on
  one GPU in float64. Stage 2 is gradient descent on the generator input, one image at a time: about 16 s per image for
  1,000 iterations on one GPU; it runs as one process per GPU (`CUDA_VISIBLE_DEVICES=0` and `1`), never DDP.
- `OPENBLAS_CORETYPE=SkylakeX`: the usage lines of the CPU scripts export it. The processor of the server (Xeon Gold
  6448Y, Sapphire Rapids) is newer than some OpenBLAS builds, which then compute wrong matrix products without any warning.
  It is not needed on other machines.

## Deviations from the original procedure

| Deviation | Where it matters |
|---|---|
| PyTorch 2.0.1 for CUDA 11.8, Python 3.10, the Python 3 port in place of Caffe and Python 2.7 (see Stack). | A, B |
| Benchmark A uses three subjects and the 19 layers of VGG19. The upstream configuration is a reduced run (subject `sub-03` only, layers `fc6`, `fc7`, `fc8`); the `exclude_labels` key is set to `[]` because the empty key makes `predict_feature_fastl2lir.py` fail. | A |
| Point 2 of Appendix A.6: sklearn Ridge in place of `fastl2lir`. The ridge uses all `nsdgeneral` voxels (about 13,000 to 16,000), not the 500 best voxels of each output unit, and keeps $\alpha = 100$ of the Kamitani configuration without tuning it on NSD. It is computed in closed form without ever forming the dense coefficient matrix (197 GB for one early layer) or the features of the training images (220 GB per subject for the eight layers of stage 2): `src/decode_features_chunked.py`. `--verify` checks it against a real sklearn `Ridge` on a small layer. | B |
| Point 1 of Appendix A.6: no normalization of the images when computing the VGG19 features. Read as: no subtraction of the mean image colour before VGG19, RGB order. The second normalization of the Kamitani code, applied to the decoded features before stage 2, is kept. `--rgb_mean`, `--bgr` and `--no-normfeat` run the other readings. | B |
| NSD-core training uses the 40 sessions of the benchmark specification and the 27,000 single trials of the 9,000 non-shared images (not averaged), read from the HDF5 files of MindEye2. | B (Section 4.2) |
| 1,000 iterations of stage 2 in place of the 200 of the upstream scripts; ten seeds per stimulus (a seed fixes only the starting input of the generator). | B |

## Patches

| Patch | What it changes | Why |
|---|---|---|
| `config_pyfastl2lir.patch` | `config/deeprecon_pyfastl2lir_alpha100_vgg19_allunits.yaml`: subjects `sub-01`, `sub-02` switched on (upstream has only `sub-03`), the 19 layers (upstream `fc6`, `fc7`, `fc8`), in both sections `decoder` and `decoded_feature`; `exclude_labels: []`. | The configuration of benchmark A. `alpha: 100`, `num: 500` and `average_sample: true` are the values of the authors and are not touched. |
| `config_sklearn_ridge.patch` | `config/deeprecon_sklearn_ridge_alpha100_vgg19_allunits.yaml`: the three subjects, the files `..._vc` (visual cortex), the layers `conv5_4`, `fc6`, `fc7`, `fc8` only. | The configuration of the optional comparison of `fastl2lir` against sklearn on the data of Shen et al. Four layers, because a dense `coef_` for `conv1_1` alone would be 197 GB per subject. This comparison was not run. The patch is included because Section 4.3.5 of the thesis lists both changed configuration files. |

No upstream code file is changed. The upstream scripts (`train_decoder_fastl2lir.py`, `predict_feature_fastl2lir.py`,
`evaluation.py`, `data/download.py`) are used as they are.

## Scripts

| File | What it does | Benchmark |
|---|---|---|
| `setup_venv.sh` | Builds the virtual environment (`.venv`): PyTorch from the CUDA 11.8 index, NumPy 1.26.4, `bdpy`, `fastl2lir`, and the packages that `requirements.txt` of the clone does not list. | A, B |
| `feature-decoding/run_A_layers.sh` | Stage 1 of benchmark A: the loop over the 19 VGG19 layers, cheapest first. Each step (train, predict) runs as three parallel workers with their own logs, five seconds apart; it stops at the first failed step and deletes the `W.mat` of a layer when its predictions are done. Started from `icnn/feature-decoding/`. | A |
| `recon_icnn_A.py` | Stage 2 for benchmark A: the 50 test images of Deeprecon, subject by subject; resumes if a `.tiff` exists. Adapted from the Takagi copy of the Kamitani example, with the paths of the file names of `feature-decoding`. With `--features-dir` it also reconstructs the `shared1000` bridge. | A, bridge |
| `diag_profile_distribution.py` | Per-unit distribution of `profile_correlation` read from `evaluation.db`: median, quartiles and the fraction of dead units, per layer and subject. | A |
| `src/link_existing_nsd_data.sh` | Links the NSD and NSD-Imagery files that the MindEye1 and Takagi reproductions already downloaded into the layout of `feature-decoding/data/`. Nothing is copied. | B |
| `src/prepare_nsdcore_icnn.py` | The NSD-core index: which rows of the HDF5 of MindEye2 are training and test trials, and which of the 73,000 images each shows. Checked trial by trial against the map of MindEye2. | B |
| `src/prepare_nsd_imagery_icnn.py` | NSD-Imagery betas to the `bdpy` file of the decoding step: `nsdgeneral` mask, divide by 300, z-score per run, 384 rows with one label each. Seven inline checks stop the run at the first failure. Writes the ground-truth tensors. | B (Section 4.2) |
| `src/decode_features_chunked.py` | Stages 0 and 1 of benchmark B in one pass: VGG19 features in RAM, the ridge in closed form, decoded features on disk. `--selftest` needs no data; `--test shared1000` makes the bridge. | B |
| `src/recon_icnn_imagery.py` | Stage 2 for benchmark B: ten seeds, one directory per seed, 1,000 iterations, `--conditions` to split the work across GPUs. | B |
| `src/export_icnn_recons.py` | Writes the `.tiff` of stage 2 as 224 px PNG files in the layout of the evaluation harness (`<subj>/<condition>/samples/{stimulus:05}_{sample:03}.png`), with the index of the stimulus inside its condition. | B |
| `src/diag_alpha_cv.py` | Cross-validation of $\alpha$ on the NSD-core training images only (five folds, ten values). It writes nothing but a table. | B (Section 4.5.5) |
| `src/diag_renorm_stats.py` | Measures the normalization of the NSD-core HDF5 of MindEye2 (global or per session). | B (Section 4.3.5) |
| `src/diag_feature_dispersion.py` | Spread of the decoded features across stimuli, imagery against vision: is there a collapse towards the training mean. | B (Section 4.5.5) |
| `src/diag_feature_identification.py` | Do the decoded features identify their own stimulus among the 12 (against the true VGG19 features). | B (Section 4.5.5) |
| `src/decode_features_fastl2lir.py` | The decoder of Shirakawa et al. (`fastl2lir`, the 500 best voxels of each unit) in streaming form, for all 19 layers; the variant of Table 4.15. `--verify` compares it with the real `FastL2LiR`. | B (Section 4.5.5) |

`NUM_VOXELS` (the size of the `nsdgeneral` mask, four subjects) is defined in `prepare_nsd_imagery_icnn.py`. The files of
`trisect/` and `experiments/` that need it import it from here, through a `sys.path` entry for the `src/` of this working
copy, and the MLP experiment of `experiments/decoder_alternatives/` holds a copy of the four values.

## Data and checkpoints

Downloads are done by hand or by the scripts of the other reproductions. Only what the code opens is fetched.

```bash
# Benchmark A: data of Shen et al. (from icnn/feature-decoding/data; upstream download.py, 3.4 GB + 64 GB + 2.7 GB)
python download.py fmri_deeprecon_fmriprep_vc
python download.py features_imagenet_training_vgg19      # long: run it in the background
python download.py features_imagenet_test_vgg19
rm -f features-ImageNet*-caffe-VGG19-*.zip               # after checking that features/ holds the 19 layer folders

# Weights of VGG19 and of the generator (A and B; from icnn/; 1.4 GB, figshare article 21564384)
mkdir -p models && cd models
wget -q -O VGG_ILSVRC_19_layers.zip https://ndownloader.figshare.com/files/38225868
wget -q -O bvlc_reference_caffenet_generator_ILSVRC2012_Training.zip https://ndownloader.figshare.com/files/52399790
unzip -q -o VGG_ILSVRC_19_layers.zip && unzip -q -o bvlc_reference_caffenet_generator_ILSVRC2012_Training.zip && rm -f *.zip
```

The two zips unpack to `models/VGG_ILSVRC_19_layers/` (`VGG_ILSVRC_19_layers.pt`, `ilsvrc_2012_mean.npy`,
`estimated_cnn_feat_std_VGG_ILSVRC_19_layers_ImgSize_224x224_chwise_dof1.mat`) and
`models/bvlc_reference_caffenet_generator_ILSVRC2012_Training/` (`generator_relu7.pt`, `relu7_act_range_3x.txt`): these
are the paths of `recon_icnn_A.py`. The paths of the Takagi copy of the script differ (it looks for `models/pytorch/...`
and `act_range/3x/fc7.txt`), which is one of the reasons for the adaptation. The same `VGG_ILSVRC_19_layers.pt` computes the
target features of benchmark B; both benchmarks therefore need the weights.

For benchmark B no feature or fMRI file is downloaded by this model; they come from the other reproductions:

- NSD-Imagery betas, the stimulus table and the 18 images: `reproductions/mindeye1/scripts/download_nsd_imagery.sh`
  (the same files, in the layout that `link_existing_nsd_data.sh` links).
- The NSD-core betas of the 40 sessions (`betas_all_subj0N_fp32_renorm.hdf5`) and `COCO_73k_subj_indices.hdf5`: the
  dataset of MindEye2, `reproductions/mindeye2/scripts/download_mindeyev2.py` (it holds them in `mindeye_II/src/mindeyev2_dataset/`).
- `nsd_expdesign.mat` and the 73,000 images (`nsd_stimuli.hdf5`, 39.5 GB): the NSD data of Takagi
  (`reproductions/takagi/README.md`, section Data and checkpoints).

## Commands

Benchmark A (the metric of Shen et al. on the data of Deeprecon; it only checks that the code works). The environment is
active, commands run from `icnn/feature-decoding/` unless marked.

```bash
CFG=config/deeprecon_pyfastl2lir_alpha100_vgg19_allunits.yaml
# stage 1, one layer at a time, from the cheapest (fc8) to the most expensive (conv1_1), three workers per step
for L in fc8 fc7 fc6 conv5_4 conv5_3 conv5_2 conv5_1 conv4_4 conv4_3 conv4_2 conv4_1 conv3_4 conv3_3 conv3_2 conv3_1 conv2_2 conv2_1 conv1_2 conv1_1; do
  python train_decoder_fastl2lir.py   $CFG -o "decoder.features.layers=[$L]" "decoded_feature.features.layers=[$L]"
  python predict_feature_fastl2lir.py $CFG -o "decoder.features.layers=[$L]" "decoded_feature.features.layers=[$L]"
  find data/feature_decoders/ImageNetTraining/deeprecon_fmriprep_pyfastl2lir_alpha100_allunits/caffe/VGG19/$L -name 'W.mat' -delete
done
python evaluation.py $CFG                       # profile_correlation, pattern_correlation, identification accuracy
python ../diag_profile_distribution.py
# stage 2, from icnn/: one process per GPU
CUDA_VISIBLE_DEVICES=0 python recon_icnn_A.py --subject sub-01
CUDA_VISIBLE_DEVICES=1 python recon_icnn_A.py --subject sub-02
```

The loop is `scripts/feature-decoding/run_A_layers.sh`: it starts each step as three parallel workers with their own
logs, five seconds apart, and stops at the first failed step. The commands above are what each step runs. The reason for one layer
at a time is disk: the weights of the first convolution layers are the largest files, and the loop deletes the `W.mat` of a
layer as soon as its predictions are done; the peak therefore stays at about 24 GB instead of 116 GB.

Benchmark B (NSD-Imagery, the thesis deliverable). `PROJECT_ROOT` is set, the data of the section above are in place.

```bash
bash src/link_existing_nsd_data.sh                       # from icnn/; then: find feature-decoding/data -xtype l   (no dead link)
cd feature-decoding
python ../src/prepare_nsd_imagery_icnn.py --subj 1 2 5 7 --data_root ./data/nsd --imagery_root ./data/nsd_imagery \
    --out_dir ./data/fmri/NSDImagery --reference_dir ./data/nsd_imagery/preprocessed_data
python ../src/prepare_nsdcore_icnn.py --subj 1 2 5 7 --betas_dir <PROJECT_ROOT>/mindeye_II/src/mindeyev2_dataset \
    --expdesign ./data/nsd/nsddata/experiments/nsd/nsd_expdesign.mat --out_dir ./data/fmri/NSDcore
python ../src/decode_features_chunked.py --selftest      # closed form against sklearn on random data, no data needed
CUDA_VISIBLE_DEVICES=0 python ../src/decode_features_chunked.py --subj 1 2     # stages 0 and 1, one subject pair per GPU
CUDA_VISIBLE_DEVICES=1 python ../src/decode_features_chunked.py --subj 5 7
cd ..
CUDA_VISIBLE_DEVICES=0 python src/recon_icnn_imagery.py --subject subj01       # stage 2: 5 conditions x 6 stimuli x 10 seeds,
CUDA_VISIBLE_DEVICES=1 python src/recon_icnn_imagery.py --subject subj02       # one subject per run (also subj05, subj07)
python src/export_icnn_recons.py                          # 1,200 PNG files in recon_B_export/, verified at the end
```

The numbers of benchmark B come from the common harness of `evaluation/`, which reads the exported layout with the adapter
of Takagi: `final_evaluations_imagery.py --model icnn --recon_root recon_B_export --gt imagery_stimuli_18.pt --out <file>.tsv`
for the image metrics, and `gnet_brain_correlation.py compute --model takagi --recon_root recon_B_export ...` (the other arguments
are in its docstring) for the brain correlation. The two lines
printed at the end of `export_icnn_recons.py` name the flags of the Takagi metric scripts (`--recon_dir`), which are not the
flags of the harness.

The `shared1000` bridge (the Table 1 row of the benchmark paper, scored with the same harness): the decoded features of the
first 100 test images, then stage 2 on them, then the harness of `evaluation/eval_bridge_icnn.py`.

```bash
cd feature-decoding && python ../src/decode_features_chunked.py --subj 1 --test shared1000 --n_test 100 --layers_recon \
    --out_dir ./data/decoded_features/NSDcoreShared1000_100     # a separate directory: see the check in main()
cd .. && python recon_icnn_A.py --subject subj01 --roi nsdgeneral --n-iter 1000 --seed 0 \
    --features-dir $PWD/feature-decoding/data/decoded_features/NSDcoreShared1000_100/nsdcore40ses_ridge_alpha100_allunits/caffe/VGG19 \
    --output-dir $PWD/recon_bridge100/normfeat
```

The variants of the bridge (`--rgb_mean`, `--no-normfeat`) use separate output directories, so that they do not overwrite
the inputs of the diagnostics. The diagnostics of Section 4.5.5 have their command in the docstring of each file.

## Notes

1. What is not here.
   - `dist_loss.py` and `recon_icnn_dists.py`: the port of the stage 2 of Shirakawa et al. (the DISTS and mean-squared-error loss
     on all 19 layers, the optimizer and the starting input of that work). It is a port of code of `KamitaniLab/spurious_reconstruction`,
     which has no licence; it is not distributed. Section 4.5.5 describes the port and what it changes on the bridge; the
     variants of Table 4.15 that use it cannot be regenerated from this folder. `decode_features_fastl2lir.py` names that
     repository only as the source of one configuration (`nsd-37ses_..._fastl2lir_a100_vgg19_allunits.yaml`); the code of the file
     is a reformulation of the same estimator and takes no line from it.
   - The 100 ground-truth images of the bridge (`recon_bridge100/gt425/`, the NSD photographs at 425 px) were written without a
     script in this folder.
2. Cross-checks that are inline. `prepare_nsdcore_icnn.py` stops if the rows of the MindEye2 HDF5 are not in the order of the
   design file (a wrong order would misalign the whole decoder without any error), and `prepare_nsd_imagery_icnn.py` stops if
   the prepared betas differ from the reference tensors of MindEye1 by more than $10^{-3}$ (they match to about $10^{-5}$).
