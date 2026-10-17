# evaluation/

The evaluation code of the thesis: the one harness that scores all six models (the five reproduced
baselines and TRISECT) on NSD-Imagery, plus the brain-correlation metric and the confidence
intervals. Every number of Chapters 4 to 8 that comes from NSD-Imagery or from the block sets goes
through the files in this folder.

## Source and licence

`final_evaluations_imagery.py`, `gnet_brain_correlation.py` and the parts of `final_evaluations.py`
that they reuse are a transcription of the evaluation code of the benchmark authors
(Kneeland et al., 2025), published in the MIT-licensed repository `MedARC-AI/MIRAGE`
(© 2023 MedARC), file `src/final_evaluations_mi_multi.ipynb`, branch `main`. The cell numbers quoted
in the comments are the cell numbers of that notebook. The logic is not changed: any change would
give numbers that cannot be compared with the published tables. `final_evaluations.py` is the
evaluation script of the original MindEye2 repository (`MedARC-AI/MindEyeV2`, MIT, © 2023 MedARC).
The licence text of MIRAGE is in `LICENSE.MIRAGE`; it is the same text as that of MindEyeV2
(`reproductions/mindeye2/LICENSE.upstream`).

Version. Commit `7dce6b5` of MIRAGE (2024-11-07), the last commit that changed the notebook. It
has 45 cells, the count that the cell numbers in the comments refer to. The `main` branch ends later,
at `7dbc13f` (2025-02-17, a change to the README only; checked on 2026-10-05). The image-metric
cells (two-way identification, PixCorr, SSIM, the five feature networks and the resizing) are the same in
every revision of the notebook from `66f611b` (2024-11-05) to `7dce6b5`: the revisions in between change
only the caption cells, the brain-correlation columns and how the number of repetitions is read
(compared cell by cell). The choice of revision among them therefore cannot change an image metric.

## Files

| File | What it does | Thesis section |
|---|---|---|
| `final_evaluations_imagery.py` | The eight image metrics (PixCorr, SSIM, AlexNet(2), AlexNet(5), Inception, CLIP, EffNet-B, SwAV), 10 samples per stimulus, one TSV row per repetition. | 4.1, 6.3 |
| `gnet_brain_correlation.py` | Brain correlation in early, higher and whole visual cortex, with the ground-truth ceiling check. | 6.3 |
| `aggregate_imagery_metrics.py` | Merges the two TSVs into Table 1 and Tables 3-4, with and without the `1/11` correction of the two-way metrics. | 4.1, 6.3 |
| `bootstrap_ci_imagery.py` | Bootstrap confidence intervals over stimuli, subjects and cells; paired and margin modes. | 6.4 |
| `evaluate_block_set.py` | The same image metrics over sets packaged as blocks of 12 stimuli, each block with its own ground truth. | 8.2 |
| `make_shared1000_blockset.py` | Packages `shared1000` as 83 blocks of 12, in the format of the proxy set. | 6.4, 8.2, 8.4 |
| `final_evaluations.py` | Benchmark A of MindEye2 on `shared1000` (original script). | 4.4 |
| `eval_bridge_mindeye2.py`, `eval_bridge_icnn.py` | The `shared1000` check, scored with the harness of benchmark B. | 4.4 |

## Notes

- Environment. MIRAGE and MindEye2 share a codebase, and the MindEye2 environment is the MIRAGE
  one (see `reproductions/mindeye2/`). The scripts of this folder import `models.py`, `utils.py` and
  `generative_models/` from the MindEye2 repository, which is not copied here
  (`reproductions/mindeye2/` has the patches and the commit).
- The off-by-one. The two-way metrics carry the `- 1` of the original notebook on purpose
  (Section 4.1). `aggregate_imagery_metrics.py` prints both readings.
- Validation gates of the harness. `--model gt --gen_rep 1` feeds the ground truth as its own
  reconstruction: PixCorr and SSIM are 1.000 and the four two-way metrics are 0.9091 (10/11). If they were
  1.0, the `- 1` was not copied. EffNet-B and SwAV do not reach 0 (0.0075 and 0.0027 on subject 1): the
  ground truth is resampled once (425 to 255 or 224), the reconstruction twice (425 to 512, then as before).
  The two-way metrics are ordinal and do not notice. The asymmetry is Kneeland's. `--roll_gt` rolls the
  ground truth by one inside each block of 6 and keeps the real reconstructions: the chance level of
  that model. `--model gt --roll_gt` is the model-free floor of the harness (about 0.409).
- Paths. Scripts that read other parts of the repository do it through arguments
  (`--recon_root`, `--gt`, ...). `evaluate_block_set.py` adds a folder of the original layout to
  `sys.path`; in this layout the import of `final_evaluations_imagery` works because both files are
  in the same folder.
- Data. The harness reads the 18 stimuli (`imagery_stimuli_18.pt`), the NSD-Imagery trial table and
  betas, the GNet weights (`gnet_multisubject.pt`) and `brain_region_masks.hdf5`. The section "Data" of
  the top-level README says where each file comes from. The weights of the feature networks
  (AlexNet, InceptionV3, CLIP, EfficientNet-B1, SwAV) are downloaded on first use.
