# experiments/decoder_alternatives/

Two alternatives to the ridge decoder of Section 5.3, both tested in Section 8.1.1: a multi-subject
MLP with 309 million parameters, and a hybrid that takes the semantic and structural codes from the MLP
and z_low from the ridge. Every later stage of the pipeline (adapter, inverse PCA, VDVAE, Versatile
Diffusion) is the same as in the final model.

## Files

| File | What it does | Thesis section |
|---|---|---|
| `mlp_encoder.py` | The `MLPEncoder`: one shared trunk with three heads, one linear adapter per subject, and its loss (cosine on the semantic code, mean L2 on the other two, optional InfoNCE). | 8.1.1 |
| `train_mlp_encoder.py` | Training loop on the four subjects at once, with resource logging. `--smoke-test` runs the same code on synthetic data. | 8.1.1 |
| `decoder_training_utils.py` | The three functions and two constants that the MLP scripts import (`load_targets`, `load_subject_split`, `info_nce_loss`, `MIN_BATCH_FOR_CTR`, `NUM_VOXELS`). | 8.1.1 |
| `test_mlp_encoder.py` | Unit tests of `mlp_encoder.py`: head shapes, scale of the output, size of the losses at random initialization. | none |
| `predict_mlp.py` | Writes the MLP predictions in the layout of the ridge scripts, so that `reconstruct.py` only needs another `--pred-dir`. `--trial-level` is the imagery mode used in the thesis. | 8.1.1 |
| `run_mlp_vs_ridge.sh` | The three generation and scoring legs: ridge on imagery, MLP on imagery, MLP on shared1000. Only the shared1000 leg enters Table 8.2. | 8.1.1 |
| `run_mlp_vs_ridge_imagery_trials.sh` | The two imagery legs with `--trial-level`, plus a control that the ridge does not move. These are the imagery numbers of Table 8.2. | 8.1.1 |
| `compare_mlp_vs_ridge.py` | Paired bootstrap, MLP minus ridge, on vision (83 blocks) and imagery (12 stimuli), and the verdict. | 8.1.1 (Table 8.2) |
| `build_hybrid_predictions.py` | Recombines the predictions of the two decoders into a hybrid prediction folder. | 8.1.1 |
| `run_hybrid_vs_ridge.sh` | Generation and scoring of the hybrid on imagery and shared1000, and of the symmetric control on shared1000. | 8.1.1 |
| `compare_hybrid_vs_ridge.py` | Per-metric criterion on vision, then the imagery reading. | 8.1.1 |

## Order of the steps

```
train_mlp_encoder.py ─► predict_mlp.py (--dataset shared1000, --dataset imagery --trial-level)
      ─► run_mlp_vs_ridge.sh (shared1000 leg) + run_mlp_vs_ridge_imagery_trials.sh ─► compare_mlp_vs_ridge.py
predict_mlp.py + the ridge predictions ─► build_hybrid_predictions.py ─► run_hybrid_vs_ridge.sh ─► compare_hybrid_vs_ridge.py
```

## Notes

- Imagery input. The MLP is trained on single trials (std about 1), while repetition-averaged betas have
  a std of about 0.30. The ridge absorbs that rescaling and the MLP does not. On imagery the predictions are
  therefore averaged instead of the betas (`run_mlp_vs_ridge_imagery_trials.sh`), and
  `compare_mlp_vs_ridge.py` reads that pair by default.
- `train_mlp_encoder.py`, `mlp_encoder.py` and `test_mlp_encoder.py` import `NUM_VOXELS` from
  `prepare_nsd_imagery_icnn.py` of the iCNN reproduction (they put `icnn/src` on `sys.path`).
- The scripts run from a project root with the Python files in `src/`, as in the layout of the
  top-level README, which also shows how to put the folders on the Python path.
- The metric files are in `results/decoder_alternatives/mlp/` and `results/decoder_alternatives/hybrid/`.
  The folders of predictions and reconstructions outside `results/` keep their original names
  (`d2_3_recons`, `g4_0_recons`, `g4_0_hybrid_*`).
- Some variable and folder names (`D2_3_DIR`, `G4_0_DIR`, `d2_3_recons`, `g4_0_*`) keep the label of the work.
  The main README, "Working labels in some names", says what each stands for.
