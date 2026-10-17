# experiments/shared1000_comparison/

TRISECT against Brain-Diffuser on the shared1000 test set, paired over blocks instead of over the 12
NSD-Imagery stimuli, and the two experiments that need that axis: the generation noise and the swap of
predicted inputs between the two models (Sections 6.4, 7.1 and 8.4).

## Files

| Part | File | What it does | Thesis section |
|---|---|---|---|
| Comparison | `prepare_bd_shared1000_blocks.py` | Puts the shared1000 reconstructions of Brain-Diffuser on the block set of TRISECT. Its test row order is reconstructed from its own data-preparation script and checked on pixels against its stored stimuli, before anything else runs. | 6.4, 7.1 |
| | `run_bd_shared1000_blocks.sh` | Scores Brain-Diffuser through the harness on the blocks, against the 425-native ground truth, then runs the three readings. No GPU is needed. | 6.4, 7.1, 8.4 |
| | `compare_trisect_vs_bd_shared1000.py` | Three readings: TRISECT minus Brain-Diffuser paired over blocks; an ordering test of the deficits on shared1000, vision and imagery; and the generation floor from replicas of the same arm. | 6.4, 7.1, 8.4 |
| Generation noise | `run_generation_noise_shared1000.sh` | Regenerates the TRISECT arm of shared1000 with the two fixes (one diffusion seed per replica, subject and stimulus; averaging before re-standardizing). | 6.4, 7.1 |
| | `generation_noise.py` | The null comparison: replicas {0,1} against {2,3} of the same arm; the true difference is zero. | 6.4, 7.1 |
| Swap | `bd_predict_shared1000.py` | Rebuilds the shared1000 test predictions of Brain-Diffuser (VDVAE, CLIP text, CLIP vision) without extracting its features again. | 8.4 |
| | `bd_zlow_layout_precheck.py` | Checks that the raw VDVAE latent predicted by Brain-Diffuser decodes correctly with the decoder of TRISECT, before any generation. | 8.4 |
| | `run_bd_swap.sh` | Generates the arms BT, TB and BB (T is TRISECT, B is Brain-Diffuser; first letter the context, second the low-level code). | 8.4 (Table 8.9) |
| | `run_bd_swap_eval.sh` | Scores the three arms with the harness. | 8.4 |
| | `compare_bd_swap.py` | Paired readings of the three arms against the TT control. | 8.4 (Table 8.9) |

## Order of the steps

```
prepare_bd_shared1000_blocks.py ─► run_bd_shared1000_blocks.sh ─► compare_trisect_vs_bd_shared1000.py
run_generation_noise_shared1000.sh ─► generation_noise.py
bd_predict_shared1000.py ─► bd_zlow_layout_precheck.py ─► run_bd_swap.sh ─► run_bd_swap_eval.sh ─► compare_bd_swap.py
```

`run_bd_shared1000_blocks.sh` calls `prepare_bd_shared1000_blocks.py` as its first step. The TRISECT arm
(`results/oracle_budget/S1K_T3_subj0N.tsv`) comes from `experiments/oracle_budget/run_oracle_shared1000.sh`.

## Notes

- The scripts read the tree of the Brain-Diffuser reproduction (`--bd-root`, default `../brain_diffuser`).
  Its files are not in this repository; `reproductions/brain_diffuser/` holds the patches and added scripts of that reproduction.
- The metric files are in `results/shared1000_comparison/`; the block index shared with the other
  comparisons on 65 blocks is `bd_block_index.json`. Arm labels (`S1K_T3_h0_*`, `S1K_BT_*`) stay, because
  they are also the `model` column of the TSVs.
- The shell scripts expect the Python files in `src/` under the project root, as in the layout of the
  top-level README.
- Some variable names (`G6_DIR`, `I2_DIR`, `F6_DELTA`, `G6_DEVICE`) keep the label of the work. The main
  README, "Working labels in some names", says what each stands for.
