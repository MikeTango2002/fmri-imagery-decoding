# experiments/structural_factor/

Whether the structural code `z_struct` (from DINOv2) matters end to end, and what it carries
(Section 8.1.4).

## Files

| File | What it does | Thesis section |
|---|---|---|
| `run_structural_ablation.sh` | Two ablations on the proxy set with true factors, at the best setting of the oracle budget (strength 0.85, mixing 0.6): `z_struct` set to zero in the whitened PCA space, and `z_struct` shuffled across the stimuli of each block. The baseline is the T2 arm of the oracle budget, which is not run again. | 8.1.4 |
| `analyze_structural_ablation.py` | Paired bootstrap CIs of the two arms against the baseline, on proxy-simple and proxy-complex images separately. | 8.1.4 (Table 8.4) |
| `traversal_figure.py` | Sweeps `z_sem` with `z_struct` fixed and the other way round, on proxy-set images, and writes labelled montages. No number comes from it: the third observation of Section 8.1.4 rests on this figure. | 8.1.4 |

## Notes

- The arms are produced by `--ablate-factor z_struct_zero` and `z_struct_shuffle` of `reconstruct.py`
  (`trisect/scripts/`). The proxy set comes from `experiments/oracle_budget/`.
- The script writes its images to `data/traversal_figure/`. They are not included in this repository.
