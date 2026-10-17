# experiments/code_width/

How wide the three latent codes need to be, and how redundant they are with each other. The final
model uses 128 components per semantic half and 32 structural channels; the scripts here are the
checks behind those choices (Sections 5.2 and 8.1.4).

## Files

| File | What it does | Thesis section |
|---|---|---|
| `pca_width_zsem.py` | Fits the whitened PCA of `z_sem` at widths 32, 64, 128, 256, 512 and 768 on the 72,000 non-shared images, and saves each basis and projection. `--no-whiten` is the un-whitened control. | 5.2 |
| `run_semantic_width_sweep.sh` | For each `d_sem` (default 256 and 512): rebuilds the targets from the saved basis, retrains the adapter, runs the adapter-vs-true-context check. | 5.2 |
| `run_structural_width_sweep.sh` | The same for the structural code (default `d_struct` 64 and 96), with a PCA refit as first step. | 5.2 |
| `sample_adapter_vs_true_context.py` | Last step of both sweeps: reconstructs held-out shared1000 images from the adapter context and from the true Versatile Diffusion context, side by side. Visual check, no metric. | 5.2 |
| `compare_zsem_txt_sources.py` | Draws the 2,000-image sample of the caption comparison (`sample_indices.npy`) and scores one human COCO caption per image against BLIP-2 OPT 2.7B (greedy) by CLIP text-to-image retrieval. The next script reads its sample. | 5.2 |
| `compare_zsem_txt_blip2_large.py` | The BLIP-2 side of the same comparison with `blip2-flan-t5-xxl` and beam 5, on the same sample. | 5.2 |
| `cross_space_redundancy.py` | Linear cross-regression R² between every ordered pair of `z_sem`, `z_struct` and `z_low`, tested on shared1000. | 8.1.4 |

## How the scripts chain

```
pca_width_zsem.py ─► run_semantic_width_sweep.sh ─┐
fit_pca_zstruct.py ─► run_structural_width_sweep.sh ─┤  each width: assemble_targets.py → train_adapter.py
                                                    └─► sample_adapter_vs_true_context.py
```

`fit_pca_zstruct.py`, `assemble_targets.py` and `train_adapter.py` are in `trisect/scripts/`. The
two sweep scripts call them as `python src/<name>.py`, the path of the layout in which every
script of the project is in one `src/` folder. The top-level README shows how to build that
folder and how to put the folders on the Python path. The log files of the sweeps are named `d<D>_1_extract_targets.log`, `d<D>_2_train_a_psi.log` and so on, as the
scripts write them.

## Notes

- The caption comparison is only partly scripted. `compare_zsem_txt_sources.py` takes one human
  caption per image, whereas the targets average the embeddings of all of them; its numbers are not
  the ones of Section 5.2. The human side with averaged captions (text-to-image R@1 of 0.468) was computed by a separate one-off step, which
  wrote `results_human_mean5.json`, and is not in this folder. The BLIP-2 side (R@1 of 0.4115)
  is the output of `compare_zsem_txt_blip2_large.py`.
- Fit sets. All PCA bases are fitted on non-shared images only.
