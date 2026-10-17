# experiments/oracle_budget/

The oracle budget of Section 8.2: how much each stage of the pipeline costs, measured by giving the
generator the true value of one stage at a time. The generator is the same in all four arms.

| Arm | Contexts of the generator | Low-level input |
|---|---|---|
| T0 | True contexts of the image | True VDVAE latent, uncompressed |
| T1 | Adapter applied to the true code | True VDVAE latent, uncompressed |
| T2 | Adapter applied to the true code | True low-level factor (PCA) |
| T3 | Adapter applied to the predicted code | Predicted low-level factor (the final model) |

The differences T0 − T1, T1 − T2 and T2 − T3 are the cost of the representation and the adapter, of the
compression of `z_low`, and of the brain decoder.

## Files

| Step | File | What it does |
|---|---|---|
| Proxy set | `make_proxy_set.py` | Builds 150 synthetic simple images and 150 COCO photographs that NSD never used, as 25 blocks in the format of the harness. The proxy set exists so that no setting is chosen on the 18 real NSD-Imagery stimuli. |
| | `caption_proxy_set.py` | Captions of the 300 images, which the true Versatile Diffusion context needs. |
| | `extract_proxy_targets.py` | Frozen `z_sem` and `z_struct` of the proxy images. |
| | `extract_proxy_vd_context.py` | True Versatile Diffusion context of the proxy images (the T0 context). |
| | `extract_proxy_vdvae_latents.py` | Raw 91,168-dimensional VDVAE latents of the proxy images (the T0 and T1 low-level input). |
| Proxy sweeps | `run_oracle_t0.sh` | T0 arm, 3 × 3 sweep of strength and mixing. |
| | `run_oracle_t1_t2.sh` | T1 and T2 arms with the same sweep, and the control with the final L2 on `z_sem` removed. |
| | `pick_oracle_constants.py` | Chooses the best (strength, mixing) of an arm from the TSVs of its sweep, with a criterion fixed before the run. |
| shared1000 | `run_oracle_shared1000_prep.sh` | CPU steps: ridge refit without the development split, its predictions, the development block set. They give the selection split for the constants of T3. |
| | `extract_shared1000_vd_context.py` | True Versatile Diffusion context of the 996 images of the shared1000 block set (called by the next script). |
| | `run_oracle_shared1000.sh` | The four arms on the 83 blocks of shared1000, each at the setting that won on a set that is not a measurement set. |
| | `run_oracle_t2_shared1000_paper_constants.sh` | T2 again at strength 0.75 and mixing 0.4, so that T2 − T3 can be read at matched constants. |
| Budget | `oracle_budget.py` | Paired bootstrap of the three differences, on the shared1000 reading and on the proxy reading. |
| | `oracle_budget_fixed_constants.py` | T2 − T3 on shared1000 at matched constants. |

## Order of the steps

```
make_proxy_set.py ─► run_oracle_t0.sh ─► run_oracle_t1_t2.sh
run_oracle_shared1000_prep.sh ─► run_oracle_shared1000.sh ─► oracle_budget.py
run_oracle_t2_shared1000_paper_constants.sh ─► oracle_budget_fixed_constants.py
```

The runners call the extraction and selection scripts themselves: `run_oracle_t0.sh` runs
`caption_proxy_set.py`, `extract_proxy_vd_context.py`, `extract_proxy_vdvae_latents.py`, the sweep,
the evaluation and `pick_oracle_constants.py`; `run_oracle_t1_t2.sh` runs `extract_proxy_targets.py`
and the same sweep and selection for T1 and T2; `run_oracle_shared1000.sh` runs
`extract_shared1000_vd_context.py` and the selection for T3. Scoring is done by
`evaluation/evaluate_block_set.py`.

## Notes

- No script builds the 200 COCO candidates of the proxy set. `make_proxy_set.py` reads
  `data/external_coco/candidates_manifest.json` and the folder `val2017_unused_by_nsd/`, which are made
  by hand in four steps:
  1. Download `nsd_stim_info_merged.csv` from the NSD bucket. It lists 2,949 `cocoId` of
     COCO val2017 used by NSD (the other 70,051 images are from train2017).
  2. Download `instances_val2017.json` from `annotations_trainval2017.zip` (`images.cocodataset.org`).
     COCO val2017 has 5,000 images, of which 2,051 are never used by NSD.
  3. Take `sorted(random.Random(0).sample(sorted(unused), 200))` and write the list of
     `{coco_id, file_name, coco_url, height, width}` as the manifest.
  4. Download the 200 JPEG files from `coco_url` into `data/external_coco/`.

  On freshly downloaded annotations these steps give the same manifest as the one used in the thesis:
  the same 200 entries, the same fields, and no id in `nsd73k_coco_ids.npy` (written by
  `trisect/scripts/derive_nsd73k_coco_ids.py`). There are two seeds: `random.Random(0)` chooses the 200
  candidates, and `make_proxy_set.py` (`seed: 0`) chooses the 150 complex images among them.
- The metric files are in `results/oracle_budget/`: one TSV per arm (`T0.tsv`, `S1K_T3_subj0N.tsv`, …),
  the setting it was read at (`*_setting_choice.json`), and every setting tried (`setting_sweep/`).
  The file names carry the arm labels of `results/README.md`, which are also the `model` column.
- The shell scripts expect the Python files in `src/` under the project root, as in the layout of the
  top-level README.
