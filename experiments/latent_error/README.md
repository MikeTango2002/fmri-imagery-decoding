# experiments/latent_error/

The analysis of Section 8.3: how far the predicted latents are from the true ones, on shared1000
(in distribution) and on NSD-Imagery (out of distribution), and how much of the low-level target is
sampling noise.

## Files

| File | What it does | Thesis section |
|---|---|---|
| `latent_prediction_error.py` | Predicted against true latents per branch and per subject, after centering on the training mean: cosine of matched and unmatched pairs, R² against the mean, ratio of norms, two-way and rank-1 identification. The four branches are reported side by side, because they share one ridge kernel and differ only in the space they are rendered into: if all collapse out of distribution, the failure is in the kernel. | 8.3 (Table 8.8) |
| `zlow_target_noise_ceiling.py` | The `z_low` target is a sample from the VDVAE posterior, not its mean. The script encodes each image twice with different seeds and measures the cosine between the two codings (0.0857 raw, 0.2032 after the PCA). This is the signal fraction of the target; a decoder that emits the posterior mean scores its square root against a single draw. | 8.3, 5.2 |

Both scripts read files that the earlier stages wrote (`trisect/scripts/`), and the NSD-Imagery
predictions of `predict_ridge_imagery.py`.
