# vdvae_lib

Vendored modules of OpenAI's "very deep VAE" (https://github.com/openai/vdvae, MIT license,
see `LICENSE.md`): `hps.py`, `utils.py`, `data.py`, `vae.py`, `vae_helpers.py`, `train_helpers.py`, `model_utils.py`.
They are the import chain of Brain-Diffuser's `scripts/vdvae_extract_features.py`.

One change to the upstream code: in `vae.py`, `DecBlock.sample` also returns the posterior mean
`qm`, and `DecBlock.forward` stores it in the stats dict next to the sample `z`
(three lines; the value of `z` and every other module are untouched).
