"""Smoke test 3 of 3: does the `init_image=`/`strength=` path of `generate_from_context()` behave like
img2img on the real pipeline? Tests 1 and 2 only use the pure-noise branch (`latents=None`).

Checks:
  1. `encode_init_image()` gives a latent of the expected shape and the whole branch runs.
  2. PixCorr(output, input) rises as `strength` falls. A script that ignores `strength` would pass
     check 1; this is the real test.
  3. strength=1.0 is "close to" the `latents=None` default at the same seed (not equal: the
     init-image branch folds the init latent in with `scheduler.add_noise`).

The init image is the real input downsampled to 64x64, not a VDVAE reconstruction. This isolates the
img2img code from reconstruction quality; a failure is a `vd_context.py` bug.

Usage (from the trisect/ folder; needs a GPU):
    python tests/smoke_vd_init_image.py --image-index 42 --seed 0 --strengths 1.0 0.75 0.4 0.15
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image

from trisect.metrics import pixcorr_single as pixcorr
from vd_context import encode_init_image, encode_true_context, generate_from_context, load_pipeline
from smoke_vd_pipeline import OUT_DIR, load_image_and_caption

INIT_RES = 64  # matches the VDVAE decoder's native output resolution (decode_zlow_to_image.py)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-index", type=int, default=42)
    ap.add_argument("--text-to-image-strength", type=float, default=0.75)
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--num-inference-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strengths", type=float, nargs="+", default=[1.0, 0.75, 0.4, 0.15],
                     help="img2img strengths to sweep, high->low so the PixCorr trend reads monotonic")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    pil_image, caption = load_image_and_caption(args.image_index)
    print(f"image {args.image_index}, caption: {caption!r}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = load_pipeline(device=device)

    prompt_embeds, image_embeddings = encode_true_context(pipe, pil_image, caption, device, do_cfg=True)

    init_image = pil_image.resize((INIT_RES, INIT_RES), resample=Image.BILINEAR)
    gt_for_corr = np.asarray(pil_image.resize((224, 224), resample=Image.BILINEAR))

    # Same height/width generate_from_context() derives internally (see `vd_context.py`); pass them
    # explicitly here so this standalone shape check matches what the real call below does.
    gen_height = pipe.image_unet.config.sample_size * pipe.vae_scale_factor
    gen_width = gen_height
    init_latents = encode_init_image(pipe, init_image, height=gen_height, width=gen_width, device=device)
    expected_channels = pipe.vae.config.latent_channels
    assert init_latents.shape[0] == 1 and init_latents.shape[1] == expected_channels, (
        f"encode_init_image() shape {tuple(init_latents.shape)} unexpected "
        f"(want batch=1, channels={expected_channels})"
    )
    print(f"encode_init_image() ok: latent shape {tuple(init_latents.shape)}")

    tiles = [("input", pil_image.resize((224, 224))), ("init(64->224)", init_image.resize((224, 224)))]
    corrs = {}

    for strength in args.strengths:
        generator = torch.Generator(device=device).manual_seed(args.seed)
        out = generate_from_context(
            pipe,
            prompt_embeds,
            image_embeddings,
            text_to_image_strength=args.text_to_image_strength,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generator,
            init_image=init_image,
            strength=strength,
        )[0]
        out_224 = out.resize((224, 224))
        corrs[strength] = pixcorr(np.asarray(out_224), gt_for_corr)
        tiles.append((f"strength={strength}", out_224))
        print(f"strength={strength}: PixCorr(out, input)={corrs[strength]:.4f}")

    # Reference: the pre-existing pure-noise default path (latents=None), same seed.
    generator = torch.Generator(device=device).manual_seed(args.seed)
    noise_out = generate_from_context(
        pipe,
        prompt_embeds,
        image_embeddings,
        text_to_image_strength=args.text_to_image_strength,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
    )[0]
    noise_out_224 = noise_out.resize((224, 224))
    tiles.append(("default (noise)", noise_out_224))

    grid = Image.new("RGB", (224 * len(tiles), 224))
    for i, (_, im) in enumerate(tiles):
        grid.paste(im, (224 * i, 0))
    out_path = os.path.join(OUT_DIR, f"init_image_smoke_idx{args.image_index}_seed{args.seed}.png")
    grid.save(out_path)
    print(f"Done. Saved {out_path}: " + " | ".join(name for name, _ in tiles))

    ordered = sorted(corrs.items(), key=lambda kv: kv[0], reverse=True)  # high strength -> low
    monotonic = all(ordered[i][1] <= ordered[i + 1][1] + 1e-6 for i in range(len(ordered) - 1))
    print(f"PixCorr vs strength, high->low: {[(s, round(c, 4)) for s, c in ordered]}")
    print(f"monotonic (lower strength -> output closer to init image): {'PASS' if monotonic else 'FAIL'}")

    if 1.0 in corrs:
        strength1_arr = np.asarray([im for name, im in tiles if name == "strength=1.0"][0], dtype=np.float32)
        noise_arr = np.asarray(noise_out_224, dtype=np.float32)
        mad = np.abs(strength1_arr - noise_arr).mean()
        print(f"mean abs pixel diff, strength=1.0 vs pure-noise default (0-255 scale): {mad:.3f} "
              f"(expect small, not necessarily ~0 -- see module docstring)")


if __name__ == "__main__":
    main()
