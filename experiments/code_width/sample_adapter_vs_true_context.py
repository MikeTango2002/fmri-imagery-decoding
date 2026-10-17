"""Visual check of the adapter: shared1000 images rebuilt from z_sem and z_struct alone, next to the
same images rebuilt from the true Versatile Diffusion context (Section 5.2).

Inputs:  --targets (targets.h5 variant) and --ckpt (checkpoint of the same train_adapter.py run)
Outputs: <out-dir>/gate0_idx<I>_seed<S>[_strengths].png, panels: input | true context | adapter

Same seed and guidance for both. The adapter sees no pixels and no caption. No pass/fail: the check
is visual. --strengths sweeps text_to_image_strength for the adapter panel only. At the default 0.75
the text half (z_sem) weighs 75% in the dual-guided mix and the half that carries z_struct 25%; a
real structural difference can be hard to see. Needs a GPU.

Usage:
    python sample_adapter_vs_true_context.py --n-images 6 --seed 0 [--strengths 0.75,0.5,0.25,0.0]
"""
import argparse
import os

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw

from adapter_model import APsiMinimal
from vd_context import encode_true_context, encode_unconditional_context, generate_from_context, load_pipeline
from smoke_vd_pipeline import ANNOTS_PATH, IMAGES_H5, load_image_and_caption

TARGETS_H5 = "data/targets/targets.h5"
SHARED1000_MASK_PATH = "data/targets/shared1000_mask.npy"
CKPT_PATH = "data/targets/a_psi_checkpoint.pt"
OUT_DIR = "data/gate0"


def label_panel(panel, text):
    """Stamp a small readable label at the bottom of a 224x224 panel (white text, black bar)."""
    panel = panel.copy()
    draw = ImageDraw.Draw(panel)
    bar_height = 16
    draw.rectangle([0, panel.height - bar_height, panel.width, panel.height], fill=(0, 0, 0))
    draw.text((3, panel.height - bar_height + 2), text, fill=(255, 255, 255))
    return panel


def pick_gate_indices(n_images):
    mask = np.load(SHARED1000_MASK_PATH)
    shared_idx = np.where(mask)[0]
    annots = np.load(ANNOTS_PATH, allow_pickle=True)
    has_caption = (annots != "").any(axis=1)
    shared_idx = [i for i in shared_idx if has_caption[i]]
    return sorted(shared_idx[:n_images])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=6)
    ap.add_argument("--text-to-image-strength", type=float, default=0.75)
    ap.add_argument("--strengths", type=str, default=None,
                     help="comma-separated text_to_image_strength values to compare for the "
                          "A_psi-context panel only, e.g. 0.75,0.5,0.25,0.0 -- one extra panel "
                          "per value, same seed/z_sem/z_struct each time. Overrides "
                          "--text-to-image-strength for those panels; the TRUE-context panel "
                          "still uses --text-to-image-strength alone, unswept. Default: just "
                          "the single --text-to-image-strength value (old single-panel behavior).")
    ap.add_argument("--guidance-scale", type=float, default=7.5)
    ap.add_argument("--num-inference-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", type=str, default=CKPT_PATH)
    ap.add_argument("--targets", type=str, default=TARGETS_H5,
                     help="assembled targets.h5 variant (assemble_targets.py --struct-d)")
    ap.add_argument("--out-dir", type=str, default=OUT_DIR)
    args = ap.parse_args()

    strengths = [float(s) for s in args.strengths.split(",")] if args.strengths else [args.text_to_image_strength]

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device)
    with h5py.File(args.targets, "r") as f:
        targets_struct_channels = f["z_struct"].shape[-1]
        targets_sem_dim = f["z_sem"].shape[-1]
    struct_channels = ckpt.get("struct_channels", targets_struct_channels)
    sem_dim = ckpt.get("sem_dim", targets_sem_dim)
    assert struct_channels == targets_struct_channels, (
        f"checkpoint was trained with struct_channels={struct_channels}, but --targets "
        f"{args.targets} has z_struct channels={targets_struct_channels} -- mismatched pair"
    )
    assert sem_dim == targets_sem_dim, (
        f"checkpoint was trained with sem_dim={sem_dim}, but --targets "
        f"{args.targets} has z_sem dim={targets_sem_dim} -- mismatched pair"
    )

    a_psi = APsiMinimal(struct_channels=struct_channels, sem_dim=sem_dim).to(device)
    a_psi.load_state_dict(ckpt["state_dict"])
    a_psi.eval()

    pipe = load_pipeline(device=device)
    neg_prompt_embeds, neg_image_embeddings = encode_unconditional_context(pipe, device)

    with h5py.File(args.targets, "r") as targets_f:
        gate_indices = pick_gate_indices(args.n_images)
        print(f"gate images (shared1000, held out from A_psi training): {gate_indices}")

        for idx in gate_indices:
            pil_image, caption = load_image_and_caption(idx)
            generator = torch.Generator(device=device).manual_seed(args.seed)

            true_prompt_embeds, true_image_embeddings = encode_true_context(pipe, pil_image, caption, device, do_cfg=True)
            true_out = generate_from_context(
                pipe,
                true_prompt_embeds,
                true_image_embeddings,
                text_to_image_strength=args.text_to_image_strength,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
            )[0]

            z_sem = torch.from_numpy(targets_f["z_sem"][idx].astype(np.float32)).unsqueeze(0).to(device)
            z_struct = torch.from_numpy(targets_f["z_struct"][idx].astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_prompt_embeds, pred_image_embeddings = a_psi(z_sem, z_struct)
            cfg_prompt_embeds = torch.cat([neg_prompt_embeds, pred_prompt_embeds.to(neg_prompt_embeds.dtype)])
            cfg_image_embeddings = torch.cat([neg_image_embeddings, pred_image_embeddings.to(neg_image_embeddings.dtype)])

            a_psi_panels = []
            for s in strengths:
                generator = torch.Generator(device=device).manual_seed(args.seed)
                a_psi_out = generate_from_context(
                    pipe,
                    cfg_prompt_embeds,
                    cfg_image_embeddings,
                    text_to_image_strength=s,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator,
                )[0]
                a_psi_panels.append(label_panel(a_psi_out.resize((224, 224)), f"A_psi s={s:g}"))

            n_panels = 2 + len(a_psi_panels)
            grid = Image.new("RGB", (224 * n_panels, 224))
            grid.paste(label_panel(pil_image.resize((224, 224)), "input"), (0, 0))
            grid.paste(label_panel(true_out.resize((224, 224)), f"true s={args.text_to_image_strength:g}"), (224, 0))
            for i, panel in enumerate(a_psi_panels):
                grid.paste(panel, (224 * (2 + i), 0))

            suffix = "_strengths" if args.strengths else ""
            out_path = os.path.join(args.out_dir, f"gate0_idx{idx}_seed{args.seed}{suffix}.png")
            grid.save(out_path)
            print(f"saved {out_path} (input | true-context | A_psi-context x {len(a_psi_panels)} strength(s))")

    print("Done.")


if __name__ == "__main__":
    main()
