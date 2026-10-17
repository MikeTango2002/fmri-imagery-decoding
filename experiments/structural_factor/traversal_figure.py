"""Traversal figure: fix z_struct and sweep z_sem, then fix z_sem and sweep z_struct (Section 8.1.4).

Inputs:  the proxy set (make_proxy_set.py), the frozen PCA bases of z_sem and z_struct, a_psi_checkpoint.pt
Outputs: data/traversal_figure/<panel>/{ref.png, srcNN.png, genNN.png}, <panel>_grid.png (labelled montages),
         manifest.json (proxy indices, angles, seeds, Versatile Diffusion settings)

The visual counterpart of the cross-regression R^2 = 0.465 (z_sem -> z_struct): how that redundancy shows in generated
images, and whether z_struct carries orientation on proxy-simple bars (the `angle` of the manifest). No pass/fail
criterion; nothing quantitative is read off the figure. One seed for every column, and a visible difference comes
from the swapped latent, not from the noise draw.

Usage (GPU, .venv_targets active, after make_proxy_set.py):
    python traversal_figure.py
    python traversal_figure.py --n-cols 4 --num-inference-steps 30   # faster smoke test
"""
import argparse
import json
import os

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from adapter_model import APsiMinimal
# The z_sem/z_struct encoder lives in the script that writes those targets to disk for the whole
# proxy set; the formulas have one copy.
from extract_proxy_targets import N_COMPLEX, TargetEncoder
from vd_context import encode_unconditional_context, generate_from_context, load_pipeline

PROXY_DIR_DEFAULT = "data/proxy_set"
PCA_ZSEM_DIR_DEFAULT = "data/targets/pca_zsem/d128"
PCA_ZSTRUCT_DIR_DEFAULT = "data/targets/pca_zstruct"
CKPT_DEFAULT = "data/targets/a_psi_checkpoint.pt"
TARGETS_H5_DEFAULT = "data/targets/targets.h5"
OUT_DIR_DEFAULT = "data/traversal_figure"
N_COLS_DEFAULT = 6
FAMILY_ORDER = ["bar", "cross", "shape_filled", "shape_hollow", "circle"]

GUIDANCE_SCALE_DEFAULT = 7.5
NUM_INFERENCE_STEPS_DEFAULT = 50
MIX_DEFAULT = 0.4  # brain_diffuser's own constant, same default reconstruct.py uses


def load_manifest(proxy_dir):
    with open(os.path.join(proxy_dir, "manifest.json")) as f:
        return json.load(f)


def load_proxy_image(proxy_dir, family, index):
    return Image.open(os.path.join(proxy_dir, "images", family, f"{index:03d}.png")).convert("RGB")


def pick_evenly_spaced(n_total, k, offset=0):
    idx = np.linspace(0, n_total - 1, k).round().astype(int)
    return [int(i) for i in np.clip(idx + offset, 0, n_total - 1)]


def select_panels(manifest, n_cols):
    """Returns a dict panel_name -> {family, ref_family_dir, ref_index, ref_is_struct,
    varying_indices, varying_labels}. No brain data, no PCA yet, just which proxy-set images
    each panel needs; encoding can be done once over the union."""
    simple = manifest["simple"]
    by_family = {fam: sorted(e["index"] for e in simple if e["family"] == fam) for fam in FAMILY_ORDER}

    bars = sorted((e for e in simple if e["family"] == "bar"), key=lambda e: e["angle"])
    bar_pick_ranks = pick_evenly_spaced(len(bars), n_cols)
    bar_picks = [bars[r] for r in bar_pick_ranks]

    sem_sweep_indices = [by_family[fam][0] for fam in FAMILY_ORDER]  # one per family, "category changes"
    struct_ref_index = by_family["shape_filled"][1 % len(by_family["shape_filled"])]  # a filled polygon, distinct from sem_sweep's own shape_filled[0]
    rotate_sem_ref_index = by_family["circle"][1 % len(by_family["circle"])]  # untouched by the struct sweep -> not tautological

    n_complex_a = pick_evenly_spaced(N_COMPLEX, n_cols, offset=3)
    n_complex_b = pick_evenly_spaced(N_COMPLEX, n_cols, offset=17)
    complex_struct_ref = 0
    complex_sem_ref = 1

    # Panels, one pair per proxy family:
    #   simple_sweep_sem      z_struct fixed to a shape_filled image, z_sem swept over one image per family:
    #                         "category changes, layout persists" if the factors are disentangled.
    #   simple_rotate_struct  z_sem fixed to a circle (a family the struct sweep does not use; the panel is not
    #                         tautological), z_struct swept over --n-cols bars ordered by `angle`: "the object
    #                         rotates, the category persists". The one panel with an unambiguous orientation.
    #   complex_sweep_sem     the same idea on proxy-complex photographs, z_struct fixed to one photo.
    #   complex_vary_struct   z_sem fixed to one photo, z_struct swept over other photos (no orientation metadata,
    #                         so a vary-structure probe, not a rotation).
    return {
        "simple_sweep_sem": dict(
            family="simple", ref_is_struct=True, ref_index=struct_ref_index,
            varying_indices=sem_sweep_indices,
            varying_labels=FAMILY_ORDER,
        ),
        "simple_rotate_struct": dict(
            family="simple", ref_is_struct=False, ref_index=rotate_sem_ref_index,
            varying_indices=[e["index"] for e in bar_picks],
            varying_labels=[f"{e['angle']:.0f}deg" for e in bar_picks],
        ),
        "complex_sweep_sem": dict(
            family="complex", ref_is_struct=True, ref_index=complex_struct_ref,
            varying_indices=n_complex_a,
            varying_labels=[f"img{i:03d}" for i in n_complex_a],
        ),
        "complex_vary_struct": dict(
            family="complex", ref_is_struct=False, ref_index=complex_sem_ref,
            varying_indices=n_complex_b,
            varying_labels=[f"img{i:03d}" for i in n_complex_b],
        ),
    }


def label_row(images, labels, cell=128, margin_top=18):
    """One row of thumbnails with a small text label under each, used for both the source strip
    and the generated strip of a panel grid."""
    font = ImageFont.load_default()
    row = Image.new("RGB", (cell * len(images), cell + margin_top), (255, 255, 255))
    draw = ImageDraw.Draw(row)
    for i, (im, lbl) in enumerate(zip(images, labels)):
        row.paste(im.resize((cell, cell), Image.BILINEAR), (i * cell, 0))
        draw.text((i * cell + 4, cell + 2), lbl, fill=(0, 0, 0), font=font)
    return row


def make_panel_grid(panel_name, ref_image, ref_label, source_images, source_labels, gen_images):
    ref_col = label_row([ref_image], [ref_label])
    src_row = label_row(source_images, source_labels)
    gen_row = label_row(gen_images, [f"gen {i}" for i in range(len(gen_images))])

    gap = 16
    w = ref_col.width + gap + src_row.width
    h = max(src_row.height + gen_row.height, ref_col.height)
    grid = Image.new("RGB", (w, h), (255, 255, 255))
    grid.paste(ref_col, (0, 0))
    grid.paste(src_row, (ref_col.width + gap, 0))
    grid.paste(gen_row, (ref_col.width + gap, src_row.height))
    draw = ImageDraw.Draw(grid)
    draw.text((4, h - 12), panel_name, fill=(120, 120, 120), font=ImageFont.load_default())
    return grid


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proxy-dir", default=PROXY_DIR_DEFAULT)
    ap.add_argument("--pca-zsem-dir", default=PCA_ZSEM_DIR_DEFAULT)
    ap.add_argument("--pca-zstruct-dir", default=PCA_ZSTRUCT_DIR_DEFAULT)
    ap.add_argument("--ckpt", default=CKPT_DEFAULT)
    ap.add_argument("--targets", default=TARGETS_H5_DEFAULT,
                     help="only read for z_sem/z_struct dim cross-check, same as reconstruct.py")
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--n-cols", type=int, default=N_COLS_DEFAULT,
                     help="columns for the rotate/vary-struct panels and the complex sem-sweep "
                          "panel; the simple sem-sweep panel always uses one column per family (5)")
    ap.add_argument("--mixing", type=float, default=MIX_DEFAULT)
    ap.add_argument("--guidance-scale", type=float, default=GUIDANCE_SCALE_DEFAULT)
    ap.add_argument("--num-inference-steps", type=int, default=NUM_INFERENCE_STEPS_DEFAULT)
    ap.add_argument("--seed", type=int, default=0, help="same seed reused for every column of a "
                     "given panel, so differences reflect the latent, not sampling noise")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = load_manifest(args.proxy_dir)
    panels = select_panels(manifest, args.n_cols)

    encoder = TargetEncoder(args.pca_zsem_dir, args.pca_zstruct_dir, device)

    with h5py.File(args.targets, "r") as f:
        struct_channels = f["z_struct"].shape[-1]
        sem_dim = f["z_sem"].shape[-1]
    ckpt = torch.load(args.ckpt, map_location=device)
    a_psi = APsiMinimal(
        struct_channels=ckpt.get("struct_channels", struct_channels),
        sem_dim=ckpt.get("sem_dim", sem_dim),
    ).to(device)
    a_psi.load_state_dict(ckpt["state_dict"])
    a_psi.eval()

    print("Loading Versatile Diffusion pipeline ...")
    pipe = load_pipeline(device=device)
    neg_prompt_embeds, neg_image_embeddings = encode_unconditional_context(pipe, device)

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_out = {"seed": args.seed, "mixing": args.mixing,
                     "guidance_scale": args.guidance_scale,
                     "num_inference_steps": args.num_inference_steps, "panels": {}}

    for panel_name, spec in panels.items():
        family = spec["family"]
        print(f"\n=== {panel_name} ({family}, {len(spec['varying_indices'])} columns) ===")
        panel_dir = os.path.join(args.out_dir, panel_name)
        os.makedirs(panel_dir, exist_ok=True)

        ref_image = load_proxy_image(args.proxy_dir, family, spec["ref_index"])
        source_images = [load_proxy_image(args.proxy_dir, family, i) for i in spec["varying_indices"]]

        z_sem_ref, z_struct_ref = encoder.encode([ref_image])
        z_sem_varying, z_struct_varying = encoder.encode(source_images)

        ref_image.save(os.path.join(panel_dir, "ref.png"))
        gen_images = []
        for j in range(len(source_images)):
            source_images[j].save(os.path.join(panel_dir, f"src{j:02d}.png"))
            if spec["ref_is_struct"]:
                z_sem_used, z_struct_used = z_sem_varying[j], z_struct_ref[0]
            else:
                z_sem_used, z_struct_used = z_sem_ref[0], z_struct_varying[j]

            # z_sem_txt is not computed (the proxy captions did not exist when the figure was made, and nothing
            # quantitative is read off it): z_sem_img fills both streams of the adapter input. The adapter was trained
            # on real image/text pairs; this is mildly off-distribution, but the streams are highly correlated.
            z_sem_t = torch.from_numpy(np.stack([z_sem_used, z_sem_used])[None]).to(device)  # (1,2,128), img-stream reused as txt-stream
            z_struct_t = torch.from_numpy(z_struct_used[None]).to(device)  # (1,8,8,32)
            with torch.no_grad():
                prompt_embeds, image_embeddings = a_psi(z_sem_t, z_struct_t)
            cfg_prompt = torch.cat([neg_prompt_embeds, prompt_embeds.to(neg_prompt_embeds.dtype)])
            cfg_image = torch.cat([neg_image_embeddings, image_embeddings.to(neg_image_embeddings.dtype)])

            # No z_low and no VDVAE: init_image=None denoises from pure noise (strength 1.0, the context-only path
            # of sample_adapter_vs_true_context.py). That shows what z_sem and z_struct alone put into the image.
            generator = torch.Generator(device=device).manual_seed(args.seed)
            out = generate_from_context(
                pipe, cfg_prompt, cfg_image,
                text_to_image_strength=args.mixing,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                generator=generator, init_image=None,
            )[0]
            out.save(os.path.join(panel_dir, f"gen{j:02d}.png"))
            gen_images.append(out)
            print(f"  col {j} ({spec['varying_labels'][j]}) done", flush=True)

        ref_label = "FIXED struct" if spec["ref_is_struct"] else "FIXED sem"
        grid = make_panel_grid(panel_name, ref_image, ref_label, source_images,
                                spec["varying_labels"], gen_images)
        grid.save(os.path.join(args.out_dir, f"{panel_name}_grid.png"))

        manifest_out["panels"][panel_name] = {
            "family": family,
            "ref_is_struct": spec["ref_is_struct"],
            "ref_index": spec["ref_index"],
            "varying_indices": spec["varying_indices"],
            "varying_labels": spec["varying_labels"],
        }

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest_out, f, indent=2)
    print(f"\nDone. Panels under {args.out_dir}/")


if __name__ == "__main__":
    main()
