"""Build the proxy stimulus set of the oracle budget: 150 synthetic simple images, 150 real photographs.

Inputs:  data/external_coco/val2017_unused_by_nsd/ and its candidates manifest, data/targets/nsd73k_coco_ids.npy
Outputs: data/proxy_set/{images/, blocks/block{b:02d}_gt.pt, manifest.json, contact_sheet.png}
         25 blocks of 18 images (3, 425, 425) in [0, 1]; the harness reads them with `--model template`

The proxy set is a benchmark-shaped stimulus set that is not the 18 real stimuli: choosing settings on those
would be selection on the test set. proxy-simple (bars, polygons, crosses, circles on a uniform background)
is the family where TRISECT loses to Brain-Diffuser. proxy-complex is COCO val2017 that NSD never used,
so it is outside every PCA and adapter fit and outside shared1000. Slots 12-17 repeat slots 6-11 and are
ignored: the harness truncates to the first 12. Chance for two-way identification is 40.91% (10/11).

Usage (from the folder that holds data/):
    python make_proxy_set.py
"""
import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw

IMAGES_H5_DEFAULT = "data/mindeyev2_dataset/coco_images_224_float16.hdf5"
SHARED1000_MASK_DEFAULT = "data/targets/shared1000_mask.npy"
NSD73K_COCO_IDS_DEFAULT = "data/targets/nsd73k_coco_ids.npy"
EXTERNAL_COCO_DIR_DEFAULT = "data/external_coco/val2017_unused_by_nsd"
EXTERNAL_COCO_MANIFEST_DEFAULT = "data/external_coco/candidates_manifest.json"
OUT_DIR_DEFAULT = "data/proxy_set"

STIM_SIZE = 425     # matches imagery_stimuli_18.pt's native resolution
N_SIMPLE = 150
N_COMPLEX = 150
N_STIM = 6          # per block, per family; matches final_evaluations_imagery.py's N_STIM
SEED = 0

FAMILIES = ["bar", "shape_filled", "shape_hollow", "cross", "circle"]


# ------------------------------------------------------------- proxy-simple ---

def rand_color(rng, avoid=None, min_dist=80.0):
    """Uniform random RGB, resampled until it is visually distinct from `avoid` (if given) --
    otherwise a low-contrast draw could make a shape nearly invisible against its background."""
    while True:
        c = rng.integers(0, 256, size=3)
        if avoid is None or float(np.linalg.norm(c.astype(np.float64) - np.asarray(avoid))) >= min_dist:
            return tuple(int(x) for x in c)


def _bar_corners(cx, cy, angle_deg, length, width):
    theta = np.radians(angle_deg)
    dx, dy = np.cos(theta), np.sin(theta)
    nx, ny = -dy, dx
    hl, hw = length / 2, width / 2
    return [
        (cx + dx * hl + nx * hw, cy + dy * hl + ny * hw),
        (cx + dx * hl - nx * hw, cy + dy * hl - ny * hw),
        (cx - dx * hl - nx * hw, cy - dy * hl - ny * hw),
        (cx - dx * hl + nx * hw, cy - dy * hl + ny * hw),
    ]


def make_simple_image(rng, family):
    """One STIM_SIZE x STIM_SIZE synthetic RGB image, one shape family, uniform background."""
    bg = rand_color(rng)
    img = Image.new("RGB", (STIM_SIZE, STIM_SIZE), bg)
    draw = ImageDraw.Draw(img)
    fg = rand_color(rng, avoid=bg)
    cx = STIM_SIZE / 2 + float(rng.integers(-40, 41))
    cy = STIM_SIZE / 2 + float(rng.integers(-40, 41))

    if family == "bar":
        angle = float(rng.uniform(0, 180))
        length = float(rng.uniform(0.45, 0.85)) * STIM_SIZE
        width = float(rng.uniform(15, 45))
        draw.polygon(_bar_corners(cx, cy, angle, length, width), fill=fg)
        params = {"angle": angle, "length": length, "width": width}

    elif family in ("shape_filled", "shape_hollow"):
        n_sides = int(rng.choice([3, 4, 5, 6, 8]))
        radius = float(rng.uniform(60, 150))
        rotation = float(rng.uniform(0, 360))
        if family == "shape_filled":
            draw.regular_polygon((cx, cy, radius), n_sides=n_sides, rotation=rotation, fill=fg)
            params = {"n_sides": n_sides, "radius": radius, "rotation": rotation}
        else:
            stroke = int(rng.integers(6, 18))
            draw.regular_polygon((cx, cy, radius), n_sides=n_sides, rotation=rotation,
                                 outline=fg, width=stroke)
            params = {"n_sides": n_sides, "radius": radius, "rotation": rotation, "stroke": stroke}

    elif family == "cross":
        angle = float(rng.uniform(0, 90))
        length = float(rng.uniform(0.5, 0.85)) * STIM_SIZE
        width = float(rng.uniform(20, 45))
        draw.polygon(_bar_corners(cx, cy, angle, length, width), fill=fg)
        draw.polygon(_bar_corners(cx, cy, angle + 90, length, width), fill=fg)
        params = {"angle": angle, "length": length, "width": width}

    elif family == "circle":
        radius = float(rng.uniform(50, 160))
        filled = bool(rng.integers(0, 2))
        bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
        if filled:
            draw.ellipse(bbox, fill=fg)
            params = {"radius": radius, "filled": True}
        else:
            stroke = int(rng.integers(6, 18))
            draw.ellipse(bbox, outline=fg, width=stroke)
            params = {"radius": radius, "filled": False, "stroke": stroke}

    else:
        raise ValueError(family)

    meta = {"family": family, "bg": list(bg), "fg": list(fg), "cx": cx, "cy": cy, **params}
    return img, meta


def make_all_simple(rng, n):
    counts = [n // len(FAMILIES)] * len(FAMILIES)
    for i in range(n - sum(counts)):
        counts[i] += 1
    imgs, metas = [], []
    for family, count in zip(FAMILIES, counts):
        for _ in range(count):
            img, meta = make_simple_image(rng, family)
            imgs.append(img)
            metas.append(meta)
    order = rng.permutation(len(imgs))  # interleave families instead of one block each
    return [imgs[i] for i in order], [metas[i] for i in order]


# ------------------------------------------------------------ proxy-complex ---

def make_all_complex_external(rng, external_dir: str, manifest_path: str, n: int):
    """Default source: COCO val2017 images that NSD never put in its 73k pool at all.
    The candidate list and the folder of jpgs come from the procedure in the README."""
    manifest = json.load(open(manifest_path))
    coco_ids = sorted(c["coco_id"] for c in manifest)
    assert len(coco_ids) >= n, f"{manifest_path} has only {len(coco_ids)} candidates, need {n}"
    chosen = np.sort(rng.choice(coco_ids, size=n, replace=False))

    imgs = []
    for cid in chosen:
        p = Path(external_dir) / f"{int(cid):012d}.jpg"
        imgs.append(Image.open(p).convert("RGB").resize((STIM_SIZE, STIM_SIZE), Image.BILINEAR))
    return imgs, chosen


def make_all_complex(rng, images_h5: str, mask_path: str, n: int):
    """Earlier source: sampled from shared1000. Kept only as an explicit
    `--complex-source shared1000` reference. It is not the default because it spends the
    shared1000 reservation and does not hold out the PCA and adapter training pool as intended
    (see the module docstring)."""
    mask = np.load(mask_path)
    shared_ids = np.flatnonzero(mask)
    assert len(shared_ids) == 1000, f"{mask_path}: {len(shared_ids)} True entries, expected 1000"
    chosen = np.sort(rng.choice(shared_ids, size=n, replace=False))

    with h5py.File(images_h5, "r") as f:
        raw = f["images"][chosen.tolist()]  # (n,3,224,224) float16 in [0,1]
    raw = raw.astype(np.float32)

    imgs = []
    for row in raw:
        arr = (np.transpose(row, (1, 2, 0)) * 255.0).clip(0, 255).astype(np.uint8)
        imgs.append(Image.fromarray(arr).resize((STIM_SIZE, STIM_SIZE), Image.BILINEAR))
    return imgs, chosen


# --------------------------------------------------------------- block gts ---

def pil_to_chw_float(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def write_blocks(simple_imgs, complex_imgs, out_dir: Path):
    n_blocks = len(simple_imgs) // N_STIM
    assert len(simple_imgs) == n_blocks * N_STIM and len(complex_imgs) == n_blocks * N_STIM
    (out_dir / "blocks").mkdir(parents=True, exist_ok=True)
    for b in range(n_blocks):
        s = torch.stack([pil_to_chw_float(im) for im in simple_imgs[b * N_STIM:(b + 1) * N_STIM]])
        c = torch.stack([pil_to_chw_float(im) for im in complex_imgs[b * N_STIM:(b + 1) * N_STIM]])
        gt = torch.cat([s, c, c], dim=0)  # (18,3,425,425); slots 12-17 are the inert placeholder
        assert tuple(gt.shape) == (18, 3, STIM_SIZE, STIM_SIZE)
        assert -1e-4 <= float(gt.min()) and float(gt.max()) <= 1.0 + 1e-4
        torch.save(gt, out_dir / "blocks" / f"block{b:02d}_gt.pt")
    return n_blocks


# ------------------------------------------------------------- contact sheet ---

def make_contact_sheet(simple_imgs, complex_imgs, out_path: Path, thumb=48, cols=15):
    def grid(imgs):
        rows = (len(imgs) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * thumb, rows * thumb), (30, 30, 30))
        for i, im in enumerate(imgs):
            r, c = divmod(i, cols)
            sheet.paste(im.resize((thumb, thumb), Image.BILINEAR), (c * thumb, r * thumb))
        return sheet

    simple_grid, complex_grid = grid(simple_imgs), grid(complex_imgs)
    gap = 12
    w = max(simple_grid.width, complex_grid.width)
    h = simple_grid.height + complex_grid.height + gap
    sheet = Image.new("RGB", (w, h), (255, 255, 255))
    sheet.paste(simple_grid, (0, 0))
    sheet.paste(complex_grid, (0, simple_grid.height + gap))
    sheet.save(out_path)


# ------------------------------------------------------------------- main ---

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-h5", default=IMAGES_H5_DEFAULT)
    ap.add_argument("--mask", default=SHARED1000_MASK_DEFAULT)
    ap.add_argument("--complex-source", choices=["external_coco", "shared1000"],
                    default="external_coco",
                    help="external_coco (default): COCO val2017 images outside NSD's "
                         "73k pool. shared1000: reference of the earlier version only, see module docstring.")
    ap.add_argument("--nsd73k-coco-ids", default=NSD73K_COCO_IDS_DEFAULT)
    ap.add_argument("--external-dir", default=EXTERNAL_COCO_DIR_DEFAULT)
    ap.add_argument("--external-manifest", default=EXTERNAL_COCO_MANIFEST_DEFAULT)
    ap.add_argument("--out-dir", default=OUT_DIR_DEFAULT)
    ap.add_argument("--n-simple", type=int, default=N_SIMPLE)
    ap.add_argument("--n-complex", type=int, default=N_COMPLEX)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    if args.complex_source == "shared1000" and not Path(args.mask).exists():
        raise SystemExit(f"missing {args.mask} -- run derive_shared1000_mask.py first")
    if args.complex_source == "external_coco" and not Path(args.nsd73k_coco_ids).exists():
        raise SystemExit(f"missing {args.nsd73k_coco_ids} -- run derive_nsd73k_coco_ids.py first")
    if args.n_simple % N_STIM or args.n_complex % N_STIM:
        raise SystemExit(f"--n-simple/--n-complex must be multiples of {N_STIM}")

    out_dir = Path(args.out_dir)
    (out_dir / "images" / "simple").mkdir(parents=True, exist_ok=True)
    (out_dir / "images" / "complex").mkdir(parents=True, exist_ok=True)

    rng_simple = np.random.default_rng(args.seed)
    rng_complex = np.random.default_rng(args.seed + 1)

    print(f"generating {args.n_simple} proxy-simple images ({FAMILIES})...")
    simple_imgs, simple_metas = make_all_simple(rng_simple, args.n_simple)
    for i, img in enumerate(simple_imgs):
        img.save(out_dir / "images" / "simple" / f"{i:03d}.png")

    print(f"sampling {args.n_complex} proxy-complex images ({args.complex_source})...")
    if args.complex_source == "external_coco":
        complex_imgs, complex_ids = make_all_complex_external(
            rng_complex, args.external_dir, args.external_manifest, args.n_complex)
    else:
        complex_imgs, complex_ids = make_all_complex(rng_complex, args.images_h5, args.mask,
                                                     args.n_complex)
    for i, img in enumerate(complex_imgs):
        img.save(out_dir / "images" / "complex" / f"{i:03d}.png")

    # Automated membership check. proxy-simple has no 73k id at all; it cannot appear in any
    # fit by construction: there is nothing to check beyond "it is not read from the 73k pool".
    if args.complex_source == "external_coco":
        # Every proxy-complex cocoId must be OUTSIDE the full 73k pool, i.e. never used by anyone,
        # in any subject, to fit the PCA bases or to train the adapter.
        nsd73k_ids = np.load(args.nsd73k_coco_ids)
        leaked = np.isin(complex_ids, nsd73k_ids)
        assert not leaked.any(), (
            f"{int(leaked.sum())} proxy-complex cocoIds ARE in NSD's 73k pool -- "
            "they could leak into a PCA fit or A_psi training")
        assert len(set(complex_ids.tolist())) == len(complex_ids), "duplicate proxy-complex ids"
        print(f"  not-in-73k-pool check PASSED: none of the {len(complex_ids)} proxy-complex "
              f"cocoIds are in NSD's 73k pool (verified against {args.nsd73k_coco_ids})")
    else:
        mask = np.load(args.mask)
        in_shared1000 = mask[complex_ids]
        assert bool(in_shared1000.all()), (
            f"{int((~in_shared1000).sum())} proxy-complex ids are NOT in shared1000 -- "
            "they could leak into a PCA fit or A_psi training")
        assert len(set(complex_ids.tolist())) == len(complex_ids), "duplicate proxy-complex ids"
        print(f"  PCA-exclusion check PASSED: all {len(complex_ids)} proxy-complex ids "
              f"are in shared1000 (verified against {args.mask})")

    print("writing blocks...")
    n_blocks = write_blocks(simple_imgs, complex_imgs, out_dir)

    print("writing contact sheet...")
    make_contact_sheet(simple_imgs, complex_imgs, out_dir / "contact_sheet.png")

    manifest = {
        "seed": args.seed,
        "stim_size": STIM_SIZE,
        "n_simple": args.n_simple,
        "n_complex": args.n_complex,
        "n_stim_per_block": N_STIM,
        "n_blocks": n_blocks,
        "families": FAMILIES,
        "complex_source": args.complex_source,
        "shared1000_mask": str(args.mask) if args.complex_source == "shared1000" else None,
        "nsd73k_coco_ids": str(args.nsd73k_coco_ids) if args.complex_source == "external_coco" else None,
        "pca_exclusion_check": "PASSED",
        "note_2afc_chance": "40.91% (10/11), the harness's own off-by-one -- not computed here, "
                            "see final_evaluations_imagery.py",
        "simple": [{"index": i, **m} for i, m in enumerate(simple_metas)],
        "complex": [{"index": i, ("coco_id" if args.complex_source == "external_coco"
                                   else "img73k_id"): int(cid)}
                    for i, cid in enumerate(complex_ids)],
        "blocks": [{"block": b, "simple_indices": list(range(b * N_STIM, (b + 1) * N_STIM)),
                    "complex_indices": list(range(b * N_STIM, (b + 1) * N_STIM)),
                    "gt_path": f"blocks/block{b:02d}_gt.pt"} for b in range(n_blocks)],
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    total = args.n_simple + args.n_complex
    print(f"\nwrote {total} images ({args.n_simple} simple + {args.n_complex} complex), "
          f"{n_blocks} blocks, manifest and contact sheet to {out_dir}")


if __name__ == "__main__":
    main()
