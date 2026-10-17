"""Assemble the frozen training targets (z_sem, z_struct) into one HDF5. Glue only: CPU, under a minute.

Inputs (under data/targets/): pca_zsem/d128/{z_sem_img.npy, pca_basis.npz} (`pca_width_zsem.py`),
  clip_vitl14_text_raw_{source}.npy (`extract_clip_text_raw.py`), pca_zstruct/z_struct.npy (`fit_pca_zstruct.py`),
  shared1000_mask.npy (carried through: marks the held-out NSD-core test rows)
Outputs: targets.h5 (blip2) or targets_humantxt.h5, with z_sem (73000, 2, 128), z_struct, shared1000_mask.

z_sem^txt = L2(P_sem @ CLIP_text(caption)): the frozen image basis is applied, never fitted; no text-specific
PCA exists. blip2 is the default because it works for any image; human captions only exist for COCO (the
retrieval comparison of Section 5.2). --sem-d / --struct-d pick a width of the sweeps; --no-whiten uses the
un-whitened bases and writes a suffixed file; the whitened targets.h5 family is never overwritten.

Usage (from the trisect/ folder):
    python scripts/assemble_targets.py --text-source blip2 --sem-d 256
"""
import argparse
import os

import h5py
import numpy as np
from trisect.targets import l2_normalize

SHARED_MASK_PATH = "data/targets/shared1000_mask.npy"
SEM_D_DEFAULT = 128     # default width; matches pca_width_zsem.py's component sweep
STRUCT_D_DEFAULT = 32   # default width; matches fit_pca_zstruct.py's D_OUT_DEFAULT


def main():
    args = argparse.ArgumentParser()
    args.add_argument("--text-source", choices=["human", "blip2"], default="blip2")
    args.add_argument("--struct-d", type=int, default=STRUCT_D_DEFAULT,
                       help="which pca_zstruct{,_d<D>}/ variant to assemble (default 32, the spec value)")
    args.add_argument("--sem-d", type=int, default=SEM_D_DEFAULT,
                       help="which pca_zsem/d<D>/ variant to assemble (default 128, the architecture value)")
    args.add_argument("--no-whiten", action="store_true",
                       help="assemble from the un-whitened pca_zsem_nowhiten/pca_zstruct_nowhiten "
                            "bases instead of the default whitened ones (both must already exist, "
                            "see pca_width_zsem.py/fit_pca_zstruct.py --no-whiten)")
    args = args.parse_args()
    whiten = not args.no_whiten

    zsem_base = "data/targets/pca_zsem" if whiten else "data/targets/pca_zsem_nowhiten"
    z_sem_img_path = os.path.join(zsem_base, f"d{args.sem_d}", "z_sem_img.npy")
    p_sem_basis_path = os.path.join(zsem_base, f"d{args.sem_d}", "pca_basis.npz")

    struct_base = "pca_zstruct" if whiten else "pca_zstruct_nowhiten"
    struct_dir = f"data/targets/{struct_base}" if args.struct_d == STRUCT_D_DEFAULT \
        else f"data/targets/{struct_base}_d{args.struct_d}"
    z_struct_path = os.path.join(struct_dir, "z_struct.npy")

    clip_text_raw_path = f"data/targets/clip_vitl14_text_raw_{args.text_source}.npy"
    name_bits = ["targets"]
    if not whiten:
        name_bits.append("nowhiten")
    if args.text_source != "blip2":
        name_bits.append("humantxt")
    if args.struct_d != STRUCT_D_DEFAULT:
        name_bits.append(f"structd{args.struct_d}")
    if args.sem_d != SEM_D_DEFAULT:
        name_bits.append(f"semd{args.sem_d}")
    out_path = "data/targets/" + "_".join(name_bits) + ".h5"

    print(f"Loading z_sem^img: {z_sem_img_path}")
    z_sem_img = np.load(z_sem_img_path)  # (N, sem_d) float16, already L2-normalized
    n = z_sem_img.shape[0]
    assert z_sem_img.shape == (n, args.sem_d)

    print(f"Loading frozen P_sem basis: {p_sem_basis_path}")
    basis = np.load(p_sem_basis_path)
    components = basis["components"].astype(np.float32)  # (128, 768)
    mean = basis["mean"].astype(np.float32)               # (768,)
    explained_variance = basis["explained_variance"].astype(np.float32)  # (128,), for whitening
    basis_whiten = bool(basis["whiten"])
    assert basis_whiten == whiten, (
        f"{p_sem_basis_path} was fit with whiten={basis_whiten}, but --no-whiten requests "
        f"whiten={whiten} -- mismatched basis/flag pair"
    )

    print(f"Loading raw CLIP text embeddings ({args.text_source}): {clip_text_raw_path}")
    text_raw = np.load(clip_text_raw_path).astype(np.float32)  # (N, 768)
    assert text_raw.shape[0] == n, f"text_raw has {text_raw.shape[0]} rows, expected {n}"

    print(f"Projecting text stream through the frozen (image-fit) P_sem basis (whiten={whiten}) ...")
    centered = text_raw - mean[None, :]
    projected = centered @ components.T  # (N, 128)
    if whiten:
        projected = projected / np.sqrt(np.clip(explained_variance, 1e-12, None))[None, :]
    z_sem_txt = l2_normalize(projected).astype(np.float16)

    z_sem = np.stack([z_sem_img, z_sem_txt], axis=1)  # (N, 2, 128)
    print(f"z_sem: {z_sem.shape} {z_sem.dtype}")

    print(f"Loading z_struct (memmap): {z_struct_path}")
    z_struct = np.load(z_struct_path, mmap_mode="r")
    assert z_struct.shape[0] == n, f"z_struct has {z_struct.shape[0]} rows, expected {n}"
    print(f"z_struct: {z_struct.shape} {z_struct.dtype}")

    print(f"Loading shared1000 mask: {SHARED_MASK_PATH}")
    shared_mask = np.load(SHARED_MASK_PATH)
    assert shared_mask.shape == (n,)

    print(f"Writing {out_path} ...")
    with h5py.File(out_path, "w") as f:
        f.create_dataset("z_sem", data=z_sem, dtype="float16")
        f.create_dataset("z_struct", data=z_struct[:], dtype="float16")
        f.create_dataset("shared1000_mask", data=shared_mask, dtype="bool")
        f.attrs["z_sem_order"] = "img, txt (axis 1)"
        f.attrs["z_sem_d"] = args.sem_d
        f.attrs["z_sem_txt_source"] = args.text_source
        f.attrs["z_struct_shared_pca"] = True
        f.attrs["z_struct_d"] = args.struct_d
        f.attrs["shared1000_excluded_from_all_fits"] = True
        f.attrs["whiten"] = whiten

    print("Done.")


if __name__ == "__main__":
    main()
