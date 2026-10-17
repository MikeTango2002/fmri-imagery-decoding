"""The reconstruction step: brain-predicted (z_sem_img, z_sem_txt, z_struct, z_low) -> `--gen-rep` images per row.

Inputs:  `predict_ridge_imagery.py` (30 rows per subject) or `predict_ridge_shared1000.py` (--dataset nsdcore)
Outputs: <out-dir>/subj<NN>/<cond>/sample<NNN>/<stim>.png, the --path_template of the Kneeland harness

Per row: z_low -> PCA^-1 -> VDVAE decoder -> 64x64 init image; (z_sem, z_struct) -> adapter -> positive half
of VD's context, CFG-doubled with the fixed negative half; then frozen VD dual-guided generation. strength
0.75 and mixing 0.4 are Brain-Diffuser's constants, used by all five models (0.4 is the text weight).
nsdcore: shared1000 rows in "blockNN" pseudo-conditions of --block-size 12; --harness-blocks writes the harness
layout (6 simple + 6 complex per block) for `evaluate_block_set.py`. proxy: the 300 proxy images, no 73k id,
so only --oracle-targets ctx|all. --oracle-targets swaps the ridge prediction for the real image's target:
none=T3, all=T2 (T1 with --zlow-raw), ctx=T0 (Table 8.5). --ablate-factor zeroes or shuffles z_sem / z_struct.
Usage (from the trisect/ folder; needs a GPU):
    python scripts/reconstruct.py --subj 1 2 5 7 --gen-rep 10
"""
import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "vdvae_lib"))
from model_utils import load_vaes, set_up_data  # noqa: E402  (needs the sys.path insert above)

from decode_zlow_to_image import (  # noqa: E402
    RAW_LATENTS_PATH, RAW_LAYER_DIMS_PATH, decode_raw_batch, decode_zlow_batch,
    layer_bounds, layer_shapes, load_pca_basis, project_raw_to_zlow,
)

VDVAE_DECODE_CHUNK = 64  # decode_zlow_batch/decode_raw_batch hold the whole batch on the GPU
# at once (forward_manual_latents has no internal chunking); fine for the 30-row real-imagery
# legs but OOMs a 16GB card on shared1000's up-to-996-row harness-blocks batches (measured:
# 996 rows tried to allocate 7.78 GiB on top of the loaded VD+VDVAE models). Chunking here
# rather than inside decode_zlow_to_image.py, which other callers rely on staying untouched.
# This is a 16GB-card constraint, not a correctness change.


def decode_batched(fn, rows, *fn_args):
    """fn(rows_chunk, *fn_args) for each <=VDVAE_DECODE_CHUNK-row slice of `rows`, concatenated
    back along axis 0. fn is decode_zlow_batch or decode_raw_batch, both (n, ...) -> (n,64,64,3)
    uint8 numpy."""
    if len(rows) <= VDVAE_DECODE_CHUNK:
        return fn(rows, *fn_args)
    chunks = [fn(rows[i:i + VDVAE_DECODE_CHUNK], *fn_args)
              for i in range(0, len(rows), VDVAE_DECODE_CHUNK)]
    return np.concatenate(chunks, axis=0)
from extract_vdvae_latents import DEVICE, H_DICT, dotdict  # noqa: E402
from adapter_model import APsiMinimal  # noqa: E402
from vd_context import (  # noqa: E402
    encode_unconditional_context, generate_from_context, load_pipeline,
)

SUBJECTS = (1, 2, 5, 7)
BRANCHES = ("z_sem_img", "z_sem_txt", "z_struct", "z_low")
N_STIM = 6  # stimuli per condition, matches final_evaluations_imagery.py's N_STIM

PRED_DIR_DEFAULT = "data/betas_trisect/imagery_predictions"
NSDCORE_PRED_DIR_DEFAULT = "data/betas_trisect/nsdcore_predictions"
PCA_ZLOW_DIR_DEFAULT = "data/targets/pca_zlow"
CKPT_DEFAULT = "data/targets/a_psi_checkpoint.pt"
TARGETS_H5_DEFAULT = "data/targets/targets.h5"
A_PSI_TRUE_CONTEXT_DEFAULT = "data/targets/a_psi_true_context.h5"  # T0's real-context arm
OUT_DIR_DEFAULT = "data/imagery_recons_b3"
NSDCORE_OUT_DIR_DEFAULT = "data/nsdcore_recons_b3"
BLOCK_SIZE_DEFAULT = 12  # --dataset nsdcore only, see module docstring

# --dataset proxy: flat 0-299 index, 0-149 simple / 150-299 complex; must match
# extract_proxy_vd_context.py / extract_proxy_vdvae_latents.py / caption_proxy_set.py.
PROXY_DIR_DEFAULT = "data/proxy_set"
PROXY_CTX_DEFAULT = "data/targets/a_psi_true_context_proxyset.h5"
BLOCKSET_DIR_DEFAULT = "data/shared1000_blockset"
PROXY_TARGETS_DEFAULT = "data/targets/targets_proxyset.h5"
PROXY_RAW_LATENTS_DEFAULT = "data/targets/vdvae_latents_proxy_raw.npy"
PROXY_RAW_LAYER_DIMS_DEFAULT = "data/targets/vdvae_latents_proxy_raw_layer_dims.json"
PROXY_OUT_DIR_DEFAULT = "data/proxy_recons_b3"
PROXY_N_SIMPLE = 150

# Brain-Diffuser's own constants (versatilediffusion_reconstruct_images_imagery.py); see the module
# docstring for the mixed_ratio -> text_to_image_strength derivation.
DIFF_STRENGTH_DEFAULT = 0.75  # VDVAE init-image denoising strength (BD's diff_str)
MIX_DEFAULT = 0.4             # text weight in the dual-guided mix (BD's mixing == text_to_image_strength)
GUIDANCE_SCALE_DEFAULT = 7.5
NUM_INFERENCE_STEPS_DEFAULT = 50


def decode_pixel_batch(z_low, res, device):
    """z_low (n, res*res*3) raw pixel intensities in [0,255], channel-last flatten (same layout
    as extract_pixel_low_target.py) -> (n, 64, 64, 3) uint8, bilinear upsampled, pixel z_low's
    own decode, no VDVAE/PCA involved (module docstring, --zlow-mode pixel)."""
    n = z_low.shape[0]
    img = z_low.reshape(n, res, res, 3).astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(0, 3, 1, 2).contiguous().to(device)  # (n,3,res,res)
    img = F.interpolate(img, size=(64, 64), mode="bilinear", align_corners=False).clamp(0, 1)
    img = (img.permute(0, 2, 3, 1) * 255.0).round().clamp(0, 255).byte().cpu().numpy()
    return img


def load_oracle_rows_by_id(h5_dataset, ids):
    """Fancy-index an h5py dataset by a possibly-unsorted array of row ids, h5py's own fancy
    indexing requires strictly increasing, duplicate-free index lists; sort/dedupe here and
    remap back to the caller's original row order (oracle lookups into targets.h5, whose rows are
    the 73k COCO/NSD image id directly, see assemble_targets.py)."""
    ids = np.asarray(ids)
    order = np.argsort(ids)
    unique_ids, inverse = np.unique(ids[order], return_inverse=True)
    rows = h5_dataset[unique_ids.tolist()]
    out = np.empty((len(ids),) + rows.shape[1:], dtype=rows.dtype)
    out[order] = rows[inverse]
    return out


def load_oracle_ctx(path, ids, device):
    """T0's context arm: read the real VD context straight from the output of
    extract_vd_context.py instead of running the adapter. Unlike targets.h5/vdvae_latents_31l.npy this file only
    covers a fixed 6000+300-image subset (module docstring); looked up via each group's
    image_indices_73k, not row-by-id directly."""
    ids = [int(i) for i in ids]
    with h5py.File(path, "r") as f:
        index = {}
        for grp in ("train", "val"):
            for local_i, img_id in enumerate(f[grp]["image_indices_73k"][:]):
                index[int(img_id)] = (grp, local_i)
        missing = [i for i in ids if i not in index]
        if missing:
            raise KeyError(
                f"--oracle-targets ctx: {len(missing)} image id(s) not covered by {path} (only "
                f"{len(index)} images extracted) -- e.g. {missing[:5]}. extract_vd_context.py "
                "needs to be extended to this row set first."
            )
        prompt = np.empty((len(ids), 77, 768), dtype=np.float32)
        image = np.empty((len(ids), 257, 768), dtype=np.float32)
        for i, img_id in enumerate(ids):
            grp, local_i = index[img_id]
            prompt[i] = f[grp]["prompt_embeds"][local_i]
            image[i] = f[grp]["image_embeddings"][local_i]
    return torch.from_numpy(prompt).to(device), torch.from_numpy(image).to(device)


def load_bd_bridge_index(path):
    """The 2x2 swap with Brain-Diffuser: output of prepare_bd_shared1000_blocks.py, i.e. the img73k-id ->
    BD-test-row map per subject (test_ids, BD's own subject-specific order) and which of the 83
    shared1000 blocks survive in all four subjects' BD test sets (blocks_kept)."""
    with open(path) as f:
        return json.load(f)


def bd_row_lookup(bd_index, subj):
    ids = bd_index["test_ids"][f"subj{subj:02d}"]
    return {int(v): i for i, v in enumerate(ids)}


def load_bd_ctx(bd_root, subj, ids, row_of, device):
    """--bd-context: Brain-Diffuser's predicted cliptext/clipvision for the given img73k ids,
    rebuilt by bd_predict_shared1000.py (bit-identical to what *_regression.py would have
    written; see that script's docstring). Row order in the BD array is BD's own
    subject-specific test order (bd_block_index.json's row_order note), not ascending id; the
    row_of lookup handles this."""
    cliptext = np.load(os.path.join(bd_root, "data", "predicted_features", f"subj{subj:02d}",
                                     "nsd_cliptext_predtest_nsdgeneral.npy"), mmap_mode="r")
    clipvision = np.load(os.path.join(bd_root, "data", "predicted_features", f"subj{subj:02d}",
                                       "nsd_clipvision_predtest_nsdgeneral.npy"), mmap_mode="r")
    rows = np.array([row_of[int(i)] for i in ids])
    prompt = np.asarray(cliptext[rows]).astype(np.float32)      # (n,77,768)
    image = np.asarray(clipvision[rows]).astype(np.float32)     # (n,257,768)
    return torch.from_numpy(prompt).to(device), torch.from_numpy(image).to(device)


def load_bd_zlow_raw(bd_root, subj, ids, row_of):
    """--bd-zlow: Brain-Diffuser's predicted raw 91168-dim VDVAE latent, same row_of lookup as
    load_bd_ctx(). Decoded through decode_raw_batch()/decode_zlow_to_image.py, i.e. the SAME
    31-layer hierarchical reshape TRISECT's own raw latents use, because both are extracted from the
    identical frozen imagenet64 VDVAE (extract_vdvae_latents.py is adapted from Brain-Diffuser's
    own vdvae_extract_features.py). The decode was checked once by a sanity check before the
    runs, and is not re-verified here."""
    vdvae = np.load(os.path.join(bd_root, "data", "predicted_features", f"subj{subj:02d}",
                                  f"nsd_vdvae_nsdgeneral_pred_sub{subj}_31l_alpha50k.npy"),
                     mmap_mode="r")
    rows = np.array([row_of[int(i)] for i in ids])
    return np.asarray(vdvae[rows]).astype(np.float32)


def condition_and_local_stim(label):
    cond, global_id = label.rsplit("-", 1)
    return cond, int(global_id) % N_STIM


def diffusion_seed(base_seed: int, rep: int, subj: int, stim_key: str) -> int:
    """Seed of the diffusion generator for one (replica, subject, stimulus).

    An earlier scheme, `args.seed + r`, keyed the generator on the replica index alone; every
    stimulus of a given replica (across every subject/block in the run) shared IDENTICAL initial
    noise: only 4 distinct noise draws ever existed for a whole --gen-rep 4 run. That produced a
    non-centered generation null (+0.527pp offset), because a lucky/unlucky noise draw correlates
    with the metric across all stimuli at once, instead of averaging out (Section 7.1). Salting with
    (subj, stim_key) gives every (replica, stimulus, subject) triple its own independent noise
    while staying a pure function of --seed (deterministic, no RNG state threaded through the row
    loop). It uses sha256 rather than Python's hash() because the latter is salted per-process
    (PYTHONHASHSEED) and would not reproduce across runs/machines."""
    key = f"{base_seed}|{rep}|{subj}|{stim_key}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


def load_imagery_predictions(pred_dir: Path, subj: int):
    subj_dir = pred_dir / f"subj{subj:02d}"
    labels = np.load(subj_dir / "labels.npy")
    z_sem_img = np.load(subj_dir / "z_sem_img_pred.npy")   # (30, 128)
    z_sem_txt = np.load(subj_dir / "z_sem_txt_pred.npy")   # (30, 128)
    z_struct = np.load(subj_dir / "z_struct_pred.npy")     # (30, 8, 8, 32)
    z_low = np.load(subj_dir / "z_low_pred.npy")           # (30, d_low)
    z_sem = np.stack([z_sem_img, z_sem_txt], axis=1)        # (30, 2, 128)
    row_meta = [condition_and_local_stim(str(lbl)) + (str(lbl),) for lbl in labels]
    return row_meta, z_sem, z_struct, z_low


def load_nsdcore_predictions(pred_dir: Path, subj: int, block_size: int):
    """predict_ridge_shared1000.py's per-subject output: (1000,)+native_shape predictions,
    sorted by ascending 73k id, no notion of "condition"; rows are grouped into block_size
    pseudo-conditions "blockNN" in that same order, see module docstring."""
    subj_dir = pred_dir / f"subj{subj:02d}"
    img73k_ids = np.load(subj_dir / "img73k_ids.npy")       # (1000,)
    z_sem_img = np.load(subj_dir / "z_sem_img_pred.npy")    # (1000, 128)
    z_sem_txt = np.load(subj_dir / "z_sem_txt_pred.npy")    # (1000, 128)
    z_struct = np.load(subj_dir / "z_struct_pred.npy")      # (1000, 8, 8, 32)
    z_low = np.load(subj_dir / "z_low_pred.npy")            # (1000, d_low)
    z_sem = np.stack([z_sem_img, z_sem_txt], axis=1)         # (1000, 2, 128)
    row_meta = [(f"block{i // block_size:02d}", i % block_size, f"img{img73k_ids[i]:05d}")
                for i in range(len(img73k_ids))]
    return row_meta, z_sem, z_struct, z_low, img73k_ids


def harness_block_rows(row_meta, img73k_ids, blockset_dir):
    """--harness-blocks: re-express nsdcore's "blockNN" pseudo-conditions in the layout that
    evaluate_block_set.py (and so Kneeland's own harness) expects: one output directory per block,
    the block's 12 rows split 6+6 into "imagerysimple" and "imagerycomplex". Those two names are STRUCTURAL halves here, not stimulus types: every
    shared1000 image is a natural photograph (make_shared1000_blockset.py's docstring).

    Returns (row_meta, path_units, n_keep): path_units[i] is the directory index row i is written
    under (the block, not the subject; the subject slot of the path is taken by the block;
    per-subject arms must use separate --out-dir). Rows past the last full block are dropped, and
    the block ids are checked against the blockset manifest so the reconstructions cannot silently
    misalign with the ground truth they will be scored against.
    """
    with open(os.path.join(blockset_dir, "manifest.json")) as f:
        manifest = json.load(f)
    n_stim = manifest["n_stim_per_block"]           # 6 per condition
    n_test = 2 * n_stim                             # 12 per block
    n_blocks = manifest["n_blocks"]
    n_keep = n_blocks * n_test
    if len(row_meta) < n_keep:
        raise ValueError(f"{len(row_meta)} prediction rows but the blockset declares {n_blocks} "
                          f"blocks of {n_test} ({n_keep} rows)")

    expected = [i for b in manifest["blocks"] for i in b["img73k_ids"]]
    got = [int(i) for i in img73k_ids[:n_keep]]
    if expected != got:
        raise ValueError(f"{blockset_dir}/manifest.json block ids do not match the prediction row "
                          "order -- the ground truth would be scored against the wrong images")

    new_meta, path_units = [], []
    for i in range(n_keep):
        block, within = divmod(i, n_test)
        cond = "imagerysimple" if within < n_stim else "imagerycomplex"
        new_meta.append((cond, within % n_stim, row_meta[i][2]))
        path_units.append(block)
    return new_meta, path_units, n_keep


def load_proxy_rows(proxy_dir: str, blocks):
    """Rows of the proxy set (make_proxy_set.py), grouped by block: a block index stands in for
    --subj (module docstring), matching manifest.json's own block layout: 6
    imagerysimple + 6 imagerycomplex rows per block, flat-indexed into the 0-299 proxy pool
    (0-149 simple, 150-299 complex; must match extract_proxy_vd_context.py /
    extract_proxy_vdvae_latents.py / caption_proxy_set.py). Returns {block: (row_meta, flat_idx)}
    with row_meta in the same (cond, stim_local, display_label) shape the other two loaders use,
    so the main loop below needs no proxy-specific branch past this point."""
    with open(os.path.join(proxy_dir, "manifest.json")) as f:
        manifest = json.load(f)
    sel_blocks = blocks if blocks is not None else list(range(manifest["n_blocks"]))
    out = {}
    for b in sel_blocks:
        bm = manifest["blocks"][b]
        row_meta, flat_idx = [], []
        for local_i, si in enumerate(bm["simple_indices"]):
            row_meta.append(("imagerysimple", local_i, f"proxy-simple-b{b:02d}-{local_i}"))
            flat_idx.append(si)                          # 0-149
        for local_i, ci in enumerate(bm["complex_indices"]):
            row_meta.append(("imagerycomplex", local_i, f"proxy-complex-b{b:02d}-{local_i}"))
            flat_idx.append(PROXY_N_SIMPLE + ci)          # 150-299
        out[b] = (row_meta, np.array(flat_idx, dtype=np.int64))
    return out


def load_proxy_oracle_ctx(path, flat_idx, device):
    """T0's context arm for --dataset proxy: the output of extract_proxy_vd_context.py is FLAT
    (no train/val split, no 73k-id lookup; see the module docstring), and this is a direct
    row-index read instead of the id->(group,local_i) indirection of load_oracle_ctx(). flat_idx is already
    strictly increasing within one block (simple_indices then complex_indices, both ascending),
    so h5py's fancy-indexing constraint is satisfied without the sort/dedupe load_oracle_rows_by_id
    needs for arbitrary id order."""
    idx = [int(i) for i in flat_idx]
    with h5py.File(path, "r") as f:
        prompt = f["prompt_embeds"][idx].astype(np.float32)
        image = f["image_embeddings"][idx].astype(np.float32)
    return torch.from_numpy(prompt).to(device), torch.from_numpy(image).to(device)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subj", type=int, nargs="+", default=list(SUBJECTS),
                     help="--dataset imagery/nsdcore only (must be a subset of 1,2,5,7) -- unused "
                          "for --dataset proxy, which iterates --proxy-blocks instead")
    ap.add_argument("--dataset", choices=["imagery", "nsdcore", "proxy"], default="imagery",
                     help="imagery: the 30 NSD-Imagery rows (default). nsdcore: the shared1000 "
                          "held-out predictions from predict_ridge_shared1000.py, grouped "
                          "into --block-size pseudo-conditions (see module docstring). proxy: "
                          "the proxy set (make_proxy_set.py), --oracle-targets ctx or all only "
                          "(T0-T2, see module docstring).")
    ap.add_argument("--block-size", type=int, default=BLOCK_SIZE_DEFAULT,
                     help="--dataset nsdcore only: rows per pseudo-condition block")
    ap.add_argument("--proxy-dir", type=str, default=PROXY_DIR_DEFAULT,
                     help="--dataset proxy only: make_proxy_set.py's output directory")
    ap.add_argument("--proxy-blocks", type=int, nargs="+", default=None,
                     help="--dataset proxy only: which block indices to reconstruct (default: all)")
    ap.add_argument("--harness-blocks", action="store_true",
                     help="--dataset nsdcore only: write the harness's own per-block layout "
                          "(one directory per block of 12, split 6+6 into imagerysimple/"
                          "imagerycomplex) instead of one directory per subject, so "
                          "evaluate_block_set.py can score this run with the same 11-distractor "
                          "pool as the proxy arms. Requires --blockset-dir's manifest, whose "
                          "block ids are checked against the prediction row order.")
    ap.add_argument("--blockset-dir", type=str, default=BLOCKSET_DIR_DEFAULT,
                     help="--harness-blocks only: make_shared1000_blockset.py's output")
    ap.add_argument("--proxy-targets", type=str, default=PROXY_TARGETS_DEFAULT,
                     help="--dataset proxy with --oracle-targets all (T1/T2) only: "
                          "extract_proxy_targets.py's flat-indexed z_sem/z_struct")
    ap.add_argument("--proxy-ctx", type=str, default=PROXY_CTX_DEFAULT,
                     help="--dataset proxy only: extract_proxy_vd_context.py's output")
    ap.add_argument("--proxy-raw-latents", type=str, default=PROXY_RAW_LATENTS_DEFAULT,
                     help="--dataset proxy only: extract_proxy_vdvae_latents.py's output")
    ap.add_argument("--proxy-raw-layer-dims", type=str, default=PROXY_RAW_LAYER_DIMS_DEFAULT,
                     help="--dataset proxy only: extract_proxy_vdvae_latents.py's layer_dims sidecar")
    ap.add_argument("--gen-rep", type=int, default=10, help="reconstructions per stimulus "
                     "(Kneeland et al. use 10; use 1 for a smoke test)")
    ap.add_argument("--limit-rows", type=int, default=None,
                     help="debug: only reconstruct the first N rows per subject")
    ap.add_argument("--strength", type=float, default=DIFF_STRENGTH_DEFAULT,
                     help="VDVAE init-image denoising strength (BD's diff_str)")
    ap.add_argument("--mixing", type=float, default=MIX_DEFAULT,
                     help="text_to_image_strength / BD's mixing (text weight, see module docstring)")
    ap.add_argument("--guidance-scale", type=float, default=GUIDANCE_SCALE_DEFAULT)
    ap.add_argument("--num-inference-steps", type=int, default=NUM_INFERENCE_STEPS_DEFAULT)
    ap.add_argument("--seed", type=int, default=0,
                     help="base seed: each (rep, subj, stimulus) draws its own diffusion noise "
                          "via diffusion_seed() -- NOT bit-identical to runs of the earlier "
                          "scheme, which reused one seed per rep across every stimulus")
    ap.add_argument("--pred-dir", type=str, default=None,
                     help=f"default: {PRED_DIR_DEFAULT} (imagery) / {NSDCORE_PRED_DIR_DEFAULT} "
                          "(nsdcore)")
    ap.add_argument("--zlow-mode", choices=["vdvae", "pixel", "raw"], default="vdvae",
                     help="vdvae (default): z_low_pred -> PCA^-1 -> VDVAE decoder (91168-dim "
                          "latent path). raw: z_low_pred is ALREADY a "
                          "ridge-predicted raw 91168-dim VDVAE latent (no PCA involved anywhere, "
                          "not even the ridge target) -> VDVAE decoder directly, via the same "
                          "decode_raw_batch() primitive --zlow-raw uses for the ORACLE raw "
                          "latent -- this is its predicted-not-oracle counterpart. pixel: "
                          "z_low_pred is already resR*resR*3 raw pixels "
                          "(pixel_res), just reshaped and upsampled -- no VDVAE/PCA loaded.")
    ap.add_argument("--pixel-res", type=int, default=16,
                     help="--zlow-mode pixel only: z_low_pred's resR (e.g. 16 for pixel16)")
    ap.add_argument("--filter-sharpness", type=float, default=1.0,
                     help="PIL ImageEnhance.Sharpness factor applied to the 64x64 init image "
                          "before it is encoded (MIRAGE uses 20). 1.0 = off (default, no-op).")
    ap.add_argument("--filter-contrast", type=float, default=1.0,
                     help="PIL ImageEnhance.Contrast factor applied after sharpness (MIRAGE "
                          "uses 1.5). 1.0 = off (default, no-op).")
    ap.add_argument("--pca-zlow-dir", type=str, default=PCA_ZLOW_DIR_DEFAULT)
    ap.add_argument("--ckpt", type=str, default=CKPT_DEFAULT)
    ap.add_argument("--targets", type=str, default=TARGETS_H5_DEFAULT,
                     help="assembled targets.h5, read for z_struct/z_sem shape cross-check and, "
                          "with --oracle-targets zsem_zstruct/all/ctx, for the oracle rows themselves")
    ap.add_argument("--oracle-targets", choices=["none", "ctx", "zsem_zstruct", "zlow", "all"],
                     default="none",
                     help="the T0-T3 oracle arms: swap the named branch(es) for the real "
                          "image's target row instead of the ridge prediction. --dataset nsdcore "
                          "or proxy only (see module docstring). "
                          "none=T3 (default, no swap), "
                          "all=T2 (T1 with --zlow-raw), ctx=T0 (with --zlow-raw).")
    ap.add_argument("--zlow-raw", action="store_true",
                     help="with --oracle-targets zlow/all/ctx: use the uncompressed 91168-dim "
                          "VDVAE latent (vdvae_latents_31l.npy) instead of the real image's "
                          "PCA-space z_low -- decode_zlow_to_image.py's --raw primitive.")
    ap.add_argument("--a-psi-context", type=str, default=A_PSI_TRUE_CONTEXT_DEFAULT,
                     help="--oracle-targets ctx only: extract_vd_context.py's output")
    ap.add_argument("--ablate-factor", choices=["none", "z_struct_zero", "z_struct_shuffle",
                     "z_sem_zero", "z_sem_shuffle"], default="none",
                     help="null out or permute one factor right before it reaches A_psi, "
                          "after any --oracle-targets swap. "
                          "*_zero sets the factor to 0 in the whitened PCA space (the "
                          "least-informative in-distribution value -- 'how much does it "
                          "contribute'). *_shuffle permutes the factor across the stimuli of the "
                          "current run/block, fixed seed=--seed, printed -- 'does A_psi use it, "
                          "and is it misled by a wrong one'. No effect with --oracle-targets ctx "
                          "(A_psi is bypassed there).")
    ap.add_argument("--out-dir", type=str, default=None,
                     help=f"default: {OUT_DIR_DEFAULT} (imagery) / {NSDCORE_OUT_DIR_DEFAULT} "
                          "(nsdcore)")
    ap.add_argument("--bd-context", action="store_true",
                     help="2x2 swap with Brain-Diffuser: read the VD "
                          "cross-attention context straight from Brain-Diffuser's predicted "
                          "cliptext/clipvision (bd_predict_shared1000.py), bypassing A_psi "
                          "entirely -- the 'B' half of the context axis (BT/BB arms). Requires "
                          "--dataset nsdcore --harness-blocks; mutually exclusive with "
                          "--oracle-targets.")
    ap.add_argument("--bd-zlow", action="store_true",
                     help="decode the VDVAE init image from Brain-Diffuser's predicted raw "
                          "91168-dim VDVAE latent instead of TRISECT's ridge z_low -- the 'B' "
                          "half of the low-level axis (TB/BB arms). Same requirements as "
                          "--bd-context.")
    ap.add_argument("--bd-root", type=str, default="../brain_diffuser",
                     help="--bd-context/--bd-zlow: brain_diffuser clone, for "
                          "data/predicted_features/subjNN/nsd_{cliptext,clipvision,vdvae}_..."
                          ".npy (bd_predict_shared1000.py's output)")
    ap.add_argument("--bd-bridge-index", type=str,
                     default="data/results/shared1000_comparison/bd_block_index.json",
                     help="--bd-context/--bd-zlow: prepare_bd_shared1000_blocks.py's output, for the "
                          "img73k-id -> BD-test-row map and the kept-block set")
    args = ap.parse_args()

    oracle_zlow = args.oracle_targets in ("zlow", "all", "ctx")
    oracle_zsem_zstruct = args.oracle_targets in ("zsem_zstruct", "all", "ctx")
    oracle_ctx = args.oracle_targets == "ctx"
    if args.dataset == "proxy":
        if args.oracle_targets not in ("ctx", "all"):
            raise ValueError("--dataset proxy only supports --oracle-targets ctx (T0) or all "
                              "(T1/T2, module docstring): every other mode keeps part of the "
                              "pipeline ridge-predicted, and no subject ever saw a proxy image")
        if args.oracle_targets == "ctx" and not args.zlow_raw:
            raise ValueError("--dataset proxy --oracle-targets ctx is T0, which is defined with "
                              "--zlow-raw (module docstring)")
    elif args.oracle_targets != "none" and args.dataset != "nsdcore":
        raise ValueError("--oracle-targets requires --dataset nsdcore: oracle rows are keyed by "
                          "the 73k COCO/NSD image id, which NSD-Imagery's 18 stimuli don't have "
                          "(module docstring)")
    if args.harness_blocks and args.dataset != "nsdcore":
        raise ValueError("--harness-blocks is a --dataset nsdcore layout: the proxy set already "
                          "writes the harness's block layout, and NSD-Imagery has no blocks")
    if args.zlow_raw and not oracle_zlow:
        raise ValueError("--zlow-raw only makes sense with --oracle-targets zlow/all/ctx -- there "
                          "is no 'raw predicted' z_low, ridge only predicts in PCA space")
    if oracle_zlow and args.zlow_mode != "vdvae":
        raise ValueError("--oracle-targets zlow/all/ctx requires --zlow-mode vdvae (the VDVAE "
                          "latent path); it has no meaning for --zlow-mode pixel/raw -- the oracle "
                          "raw latent is already reached via --zlow-raw, not --zlow-mode raw "
                          "(that flag is for a ridge-PREDICTED raw latent)")
    if args.ablate_factor != "none" and oracle_ctx:
        raise ValueError("--ablate-factor has no effect with --oracle-targets ctx: the VD context "
                          "is read straight from a_psi_true_context.h5, A_psi (and z_sem/z_struct) "
                          "is bypassed entirely")
    if args.bd_context or args.bd_zlow:
        if args.dataset != "nsdcore" or not args.harness_blocks:
            raise ValueError("--bd-context/--bd-zlow require --dataset nsdcore --harness-blocks "
                              "(the swap works on the 65-block Brain-Diffuser bridge subset, "
                              "prepare_bd_shared1000_blocks.py)")
        if args.oracle_targets != "none":
            raise ValueError("--bd-context/--bd-zlow are mutually exclusive with --oracle-targets")

    bd_index = kept_blocks = None
    if args.bd_context or args.bd_zlow:
        bd_index = load_bd_bridge_index(args.bd_bridge_index)
        kept_blocks = set(bd_index["blocks_kept"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dataset == "imagery":
        pred_dir = Path(args.pred_dir or PRED_DIR_DEFAULT)
        out_dir = Path(args.out_dir or OUT_DIR_DEFAULT)
    elif args.dataset == "nsdcore":
        pred_dir = Path(args.pred_dir or NSDCORE_PRED_DIR_DEFAULT)
        out_dir = Path(args.out_dir or NSDCORE_OUT_DIR_DEFAULT)
    else:  # proxy
        pred_dir = None
        out_dir = Path(args.out_dir or PROXY_OUT_DIR_DEFAULT)

    if args.zlow_mode in ("vdvae", "raw"):
        print("Loading VDVAE (ImageNet64, EMA weights) ...")
        H = dotdict(dict(H_DICT))
        H, _ = set_up_data(H)
        ema_vae = load_vaes(H)
        pca_basis = load_pca_basis(args.pca_zlow_dir) if args.zlow_mode == "vdvae" else None
    else:
        print(f"--zlow-mode pixel (res={args.pixel_res}): VDVAE not loaded.")
        ema_vae = pca_basis = None

    # The proxy set has no row in pca_zlow/z_low.npy; its PCA-space z_low (T2) is projected
    # from the same raw latents T1 decodes directly; both arms therefore need them loaded.
    proxy_zlow_from_raw = args.dataset == "proxy" and oracle_zlow and not args.zlow_raw
    raw_shapes = raw_bounds = raw_latents_mmap = None
    if args.zlow_raw or proxy_zlow_from_raw or args.zlow_mode == "raw" or args.bd_zlow:
        layer_dims_path = args.proxy_raw_layer_dims if args.dataset == "proxy" else RAW_LAYER_DIMS_PATH
        with open(layer_dims_path) as f:
            raw_layer_dims = json.load(f)["layer_dims"]
        raw_shapes, raw_bounds = layer_shapes(raw_layer_dims), layer_bounds(raw_layer_dims)
    if args.zlow_raw or proxy_zlow_from_raw:
        # --zlow-mode raw needs only shapes/bounds (its latent is already predicted, in
        # z_low_pred.npy): the oracle mmap of the ground-truth 73k-indexed latents is not
        # relevant there and would need a subj-73k id this dataset does not have (NSD-Imagery).
        latents_path = args.proxy_raw_latents if args.dataset == "proxy" else RAW_LATENTS_PATH
        raw_latents_mmap = np.load(latents_path, mmap_mode="r")
    oracle_zlow_pca_mmap = None
    if oracle_zlow and not args.zlow_raw and not proxy_zlow_from_raw:
        oracle_zlow_pca_mmap = np.load(
            os.path.join(args.pca_zlow_dir, "z_low.npy"), mmap_mode="r")

    ckpt = torch.load(args.ckpt, map_location=device)
    with h5py.File(args.targets, "r") as f:
        targets_struct_channels = f["z_struct"].shape[-1]
        targets_sem_dim = f["z_sem"].shape[-1]
    struct_channels = ckpt.get("struct_channels", targets_struct_channels)
    sem_dim = ckpt.get("sem_dim", targets_sem_dim)
    a_psi = APsiMinimal(struct_channels=struct_channels, sem_dim=sem_dim).to(device)
    a_psi.load_state_dict(ckpt["state_dict"])
    a_psi.eval()

    print("Loading Versatile Diffusion pipeline ...")
    pipe = load_pipeline(device=device)
    neg_prompt_embeds, neg_image_embeddings = encode_unconditional_context(pipe, device)

    if args.dataset == "proxy":
        proxy_rows_by_block = load_proxy_rows(args.proxy_dir, args.proxy_blocks)
        units = sorted(proxy_rows_by_block.keys())
    else:
        units = args.subj

    n_total_images = 0
    t0 = time.time()
    for subj in units:
        flat_idx = None
        path_units = None
        if args.dataset == "imagery":
            row_meta, z_sem, z_struct, z_low = load_imagery_predictions(pred_dir, subj)
            img73k_ids = None
        elif args.dataset == "nsdcore":
            row_meta, z_sem, z_struct, z_low, img73k_ids = load_nsdcore_predictions(
                pred_dir, subj, args.block_size)
            if args.harness_blocks:
                row_meta, path_units, n_keep = harness_block_rows(
                    row_meta, img73k_ids, args.blockset_dir)
                z_sem, z_struct, z_low = z_sem[:n_keep], z_struct[:n_keep], z_low[:n_keep]
                img73k_ids = img73k_ids[:n_keep]
                print(f"  --harness-blocks: {n_keep} rows in "
                      f"{n_keep // 12} blocks of 12, ids checked against {args.blockset_dir}")
                if kept_blocks is not None:
                    # Restrict to the blocks prepare_bd_shared1000_blocks.py found present in ALL four
                    # subjects' BD test sets (65/83); the rest have no BD prediction.
                    keep_mask = np.array([pu in kept_blocks for pu in path_units])
                    row_meta = [rm for rm, k in zip(row_meta, keep_mask) if k]
                    path_units = [pu for pu, k in zip(path_units, keep_mask) if k]
                    z_sem, z_struct, z_low = z_sem[keep_mask], z_struct[keep_mask], z_low[keep_mask]
                    img73k_ids = img73k_ids[keep_mask]
                    print(f"  --bd-context/--bd-zlow: restricted to {len(kept_blocks)} "
                          f"BD-bridge blocks -> {len(row_meta)} rows")
        else:  # proxy: no ridge predictions at all, --oracle-targets ctx --zlow-raw is mandatory
            row_meta, flat_idx = proxy_rows_by_block[subj]
            z_sem = z_struct = z_low = None
            img73k_ids = None
        n_rows = len(row_meta) if args.limit_rows is None else min(args.limit_rows, len(row_meta))
        print(f"\n=== subj{subj:02d}: {n_rows}/{len(row_meta)} rows, {args.gen_rep} reps each ===")
        ids_n = img73k_ids[:n_rows] if img73k_ids is not None else None
        flat_idx_n = flat_idx[:n_rows] if flat_idx is not None else None

        if args.bd_zlow:
            print("  decoding z_low -> Brain-Diffuser's predicted RAW 91168-dim VDVAE latent "
                  "-> 64x64 init images (--bd-zlow, no PCA) ...")
            bd_row_of = bd_row_lookup(bd_index, subj)
            bd_raw_rows = load_bd_zlow_raw(args.bd_root, subj, ids_n, bd_row_of)
            init_pixels = decode_batched(decode_raw_batch, bd_raw_rows, raw_shapes, raw_bounds,
                                          ema_vae, device)
        elif oracle_zlow:
            if args.zlow_raw:
                print("  decoding z_low -> RAW 91168-dim VDVAE latent -> 64x64 init images "
                      "(--oracle-targets, no PCA) ...")
                raw_key = flat_idx_n if args.dataset == "proxy" else ids_n
                raw_rows = raw_latents_mmap[raw_key].astype(np.float32)
                init_pixels = decode_batched(decode_raw_batch, raw_rows, raw_shapes, raw_bounds,
                                              ema_vae, device)
            elif proxy_zlow_from_raw:
                print("  decoding z_low -> oracle PCA-space z_low projected from the proxy raw "
                      "latents -> VDVAE 64x64 init images (--oracle-targets, T2) ...")
                raw_rows = raw_latents_mmap[flat_idx_n].astype(np.float32)
                proxy_zlow = project_raw_to_zlow(raw_rows, pca_basis)
                init_pixels = decode_batched(decode_zlow_batch, proxy_zlow, pca_basis, ema_vae,
                                              device)
            else:
                print("  decoding z_low -> oracle PCA-space z_low -> VDVAE 64x64 init images "
                      "(--oracle-targets) ...")
                oracle_zlow_rows = np.asarray(oracle_zlow_pca_mmap[ids_n])
                init_pixels = decode_batched(decode_zlow_batch, oracle_zlow_rows, pca_basis,
                                              ema_vae, device)
        elif args.zlow_mode == "vdvae":
            print("  decoding z_low -> VDVAE 64x64 init images ...")
            init_pixels = decode_batched(decode_zlow_batch, z_low[:n_rows], pca_basis, ema_vae,
                                          device)  # (n_rows,64,64,3) uint8
        elif args.zlow_mode == "raw":
            print("  decoding z_low -> PREDICTED raw 91168-dim VDVAE latent -> 64x64 init images "
                  "(raw latent target, no PCA) ...")
            init_pixels = decode_batched(decode_raw_batch, z_low[:n_rows], raw_shapes, raw_bounds,
                                          ema_vae, device)
        else:
            print(f"  decoding z_low -> pixel{args.pixel_res} 64x64 init images (no VDVAE) ...")
            init_pixels = decode_pixel_batch(z_low[:n_rows], args.pixel_res, device)
        init_images = [Image.fromarray(init_pixels[i]) for i in range(n_rows)]
        if args.filter_sharpness != 1.0 or args.filter_contrast != 1.0:
            # MIRAGE's filter on the decoded init, same order (sharpness then contrast), applied
            # here so it never touches init_pixels/the oracle decode path.
            init_images = [ImageEnhance.Contrast(
                               ImageEnhance.Sharpness(im).enhance(args.filter_sharpness)
                           ).enhance(args.filter_contrast)
                           for im in init_images]

        if args.bd_context:
            print("  VD context <- Brain-Diffuser's predicted cliptext/clipvision "
                  "(--bd-context, A_psi bypassed) ...")
            bd_row_of = bd_row_lookup(bd_index, subj)
            pred_prompt_embeds, pred_image_embeddings = load_bd_ctx(
                args.bd_root, subj, ids_n, bd_row_of, device)
        elif oracle_ctx:
            if args.dataset == "proxy":
                print(f"  reading VD context straight from {args.proxy_ctx} "
                      "(--oracle-targets ctx, A_psi bypassed) ...")
                pred_prompt_embeds, pred_image_embeddings = load_proxy_oracle_ctx(
                    args.proxy_ctx, flat_idx_n, device)
            else:
                print(f"  reading VD context straight from {args.a_psi_context} "
                      "(--oracle-targets ctx, A_psi bypassed) ...")
                pred_prompt_embeds, pred_image_embeddings = load_oracle_ctx(
                    args.a_psi_context, ids_n, device)  # (n_rows,77,768)/(n_rows,257,768)
        else:
            if oracle_zsem_zstruct and args.dataset == "proxy":
                # Flat 0-299 read, the load_proxy_oracle_ctx() rationale applied to the targets:
                # extract_proxy_targets.py writes in the proxy set's own order; there is no
                # id -> row indirection to undo (flat_idx_n is already sorted and unique).
                print(f"  z_sem/z_struct <- {args.proxy_targets} (--oracle-targets all, T1/T2) ...")
                with h5py.File(args.proxy_targets, "r") as tf:
                    z_sem_used = tf["z_sem"][flat_idx_n].astype(np.float32)
                    z_struct_used = tf["z_struct"][flat_idx_n].astype(np.float32)
            elif oracle_zsem_zstruct:
                print("  z_sem/z_struct <- oracle rows of targets.h5 (--oracle-targets) ...")
                with h5py.File(args.targets, "r") as tf:
                    z_sem_used = load_oracle_rows_by_id(tf["z_sem"], ids_n).astype(np.float32)
                    z_struct_used = load_oracle_rows_by_id(tf["z_struct"], ids_n).astype(np.float32)
            else:
                z_sem_used = z_sem[:n_rows]
                z_struct_used = z_struct[:n_rows]

            if args.ablate_factor != "none":
                # Applied after every --oracle-targets swap above, on whatever
                # z_sem_used/z_struct_used A_psi is about to see, oracle or predicted alike.
                factor_name, _, mode = args.ablate_factor.rpartition("_")
                target = z_struct_used if factor_name == "z_struct" else z_sem_used
                if mode == "zero":
                    target = np.zeros_like(target)
                else:  # shuffle
                    rng = np.random.RandomState(args.seed)
                    perm = rng.permutation(target.shape[0])
                    print(f"  --ablate-factor {args.ablate_factor}: seed={args.seed} "
                          f"perm={perm.tolist()}")
                    target = target[perm]
                if factor_name == "z_struct":
                    z_struct_used = target
                else:
                    z_sem_used = target

            z_sem_t = torch.from_numpy(z_sem_used.astype(np.float32)).to(device)
            z_struct_t = torch.from_numpy(z_struct_used.astype(np.float32)).to(device)
            with torch.no_grad():
                pred_prompt_embeds, pred_image_embeddings = a_psi(z_sem_t, z_struct_t)  # (n_rows,77,768)/(n_rows,257,768)

        for i in range(n_rows):
            cond, stim_local, display_label = row_meta[i]
            filename = f"{stim_local:05d}.png"
            # --harness-blocks puts the BLOCK in the path's subject slot (harness_block_rows):
            # per-subject arms therefore need separate --out-dir, not separate subdirectories.
            unit = subj if path_units is None else path_units[i]
            cond_dir = out_dir / f"subj{unit:02d}" / cond
            reps_todo = [r for r in range(args.gen_rep)
                         if not (cond_dir / f"sample{r:03d}" / filename).exists()]
            if not reps_todo:
                print(f"  [{display_label}] already done, skipping ({n_total_images} images so "
                      f"far, {time.time() - t0:.0f}s elapsed)", flush=True)
                continue

            cfg_prompt_embeds = torch.cat([neg_prompt_embeds, pred_prompt_embeds[i:i + 1].to(neg_prompt_embeds.dtype)])
            cfg_image_embeddings = torch.cat([neg_image_embeddings, pred_image_embeddings[i:i + 1].to(neg_image_embeddings.dtype)])

            for r in reps_todo:
                generator = torch.Generator(device=device).manual_seed(
                    diffusion_seed(args.seed, r, subj, display_label))
                out = generate_from_context(
                    pipe,
                    cfg_prompt_embeds,
                    cfg_image_embeddings,
                    text_to_image_strength=args.mixing,
                    guidance_scale=args.guidance_scale,
                    num_inference_steps=args.num_inference_steps,
                    generator=generator,
                    init_image=init_images[i],
                    strength=args.strength,
                )[0]

                sample_dir = cond_dir / f"sample{r:03d}"
                sample_dir.mkdir(parents=True, exist_ok=True)
                out.save(sample_dir / filename)
                n_total_images += 1
                del out, generator

            # WSL2's GPU passthrough (dxgkrnl) accumulates RAM/VRAM over long-running CUDA
            # workloads independently of anything this process holds live: an explicit release per
            # row is cheap insurance against that growth, not a fix for a leak in this loop itself.
            torch.cuda.empty_cache()
            gc.collect()

            print(f"  [{display_label}] -> {len(reps_todo)} sample(s) saved "
                  f"({n_total_images} images so far, {time.time() - t0:.0f}s elapsed)", flush=True)

    print(f"\nDone. {n_total_images} images written under {out_dir}")


if __name__ == "__main__":
    main()
