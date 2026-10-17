"""Frozen targets (z_sem, z_struct) for the 300 proxy-set images, the proxy counterpart of targets.h5.

Inputs:  data/proxy_set/{images/, captions.npy}, the frozen PCA bases (data/targets/pca_zsem, pca_zstruct)
Outputs: data/targets/targets_proxyset.h5, flat 0-299 as in captions.npy:
         z_sem (300, 2, 128) [z_sem_img, z_sem_txt], z_struct (300, 8, 8, 32), both float16

The proxy set has no 73k id; the id-keyed targets.h5 cannot serve it. The formulas are those of
the target scripts of trisect/, kept in one copy in TargetEncoder (traversal_figure.py imports it); no PCA is fit.
`--txt-from-img` copies z_sem_img into the text slot (smoke test only, never for a real T1 or T2 arm).

Usage (after make_proxy_set.py and caption_proxy_set.py):
    python extract_proxy_targets.py
"""
import argparse
import os

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, CLIPVisionModelWithProjection
from trisect.pca import load_pca_basis
from trisect.targets import l2_normalize, load_proxy_images

PROXY_DIR_DEFAULT = "data/proxy_set"
CAPTIONS_DEFAULT = "data/proxy_set/captions.npy"
PCA_ZSEM_DIR_DEFAULT = "data/targets/pca_zsem/d128"
PCA_ZSTRUCT_DIR_DEFAULT = "data/targets/pca_zstruct"
OUT_DEFAULT = "data/targets/targets_proxyset.h5"

N_SIMPLE = 150
N_COMPLEX = 150
ENCODE_BATCH_SIZE = 16

CLIP_CKPT = "openai/clip-vit-large-patch14"
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

DINO_HUB_REPO = "facebookresearch/dinov2"
DINO_HUB_MODEL = "dinov2_vitl14"
DINO_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
DINO_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
DINO_GRID_NATIVE = 16
DINO_GRID_POOLED = 8


def preprocess_batch(pil_images, size=224):
    arrs = [np.asarray(im.resize((size, size), Image.BICUBIC)).astype(np.float32) / 255.0
            for im in pil_images]
    return torch.from_numpy(np.stack(arrs)).permute(0, 3, 1, 2).contiguous()  # (B,3,size,size)


def pca_transform(raw, basis):
    """sklearn PCA.transform(), reimplemented from the frozen (components, mean,
    explained_variance, whiten) tuple that the PCA scripts save, see pca_width_zsem.py /
    fit_pca_zstruct.py. `raw` is (N, d_in); returns (N, d_out)."""
    centered = raw - basis["mean"][None, :]
    projected = centered @ basis["components"].T
    if basis["whiten"]:
        projected = projected / np.sqrt(np.clip(basis["explained_variance"], 1e-12, None))[None, :]
    return projected


class TargetEncoder:
    """CLIP ViT-L/14 image embedding -> P_sem -> L2(z_sem_img); DINOv2 ViT-L/14 patch tokens,
    pooled 8x8 -> P_struct -> z_struct. Exact formulas of extract_clip_image_raw.py +
    pca_width_zsem.py (z_sem) and extract_dinov2_pooled.py + fit_pca_zstruct.py (z_struct),
    run on a handful of images at a time instead of streamed over the full 73k pool."""

    def __init__(self, sem_basis_dir, struct_basis_dir, device, zsem_l2=True):
        self.device = device
        # zsem_l2=False is an ablation arm: the raw CLIP embedding is deliberately unnormalized
        # and P_sem is whitened; the final L2 throws the norm away. The default is True:
        # traversal_figure.py and the production targets.h5 both use the normalized definition.
        self.zsem_l2 = zsem_l2
        print(f"Loading {CLIP_CKPT} ...")
        self.clip = CLIPVisionModelWithProjection.from_pretrained(
            CLIP_CKPT, torch_dtype=torch.float32).to(device).eval()
        print(f"Loading {DINO_HUB_REPO}:{DINO_HUB_MODEL} ...")
        self.dino = torch.hub.load(DINO_HUB_REPO, DINO_HUB_MODEL).to(device).eval()
        self.sem_basis = load_pca_basis(sem_basis_dir)
        self.struct_basis = load_pca_basis(struct_basis_dir)

    @torch.no_grad()
    def encode(self, pil_images):
        """-> (z_sem_img (n,128) float32, z_struct (n,8,8,32) float32)."""
        n = len(pil_images)
        imgs = preprocess_batch(pil_images).to(self.device)

        clip_in = (imgs - CLIP_MEAN.to(self.device)) / CLIP_STD.to(self.device)
        clip_raw = self.clip(pixel_values=clip_in).image_embeds.cpu().numpy().astype(np.float32)  # (n,768)
        proj = pca_transform(clip_raw, self.sem_basis)
        z_sem_img = (l2_normalize(proj) if self.zsem_l2 else proj).astype(np.float32)  # (n,128)

        dino_in = (imgs - DINO_MEAN.to(self.device)) / DINO_STD.to(self.device)
        feats = self.dino.forward_features(dino_in)["x_norm_patchtokens"]  # (n,256,1024)
        feats = feats.reshape(n, DINO_GRID_NATIVE, DINO_GRID_NATIVE, -1).permute(0, 3, 1, 2)
        pooled = F.avg_pool2d(feats, kernel_size=2, stride=2).permute(0, 2, 3, 1).contiguous()  # (n,8,8,1024)
        dino_raw = pooled.cpu().numpy().astype(np.float32).reshape(n * DINO_GRID_POOLED ** 2, -1)
        z_struct = pca_transform(dino_raw, self.struct_basis).reshape(
            n, DINO_GRID_POOLED, DINO_GRID_POOLED, -1).astype(np.float32)
        return z_sem_img, z_struct


@torch.no_grad()
def encode_z_sem_txt(captions, sem_basis, device, zsem_l2=True):
    """z_sem^txt = L2(P_sem @ CLIP_text(caption)), the text block of assemble_targets.py, with
    one caption per image (nothing to average) and the SAME frozen image-fit P_sem basis."""
    print(f"Loading {CLIP_CKPT} text encoder ...")
    tokenizer = CLIPTokenizer.from_pretrained(CLIP_CKPT)
    model = CLIPTextModelWithProjection.from_pretrained(
        CLIP_CKPT, torch_dtype=torch.float32).to(device).eval()

    raw = np.zeros((len(captions), sem_basis["components"].shape[1]), dtype=np.float32)
    for start in range(0, len(captions), ENCODE_BATCH_SIZE):
        end = min(start + ENCODE_BATCH_SIZE, len(captions))
        tokens = tokenizer(captions[start:end], padding="max_length", truncation=True,
                           max_length=77, return_tensors="pt").to(device)
        raw[start:end] = model(**tokens).text_embeds.cpu().numpy().astype(np.float32)

    proj = pca_transform(raw, sem_basis)
    return (l2_normalize(proj) if zsem_l2 else proj).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--proxy-dir", default=PROXY_DIR_DEFAULT)
    ap.add_argument("--captions", default=CAPTIONS_DEFAULT)
    ap.add_argument("--pca-zsem-dir", default=PCA_ZSEM_DIR_DEFAULT)
    ap.add_argument("--pca-zstruct-dir", default=PCA_ZSTRUCT_DIR_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--no-zsem-l2", action="store_true",
                    help="ablation arm: drop the final L2 on both "
                         "z_sem^img and z_sem^txt, keeping the whitened PCA norm. Write it to a "
                         "SEPARATE --out; A_psi was trained on L2-normalized z_sem, so this arm "
                         "measures the L2's effect at fixed A_psi, not a retrained pipeline.")
    ap.add_argument("--txt-from-img", action="store_true",
                    help="smoke test only: copy z_sem^img into the text slot instead of reading "
                         "--captions (off-distribution for A_psi, recorded in the output attrs)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    images = load_proxy_images(args.proxy_dir)
    n = len(images)
    print(f"{n} proxy images ({N_SIMPLE} simple + {N_COMPLEX} complex) from {args.proxy_dir}")

    if args.txt_from_img:
        captions = None
        print("WARNING: --txt-from-img -- z_sem^txt is a copy of z_sem^img, NOT a real caption "
              "embedding. Fine for a smoke test, not for a T1/T2 arm.")
    else:
        captions = [str(c) for c in np.load(args.captions, allow_pickle=True)]
        assert len(captions) == n, f"{len(captions)} captions vs {n} images"
        print(f"Captions from {args.captions}")

    zsem_l2 = not args.no_zsem_l2
    if not zsem_l2:
        print("--no-zsem-l2: z_sem keeps its whitened-PCA norm (ablation arm)")
    encoder = TargetEncoder(args.pca_zsem_dir, args.pca_zstruct_dir, device, zsem_l2=zsem_l2)
    sem_d = encoder.sem_basis["components"].shape[0]
    struct_d = encoder.struct_basis["components"].shape[0]
    print(f"Frozen bases: P_sem d={sem_d} whiten={encoder.sem_basis['whiten']}, "
          f"P_struct d={struct_d} whiten={encoder.struct_basis['whiten']}")

    z_sem_img = np.zeros((n, sem_d), dtype=np.float32)
    z_struct = np.zeros((n, DINO_GRID_POOLED, DINO_GRID_POOLED, struct_d), dtype=np.float32)
    for start in range(0, n, ENCODE_BATCH_SIZE):
        end = min(start + ENCODE_BATCH_SIZE, n)
        z_sem_img[start:end], z_struct[start:end] = encoder.encode(images[start:end])
        print(f"  encoded {end}/{n}", flush=True)

    if args.txt_from_img:
        z_sem_txt = z_sem_img.copy()
    else:
        z_sem_txt = encode_z_sem_txt(captions, encoder.sem_basis, device, zsem_l2=zsem_l2)

    z_sem = np.stack([z_sem_img, z_sem_txt], axis=1)  # (n, 2, sem_d)
    print(f"z_sem: {z_sem.shape} | z_struct: {z_struct.shape}")
    print(f"z_sem^img norm mean {np.linalg.norm(z_sem_img, axis=-1).mean():.4f} "
          f"(expect {'1.0' if zsem_l2 else 'sqrt(d) ~ 11.3, unnormalized'}); "
          f"z_struct std {z_struct.std():.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with h5py.File(args.out, "w") as f:
        f.create_dataset("z_sem", data=z_sem, dtype="float16")
        f.create_dataset("z_struct", data=z_struct, dtype="float16")
        f.attrs["order"] = f"0-{N_SIMPLE - 1}=simple, {N_SIMPLE}-{n - 1}=complex"
        f.attrs["z_sem_order"] = "img, txt (axis 1)"
        f.attrs["z_sem_d"] = sem_d
        f.attrs["z_sem_txt_source"] = "img_copy_SMOKE_TEST" if args.txt_from_img else "proxy_captions"
        f.attrs["zsem_l2"] = zsem_l2
        f.attrs["z_struct_shared_pca"] = True
        f.attrs["z_struct_d"] = struct_d
        f.attrs["whiten"] = encoder.sem_basis["whiten"]
        f.attrs["pca_zsem_dir"] = args.pca_zsem_dir
        f.attrs["pca_zstruct_dir"] = args.pca_zstruct_dir

    print(f"Done. Wrote {args.out}")


if __name__ == "__main__":
    main()
